"""Application-owned LLM Gateway with bounded validation repair."""

import json
import time
from collections.abc import Callable
from dataclasses import replace

from pydantic import ValidationError

from supportops.ai.gateway.contracts import (
    LLMProvider,
    LLMProviderResponse,
    LLMRequest,
)
from supportops.ai.gateway.errors import (
    LLMError,
    LLMIncompleteResponseError,
    LLMOutputValidationError,
    LLMRefusalError,
    LLMTimeoutError,
)
from supportops.ai.gateway.results import (
    LLMGatewayFailure,
    LLMGatewayResult,
    LLMInvocationStatus,
    LLMInvocationTrace,
)

Clock = Callable[[], float]

_REPAIR_INVOCATION_KIND_METADATA_KEY = "supportops_invocation_kind"
_REPAIR_ATTEMPT_METADATA_KEY = "supportops_repair_attempt"


class LLMGateway:
    """Own validation, repair policy, and provider-independent traces."""

    def __init__(
        self,
        *,
        provider: LLMProvider,
        max_repair_attempts: int,
        clock: Clock = time.perf_counter,
    ) -> None:
        if max_repair_attempts < 0:
            raise ValueError(
                "max_repair_attempts must be non-negative.",
            )

        if max_repair_attempts > 1:
            raise ValueError(
                "max_repair_attempts must not exceed 1.",
            )

        self._provider = provider
        self._max_repair_attempts = max_repair_attempts
        self._clock = clock

    async def generate(
        self,
        request: LLMRequest,
    ) -> LLMGatewayResult:
        """Generate and validate output with bounded repair attempts."""

        invocations: list[LLMInvocationTrace] = []
        current_request = request

        for repair_attempt in range(
            self._max_repair_attempts + 1,
        ):
            invocation_sequence = repair_attempt + 1
            started_at = self._clock()

            try:
                response = await self._provider.generate(
                    current_request,
                )
            except LLMError as error:
                latency_ms = _elapsed_milliseconds(
                    started_at,
                    self._clock(),
                )
                invocations.append(
                    _trace_from_error(
                        request=current_request,
                        provider_name=self._provider.provider_name,
                        invocation_sequence=invocation_sequence,
                        latency_ms=latency_ms,
                        error=error,
                    ),
                )

                if error.repairable and repair_attempt < self._max_repair_attempts:
                    current_request = _build_repair_request(
                        original_request=request,
                        repair_attempt=repair_attempt + 1,
                        validation_feedback=error.safe_summary,
                    )
                    continue

                raise LLMGatewayFailure(
                    error=error,
                    invocations=tuple(invocations),
                ) from error

            latency_ms = _elapsed_milliseconds(
                started_at,
                self._clock(),
            )

            _validate_response_provenance(
                request=current_request,
                expected_provider=self._provider.provider_name,
                response=response,
            )

            try:
                validated_output = current_request.output_schema.model_validate(
                    response.parsed_output,
                )
            except ValidationError as validation_error:
                normalized_error = LLMOutputValidationError(
                    provider_request_id=(response.provider_request_id),
                )
                invocations.append(
                    LLMInvocationTrace(
                        invocation_sequence=invocation_sequence,
                        status=(LLMInvocationStatus.VALIDATION_FAILED),
                        provider=response.provider,
                        model=response.model,
                        provider_request_id=(response.provider_request_id),
                        usage=response.usage,
                        latency_ms=latency_ms,
                        error_code=normalized_error.error_code,
                    ),
                )

                if repair_attempt < self._max_repair_attempts:
                    current_request = _build_repair_request(
                        original_request=request,
                        repair_attempt=repair_attempt + 1,
                        validation_feedback=(
                            _safe_validation_feedback(
                                validation_error,
                            )
                        ),
                    )
                    continue

                raise LLMGatewayFailure(
                    error=normalized_error,
                    invocations=tuple(invocations),
                ) from validation_error

            invocations.append(
                LLMInvocationTrace(
                    invocation_sequence=invocation_sequence,
                    status=LLMInvocationStatus.SUCCEEDED,
                    provider=response.provider,
                    model=response.model,
                    provider_request_id=(response.provider_request_id),
                    usage=response.usage,
                    latency_ms=latency_ms,
                    error_code=None,
                ),
            )

            return LLMGatewayResult(
                output=validated_output,
                invocations=tuple(invocations),
                accepted_invocation_sequence=(invocation_sequence),
            )

        raise RuntimeError(
            "LLM Gateway exhausted an unreachable execution path.",
        )


def _build_repair_request(
    *,
    original_request: LLMRequest,
    repair_attempt: int,
    validation_feedback: str,
) -> LLMRequest:
    repair_metadata = {
        **original_request.metadata,
        _REPAIR_INVOCATION_KIND_METADATA_KEY: "repair",
        _REPAIR_ATTEMPT_METADATA_KEY: str(repair_attempt),
    }

    repair_input = "\n\n".join(
        (
            original_request.input,
            (
                "BEGIN_APPLICATION_REPAIR_INSTRUCTIONS\n"
                "The previous structured response was rejected by "
                "application validation.\n"
                f"Safe validation feedback: {validation_feedback}\n"
                "Produce a complete replacement response using the "
                "same task, taxonomy, and structured output schema.\n"
                "Do not explain the correction and do not include "
                "additional fields.\n"
                "END_APPLICATION_REPAIR_INSTRUCTIONS"
            ),
        ),
    )

    return replace(
        original_request,
        input=repair_input,
        metadata=repair_metadata,
    )


def _safe_validation_feedback(
    error: ValidationError,
) -> str:
    violations = []

    for issue in error.errors(
        include_input=False,
        include_url=False,
    ):
        location = ".".join(str(location_part) for location_part in issue["loc"])

        violations.append(
            {
                "location": location or "$",
                "type": str(issue["type"]),
            },
        )

    return json.dumps(
        {
            "validation_errors": violations,
        },
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def _trace_from_error(
    *,
    request: LLMRequest,
    provider_name: str,
    invocation_sequence: int,
    latency_ms: int,
    error: LLMError,
) -> LLMInvocationTrace:
    return LLMInvocationTrace(
        invocation_sequence=invocation_sequence,
        status=_status_from_error(error),
        provider=provider_name,
        model=request.model,
        provider_request_id=error.provider_request_id,
        usage=None,
        latency_ms=latency_ms,
        error_code=error.error_code,
    )


def _status_from_error(
    error: LLMError,
) -> LLMInvocationStatus:
    if isinstance(error, LLMTimeoutError):
        return LLMInvocationStatus.TIMED_OUT

    if isinstance(error, LLMRefusalError):
        return LLMInvocationStatus.REFUSED

    if isinstance(error, LLMIncompleteResponseError):
        return LLMInvocationStatus.INCOMPLETE

    if isinstance(error, LLMOutputValidationError):
        return LLMInvocationStatus.VALIDATION_FAILED

    return LLMInvocationStatus.PROVIDER_FAILED


def _validate_response_provenance(
    *,
    request: LLMRequest,
    expected_provider: str,
    response: LLMProviderResponse,
) -> None:
    if response.provider != expected_provider:
        raise RuntimeError(
            "LLM provider response provenance does not match the configured provider.",
        )

    if response.model != request.model:
        raise RuntimeError(
            "LLM provider response model does not match the requested model.",
        )


def _elapsed_milliseconds(
    started_at: float,
    completed_at: float,
) -> int:
    elapsed_seconds = completed_at - started_at

    if elapsed_seconds < 0:
        raise RuntimeError(
            "The LLM Gateway clock moved backwards.",
        )

    return round(elapsed_seconds * 1_000)
