"""Application-owned LLM Gateway with bounded validation repair."""

import json
import time
from collections.abc import Callable, Mapping
from contextlib import AbstractContextManager
from dataclasses import replace
from types import TracebackType
from typing import Literal

from pydantic import ValidationError

from supportops.ai.gateway.contracts import (
    LLMProvider,
    LLMProviderResponse,
    LLMRequest,
    LLMTokenUsage,
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
from supportops.ai.pricing.catalog import DEFAULT_PRICING_CATALOG
from supportops.ai.pricing.estimation import estimate_llm_cost
from supportops.observability.contracts import (
    ObservabilityClient,
    ObservationScope,
)
from supportops.observability.models import (
    CostDetails,
    JsonValue,
    ObservationAttributes,
    ObservationStatus,
    ObservationType,
    ObservationUpdate,
    PricingStatus,
    UsageDetails,
)
from supportops.observability.noop import NoOpObservabilityClient

Clock = Callable[[], float]

_REPAIR_INVOCATION_KIND_METADATA_KEY = "supportops_invocation_kind"
_REPAIR_ATTEMPT_METADATA_KEY = "supportops_repair_attempt"

_GENERATION_OBSERVATION_NAME = "llm.generate"
_UNHANDLED_BUSINESS_ERROR_CODE = "unhandled_business_error"

_REQUEST_METADATA_TO_OBSERVATION: Mapping[str, str] = {
    "supportops_prompt_id": "prompt_id",
    "supportops_prompt_version": "prompt_version",
    "supportops_prompt_content_hash": "prompt_hash",
    "supportops_schema_version": "schema_version",
    "supportops_agent_run_id": "agent_run_id",
    "supportops_agent_run_attempt_id": "agent_run_attempt_id",
    "supportops_workspace_id": "workspace_id",
    "supportops_correlation_id": "correlation_id",
}

_GENERATION_METADATA_PATHS = frozenset(
    {
        ("operation",),
        ("invocation_sequence",),
        ("is_repair",),
        ("provider",),
        ("model",),
        ("prompt_id",),
        ("prompt_version",),
        ("prompt_hash",),
        ("schema_version",),
        ("agent_run_id",),
        ("agent_run_attempt_id",),
        ("workspace_id",),
        ("correlation_id",),
        ("provider_request_id",),
        ("latency_ms",),
        ("pricing_catalog_version",),
        ("pricing_found",),
    },
)


class LLMGateway:
    """Own validation, repair policy, and provider-independent traces."""

    def __init__(
        self,
        *,
        provider: LLMProvider,
        max_repair_attempts: int,
        clock: Clock = time.perf_counter,
        observability_client: ObservabilityClient | None = None,
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
        self._observability_client = (
            observability_client if observability_client is not None else NoOpObservabilityClient()
        )

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
            is_repair = invocation_sequence > 1
            started_at = self._clock()

            with _FailOpenGenerationObservation(
                client=self._observability_client,
                attributes=_build_generation_attributes(
                    request=current_request,
                    provider_name=self._provider.provider_name,
                    invocation_sequence=invocation_sequence,
                    is_repair=is_repair,
                ),
            ) as generation:
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
                    generation.complete(
                        _build_error_update(
                            error_code=error.error_code.value,
                            provider_request_id=(error.provider_request_id),
                            latency_ms=latency_ms,
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
                    generation.complete(
                        _build_completion_update(
                            status=ObservationStatus.ERROR,
                            error_code=normalized_error.error_code.value,
                            provider=response.provider,
                            model=response.model,
                            provider_request_id=(response.provider_request_id),
                            usage=response.usage,
                            latency_ms=latency_ms,
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
                generation.complete(
                    _build_completion_update(
                        status=ObservationStatus.OK,
                        error_code=None,
                        provider=response.provider,
                        model=response.model,
                        provider_request_id=(response.provider_request_id),
                        usage=response.usage,
                        latency_ms=latency_ms,
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


class _FailOpenGenerationObservation:
    """Gateway-owned fail-open boundary around one generation observation."""

    def __init__(
        self,
        *,
        client: ObservabilityClient,
        attributes: ObservationAttributes,
    ) -> None:
        self._client = client
        self._attributes = attributes
        self._manager: AbstractContextManager[ObservationScope] | None = None
        self._scope: ObservationScope | None = None
        self._completed = False

    def __enter__(self) -> "_FailOpenGenerationObservation":
        try:
            self._manager = self._client.start_observation(
                self._attributes,
            )
            self._scope = self._manager.__enter__()
        except Exception:
            self._manager = None
            self._scope = None

        return self

    def complete(self, update: ObservationUpdate) -> None:
        if self._scope is None or self._completed:
            return

        try:
            self._scope.update(update)
            self._completed = True
        except Exception:
            return

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> Literal[False]:
        del exc, traceback

        if self._manager is None:
            return False

        try:
            if exc_type is not None and not self._completed:
                self.complete(
                    ObservationUpdate(
                        status=ObservationStatus.ERROR,
                        error_code=_UNHANDLED_BUSINESS_ERROR_CODE,
                    ),
                )

            self._manager.__exit__(None, None, None)
        except Exception:
            return False

        return False


def _build_generation_attributes(
    *,
    request: LLMRequest,
    provider_name: str,
    invocation_sequence: int,
    is_repair: bool,
) -> ObservationAttributes:
    metadata: dict[str, JsonValue] = {
        "operation": request.operation.value,
        "invocation_sequence": invocation_sequence,
        "is_repair": is_repair,
        "provider": provider_name,
        "model": request.model,
    }

    for source_key, target_key in _REQUEST_METADATA_TO_OBSERVATION.items():
        value = request.metadata.get(source_key)
        if value is not None:
            metadata[target_key] = value

    return ObservationAttributes(
        name=_GENERATION_OBSERVATION_NAME,
        observation_type=ObservationType.GENERATION,
        metadata=metadata,
        metadata_paths=_GENERATION_METADATA_PATHS,
        input_paths=frozenset(),
        output_paths=frozenset(),
        provider=provider_name,
        model=request.model,
    )


def _build_error_update(
    *,
    error_code: str,
    provider_request_id: str | None,
    latency_ms: int,
) -> ObservationUpdate:
    metadata: dict[str, JsonValue] = {
        "latency_ms": latency_ms,
    }

    if provider_request_id is not None:
        metadata["provider_request_id"] = provider_request_id

    return ObservationUpdate(
        status=ObservationStatus.ERROR,
        metadata=metadata,
        error_code=error_code,
    )


def _build_completion_update(
    *,
    status: ObservationStatus,
    error_code: str | None,
    provider: str,
    model: str,
    provider_request_id: str | None,
    usage: LLMTokenUsage | None,
    latency_ms: int,
) -> ObservationUpdate:
    metadata: dict[str, JsonValue] = {
        "latency_ms": latency_ms,
    }

    if provider_request_id is not None:
        metadata["provider_request_id"] = provider_request_id

    usage_details, cost_details, pricing_metadata = _map_usage_and_cost(
        provider=provider,
        model=model,
        usage=usage,
    )
    metadata.update(pricing_metadata)

    return ObservationUpdate(
        status=status,
        metadata=metadata,
        usage=usage_details,
        cost=cost_details,
        error_code=error_code,
    )


def _map_usage_and_cost(
    *,
    provider: str,
    model: str,
    usage: LLMTokenUsage | None,
) -> tuple[
    UsageDetails | None,
    CostDetails | None,
    dict[str, JsonValue],
]:
    if usage is None:
        return (
            None,
            None,
            {},
        )

    usage_details = _to_usage_details(usage)

    if usage_details is None:
        return (
            None,
            None,
            {},
        )

    cost_estimate = estimate_llm_cost(
        provider=provider,
        model=model,
        usage=usage,
        catalog=DEFAULT_PRICING_CATALOG,
    )
    pricing_metadata: dict[str, JsonValue] = {
        "pricing_catalog_version": (cost_estimate.pricing_catalog_version),
        "pricing_found": cost_estimate.pricing_found,
    }

    if not cost_estimate.pricing_found:
        return (
            usage_details,
            CostDetails(
                pricing_status=PricingStatus.UNKNOWN,
                pricing_catalog_version=(cost_estimate.pricing_catalog_version),
            ),
            pricing_metadata,
        )

    return (
        usage_details,
        CostDetails(
            pricing_status=PricingStatus.KNOWN,
            input_cost=cost_estimate.estimated_input_cost_usd,
            cached_input_cost=(cost_estimate.estimated_cached_input_cost_usd),
            output_cost=cost_estimate.estimated_output_cost_usd,
            total_cost=None,
            pricing_catalog_version=(cost_estimate.pricing_catalog_version),
        ),
        pricing_metadata,
    )


def _to_usage_details(
    usage: LLMTokenUsage,
) -> UsageDetails | None:
    try:
        input_tokens = usage.input_tokens
        cached_input_tokens = usage.cached_input_tokens
        output_tokens = usage.output_tokens
        reasoning_tokens = usage.reasoning_tokens

        if (
            cached_input_tokens is not None
            and input_tokens is not None
            and cached_input_tokens > input_tokens
        ):
            return None

        if (
            reasoning_tokens is not None
            and output_tokens is not None
            and reasoning_tokens > output_tokens
        ):
            return None

        mapped_input_tokens = input_tokens

        if cached_input_tokens is not None and input_tokens is not None:
            mapped_input_tokens = input_tokens - cached_input_tokens

        mapped_output_tokens = output_tokens

        if reasoning_tokens is not None and output_tokens is not None:
            mapped_output_tokens = output_tokens - reasoning_tokens

        component_values = tuple(
            value
            for value in (
                mapped_input_tokens,
                cached_input_tokens,
                mapped_output_tokens,
                reasoning_tokens,
            )
            if value is not None
        )

        return UsageDetails(
            input_tokens=mapped_input_tokens,
            cached_input_tokens=cached_input_tokens,
            output_tokens=mapped_output_tokens,
            reasoning_tokens=reasoning_tokens,
            total_tokens=None if component_values else usage.total_tokens,
        )
    except Exception:
        return None


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
