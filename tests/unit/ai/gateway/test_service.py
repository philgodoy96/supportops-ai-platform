"""Unit tests for application-owned LLM Gateway reliability behavior."""

from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass

import pytest

from supportops.ai.gateway.contracts import (
    LLMOperation,
    LLMProviderResponse,
    LLMRequest,
    LLMTokenUsage,
)
from supportops.ai.gateway.errors import (
    LLMAuthenticationError,
    LLMErrorCode,
    LLMIncompleteResponseError,
    LLMOutputValidationError,
    LLMProviderUnavailableError,
    LLMRefusalError,
    LLMTimeoutError,
)
from supportops.ai.gateway.results import (
    LLMGatewayFailure,
    LLMInvocationStatus,
)
from supportops.ai.gateway.service import LLMGateway
from supportops.ai.schemas.ticket_classification import (
    TICKET_CLASSIFICATION_SCHEMA_VERSION,
    TicketClassificationResult,
)

_MODEL = "test-model"
_PROVIDER = "test-provider"


class SequenceClock:
    """Deterministic clock returning a configured sequence."""

    def __init__(
        self,
        values: Iterable[float],
    ) -> None:
        self._values = iter(values)

    def __call__(self) -> float:
        return next(self._values)


@dataclass(frozen=True, slots=True)
class ProviderErrorOutcome:
    """Explicit provider failure used by the recording test provider."""

    error: Exception


class RecordingProvider:
    """Provider double recording every request in invocation order."""

    def __init__(
        self,
        outcomes: Iterable[LLMProviderResponse | ProviderErrorOutcome],
    ) -> None:
        self._outcomes = deque(outcomes)
        self.requests: list[LLMRequest] = []
        self.closed = False

    @property
    def provider_name(self) -> str:
        return _PROVIDER

    async def generate(
        self,
        request: LLMRequest,
    ) -> LLMProviderResponse:
        self.requests.append(request)

        if not self._outcomes:
            raise AssertionError(
                "No provider outcome remains.",
            )

        outcome = self._outcomes.popleft()

        if isinstance(outcome, ProviderErrorOutcome):
            raise outcome.error

        return outcome

    async def close(self) -> None:
        self.closed = True


def _request() -> LLMRequest:
    return LLMRequest(
        operation=LLMOperation.TICKET_CLASSIFICATION,
        model=_MODEL,
        instructions="Classify the supplied support ticket.",
        input=(
            "BEGIN_UNTRUSTED_TICKET_JSON\n"
            '{"description":"Invoice question"}\n'
            "END_UNTRUSTED_TICKET_JSON"
        ),
        output_schema=TicketClassificationResult,
        timeout_seconds=12,
        metadata={
            "prompt_id": "ticket-classification",
            "prompt_version": "1",
        },
    )


def _valid_output(
    *,
    category: str = "billing",
) -> dict[str, object]:
    return {
        "category": category,
        "intent": "ask_question",
        "urgency": "normal",
        "sentiment": "neutral",
        "requires_human_review": False,
        "summary": "The customer is asking about an invoice charge.",
        "schema_version": TICKET_CLASSIFICATION_SCHEMA_VERSION,
    }


def _response(
    parsed_output: dict[str, object],
    *,
    provider_request_id: str,
    usage: LLMTokenUsage | None = None,
    provider: str = _PROVIDER,
    model: str = _MODEL,
) -> LLMProviderResponse:
    return LLMProviderResponse(
        parsed_output=parsed_output,
        provider=provider,
        model=model,
        provider_request_id=provider_request_id,
        usage=usage,
        finish_reason="completed",
    )


async def test_returns_application_validated_output_and_trace() -> None:
    usage = LLMTokenUsage(
        input_tokens=100,
        output_tokens=20,
        total_tokens=120,
    )
    provider = RecordingProvider(
        (
            _response(
                _valid_output(),
                provider_request_id="request-1",
                usage=usage,
            ),
        ),
    )
    gateway = LLMGateway(
        provider=provider,
        max_repair_attempts=1,
        clock=SequenceClock(
            (
                10.000,
                10.125,
            ),
        ),
    )

    result = await gateway.generate(_request())

    assert isinstance(
        result.output,
        TicketClassificationResult,
    )
    assert result.output.category.value == "billing"
    assert result.accepted_invocation_sequence == 1
    assert len(result.invocations) == 1

    invocation = result.invocations[0]

    assert invocation.invocation_sequence == 1
    assert invocation.status is LLMInvocationStatus.SUCCEEDED
    assert invocation.provider == _PROVIDER
    assert invocation.model == _MODEL
    assert invocation.provider_request_id == "request-1"
    assert invocation.usage == usage
    assert invocation.latency_ms == 125
    assert invocation.error_code is None


async def test_invalid_output_is_repaired_once() -> None:
    provider = RecordingProvider(
        (
            _response(
                {
                    "category": "invented_category",
                },
                provider_request_id="request-1",
            ),
            _response(
                _valid_output(),
                provider_request_id="request-2",
            ),
        ),
    )
    gateway = LLMGateway(
        provider=provider,
        max_repair_attempts=1,
        clock=SequenceClock(
            (
                1.000,
                1.100,
                2.000,
                2.250,
            ),
        ),
    )

    result = await gateway.generate(_request())

    assert len(provider.requests) == 2
    assert len(result.invocations) == 2
    assert result.invocations[0].status is LLMInvocationStatus.VALIDATION_FAILED
    assert result.invocations[1].status is LLMInvocationStatus.SUCCEEDED
    assert result.accepted_invocation_sequence == 2


async def test_repair_preserves_task_contract_and_schema() -> None:
    provider = RecordingProvider(
        (
            _response(
                {
                    "category": "invalid",
                },
                provider_request_id="request-1",
            ),
            _response(
                _valid_output(),
                provider_request_id="request-2",
            ),
        ),
    )
    gateway = LLMGateway(
        provider=provider,
        max_repair_attempts=1,
    )
    original_request = _request()

    await gateway.generate(original_request)

    initial_request = provider.requests[0]
    repair_request = provider.requests[1]

    assert repair_request.instructions == original_request.instructions
    assert repair_request.output_schema is original_request.output_schema
    assert repair_request.operation is original_request.operation
    assert repair_request.model == original_request.model
    assert repair_request.timeout_seconds == original_request.timeout_seconds
    assert original_request.input in repair_request.input
    assert "BEGIN_APPLICATION_REPAIR_INSTRUCTIONS" in repair_request.input
    assert repair_request.metadata["prompt_id"] == (initial_request.metadata["prompt_id"])
    assert repair_request.metadata["prompt_version"] == "1"
    assert repair_request.metadata["supportops_invocation_kind"] == "repair"
    assert repair_request.metadata["supportops_repair_attempt"] == "1"


async def test_repair_feedback_excludes_invalid_field_value() -> None:
    sensitive_invalid_value = "do-not-repeat-this-value"
    provider = RecordingProvider(
        (
            _response(
                {
                    **_valid_output(),
                    "category": sensitive_invalid_value,
                },
                provider_request_id="request-1",
            ),
            _response(
                _valid_output(),
                provider_request_id="request-2",
            ),
        ),
    )
    gateway = LLMGateway(
        provider=provider,
        max_repair_attempts=1,
    )

    await gateway.generate(_request())

    repair_request = provider.requests[1]

    assert sensitive_invalid_value not in repair_request.input
    assert '"location":"category"' in repair_request.input
    assert '"type":"enum"' in repair_request.input


async def test_repair_exhaustion_preserves_both_invocations() -> None:
    provider = RecordingProvider(
        (
            _response(
                {
                    "category": "invalid-one",
                },
                provider_request_id="request-1",
            ),
            _response(
                {
                    "category": "invalid-two",
                },
                provider_request_id="request-2",
            ),
        ),
    )
    gateway = LLMGateway(
        provider=provider,
        max_repair_attempts=1,
    )

    with pytest.raises(LLMGatewayFailure) as captured:
        await gateway.generate(_request())

    failure = captured.value

    assert isinstance(
        failure.error,
        LLMOutputValidationError,
    )
    assert failure.terminal is True
    assert failure.retryable is False
    assert len(failure.invocations) == 2
    assert all(
        invocation.status is LLMInvocationStatus.VALIDATION_FAILED
        for invocation in failure.invocations
    )


async def test_zero_repair_budget_fails_after_initial_validation() -> None:
    provider = RecordingProvider(
        (
            _response(
                {
                    "category": "invalid",
                },
                provider_request_id="request-1",
            ),
        ),
    )
    gateway = LLMGateway(
        provider=provider,
        max_repair_attempts=0,
    )

    with pytest.raises(LLMGatewayFailure) as captured:
        await gateway.generate(_request())

    assert len(provider.requests) == 1
    assert len(captured.value.invocations) == 1


async def test_incomplete_response_is_repaired_once() -> None:
    provider = RecordingProvider(
        (
            ProviderErrorOutcome(
                LLMIncompleteResponseError(
                    provider_request_id="request-1",
                ),
            ),
            _response(
                _valid_output(),
                provider_request_id="request-2",
            ),
        ),
    )
    gateway = LLMGateway(
        provider=provider,
        max_repair_attempts=1,
    )

    result = await gateway.generate(_request())

    assert len(provider.requests) == 2
    assert result.invocations[0].status is LLMInvocationStatus.INCOMPLETE
    assert result.invocations[1].status is LLMInvocationStatus.SUCCEEDED


async def test_refusal_is_never_repaired() -> None:
    provider = RecordingProvider(
        (
            ProviderErrorOutcome(
                LLMRefusalError(
                    provider_request_id="request-1",
                ),
            ),
            _response(
                _valid_output(),
                provider_request_id="request-2",
            ),
        ),
    )
    gateway = LLMGateway(
        provider=provider,
        max_repair_attempts=1,
    )

    with pytest.raises(LLMGatewayFailure) as captured:
        await gateway.generate(_request())

    assert len(provider.requests) == 1
    assert captured.value.terminal is True
    assert captured.value.repairable is False
    assert captured.value.invocations[0].status is LLMInvocationStatus.REFUSED


@pytest.mark.parametrize(
    "provider_error",
    [
        LLMTimeoutError(
            provider_request_id="request-1",
        ),
        LLMProviderUnavailableError(
            provider_request_id="request-1",
        ),
        LLMAuthenticationError(
            provider_request_id="request-1",
        ),
    ],
)
async def test_operational_provider_failures_are_not_repaired(
    provider_error: Exception,
) -> None:
    provider = RecordingProvider(
        (
            ProviderErrorOutcome(provider_error),
            _response(
                _valid_output(),
                provider_request_id="request-2",
            ),
        ),
    )
    gateway = LLMGateway(
        provider=provider,
        max_repair_attempts=1,
    )

    with pytest.raises(LLMGatewayFailure):
        await gateway.generate(_request())

    assert len(provider.requests) == 1


async def test_timeout_trace_is_marked_timed_out() -> None:
    provider = RecordingProvider(
        (
            ProviderErrorOutcome(
                LLMTimeoutError(
                    provider_request_id="request-1",
                ),
            ),
        ),
    )
    gateway = LLMGateway(
        provider=provider,
        max_repair_attempts=1,
    )

    with pytest.raises(LLMGatewayFailure) as captured:
        await gateway.generate(_request())

    invocation = captured.value.invocations[0]

    assert invocation.status is LLMInvocationStatus.TIMED_OUT
    assert invocation.error_code is LLMErrorCode.TIMEOUT
    assert invocation.provider_request_id == "request-1"


async def test_gateway_does_not_fallback_to_another_provider() -> None:
    provider = RecordingProvider(
        (
            ProviderErrorOutcome(
                LLMProviderUnavailableError(
                    provider_request_id="request-1",
                ),
            ),
        ),
    )
    gateway = LLMGateway(
        provider=provider,
        max_repair_attempts=1,
    )

    with pytest.raises(LLMGatewayFailure):
        await gateway.generate(_request())

    assert len(provider.requests) == 1
    assert provider.provider_name == _PROVIDER


async def test_rejects_provider_provenance_mismatch() -> None:
    provider = RecordingProvider(
        (
            _response(
                _valid_output(),
                provider_request_id="request-1",
                provider="different-provider",
            ),
        ),
    )
    gateway = LLMGateway(
        provider=provider,
        max_repair_attempts=1,
    )

    with pytest.raises(
        RuntimeError,
        match="provenance does not match",
    ):
        await gateway.generate(_request())


async def test_rejects_provider_model_mismatch() -> None:
    provider = RecordingProvider(
        (
            _response(
                _valid_output(),
                provider_request_id="request-1",
                model="different-model",
            ),
        ),
    )
    gateway = LLMGateway(
        provider=provider,
        max_repair_attempts=1,
    )

    with pytest.raises(
        RuntimeError,
        match="model does not match",
    ):
        await gateway.generate(_request())


def test_rejects_unbounded_repair_configuration() -> None:
    provider = RecordingProvider(())

    with pytest.raises(
        ValueError,
        match="must not exceed 1",
    ):
        LLMGateway(
            provider=provider,
            max_repair_attempts=2,
        )


def test_rejects_negative_repair_configuration() -> None:
    provider = RecordingProvider(())

    with pytest.raises(
        ValueError,
        match="must be non-negative",
    ):
        LLMGateway(
            provider=provider,
            max_repair_attempts=-1,
        )
