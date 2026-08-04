"""Unit tests for application-owned LLM Gateway reliability behavior."""

from collections import deque
from collections.abc import Iterable
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from decimal import Decimal
from types import TracebackType
from typing import Any, Literal, cast

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
from supportops.ai.pricing.catalog import PRICING_CATALOG_VERSION
from supportops.ai.schemas.ticket_classification import (
    TICKET_CLASSIFICATION_SCHEMA_VERSION,
    TicketClassificationResult,
)
from supportops.observability.contracts import TraceScope
from supportops.observability.models import (
    ObservabilityProvider,
    ObservationAttributes,
    ObservationStatus,
    ObservationType,
    ObservationUpdate,
    PricingStatus,
    TraceAttributes,
)

_MODEL = "test-model"
_PROVIDER = "test-provider"
_MOCK_MODEL = "mock-ticket-classifier-v1"
_MOCK_PROVIDER = "mock"


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
        *,
        provider_name: str = _PROVIDER,
    ) -> None:
        self._outcomes = deque(outcomes)
        self._provider_name = provider_name
        self.requests: list[LLMRequest] = []
        self.closed = False

    @property
    def provider_name(self) -> str:
        return self._provider_name

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


@dataclass
class RecordingObservationScope:
    """Observation scope that records updates."""

    attributes: ObservationAttributes
    updates: list[ObservationUpdate] = field(default_factory=list)
    update_error: Exception | None = None
    observation_id: str | None = "observation-1"

    def update(self, update: ObservationUpdate) -> None:
        if self.update_error is not None:
            raise self.update_error
        self.updates.append(update)

    def start_observation(
        self,
        attributes: ObservationAttributes,
    ) -> AbstractContextManager["RecordingObservationScope"]:
        del attributes
        raise AssertionError("Nested observations are not expected.")

    def record_event(self, event: object) -> None:
        del event


class RecordingObservationManager(AbstractContextManager[RecordingObservationScope]):
    """Context manager for one recorded observation."""

    def __init__(
        self,
        *,
        scope: RecordingObservationScope,
        exit_error: Exception | None = None,
    ) -> None:
        self._scope = scope
        self._exit_error = exit_error
        self.exit_args: (
            tuple[
                type[BaseException] | None,
                BaseException | None,
                TracebackType | None,
            ]
            | None
        ) = None

    def __enter__(self) -> RecordingObservationScope:
        return self._scope

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> Literal[False]:
        self.exit_args = (exc_type, exc, traceback)
        if self._exit_error is not None:
            raise self._exit_error
        return False


class RecordingObservabilityClient:
    """Observability double satisfying ObservabilityClient for gateway tests."""

    def __init__(self) -> None:
        self.started_attributes: list[ObservationAttributes] = []
        self.scopes: list[RecordingObservationScope] = []
        self.managers: list[RecordingObservationManager] = []
        self.start_error: Exception | None = None
        self.update_error: Exception | None = None
        self.exit_error: Exception | None = None
        self.enabled = True
        self.shutdown_calls = 0

    @property
    def provider(self) -> ObservabilityProvider:
        return ObservabilityProvider.NOOP

    def start_trace(
        self,
        attributes: TraceAttributes,
    ) -> AbstractContextManager[TraceScope]:
        del attributes
        raise AssertionError("Gateway must not start traces.")

    def start_observation(
        self,
        attributes: ObservationAttributes,
    ) -> AbstractContextManager[RecordingObservationScope]:
        if self.start_error is not None:
            raise self.start_error

        self.started_attributes.append(attributes)
        scope = RecordingObservationScope(
            attributes=attributes,
            update_error=self.update_error,
        )
        manager = RecordingObservationManager(
            scope=scope,
            exit_error=self.exit_error,
        )
        self.scopes.append(scope)
        self.managers.append(manager)
        return manager

    def record_event(self, event: object) -> None:
        del event

    def flush(self) -> None:
        return None

    def shutdown(self) -> None:
        self.shutdown_calls += 1


def _request(
    *,
    model: str = _MODEL,
    metadata: dict[str, str] | None = None,
) -> LLMRequest:
    return LLMRequest(
        operation=LLMOperation.TICKET_CLASSIFICATION,
        model=model,
        instructions="Classify the supplied support ticket.",
        input=(
            "BEGIN_UNTRUSTED_TICKET_JSON\n"
            '{"description":"Invoice question"}\n'
            "END_UNTRUSTED_TICKET_JSON"
        ),
        output_schema=TicketClassificationResult,
        timeout_seconds=12,
        metadata=metadata
        or {
            "prompt_id": "ticket-classification",
            "prompt_version": "1",
        },
    )


def _observability_metadata() -> dict[str, str]:
    return {
        "supportops_workspace_id": "workspace-1",
        "supportops_agent_run_id": "agent-run-1",
        "supportops_agent_run_attempt_id": "attempt-1",
        "supportops_correlation_id": "correlation-1",
        "supportops_prompt_id": "ticket-classification",
        "supportops_prompt_version": "1",
        "supportops_prompt_content_hash": "hash-1",
        "supportops_schema_version": (TICKET_CLASSIFICATION_SCHEMA_VERSION),
        "llm_invocation_id": "should-not-export",
        "execution_request_id": "should-not-export",
    }


def _observability_request(
    *,
    model: str = _MODEL,
) -> LLMRequest:
    return _request(
        model=model,
        metadata=_observability_metadata(),
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


def _assert_observation_is_content_free(
    attributes: ObservationAttributes,
    updates: Iterable[ObservationUpdate],
) -> None:
    assert attributes.input_paths == frozenset()
    assert attributes.output_paths == frozenset()
    assert attributes.input_data is None

    serialized = repr(attributes.metadata)
    for update in updates:
        serialized += repr(update.metadata)
        serialized += repr(update.output_data)
        serialized += repr(update.status_message)

    assert "BEGIN_UNTRUSTED_TICKET_JSON" not in serialized
    assert "Invoice question" not in serialized
    assert "Classify the supplied support ticket." not in serialized
    assert "llm_invocation_id" not in attributes.metadata
    assert "execution_request_id" not in attributes.metadata
    assert "should-not-export" not in serialized


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


async def test_successful_provider_request_creates_one_generation() -> None:
    usage = LLMTokenUsage(
        input_tokens=120,
        cached_input_tokens=20,
        output_tokens=30,
        reasoning_tokens=10,
        total_tokens=150,
    )
    observability = RecordingObservabilityClient()
    provider = RecordingProvider(
        (
            _response(
                _valid_output(),
                provider_request_id="request-1",
                usage=usage,
                provider=_MOCK_PROVIDER,
                model=_MOCK_MODEL,
            ),
        ),
        provider_name=_MOCK_PROVIDER,
    )
    gateway = LLMGateway(
        provider=provider,
        max_repair_attempts=1,
        observability_client=cast(Any, observability),
        clock=SequenceClock((1.0, 1.05)),
    )

    result = await gateway.generate(
        _observability_request(model=_MOCK_MODEL),
    )

    assert len(result.invocations) == 1
    assert len(observability.scopes) == 1

    attributes = observability.started_attributes[0]
    update = observability.scopes[0].updates[0]

    assert attributes.name == "llm.generate"
    assert attributes.observation_type is ObservationType.GENERATION
    assert attributes.provider == _MOCK_PROVIDER
    assert attributes.model == _MOCK_MODEL
    assert attributes.metadata["operation"] == (LLMOperation.TICKET_CLASSIFICATION.value)
    assert attributes.metadata["invocation_sequence"] == 1
    assert attributes.metadata["is_repair"] is False
    assert attributes.metadata["provider"] == _MOCK_PROVIDER
    assert attributes.metadata["model"] == _MOCK_MODEL
    assert attributes.metadata["prompt_id"] == "ticket-classification"
    assert attributes.metadata["prompt_version"] == "1"
    assert attributes.metadata["prompt_hash"] == "hash-1"
    assert attributes.metadata["schema_version"] == (TICKET_CLASSIFICATION_SCHEMA_VERSION)
    assert attributes.metadata["agent_run_id"] == "agent-run-1"
    assert attributes.metadata["workspace_id"] == "workspace-1"
    assert attributes.metadata["correlation_id"] == "correlation-1"
    assert attributes.input_paths == frozenset()
    assert attributes.output_paths == frozenset()

    assert update.status is ObservationStatus.OK
    assert update.error_code is None
    assert update.usage is not None
    assert update.usage.input_tokens == 100
    assert update.usage.cached_input_tokens == 20
    assert update.usage.output_tokens == 20
    assert update.usage.reasoning_tokens == 10
    assert update.usage.total_tokens is None
    assert update.cost is not None
    assert update.cost.pricing_status is PricingStatus.KNOWN
    assert update.cost.input_cost == Decimal("0")
    assert update.cost.cached_input_cost == Decimal("0")
    assert update.cost.output_cost == Decimal("0")
    assert update.cost.total_cost is None
    assert update.metadata["provider_request_id"] == "request-1"
    assert update.metadata["latency_ms"] == 50
    assert update.metadata["pricing_found"] is True
    assert update.metadata["pricing_catalog_version"] == (PRICING_CATALOG_VERSION)
    _assert_observation_is_content_free(attributes, observability.scopes[0].updates)
    assert observability.managers[0].exit_args == (None, None, None)


async def test_unknown_pricing_omits_cost_values() -> None:
    usage = LLMTokenUsage(
        input_tokens=10,
        cached_input_tokens=0,
        output_tokens=5,
        total_tokens=15,
    )
    observability = RecordingObservabilityClient()
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
        max_repair_attempts=0,
        observability_client=cast(Any, observability),
    )

    await gateway.generate(_observability_request())

    update = observability.scopes[0].updates[0]

    assert update.usage is not None
    assert update.cost is not None
    assert update.cost.pricing_status is PricingStatus.UNKNOWN
    assert update.cost.input_cost is None
    assert update.cost.cached_input_cost is None
    assert update.cost.output_cost is None
    assert update.cost.total_cost is None
    assert update.metadata["pricing_found"] is False
    assert update.metadata["pricing_catalog_version"] == (PRICING_CATALOG_VERSION)


async def test_missing_usage_omits_usage_and_cost() -> None:
    observability = RecordingObservabilityClient()
    provider = RecordingProvider(
        (
            _response(
                _valid_output(),
                provider_request_id="request-1",
                usage=None,
            ),
        ),
    )
    gateway = LLMGateway(
        provider=provider,
        max_repair_attempts=0,
        observability_client=cast(Any, observability),
    )

    await gateway.generate(_observability_request())

    update = observability.scopes[0].updates[0]

    assert update.usage is None
    assert update.cost is None
    assert "pricing_found" not in update.metadata
    assert "pricing_catalog_version" not in update.metadata


async def test_validation_failure_and_repair_create_two_observations() -> None:
    observability = RecordingObservabilityClient()
    provider = RecordingProvider(
        (
            _response(
                {"category": "invalid"},
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
        observability_client=cast(Any, observability),
    )

    result = await gateway.generate(_observability_request())

    assert len(result.invocations) == 2
    assert len(observability.scopes) == 2
    assert observability.started_attributes[0].metadata["is_repair"] is False
    assert observability.started_attributes[0].metadata["invocation_sequence"] == 1
    assert observability.started_attributes[1].metadata["is_repair"] is True
    assert observability.started_attributes[1].metadata["invocation_sequence"] == 2
    assert observability.scopes[0].updates[0].status is ObservationStatus.ERROR
    assert observability.scopes[0].updates[0].error_code == (
        LLMErrorCode.OUTPUT_VALIDATION_FAILED.value
    )
    assert observability.scopes[1].updates[0].status is ObservationStatus.OK
    assert result.invocations[0].status is LLMInvocationStatus.VALIDATION_FAILED
    assert result.invocations[1].status is LLMInvocationStatus.SUCCEEDED


async def test_zero_repair_budget_creates_one_error_observation() -> None:
    observability = RecordingObservabilityClient()
    provider = RecordingProvider(
        (
            _response(
                {"category": "invalid"},
                provider_request_id="request-1",
            ),
        ),
    )
    gateway = LLMGateway(
        provider=provider,
        max_repair_attempts=0,
        observability_client=cast(Any, observability),
    )

    with pytest.raises(LLMGatewayFailure) as captured:
        await gateway.generate(_observability_request())

    assert len(observability.scopes) == 1
    assert observability.scopes[0].updates[0].status is ObservationStatus.ERROR
    assert isinstance(captured.value.error, LLMOutputValidationError)
    assert len(captured.value.invocations) == 1


async def test_non_repairable_provider_failure_creates_one_error_observation() -> None:
    observability = RecordingObservabilityClient()
    provider = RecordingProvider(
        (
            ProviderErrorOutcome(
                LLMTimeoutError(provider_request_id="request-1"),
            ),
        ),
    )
    gateway = LLMGateway(
        provider=provider,
        max_repair_attempts=1,
        observability_client=cast(Any, observability),
    )

    with pytest.raises(LLMGatewayFailure) as captured:
        await gateway.generate(_observability_request())

    assert len(observability.scopes) == 1
    update = observability.scopes[0].updates[0]
    assert update.status is ObservationStatus.ERROR
    assert update.error_code == LLMErrorCode.TIMEOUT.value
    assert update.metadata["provider_request_id"] == "request-1"
    assert captured.value.error_code is LLMErrorCode.TIMEOUT
    assert len(captured.value.invocations) == 1
    assert observability.managers[0].exit_args == (None, None, None)


async def test_observability_start_failure_preserves_success() -> None:
    observability = RecordingObservabilityClient()
    observability.start_error = RuntimeError("start failed")
    provider = RecordingProvider(
        (
            _response(
                _valid_output(),
                provider_request_id="request-1",
            ),
        ),
    )
    gateway = LLMGateway(
        provider=provider,
        max_repair_attempts=0,
        observability_client=cast(Any, observability),
    )

    result = await gateway.generate(_request())

    assert result.accepted_invocation_sequence == 1
    assert len(observability.scopes) == 0


async def test_observability_update_failure_preserves_success() -> None:
    observability = RecordingObservabilityClient()
    observability.update_error = RuntimeError("update failed")
    provider = RecordingProvider(
        (
            _response(
                _valid_output(),
                provider_request_id="request-1",
            ),
        ),
    )
    gateway = LLMGateway(
        provider=provider,
        max_repair_attempts=0,
        observability_client=cast(Any, observability),
    )

    result = await gateway.generate(_request())

    assert result.accepted_invocation_sequence == 1
    assert observability.scopes[0].updates == []


async def test_observability_exit_failure_preserves_success() -> None:
    observability = RecordingObservabilityClient()
    observability.exit_error = RuntimeError("exit failed")
    provider = RecordingProvider(
        (
            _response(
                _valid_output(),
                provider_request_id="request-1",
            ),
        ),
    )
    gateway = LLMGateway(
        provider=provider,
        max_repair_attempts=0,
        observability_client=cast(Any, observability),
    )

    result = await gateway.generate(_request())

    assert result.accepted_invocation_sequence == 1
    assert len(observability.scopes[0].updates) == 1


async def test_business_exception_is_preserved_exactly() -> None:
    observability = RecordingObservabilityClient()
    provider_error = LLMTimeoutError(provider_request_id="request-1")
    provider = RecordingProvider(
        (ProviderErrorOutcome(provider_error),),
    )
    gateway = LLMGateway(
        provider=provider,
        max_repair_attempts=0,
        observability_client=cast(Any, observability),
    )

    with pytest.raises(LLMGatewayFailure) as captured:
        await gateway.generate(_request())

    assert captured.value.error is provider_error
    assert observability.managers[0].exit_args == (None, None, None)


async def test_provenance_mismatch_records_unhandled_error_observation() -> None:
    observability = RecordingObservabilityClient()
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
        max_repair_attempts=0,
        observability_client=cast(Any, observability),
    )

    with pytest.raises(RuntimeError, match="provenance does not match"):
        await gateway.generate(_request())

    assert len(observability.scopes) == 1
    update = observability.scopes[0].updates[0]
    assert update.status is ObservationStatus.ERROR
    assert update.error_code == "unhandled_business_error"
