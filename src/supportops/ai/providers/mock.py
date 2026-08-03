"""Deterministic mock provider for local execution and automated tests."""

from collections import deque
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType

from supportops.ai.gateway.contracts import (
    LLMProviderResponse,
    LLMRequest,
    LLMTokenUsage,
)
from supportops.ai.gateway.errors import (
    LLMIncompleteResponseError,
    LLMInvalidRequestError,
    LLMProviderUnavailableError,
    LLMRefusalError,
    LLMTimeoutError,
)
from supportops.ai.gateway.tool_decisions import (
    LLMProviderFunctionCallResponse,
    LLMProviderToolDecisionRequest,
)
from supportops.ai.providers.mock_tool_decisions import (
    MockToolDecisionOutcome,
    MockToolDecisionOutcomeKind,
    MockToolDecisionOutcomeQueueExhaustedError,
)
from supportops.ai.schemas.ticket_classification import (
    TICKET_CLASSIFICATION_SCHEMA_VERSION,
    TicketCategory,
    TicketIntent,
    TicketSentiment,
    TicketUrgency,
)

MOCK_LLM_PROVIDER_NAME = "mock"
MOCK_TICKET_CLASSIFIER_MODEL = "mock-ticket-classifier-v1"


class MockLLMOutcomeKind(StrEnum):
    """Explicit deterministic outcomes supported by the mock provider."""

    SUCCESS = "success"
    REFUSAL = "refusal"
    TIMEOUT = "timeout"
    RETRYABLE_PROVIDER_ERROR = "retryable_provider_error"
    TERMINAL_PROVIDER_ERROR = "terminal_provider_error"
    INCOMPLETE_RESPONSE = "incomplete_response"


@dataclass(frozen=True, slots=True)
class MockLLMOutcome:
    """One explicitly configured result for a mock provider invocation."""

    kind: MockLLMOutcomeKind
    parsed_output: Mapping[str, object] | None = None
    usage: LLMTokenUsage | None = None
    finish_reason: str | None = None

    def __post_init__(self) -> None:
        if self.kind is MockLLMOutcomeKind.SUCCESS:
            if self.parsed_output is None:
                raise ValueError(
                    "A successful mock outcome requires parsed_output.",
                )
        elif self.parsed_output is not None:
            raise ValueError(
                "Only successful mock outcomes may define parsed_output.",
            )

        if self.parsed_output is not None:
            object.__setattr__(
                self,
                "parsed_output",
                MappingProxyType(dict(self.parsed_output)),
            )

        if self.finish_reason is not None:
            _validate_required_text(
                self.finish_reason,
                field_name="finish_reason",
            )

    @classmethod
    def success(
        cls,
        parsed_output: Mapping[str, object],
        *,
        usage: LLMTokenUsage | None = None,
        finish_reason: str = "completed",
    ) -> "MockLLMOutcome":
        """Create one successful structured mock response."""

        return cls(
            kind=MockLLMOutcomeKind.SUCCESS,
            parsed_output=parsed_output,
            usage=usage,
            finish_reason=finish_reason,
        )

    @classmethod
    def refusal(cls) -> "MockLLMOutcome":
        """Create one explicit provider refusal."""

        return cls(kind=MockLLMOutcomeKind.REFUSAL)

    @classmethod
    def timeout(cls) -> "MockLLMOutcome":
        """Create one deterministic timeout failure."""

        return cls(kind=MockLLMOutcomeKind.TIMEOUT)

    @classmethod
    def retryable_provider_error(cls) -> "MockLLMOutcome":
        """Create one retryable provider-unavailable failure."""

        return cls(
            kind=MockLLMOutcomeKind.RETRYABLE_PROVIDER_ERROR,
        )

    @classmethod
    def terminal_provider_error(cls) -> "MockLLMOutcome":
        """Create one terminal invalid-request failure."""

        return cls(
            kind=MockLLMOutcomeKind.TERMINAL_PROVIDER_ERROR,
        )

    @classmethod
    def incomplete_response(cls) -> "MockLLMOutcome":
        """Create one recoverable incomplete-response failure."""

        return cls(
            kind=MockLLMOutcomeKind.INCOMPLETE_RESPONSE,
        )

    @classmethod
    def human_approved_escalation_executed(
        cls,
        *,
        usage: LLMTokenUsage | None = None,
    ) -> "MockLLMOutcome":
        """Script a recommendation after approved escalation execution."""

        return cls.success(
            {
                "recommended_action": "recommend_escalation",
                "response_text": (
                    "The approved escalation was executed. "
                    "Share the confirmed queue handoff with the requester."
                ),
                "requires_human_review": True,
                "decision_summary": ("Approved escalation executed to the target queue."),
                "schema_version": "support-recommendation-v1",
            },
            usage=usage,
        )

    @classmethod
    def human_approved_escalation_rejected(
        cls,
        *,
        usage: LLMTokenUsage | None = None,
    ) -> "MockLLMOutcome":
        """Script a recommendation after rejected escalation."""

        return cls.success(
            {
                "recommended_action": "respond",
                "response_text": (
                    "The sensitive escalation was rejected and was not "
                    "executed. Continue with a grounded non-escalation reply."
                ),
                "requires_human_review": False,
                "decision_summary": ("Escalation was rejected and was not executed."),
                "schema_version": "support-recommendation-v1",
            },
            usage=usage,
        )

    @classmethod
    def human_approved_escalation_expired(
        cls,
        *,
        usage: LLMTokenUsage | None = None,
    ) -> "MockLLMOutcome":
        """Script a recommendation after expired escalation approval."""

        return cls.success(
            {
                "recommended_action": "respond",
                "response_text": (
                    "The sensitive escalation approval expired and was not "
                    "executed. Continue with a grounded non-escalation reply."
                ),
                "requires_human_review": False,
                "decision_summary": ("Escalation approval expired and was not executed."),
                "schema_version": "support-recommendation-v1",
            },
            usage=usage,
        )


class MockLLMOutcomeQueueExhaustedError(RuntimeError):
    """Raised when a strict test outcome queue has no remaining result."""


class MockLLMProvider:
    """Deterministic network-free implementation of the LLM provider contract."""

    def __init__(
        self,
        *,
        model: str = MOCK_TICKET_CLASSIFIER_MODEL,
        outcomes: Iterable[MockLLMOutcome] = (),
        default_outcome: MockLLMOutcome | None = None,
        tool_decision_outcomes: Iterable[MockToolDecisionOutcome] = (),
        default_tool_decision_outcome: MockToolDecisionOutcome | None = None,
    ) -> None:
        _validate_required_text(model, field_name="model")

        self._model = model
        self._outcomes = deque(outcomes)
        self._default_outcome: MockLLMOutcome | None = (
            _default_success_outcome() if default_outcome is None else default_outcome
        )
        self._tool_decision_outcomes = deque(tool_decision_outcomes)
        self._default_tool_decision_outcome = default_tool_decision_outcome
        self._invocation_count = 0
        self._closed = False

    @classmethod
    def with_strict_outcomes(
        cls,
        outcomes: Iterable[MockLLMOutcome],
        *,
        model: str = MOCK_TICKET_CLASSIFIER_MODEL,
    ) -> "MockLLMProvider":
        """Create a provider that fails when its configured queue is exhausted."""

        provider = cls(
            model=model,
            outcomes=outcomes,
        )
        provider._default_outcome = None
        return provider

    @classmethod
    def with_strict_tool_decisions(
        cls,
        outcomes: Iterable[MockToolDecisionOutcome],
        *,
        model: str = MOCK_TICKET_CLASSIFIER_MODEL,
    ) -> "MockLLMProvider":
        """Create a provider with an explicit finite tool-decision script."""

        return cls(
            model=model,
            tool_decision_outcomes=outcomes,
            default_tool_decision_outcome=None,
        )

    @property
    def provider_name(self) -> str:
        """Return the explicit mock provider identity."""

        return MOCK_LLM_PROVIDER_NAME

    @property
    def model(self) -> str:
        """Return the configured mock model identifier."""

        return self._model

    @property
    def invocation_count(self) -> int:
        """Return the number of logical mock provider requests."""

        return self._invocation_count

    async def generate(
        self,
        request: LLMRequest,
    ) -> LLMProviderResponse:
        """Return one deterministic configured outcome without network access."""

        if self._closed:
            raise RuntimeError("Mock LLM provider is closed.")

        if request.model != self._model:
            raise LLMInvalidRequestError()

        outcome = self._next_outcome()
        self._invocation_count += 1

        provider_request_id = f"mock-request-{self._invocation_count}"

        if outcome.kind is MockLLMOutcomeKind.SUCCESS:
            if outcome.parsed_output is None:
                raise RuntimeError(
                    "Successful mock outcome has no parsed output.",
                )

            return LLMProviderResponse(
                parsed_output=outcome.parsed_output,
                provider=self.provider_name,
                model=self._model,
                provider_request_id=provider_request_id,
                usage=outcome.usage,
                finish_reason=outcome.finish_reason,
            )

        if outcome.kind is MockLLMOutcomeKind.REFUSAL:
            raise LLMRefusalError(
                provider_request_id=provider_request_id,
            )

        if outcome.kind is MockLLMOutcomeKind.TIMEOUT:
            raise LLMTimeoutError(
                provider_request_id=provider_request_id,
            )

        if outcome.kind is MockLLMOutcomeKind.RETRYABLE_PROVIDER_ERROR:
            raise LLMProviderUnavailableError(
                provider_request_id=provider_request_id,
            )

        if outcome.kind is MockLLMOutcomeKind.TERMINAL_PROVIDER_ERROR:
            raise LLMInvalidRequestError(
                provider_request_id=provider_request_id,
            )

        if outcome.kind is MockLLMOutcomeKind.INCOMPLETE_RESPONSE:
            raise LLMIncompleteResponseError(
                provider_request_id=provider_request_id,
            )

        raise RuntimeError(
            f"Unsupported mock LLM outcome: {outcome.kind}.",
        )

    async def decide(
        self,
        request: LLMProviderToolDecisionRequest,
    ) -> LLMProviderFunctionCallResponse:
        """Return one deterministic scripted tool decision without network access."""

        if self._closed:
            raise RuntimeError("Mock LLM provider is closed.")

        if request.model != self._model:
            raise LLMInvalidRequestError()

        outcome = self._next_tool_decision_outcome()
        self._invocation_count += 1

        provider_request_id = f"mock-request-{self._invocation_count}"
        provider_tool_call_id = f"mock-tool-call-{self._invocation_count}"

        if outcome.kind is MockToolDecisionOutcomeKind.FUNCTION_CALL:
            if (
                outcome.function_name is None
                or outcome.arguments_json is None
                or outcome.finish_reason is None
            ):
                raise RuntimeError(
                    "Function-call mock outcome is missing required fields.",
                )

            return LLMProviderFunctionCallResponse(
                provider_tool_call_id=provider_tool_call_id,
                function_name=outcome.function_name,
                arguments_json=outcome.arguments_json,
                provider=self.provider_name,
                model=self._model,
                provider_request_id=provider_request_id,
                usage=outcome.usage,
                finish_reason=outcome.finish_reason,
            )

        if outcome.kind is MockToolDecisionOutcomeKind.REFUSAL:
            raise LLMRefusalError(
                provider_request_id=provider_request_id,
            )

        if outcome.kind is MockToolDecisionOutcomeKind.TIMEOUT:
            raise LLMTimeoutError(
                provider_request_id=provider_request_id,
            )

        if outcome.kind is MockToolDecisionOutcomeKind.RETRYABLE_PROVIDER_ERROR:
            raise LLMProviderUnavailableError(
                provider_request_id=provider_request_id,
            )

        if outcome.kind is MockToolDecisionOutcomeKind.TERMINAL_PROVIDER_ERROR:
            raise LLMInvalidRequestError(
                provider_request_id=provider_request_id,
            )

        if outcome.kind is MockToolDecisionOutcomeKind.INCOMPLETE_RESPONSE:
            raise LLMIncompleteResponseError(
                provider_request_id=provider_request_id,
            )

        raise RuntimeError(
            f"Unsupported mock tool-decision outcome: {outcome.kind}.",
        )

    async def close(self) -> None:
        """Close the mock provider lifecycle idempotently."""

        self._closed = True

    def _next_outcome(self) -> MockLLMOutcome:
        if self._outcomes:
            return self._outcomes.popleft()

        if self._default_outcome is not None:
            return self._default_outcome

        raise MockLLMOutcomeQueueExhaustedError(
            "The strict mock LLM outcome queue is exhausted.",
        )

    def _next_tool_decision_outcome(self) -> MockToolDecisionOutcome:
        if self._tool_decision_outcomes:
            return self._tool_decision_outcomes.popleft()

        if self._default_tool_decision_outcome is not None:
            return self._default_tool_decision_outcome

        raise MockToolDecisionOutcomeQueueExhaustedError(
            "The strict mock tool-decision outcome queue is exhausted.",
        )


def _default_success_outcome() -> MockLLMOutcome:
    return MockLLMOutcome.success(
        {
            "category": TicketCategory.OTHER.value,
            "intent": TicketIntent.OTHER.value,
            "urgency": TicketUrgency.NORMAL.value,
            "sentiment": TicketSentiment.NEUTRAL.value,
            "requires_human_review": False,
            "summary": ("The mock provider produced a deterministic baseline classification."),
            "schema_version": TICKET_CLASSIFICATION_SCHEMA_VERSION,
        },
        usage=LLMTokenUsage(
            input_tokens=120,
            cached_input_tokens=None,
            output_tokens=24,
            reasoning_tokens=None,
            total_tokens=144,
        ),
    )


def _validate_required_text(
    value: str,
    *,
    field_name: str,
) -> None:
    if not value:
        raise ValueError(f"{field_name} is required.")

    if value != value.strip():
        raise ValueError(
            f"{field_name} must not contain surrounding whitespace.",
        )
