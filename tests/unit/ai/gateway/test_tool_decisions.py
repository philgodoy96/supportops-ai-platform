"""Unit tests for provider-independent LLM tool decisions."""

from collections.abc import Callable, Iterable
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from decimal import Decimal
from types import TracebackType
from typing import Annotated, Any, Literal, cast
from uuid import UUID

import pytest
from pydantic import (
    Field,
    StrictBool,
    StringConstraints,
)

from supportops.agent_tools.domain.contracts import (
    StrictToolSchema,
    ToolAuditPolicy,
    ToolDefinition,
    ToolFailurePolicy,
    ToolSafetyLevel,
)
from supportops.agent_tools.tools.escalate_ticket import (
    create_escalate_ticket_definition,
)
from supportops.ai.gateway.contracts import (
    LLMOperation,
    LLMTokenUsage,
)
from supportops.ai.gateway.errors import (
    LLMErrorCode,
    LLMTimeoutError,
)
from supportops.ai.gateway.results import (
    LLMGatewayFailure,
    LLMInvocationStatus,
)
from supportops.ai.gateway.tool_decisions import (
    COMPLETE_SUPPORT_ANALYSIS_CONTROL_NAME,
    LLMExecutableToolCallDecision,
    LLMHumanApprovedToolDecisionRequest,
    LLMProviderFunctionCallResponse,
    LLMProviderToolDecisionRequest,
    LLMTerminalControlDecision,
    LLMTerminalControlDefinition,
    LLMToolDecisionGateway,
    LLMToolDecisionProvider,
    LLMToolDecisionRequest,
)
from supportops.ai.pricing.catalog import PRICING_CATALOG_VERSION
from supportops.ai.schemas.human_approved_support_decision import (
    COMPLETE_HUMAN_APPROVED_SUPPORT_ANALYSIS_CONTROL,
    COMPLETE_HUMAN_APPROVED_SUPPORT_ANALYSIS_CONTROL_NAME,
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

DOCUMENT_ID = UUID("11111111-1111-4111-8111-111111111111")
_MOCK_PROVIDER = "mock"
_MOCK_MODEL = "mock-ticket-classifier-v1"
_DEFAULT_USAGE = object()


class SearchKnowledgeInput(StrictToolSchema):
    """Strict model-visible knowledge search arguments."""

    top_k: Annotated[
        int,
        Field(
            strict=True,
            ge=1,
            le=10,
        ),
    ]
    document_ids: tuple[UUID, ...] | None


class SearchKnowledgeOutput(StrictToolSchema):
    """Minimal synthetic output required by ToolDefinition."""

    result_count: Annotated[
        int,
        Field(
            strict=True,
            ge=0,
        ),
    ]


class CompleteSupportAnalysisInput(StrictToolSchema):
    """Strict terminal control arguments."""

    recommended_action: Literal[
        "respond",
        "request_more_information",
        "recommend_escalation",
    ]
    evidence_sufficient: StrictBool
    requires_human_review: StrictBool
    decision_summary: Annotated[
        str,
        StringConstraints(
            strip_whitespace=True,
            min_length=1,
            max_length=500,
        ),
    ]


class StubToolDecisionProvider:
    """Deterministic provider test double."""

    def __init__(
        self,
        outcome: (LLMProviderFunctionCallResponse | Exception),
        *,
        provider_name: str = "stub",
    ) -> None:
        self._outcome = outcome
        self._provider_name = provider_name
        self.requests: list[LLMProviderToolDecisionRequest] = []
        self.close_calls = 0

    @property
    def provider_name(self) -> str:
        return self._provider_name

    async def decide(
        self,
        request: LLMProviderToolDecisionRequest,
    ) -> LLMProviderFunctionCallResponse:
        self.requests.append(request)

        if isinstance(self._outcome, Exception):
            raise self._outcome

        return self._outcome

    async def close(self) -> None:
        self.close_calls += 1


def create_search_definition(
    *,
    name: str = "search_knowledge",
    version: int = 1,
    safety_level: ToolSafetyLevel = (ToolSafetyLevel.READ_ONLY),
) -> ToolDefinition:
    """Create one synthetic executable tool definition."""

    return ToolDefinition(
        name=name,
        version=version,
        description=("Search active workspace-scoped runbook evidence."),
        input_schema=SearchKnowledgeInput,
        output_schema=SearchKnowledgeOutput,
        safety_level=safety_level,
        timeout_seconds=15,
        failure_policy=(ToolFailurePolicy.RETRY_AGENT_RUN),
        audit_policy=(ToolAuditPolicy.SAFE_PROJECTION),
    )


def create_terminal_control(
    *,
    name: str = (COMPLETE_SUPPORT_ANALYSIS_CONTROL_NAME),
) -> LLMTerminalControlDefinition:
    """Create the approved terminal control definition."""

    return LLMTerminalControlDefinition(
        name=name,
        version=1,
        description=("Complete controlled support analysis without executing an external action."),
        input_schema=CompleteSupportAnalysisInput,
    )


def create_request(
    *,
    tools: tuple[ToolDefinition, ...] | None = None,
    terminal_control: (LLMTerminalControlDefinition | None) = None,
) -> LLMToolDecisionRequest:
    """Create a valid application-owned tool decision request."""

    return LLMToolDecisionRequest(
        operation=(LLMOperation.SUPPORT_ACTION_DECISION),
        model="stub-support-model-v1",
        instructions=("Select one approved function for the controlled support workflow."),
        input=('{"ticket":{"subject":"Reset access"},"classification":{"category":"how_to"}}'),
        tools=((create_search_definition(),) if tools is None else tools),
        terminal_control=(terminal_control or create_terminal_control()),
        timeout_seconds=20,
        metadata={
            "agent_run_id": ("22222222-2222-4222-8222-222222222222"),
        },
    )


def create_tool_call_response(
    *,
    function_name: str = "search_knowledge",
    arguments_json: str = ('{"top_k":5,"document_ids":null}'),
    provider: str = "stub",
    model: str = "stub-support-model-v1",
    usage: LLMTokenUsage | object | None = _DEFAULT_USAGE,
) -> LLMProviderFunctionCallResponse:
    """Create one successful synthetic function call."""

    resolved_usage: LLMTokenUsage | None
    if usage is _DEFAULT_USAGE:
        resolved_usage = LLMTokenUsage(
            input_tokens=30,
            output_tokens=10,
            total_tokens=40,
        )
    else:
        resolved_usage = cast(LLMTokenUsage | None, usage)

    return LLMProviderFunctionCallResponse(
        provider_tool_call_id="tool-call-1",
        function_name=function_name,
        arguments_json=arguments_json,
        provider=provider,
        model=model,
        provider_request_id="provider-request-1",
        usage=resolved_usage,
        finish_reason="completed",
    )


def create_clock(
    *values: float,
) -> Callable[[], float]:
    """Return deterministic sequential clock values."""

    clock_values = iter(values)

    return lambda: next(clock_values)


def test_application_request_projects_provider_only_metadata() -> None:
    request = create_request()

    provider_request = request.to_provider_request()

    assert provider_request.operation is LLMOperation.SUPPORT_ACTION_DECISION
    assert provider_request.tool_choice == "required"
    assert provider_request.parallel_tool_calls is False
    assert [function.name for function in provider_request.functions] == [
        "search_knowledge",
        COMPLETE_SUPPORT_ANALYSIS_CONTROL_NAME,
    ]

    executable_payload = provider_request.functions[0].model_dump(mode="json")

    assert executable_payload["strict"] is True
    assert "safety_level" not in executable_payload
    assert "timeout_seconds" not in executable_payload
    assert "output_schema" not in executable_payload
    assert "failure_policy" not in executable_payload
    assert "audit_policy" not in executable_payload


def test_application_request_supports_terminal_only_decision() -> None:
    request = create_request(tools=())

    provider_request = request.to_provider_request()

    assert request.tools == ()
    assert provider_request.tool_choice == "required"
    assert provider_request.parallel_tool_calls is False
    assert [function.name for function in provider_request.functions] == [
        COMPLETE_SUPPORT_ANALYSIS_CONTROL_NAME,
    ]
    assert provider_request.functions[0].strict is True


def test_application_request_copies_and_freezes_metadata() -> None:
    metadata = {"correlation_id": "corr-1"}
    request = LLMToolDecisionRequest(
        operation=(LLMOperation.SUPPORT_ACTION_DECISION),
        model="stub-support-model-v1",
        instructions="Select one approved function.",
        input='{"ticket_id":"ticket-1"}',
        tools=(create_search_definition(),),
        terminal_control=create_terminal_control(),
        timeout_seconds=20,
        metadata=metadata,
    )

    metadata["correlation_id"] = "changed"

    assert request.metadata == {"correlation_id": "corr-1"}

    with pytest.raises(TypeError):
        cast(
            dict[str, str],
            request.metadata,
        )["request_id"] = "request-1"


def test_request_rejects_non_decision_operation() -> None:
    with pytest.raises(
        ValueError,
        match="support_action_decision",
    ):
        LLMToolDecisionRequest(
            operation=LLMOperation.TICKET_CLASSIFICATION,
            model="stub-support-model-v1",
            instructions="Select one approved function.",
            input='{"ticket_id":"ticket-1"}',
            tools=(create_search_definition(),),
            terminal_control=create_terminal_control(),
            timeout_seconds=20,
        )


def test_request_rejects_non_read_only_tool() -> None:
    with pytest.raises(
        ValueError,
        match="only read-only tools",
    ):
        create_request(
            tools=(create_search_definition(safety_level=(ToolSafetyLevel.SENSITIVE_WRITE)),)
        )


def test_request_rejects_multiple_versions_of_same_name() -> None:
    with pytest.raises(
        ValueError,
        match="Only one version",
    ):
        create_request(
            tools=(
                create_search_definition(version=1),
                create_search_definition(version=2),
            )
        )


def test_request_rejects_terminal_name_collision() -> None:
    with pytest.raises(
        ValueError,
        match="must not collide",
    ):
        create_request(
            tools=(create_search_definition(name=(COMPLETE_SUPPORT_ANALYSIS_CONTROL_NAME)),)
        )


def test_terminal_control_schema_is_strict() -> None:
    provider_definition = create_terminal_control().to_provider_definition()
    required = cast(
        list[str],
        provider_definition.input_schema["required"],
    )

    assert provider_definition.strict is True
    assert provider_definition.input_schema["additionalProperties"] is False
    assert set(required) == {
        "recommended_action",
        "evidence_sufficient",
        "requires_human_review",
        "decision_summary",
    }


async def test_provider_protocol_supports_decision_and_close() -> None:
    provider: LLMToolDecisionProvider = StubToolDecisionProvider(create_tool_call_response())

    response = await provider.decide(create_request().to_provider_request())
    await provider.close()

    assert response.function_name == "search_knowledge"
    assert response.provider == "stub"


async def test_gateway_returns_validated_executable_tool_call() -> None:
    provider = StubToolDecisionProvider(
        create_tool_call_response(
            arguments_json=(f'{{"document_ids":["{DOCUMENT_ID}"],"top_k":5}}')
        )
    )
    gateway = LLMToolDecisionGateway(
        provider=provider,
        clock=create_clock(10.0, 10.025),
    )

    result = await gateway.decide(create_request())

    assert len(provider.requests) == 1
    assert isinstance(
        result.decision,
        LLMExecutableToolCallDecision,
    )
    assert result.decision.provider_tool_call_id == ("tool-call-1")
    assert result.decision.tool_name == ("search_knowledge")
    assert result.decision.tool_version == 1
    assert isinstance(
        result.decision.arguments,
        SearchKnowledgeInput,
    )
    assert result.decision.arguments.top_k == 5
    assert result.decision.arguments.document_ids == (DOCUMENT_ID,)
    assert result.accepted_invocation_sequence == 1
    assert result.invocations[0].status is (LLMInvocationStatus.SUCCEEDED)
    assert result.invocations[0].latency_ms == 25
    assert result.invocations[0].usage == LLMTokenUsage(
        input_tokens=30,
        output_tokens=10,
        total_tokens=40,
    )


async def test_gateway_returns_validated_terminal_control() -> None:
    provider = StubToolDecisionProvider(
        create_tool_call_response(
            function_name=(COMPLETE_SUPPORT_ANALYSIS_CONTROL_NAME),
            arguments_json=(
                "{"
                '"recommended_action":"respond",'
                '"evidence_sufficient":true,'
                '"requires_human_review":false,'
                '"decision_summary":'
                '"Relevant evidence is available."'
                "}"
            ),
        )
    )
    gateway = LLMToolDecisionGateway(
        provider=provider,
        clock=create_clock(20.0, 20.01),
    )

    result = await gateway.decide(create_request())

    assert isinstance(
        result.decision,
        LLMTerminalControlDecision,
    )
    assert result.decision.control_name == (COMPLETE_SUPPORT_ANALYSIS_CONTROL_NAME)
    assert result.decision.control_version == 1
    assert isinstance(
        result.decision.output,
        CompleteSupportAnalysisInput,
    )
    assert result.decision.output.recommended_action == "respond"
    assert result.decision.output.evidence_sufficient is True


async def test_gateway_accepts_terminal_control_without_executable_tools() -> None:
    provider = StubToolDecisionProvider(
        create_tool_call_response(
            function_name=(COMPLETE_SUPPORT_ANALYSIS_CONTROL_NAME),
            arguments_json=(
                "{"
                '"recommended_action":"request_more_information",'
                '"evidence_sufficient":false,'
                '"requires_human_review":false,'
                '"decision_summary":'
                '"Additional diagnostic details are required."'
                "}"
            ),
        )
    )
    gateway = LLMToolDecisionGateway(
        provider=provider,
        clock=create_clock(20.0, 20.01),
    )

    result = await gateway.decide(create_request(tools=()))

    assert len(provider.requests) == 1
    assert len(provider.requests[0].functions) == 1
    assert provider.requests[0].functions[0].name == (COMPLETE_SUPPORT_ANALYSIS_CONTROL_NAME)
    assert isinstance(
        result.decision,
        LLMTerminalControlDecision,
    )
    assert isinstance(
        result.decision.output,
        CompleteSupportAnalysisInput,
    )
    assert result.decision.output.recommended_action == ("request_more_information")
    assert result.decision.output.evidence_sufficient is False


@pytest.mark.parametrize(
    "arguments_json",
    [
        "{malformed-json",
        "[]",
        '"not-an-object"',
        "null",
    ],
)
async def test_gateway_rejects_invalid_argument_json(
    arguments_json: str,
) -> None:
    provider = StubToolDecisionProvider(create_tool_call_response(arguments_json=arguments_json))
    gateway = LLMToolDecisionGateway(
        provider=provider,
        clock=create_clock(10.0, 10.005),
    )

    with pytest.raises(LLMGatewayFailure) as exc_info:
        await gateway.decide(create_request())

    failure = exc_info.value

    assert failure.error_code is (LLMErrorCode.TOOL_DECISION_VALIDATION_FAILED)
    assert failure.retryable is False
    assert failure.terminal is True
    assert failure.repairable is False
    assert len(provider.requests) == 1
    assert failure.invocations[0].status is (LLMInvocationStatus.VALIDATION_FAILED)


async def test_gateway_rejects_unknown_function_name() -> None:
    provider = StubToolDecisionProvider(create_tool_call_response(function_name="invented_tool"))
    gateway = LLMToolDecisionGateway(
        provider=provider,
        clock=create_clock(10.0, 10.005),
    )

    with pytest.raises(LLMGatewayFailure) as exc_info:
        await gateway.decide(create_request())

    assert exc_info.value.error_code is (LLMErrorCode.TOOL_DECISION_VALIDATION_FAILED)
    assert len(provider.requests) == 1


async def test_gateway_rejects_model_supplied_workspace_id() -> None:
    provider = StubToolDecisionProvider(
        create_tool_call_response(
            arguments_json=(
                "{"
                '"top_k":5,'
                '"document_ids":null,'
                '"workspace_id":'
                '"99999999-9999-4999-8999-999999999999"'
                "}"
            )
        )
    )
    gateway = LLMToolDecisionGateway(
        provider=provider,
        clock=create_clock(10.0, 10.005),
    )

    with pytest.raises(LLMGatewayFailure) as exc_info:
        await gateway.decide(create_request())

    assert exc_info.value.error_code is (LLMErrorCode.TOOL_DECISION_VALIDATION_FAILED)


async def test_gateway_rejects_invalid_terminal_control() -> None:
    provider = StubToolDecisionProvider(
        create_tool_call_response(
            function_name=(COMPLETE_SUPPORT_ANALYSIS_CONTROL_NAME),
            arguments_json=(
                "{"
                '"recommended_action":"respond",'
                '"evidence_sufficient":"yes",'
                '"requires_human_review":false,'
                '"decision_summary":"Evidence exists."'
                "}"
            ),
        )
    )
    gateway = LLMToolDecisionGateway(
        provider=provider,
        clock=create_clock(10.0, 10.005),
    )

    with pytest.raises(LLMGatewayFailure) as exc_info:
        await gateway.decide(create_request())

    assert exc_info.value.error_code is (LLMErrorCode.TOOL_DECISION_VALIDATION_FAILED)


async def test_provider_failure_preserves_normalized_trace() -> None:
    provider = StubToolDecisionProvider(LLMTimeoutError(provider_request_id="provider-request-1"))
    gateway = LLMToolDecisionGateway(
        provider=provider,
        clock=create_clock(10.0, 10.05),
    )

    with pytest.raises(LLMGatewayFailure) as exc_info:
        await gateway.decide(create_request())

    failure = exc_info.value

    assert failure.error_code is LLMErrorCode.TIMEOUT
    assert failure.retryable is True
    assert len(provider.requests) == 1
    assert failure.invocations[0].status is (LLMInvocationStatus.TIMED_OUT)
    assert failure.invocations[0].latency_ms == 50


async def test_gateway_rejects_provider_provenance_mismatch() -> None:
    provider = StubToolDecisionProvider(create_tool_call_response(provider="unexpected-provider"))
    gateway = LLMToolDecisionGateway(
        provider=provider,
        clock=create_clock(10.0, 10.005),
    )

    with pytest.raises(
        RuntimeError,
        match="provenance does not match",
    ):
        await gateway.decide(create_request())


async def test_gateway_rejects_model_provenance_mismatch() -> None:
    provider = StubToolDecisionProvider(create_tool_call_response(model="unexpected-model"))
    gateway = LLMToolDecisionGateway(
        provider=provider,
        clock=create_clock(10.0, 10.005),
    )

    with pytest.raises(
        RuntimeError,
        match="model does not match",
    ):
        await gateway.decide(create_request())


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


def _observability_metadata() -> dict[str, str]:
    return {
        "supportops_workspace_id": "workspace-1",
        "supportops_agent_run_id": "agent-run-1",
        "supportops_agent_run_attempt_id": "attempt-1",
        "supportops_correlation_id": "correlation-1",
        "supportops_prompt_id": "controlled-support-decision",
        "supportops_prompt_version": "1",
        "supportops_prompt_content_hash": "hash-1",
        "supportops_schema_version": "controlled-support-decision-v1",
        "llm_invocation_id": "should-not-export",
        "execution_request_id": "should-not-export",
    }


def _observability_request(
    *,
    model: str = "stub-support-model-v1",
) -> LLMToolDecisionRequest:
    return LLMToolDecisionRequest(
        operation=LLMOperation.SUPPORT_ACTION_DECISION,
        model=model,
        instructions=("Select one approved function for the controlled support workflow."),
        input=('{"ticket":{"subject":"Reset access"},"classification":{"category":"how_to"}}'),
        tools=(create_search_definition(),),
        terminal_control=create_terminal_control(),
        timeout_seconds=20,
        metadata=_observability_metadata(),
    )


def _human_approved_observability_request(
    *,
    model: str = "stub-support-model-v1",
) -> LLMHumanApprovedToolDecisionRequest:
    return LLMHumanApprovedToolDecisionRequest(
        operation=LLMOperation.SUPPORT_ACTION_DECISION,
        model=model,
        instructions=("Select one approved sensitive or terminal function."),
        input=('{"ticket":{"subject":"Escalate billing"},"classification":{"category":"billing"}}'),
        sensitive_tools=(create_escalate_ticket_definition(),),
        terminal_control=(COMPLETE_HUMAN_APPROVED_SUPPORT_ANALYSIS_CONTROL),
        timeout_seconds=20,
        prompt_id="human-approved-support-decision",
        prompt_version=1,
        metadata=_observability_metadata(),
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

    assert "Reset access" not in serialized
    assert "Select one approved function" not in serialized
    assert "Escalate billing" not in serialized
    assert "Needs billing review" not in serialized
    assert "llm_invocation_id" not in attributes.metadata
    assert "execution_request_id" not in attributes.metadata
    assert "should-not-export" not in serialized
    assert "arguments_json" not in serialized
    assert "document_ids" not in serialized
    assert "search_knowledge" not in attributes.metadata
    assert "escalate_ticket" not in attributes.metadata


async def test_successful_controlled_decision_creates_one_generation() -> None:
    usage = LLMTokenUsage(
        input_tokens=120,
        cached_input_tokens=20,
        output_tokens=30,
        reasoning_tokens=10,
        total_tokens=150,
    )
    observability = RecordingObservabilityClient()
    provider = StubToolDecisionProvider(
        create_tool_call_response(
            provider=_MOCK_PROVIDER,
            model=_MOCK_MODEL,
            usage=usage,
        ),
        provider_name=_MOCK_PROVIDER,
    )
    gateway = LLMToolDecisionGateway(
        provider=provider,
        clock=create_clock(1.0, 1.05),
        observability_client=cast(Any, observability),
    )

    result = await gateway.decide(_observability_request(model=_MOCK_MODEL))

    assert len(result.invocations) == 1
    assert len(observability.scopes) == 1

    attributes = observability.started_attributes[0]
    update = observability.scopes[0].updates[0]

    assert attributes.name == "llm.tool_decision"
    assert attributes.observation_type is ObservationType.GENERATION
    assert attributes.provider == _MOCK_PROVIDER
    assert attributes.model == _MOCK_MODEL
    assert attributes.metadata["operation"] == (LLMOperation.SUPPORT_ACTION_DECISION.value)
    assert attributes.metadata["decision_mode"] == "controlled"
    assert attributes.metadata["invocation_sequence"] == 1
    assert attributes.metadata["is_repair"] is False
    assert attributes.metadata["tool_count"] == 1
    assert attributes.metadata["sensitive_tool_count"] == 0
    assert attributes.metadata["prompt_id"] == "controlled-support-decision"
    assert attributes.metadata["prompt_version"] == "1"
    assert attributes.metadata["prompt_hash"] == "hash-1"
    assert attributes.metadata["schema_version"] == ("controlled-support-decision-v1")
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
    assert update.metadata["provider_request_id"] == "provider-request-1"
    assert update.metadata["latency_ms"] == 50
    assert update.metadata["pricing_found"] is True
    assert update.metadata["pricing_catalog_version"] == (PRICING_CATALOG_VERSION)
    assert update.metadata["decision_kind"] == "executable_tool_call"
    assert update.metadata["selected_tool_safety"] == (ToolSafetyLevel.READ_ONLY.value)
    assert "search_knowledge" not in update.metadata
    _assert_observation_is_content_free(attributes, observability.scopes[0].updates)
    assert observability.managers[0].exit_args == (None, None, None)


async def test_successful_human_approved_decision_creates_one_generation() -> None:
    observability = RecordingObservabilityClient()
    provider = StubToolDecisionProvider(
        create_tool_call_response(
            function_name="escalate_ticket",
            arguments_json=(
                '{"target_queue":"billing_operations","reason":"Needs billing review."}'
            ),
            provider=_MOCK_PROVIDER,
            model=_MOCK_MODEL,
        ),
        provider_name=_MOCK_PROVIDER,
    )
    gateway = LLMToolDecisionGateway(
        provider=provider,
        clock=create_clock(1.0, 1.04),
        observability_client=cast(Any, observability),
    )

    result = await gateway.decide_human_approved(
        _human_approved_observability_request(model=_MOCK_MODEL),
    )

    assert isinstance(result.decision, LLMExecutableToolCallDecision)
    assert len(observability.scopes) == 1

    attributes = observability.started_attributes[0]
    update = observability.scopes[0].updates[0]

    assert attributes.name == "llm.tool_decision"
    assert attributes.observation_type is ObservationType.GENERATION
    assert attributes.metadata["decision_mode"] == "human_approved"
    assert attributes.metadata["invocation_sequence"] == 1
    assert attributes.metadata["is_repair"] is False
    assert attributes.metadata["tool_count"] == 1
    assert attributes.metadata["sensitive_tool_count"] == 1
    assert attributes.metadata["prompt_id"] == ("human-approved-support-decision")
    assert attributes.metadata["prompt_version"] == 1
    assert attributes.input_paths == frozenset()
    assert attributes.output_paths == frozenset()
    assert update.status is ObservationStatus.OK
    assert update.metadata["decision_kind"] == "executable_tool_call"
    assert update.metadata["selected_tool_safety"] == (ToolSafetyLevel.SENSITIVE_WRITE.value)
    assert "escalate_ticket" not in attributes.metadata
    assert "escalate_ticket" not in update.metadata
    _assert_observation_is_content_free(attributes, observability.scopes[0].updates)


async def test_unknown_pricing_omits_cost_values() -> None:
    usage = LLMTokenUsage(
        input_tokens=10,
        cached_input_tokens=0,
        output_tokens=5,
        total_tokens=15,
    )
    observability = RecordingObservabilityClient()
    provider = StubToolDecisionProvider(
        create_tool_call_response(usage=usage),
    )
    gateway = LLMToolDecisionGateway(
        provider=provider,
        observability_client=cast(Any, observability),
    )

    await gateway.decide(_observability_request())

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
    provider = StubToolDecisionProvider(
        create_tool_call_response(usage=None),
    )
    gateway = LLMToolDecisionGateway(
        provider=provider,
        observability_client=cast(Any, observability),
    )

    await gateway.decide(_observability_request())

    update = observability.scopes[0].updates[0]

    assert update.usage is None
    assert update.cost is None
    assert "pricing_found" not in update.metadata
    assert "pricing_catalog_version" not in update.metadata


async def test_validation_failure_creates_one_error_observation() -> None:
    observability = RecordingObservabilityClient()
    provider = StubToolDecisionProvider(create_tool_call_response(arguments_json="{malformed-json"))
    gateway = LLMToolDecisionGateway(
        provider=provider,
        clock=create_clock(10.0, 10.005),
        observability_client=cast(Any, observability),
    )

    with pytest.raises(LLMGatewayFailure) as captured:
        await gateway.decide(_observability_request())

    assert len(observability.scopes) == 1
    assert observability.started_attributes[0].metadata["is_repair"] is False
    assert observability.started_attributes[0].metadata["invocation_sequence"] == 1
    assert observability.scopes[0].updates[0].status is ObservationStatus.ERROR
    assert observability.scopes[0].updates[0].error_code == (
        LLMErrorCode.TOOL_DECISION_VALIDATION_FAILED.value
    )
    assert captured.value.error_code is (LLMErrorCode.TOOL_DECISION_VALIDATION_FAILED)
    assert len(captured.value.invocations) == 1
    serialized = repr(observability.scopes[0].updates[0])
    assert "malformed" not in serialized


async def test_non_repairable_provider_failure_creates_one_error_observation() -> None:
    observability = RecordingObservabilityClient()
    provider = StubToolDecisionProvider(LLMTimeoutError(provider_request_id="provider-request-1"))
    gateway = LLMToolDecisionGateway(
        provider=provider,
        clock=create_clock(10.0, 10.05),
        observability_client=cast(Any, observability),
    )

    with pytest.raises(LLMGatewayFailure) as captured:
        await gateway.decide(_observability_request())

    assert len(observability.scopes) == 1
    update = observability.scopes[0].updates[0]
    assert update.status is ObservationStatus.ERROR
    assert update.error_code == LLMErrorCode.TIMEOUT.value
    assert update.metadata["provider_request_id"] == "provider-request-1"
    assert captured.value.error_code is LLMErrorCode.TIMEOUT
    assert len(captured.value.invocations) == 1
    assert observability.managers[0].exit_args == (None, None, None)


async def test_observability_start_failure_preserves_success() -> None:
    observability = RecordingObservabilityClient()
    observability.start_error = RuntimeError("start failed")
    provider = StubToolDecisionProvider(create_tool_call_response())
    gateway = LLMToolDecisionGateway(
        provider=provider,
        clock=create_clock(10.0, 10.025),
        observability_client=cast(Any, observability),
    )

    result = await gateway.decide(create_request())

    assert result.accepted_invocation_sequence == 1
    assert isinstance(result.decision, LLMExecutableToolCallDecision)
    assert len(observability.scopes) == 0


async def test_observability_update_failure_preserves_success() -> None:
    observability = RecordingObservabilityClient()
    observability.update_error = RuntimeError("update failed")
    provider = StubToolDecisionProvider(create_tool_call_response())
    gateway = LLMToolDecisionGateway(
        provider=provider,
        clock=create_clock(10.0, 10.025),
        observability_client=cast(Any, observability),
    )

    result = await gateway.decide(create_request())

    assert result.accepted_invocation_sequence == 1
    assert observability.scopes[0].updates == []


async def test_observability_exit_failure_preserves_success() -> None:
    observability = RecordingObservabilityClient()
    observability.exit_error = RuntimeError("exit failed")
    provider = StubToolDecisionProvider(create_tool_call_response())
    gateway = LLMToolDecisionGateway(
        provider=provider,
        clock=create_clock(10.0, 10.025),
        observability_client=cast(Any, observability),
    )

    result = await gateway.decide(create_request())

    assert result.accepted_invocation_sequence == 1
    assert len(observability.scopes[0].updates) == 1


async def test_business_exception_is_preserved_exactly() -> None:
    observability = RecordingObservabilityClient()
    provider_error = LLMTimeoutError(provider_request_id="provider-request-1")
    provider = StubToolDecisionProvider(provider_error)
    gateway = LLMToolDecisionGateway(
        provider=provider,
        clock=create_clock(10.0, 10.05),
        observability_client=cast(Any, observability),
    )

    with pytest.raises(LLMGatewayFailure) as captured:
        await gateway.decide(create_request())

    assert captured.value.error is provider_error
    assert observability.managers[0].exit_args == (None, None, None)


async def test_provenance_mismatch_records_unhandled_error_observation() -> None:
    observability = RecordingObservabilityClient()
    provider = StubToolDecisionProvider(create_tool_call_response(provider="unexpected-provider"))
    gateway = LLMToolDecisionGateway(
        provider=provider,
        clock=create_clock(10.0, 10.005),
        observability_client=cast(Any, observability),
    )

    with pytest.raises(RuntimeError, match="provenance does not match"):
        await gateway.decide(create_request())

    assert len(observability.scopes) == 1
    assert observability.scopes[0].updates[0].status is ObservationStatus.ERROR
    assert observability.scopes[0].updates[0].error_code == ("unhandled_business_error")


async def test_terminal_control_decision_omits_tool_safety() -> None:
    observability = RecordingObservabilityClient()
    provider = StubToolDecisionProvider(
        create_tool_call_response(
            function_name=(COMPLETE_SUPPORT_ANALYSIS_CONTROL_NAME),
            arguments_json=(
                "{"
                '"recommended_action":"respond",'
                '"evidence_sufficient":true,'
                '"requires_human_review":false,'
                '"decision_summary":'
                '"Relevant evidence is available."'
                "}"
            ),
            provider=_MOCK_PROVIDER,
            model=_MOCK_MODEL,
        ),
        provider_name=_MOCK_PROVIDER,
    )
    gateway = LLMToolDecisionGateway(
        provider=provider,
        clock=create_clock(1.0, 1.01),
        observability_client=cast(Any, observability),
    )

    result = await gateway.decide(_observability_request(model=_MOCK_MODEL))

    assert isinstance(result.decision, LLMTerminalControlDecision)
    update = observability.scopes[0].updates[0]
    assert update.metadata["decision_kind"] == "terminal_control"
    assert "selected_tool_safety" not in update.metadata
    assert COMPLETE_SUPPORT_ANALYSIS_CONTROL_NAME not in update.metadata
    assert COMPLETE_HUMAN_APPROVED_SUPPORT_ANALYSIS_CONTROL_NAME not in (
        repr(observability.started_attributes[0].metadata)
    )
