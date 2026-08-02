"""Unit tests for provider-independent LLM tool decisions."""

from collections.abc import Callable
from typing import Annotated, Literal, cast
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
    LLMProviderFunctionCallResponse,
    LLMProviderToolDecisionRequest,
    LLMTerminalControlDecision,
    LLMTerminalControlDefinition,
    LLMToolDecisionGateway,
    LLMToolDecisionProvider,
    LLMToolDecisionRequest,
)

DOCUMENT_ID = UUID("11111111-1111-4111-8111-111111111111")


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
    ) -> None:
        self._outcome = outcome
        self.requests: list[LLMProviderToolDecisionRequest] = []
        self.close_calls = 0

    @property
    def provider_name(self) -> str:
        return "stub"

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
        tools=tools or (create_search_definition(),),
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
) -> LLMProviderFunctionCallResponse:
    """Create one successful synthetic function call."""

    return LLMProviderFunctionCallResponse(
        provider_tool_call_id="tool-call-1",
        function_name=function_name,
        arguments_json=arguments_json,
        provider=provider,
        model=model,
        provider_request_id="provider-request-1",
        usage=LLMTokenUsage(
            input_tokens=30,
            output_tokens=10,
            total_tokens=40,
        ),
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
