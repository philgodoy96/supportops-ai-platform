"""Unit tests for controlled-support inspection read models."""

from datetime import UTC, datetime
from decimal import Decimal
from uuid import UUID

import pytest

from supportops.agent_graph.domain.state import (
    CONTROLLED_SUPPORT_WORKFLOW_NAME,
    CONTROLLED_SUPPORT_WORKFLOW_VERSION,
)
from supportops.agent_tools.domain.audit import (
    AgentToolCallStatus,
)
from supportops.agent_tools.domain.contracts import (
    ToolSafetyLevel,
)
from supportops.agent_tools.tools.service_status import (
    ServiceOperationalStatus,
)
from supportops.ai.gateway.errors import LLMErrorCode
from supportops.ai.gateway.results import (
    LLMInvocationStatus,
)
from supportops.ai.schemas.ticket_classification import (
    TicketCategory,
    TicketIntent,
    TicketSentiment,
    TicketUrgency,
)
from supportops.modules.agent_runs.domain.models import (
    AgentRunStatus,
)
from supportops.modules.controlled_support_inspection.domain.models import (
    AgentRunInspectionSummary,
    ClassificationInspection,
    ControlledSupportInspection,
    ControlledSupportInspectionStatus,
    LLMInvocationInspection,
    LLMUsageSummary,
    RecommendationInspection,
    RecommendationPromptInspection,
    ServiceStatusSummary,
    TerminalAnalysisInspection,
    ToolCallInspection,
    map_agent_run_inspection_status,
)
from supportops.modules.support_recommendations.domain.models import (
    SupportRecommendationAction,
)

_NOW = datetime(
    2026,
    8,
    2,
    18,
    0,
    tzinfo=UTC,
)

_WORKSPACE_ID = UUID("10000000-0000-4000-8000-000000000001")
_TICKET_ID = UUID("20000000-0000-4000-8000-000000000002")
_AGENT_RUN_ID = UUID("30000000-0000-4000-8000-000000000003")
_CLASSIFICATION_ID = UUID("40000000-0000-4000-8000-000000000004")
_ATTEMPT_ONE_ID = UUID("50000000-0000-4000-8000-000000000005")
_ATTEMPT_TWO_ID = UUID("60000000-0000-4000-8000-000000000006")
_TOOL_ONE_ID = UUID("70000000-0000-4000-8000-000000000007")
_TOOL_TWO_ID = UUID("80000000-0000-4000-8000-000000000008")
_INVOCATION_ONE_ID = UUID("90000000-0000-4000-8000-000000000009")
_INVOCATION_TWO_ID = UUID("a0000000-0000-4000-8000-000000000010")
_RECOMMENDATION_ID = UUID("b0000000-0000-4000-8000-000000000011")


@pytest.mark.parametrize(
    ("durable_status", "inspection_status"),
    [
        (
            AgentRunStatus.QUEUED,
            ControlledSupportInspectionStatus.QUEUED,
        ),
        (
            AgentRunStatus.RUNNING,
            ControlledSupportInspectionStatus.RUNNING,
        ),
        (
            AgentRunStatus.RETRY_SCHEDULED,
            ControlledSupportInspectionStatus.RETRYING,
        ),
        (
            AgentRunStatus.FAILED,
            ControlledSupportInspectionStatus.FAILED,
        ),
        (
            AgentRunStatus.SUCCEEDED,
            ControlledSupportInspectionStatus.COMPLETED,
        ),
    ],
)
def test_maps_every_durable_agent_run_status(
    durable_status: AgentRunStatus,
    inspection_status: ControlledSupportInspectionStatus,
) -> None:
    assert map_agent_run_inspection_status(durable_status) is inspection_status


def _agent_run(
    *,
    status: ControlledSupportInspectionStatus = (ControlledSupportInspectionStatus.RUNNING),
    attempt_count: int = 2,
) -> AgentRunInspectionSummary:
    terminal = status in {
        ControlledSupportInspectionStatus.COMPLETED,
        ControlledSupportInspectionStatus.FAILED,
    }
    has_error = status in {
        ControlledSupportInspectionStatus.RETRYING,
        ControlledSupportInspectionStatus.FAILED,
    }

    return AgentRunInspectionSummary(
        id=_AGENT_RUN_ID,
        workspace_id=_WORKSPACE_ID,
        ticket_id=_TICKET_ID,
        workflow_name=CONTROLLED_SUPPORT_WORKFLOW_NAME,
        workflow_version=(CONTROLLED_SUPPORT_WORKFLOW_VERSION),
        status=status,
        attempt_count=attempt_count,
        max_attempts=3,
        created_at=_NOW,
        first_started_at=(None if attempt_count == 0 else _NOW),
        completed_at=_NOW if terminal else None,
        last_error_code=("synthetic_failure" if has_error else None),
    )


def _classification() -> ClassificationInspection:
    return ClassificationInspection(
        id=_CLASSIFICATION_ID,
        category=TicketCategory.ACCOUNT_ACCESS,
        intent=TicketIntent.REQUEST_ACCESS,
        urgency=TicketUrgency.NORMAL,
        sentiment=TicketSentiment.NEUTRAL,
        requires_human_review=False,
        summary=("The customer needs account recovery guidance."),
        created_at=_NOW,
    )


def _tool_call(
    *,
    tool_call_id: UUID,
    attempt_id: UUID,
    attempt_number: int,
    sequence: int,
) -> ToolCallInspection:
    return ToolCallInspection(
        id=tool_call_id,
        agent_run_attempt_id=attempt_id,
        attempt_number=attempt_number,
        sequence=sequence,
        tool_name="lookup_service_status",
        tool_version=1,
        safety_level=ToolSafetyLevel.READ_ONLY,
        status=AgentToolCallStatus.SUCCEEDED,
        latency_ms=5,
        error_code=None,
        started_at=_NOW,
        finished_at=_NOW,
        result_summary=ServiceStatusSummary(
            service_name="payments-api",
            status=ServiceOperationalStatus.OPERATIONAL,
            incident_reference=None,
        ),
    )


def _invocation(
    *,
    invocation_id: UUID,
    attempt_id: UUID,
    attempt_number: int,
    sequence: int,
    status: LLMInvocationStatus = (LLMInvocationStatus.SUCCEEDED),
    cost: Decimal | None = Decimal("0.010000"),
) -> LLMInvocationInspection:
    return LLMInvocationInspection(
        id=invocation_id,
        agent_run_attempt_id=attempt_id,
        attempt_number=attempt_number,
        invocation_sequence=sequence,
        status=status,
        provider="mock",
        model="mock-model",
        prompt_id="support-tool-decision",
        prompt_version=1,
        prompt_content_hash="a" * 64,
        schema_version="provider-tool-decision-v1",
        input_tokens=10,
        cached_input_tokens=2,
        output_tokens=5,
        reasoning_tokens=1,
        total_tokens=15,
        estimated_total_cost_usd=cost,
        pricing_found=cost is not None,
        latency_ms=20,
        error_code=(None if status is LLMInvocationStatus.SUCCEEDED else LLMErrorCode.TIMEOUT),
        created_at=_NOW,
    )


def _terminal_analysis(
    *,
    action: SupportRecommendationAction = (SupportRecommendationAction.RESPOND),
) -> TerminalAnalysisInspection:
    return TerminalAnalysisInspection(
        recommended_action=action,
        evidence_sufficient=(action is not SupportRecommendationAction.REQUEST_MORE_INFORMATION),
        requires_human_review=(action is SupportRecommendationAction.RECOMMEND_ESCALATION),
        decision_summary=("The available evidence supports the selected action."),
    )


def _recommendation(
    *,
    action: SupportRecommendationAction = (SupportRecommendationAction.RESPOND),
) -> RecommendationInspection:
    return RecommendationInspection(
        id=_RECOMMENDATION_ID,
        recommended_action=action,
        response_text=("Follow the documented account recovery steps."),
        requires_human_review=(action is SupportRecommendationAction.RECOMMEND_ESCALATION),
        decision_summary=("The available evidence supports the selected action."),
        prompt=RecommendationPromptInspection(
            id="support-recommendation-draft",
            version=1,
            content_hash="b" * 64,
        ),
        provider="mock",
        model="mock-model",
        created_at=_NOW,
        citations=(),
    )


def test_usage_summary_aggregates_historical_values() -> None:
    invocations = (
        _invocation(
            invocation_id=_INVOCATION_ONE_ID,
            attempt_id=_ATTEMPT_ONE_ID,
            attempt_number=1,
            sequence=1,
        ),
        _invocation(
            invocation_id=_INVOCATION_TWO_ID,
            attempt_id=_ATTEMPT_TWO_ID,
            attempt_number=2,
            sequence=1,
            status=LLMInvocationStatus.TIMED_OUT,
            cost=None,
        ),
    )

    summary = LLMUsageSummary.from_invocations(invocations)

    assert summary.invocation_count == 2
    assert summary.successful_invocation_count == 1
    assert summary.failed_invocation_count == 1
    assert summary.input_tokens == 20
    assert summary.cached_input_tokens == 4
    assert summary.output_tokens == 10
    assert summary.reasoning_tokens == 2
    assert summary.total_tokens == 30
    assert summary.estimated_cost_usd == Decimal("0.010000")
    assert summary.unpriced_invocation_count == 1


def test_allows_sequence_restart_for_new_attempt() -> None:
    tool_calls = (
        _tool_call(
            tool_call_id=_TOOL_ONE_ID,
            attempt_id=_ATTEMPT_ONE_ID,
            attempt_number=1,
            sequence=1,
        ),
        _tool_call(
            tool_call_id=_TOOL_TWO_ID,
            attempt_id=_ATTEMPT_TWO_ID,
            attempt_number=2,
            sequence=1,
        ),
    )
    invocations = (
        _invocation(
            invocation_id=_INVOCATION_ONE_ID,
            attempt_id=_ATTEMPT_ONE_ID,
            attempt_number=1,
            sequence=1,
        ),
        _invocation(
            invocation_id=_INVOCATION_TWO_ID,
            attempt_id=_ATTEMPT_TWO_ID,
            attempt_number=2,
            sequence=1,
        ),
    )

    inspection = ControlledSupportInspection(
        agent_run=_agent_run(),
        classification=_classification(),
        tool_calls=tool_calls,
        terminal_analysis=None,
        recommendation=None,
        llm_usage=(LLMUsageSummary.from_invocations(invocations)),
        llm_invocations=invocations,
    )

    assert inspection.tool_calls == tool_calls
    assert inspection.llm_invocations == invocations


def test_rejects_tool_sequence_gap_within_attempt() -> None:
    tool_calls = (
        _tool_call(
            tool_call_id=_TOOL_ONE_ID,
            attempt_id=_ATTEMPT_ONE_ID,
            attempt_number=1,
            sequence=2,
        ),
    )

    with pytest.raises(
        ValueError,
        match="contiguous and one-based",
    ):
        ControlledSupportInspection(
            agent_run=_agent_run(attempt_count=1),
            classification=_classification(),
            tool_calls=tool_calls,
            terminal_analysis=None,
            recommendation=None,
            llm_usage=(LLMUsageSummary.from_invocations(())),
            llm_invocations=(),
        )


def test_completed_run_requires_recommendation() -> None:
    with pytest.raises(
        ValueError,
        match="require a persisted recommendation",
    ):
        ControlledSupportInspection(
            agent_run=_agent_run(
                status=(ControlledSupportInspectionStatus.COMPLETED),
                attempt_count=1,
            ),
            classification=_classification(),
            tool_calls=(),
            terminal_analysis=_terminal_analysis(),
            recommendation=None,
            llm_usage=(LLMUsageSummary.from_invocations(())),
            llm_invocations=(),
        )


def test_recommendation_action_must_match_terminal_analysis() -> None:
    with pytest.raises(
        ValueError,
        match="must match terminal analysis",
    ):
        ControlledSupportInspection(
            agent_run=_agent_run(
                status=(ControlledSupportInspectionStatus.COMPLETED),
                attempt_count=1,
            ),
            classification=_classification(),
            tool_calls=(),
            terminal_analysis=_terminal_analysis(),
            recommendation=_recommendation(
                action=(SupportRecommendationAction.RECOMMEND_ESCALATION)
            ),
            llm_usage=(LLMUsageSummary.from_invocations(())),
            llm_invocations=(),
        )


def test_queued_run_rejects_execution_progress() -> None:
    with pytest.raises(
        ValueError,
        match="must not contain execution progress",
    ):
        ControlledSupportInspection(
            agent_run=_agent_run(
                status=(ControlledSupportInspectionStatus.QUEUED),
                attempt_count=0,
            ),
            classification=_classification(),
            tool_calls=(),
            terminal_analysis=None,
            recommendation=None,
            llm_usage=(LLMUsageSummary.from_invocations(())),
            llm_invocations=(),
        )


def test_rejects_usage_summary_that_does_not_match_invocations() -> None:
    invocation = _invocation(
        invocation_id=_INVOCATION_ONE_ID,
        attempt_id=_ATTEMPT_ONE_ID,
        attempt_number=1,
        sequence=1,
    )

    with pytest.raises(
        ValueError,
        match="must match llm_invocations",
    ):
        ControlledSupportInspection(
            agent_run=_agent_run(attempt_count=1),
            classification=_classification(),
            tool_calls=(),
            terminal_analysis=None,
            recommendation=None,
            llm_usage=(LLMUsageSummary.from_invocations(())),
            llm_invocations=(invocation,),
        )
