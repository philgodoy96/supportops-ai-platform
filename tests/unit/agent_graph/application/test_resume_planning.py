"""Unit tests for approval-aware graph resume planning."""

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from pydantic import JsonValue

from supportops.agent_graph.application.approval_interrupt import (
    ApprovalInterruptPayload,
)
from supportops.agent_graph.application.resume_planning import (
    HumanApprovedGraphResumePlanner,
    HumanApprovedResumePlanningContext,
    build_approval_resume_value,
)
from supportops.agent_graph.domain.human_approved_state import (
    HumanApprovedSupportGraphState,
    create_initial_human_approved_support_state,
)
from supportops.agent_graph.domain.resume_planning import (
    ApprovalResumeDecisionStatus,
    CompletedGraphExecution,
    ContinueGraphExecution,
    IncompatibleGraphState,
    InitialGraphExecution,
    ResumeGraphExecution,
)
from supportops.modules.approvals.domain.models import (
    ApprovalRequestStatus,
)

_NOW = datetime(2026, 8, 3, 17, 0, tzinfo=UTC)


def _context() -> HumanApprovedResumePlanningContext:
    return HumanApprovedResumePlanningContext(
        workspace_id=uuid4(),
        ticket_id=uuid4(),
        agent_run_id=uuid4(),
    )


def _interrupted_state(
    context: HumanApprovedResumePlanningContext,
) -> tuple[HumanApprovedSupportGraphState, ApprovalInterruptPayload]:
    approval_request_id = uuid4()
    agent_tool_call_id = uuid4()
    expires_at = (_NOW + timedelta(days=1)).isoformat()
    proposed_input: dict[str, JsonValue] = {
        "target_queue": "engineering_support",
        "reason": "A product defect requires investigation.",
    }
    state = create_initial_human_approved_support_state(
        workspace_id=context.workspace_id,
        ticket_id=context.ticket_id,
        agent_run_id=context.agent_run_id,
    )
    state.update(
        {
            "run_context_loaded": True,
            "decision_kind": "sensitive_tool",
            "decision_invocation_id": str(uuid4()),
            "decision_summary": ("A product defect requires investigation."),
            "proposed_tool_provider_call_id": "call-1",
            "proposed_tool_name": "escalate_ticket",
            "proposed_tool_version": 1,
            "proposed_tool_input": proposed_input,
            "proposed_tool_fingerprint": "a" * 64,
            "approval_request_reason": ("A product defect requires investigation."),
            "agent_tool_call_id": str(agent_tool_call_id),
            "approval_request_id": str(approval_request_id),
            "approval_status": "pending",
            "approval_expires_at": expires_at,
        },
    )
    payload = ApprovalInterruptPayload(
        approval_request_id=approval_request_id,
        agent_tool_call_id=agent_tool_call_id,
        agent_run_id=context.agent_run_id,
        ticket_id=context.ticket_id,
        tool_name="escalate_ticket",
        tool_version=1,
        proposed_input=proposed_input,
        request_reason=("A product defect requires investigation."),
        expires_at=expires_at,
    )
    return state, payload


def _planner(
    *,
    approval: object | None = None,
    tool_call: object | None = None,
) -> HumanApprovedGraphResumePlanner:
    return HumanApprovedGraphResumePlanner(
        approval_request_repository=SimpleNamespace(
            get_by_id=AsyncMock(return_value=approval),
        ),
        tool_call_query_repository=SimpleNamespace(
            get_sensitive_by_identity=AsyncMock(
                return_value=tool_call,
            ),
        ),
    )


@pytest.mark.asyncio
async def test_empty_checkpoint_plans_initial_execution() -> None:
    plan = await _planner().plan(
        context=_context(),
        checkpoint_values={},
        checkpoint_interrupts=(),
    )

    assert isinstance(plan, InitialGraphExecution)


@pytest.mark.asyncio
async def test_non_interrupted_checkpoint_plans_continue() -> None:
    context = _context()
    state = create_initial_human_approved_support_state(
        workspace_id=context.workspace_id,
        ticket_id=context.ticket_id,
        agent_run_id=context.agent_run_id,
    )
    state["run_context_loaded"] = True

    plan = await _planner().plan(
        context=context,
        checkpoint_values=state,
        checkpoint_interrupts=(),
    )

    assert isinstance(plan, ContinueGraphExecution)


@pytest.mark.asyncio
async def test_completed_checkpoint_plans_completed_execution() -> None:
    context = _context()
    state = create_initial_human_approved_support_state(
        workspace_id=context.workspace_id,
        ticket_id=context.ticket_id,
        agent_run_id=context.agent_run_id,
    )
    state.update(
        {
            "run_context_loaded": True,
            "decision_kind": "terminal",
            "decision_invocation_id": str(uuid4()),
            "decision_summary": "Respond with available evidence.",
            "analysis_recommended_action": "respond",
            "analysis_evidence_sufficient": True,
            "analysis_requires_human_review": False,
            "recommendation_invocation_id": str(uuid4()),
            "recommendation_id": str(uuid4()),
        },
    )

    plan = await _planner().plan(
        context=context,
        checkpoint_values=state,
        checkpoint_interrupts=(),
    )

    assert isinstance(plan, CompletedGraphExecution)


@pytest.mark.asyncio
async def test_pending_approval_fails_closed() -> None:
    context = _context()
    state, payload = _interrupted_state(context)
    approval = SimpleNamespace(
        id=payload.approval_request_id,
        workspace_id=context.workspace_id,
        ticket_id=context.ticket_id,
        agent_run_id=context.agent_run_id,
        agent_tool_call_id=payload.agent_tool_call_id,
        status=ApprovalRequestStatus.PENDING,
        tool_name=payload.tool_name,
        tool_version=payload.tool_version,
        input_fingerprint="a" * 64,
        proposed_input=payload.proposed_input,
        request_reason=payload.request_reason,
        expires_at=datetime.fromisoformat(payload.expires_at),
    )
    tool_call = SimpleNamespace(
        id=payload.agent_tool_call_id,
        workspace_id=context.workspace_id,
        ticket_id=context.ticket_id,
        agent_run_id=context.agent_run_id,
        status="pending_approval",
        tool_name=payload.tool_name,
        tool_version=payload.tool_version,
        input_fingerprint="a" * 64,
        safe_input=payload.proposed_input,
    )

    plan = await _planner(
        approval=approval,
        tool_call=tool_call,
    ).plan(
        context=context,
        checkpoint_values=state,
        checkpoint_interrupts=(SimpleNamespace(value=payload.to_interrupt_value()),),
    )

    assert isinstance(plan, IncompatibleGraphState)
    assert plan.error_code == "approval_request_still_pending"


@pytest.mark.asyncio
async def test_approved_request_plans_resume() -> None:
    context = _context()
    state, payload = _interrupted_state(context)
    approval = SimpleNamespace(
        id=payload.approval_request_id,
        workspace_id=context.workspace_id,
        ticket_id=context.ticket_id,
        agent_run_id=context.agent_run_id,
        agent_tool_call_id=payload.agent_tool_call_id,
        status=ApprovalRequestStatus.APPROVED,
        tool_name=payload.tool_name,
        tool_version=payload.tool_version,
        input_fingerprint="a" * 64,
        proposed_input=payload.proposed_input,
        request_reason=payload.request_reason,
        expires_at=datetime.fromisoformat(payload.expires_at),
    )
    tool_call = SimpleNamespace(
        id=payload.agent_tool_call_id,
        workspace_id=context.workspace_id,
        ticket_id=context.ticket_id,
        agent_run_id=context.agent_run_id,
        status="pending_approval",
        tool_name=payload.tool_name,
        tool_version=payload.tool_version,
        input_fingerprint="a" * 64,
        safe_input=payload.proposed_input,
    )

    plan = await _planner(
        approval=approval,
        tool_call=tool_call,
    ).plan(
        context=context,
        checkpoint_values=state,
        checkpoint_interrupts=(SimpleNamespace(value=payload.to_interrupt_value()),),
    )

    assert isinstance(plan, ResumeGraphExecution)
    assert plan.decision_status is (ApprovalResumeDecisionStatus.APPROVED)
    assert build_approval_resume_value(plan) == {
        "approval_request_id": str(payload.approval_request_id),
        "agent_tool_call_id": str(payload.agent_tool_call_id),
        "decision_status": "approved",
    }


@pytest.mark.asyncio
async def test_tool_call_status_mismatch_fails_closed() -> None:
    context = _context()
    state, payload = _interrupted_state(context)
    approval = SimpleNamespace(
        id=payload.approval_request_id,
        workspace_id=context.workspace_id,
        ticket_id=context.ticket_id,
        agent_run_id=context.agent_run_id,
        agent_tool_call_id=payload.agent_tool_call_id,
        status=ApprovalRequestStatus.APPROVED,
        tool_name=payload.tool_name,
        tool_version=payload.tool_version,
        input_fingerprint="a" * 64,
        proposed_input=payload.proposed_input,
        request_reason=payload.request_reason,
        expires_at=datetime.fromisoformat(payload.expires_at),
    )
    tool_call = SimpleNamespace(
        id=payload.agent_tool_call_id,
        workspace_id=context.workspace_id,
        ticket_id=context.ticket_id,
        agent_run_id=context.agent_run_id,
        status="rejected",
        tool_name=payload.tool_name,
        tool_version=payload.tool_version,
        input_fingerprint="a" * 64,
        safe_input=payload.proposed_input,
    )

    plan = await _planner(
        approval=approval,
        tool_call=tool_call,
    ).plan(
        context=context,
        checkpoint_values=state,
        checkpoint_interrupts=(SimpleNamespace(value=payload.to_interrupt_value()),),
    )

    assert isinstance(plan, IncompatibleGraphState)
    assert plan.error_code == "approval_tool_call_status_mismatch"


@pytest.mark.asyncio
async def test_missing_checkpoint_with_interrupt_fails_closed() -> None:
    plan = await _planner().plan(
        context=_context(),
        checkpoint_values={},
        checkpoint_interrupts=(SimpleNamespace(value={"unexpected": "value"}),),
    )

    assert isinstance(plan, IncompatibleGraphState)
    assert plan.error_code == ("human_approved_checkpoint_interrupt_without_state")


def _approved_pair(
    context: HumanApprovedResumePlanningContext,
    payload: ApprovalInterruptPayload,
    *,
    approval_status: ApprovalRequestStatus = ApprovalRequestStatus.APPROVED,
    tool_status: str = "pending_approval",
    fingerprint: str = "a" * 64,
    proposed_input: dict[str, JsonValue] | None = None,
    workspace_id: object | None = None,
    ticket_id: object | None = None,
    agent_run_id: object | None = None,
) -> tuple[SimpleNamespace, SimpleNamespace]:
    input_data = proposed_input or payload.proposed_input
    approval = SimpleNamespace(
        id=payload.approval_request_id,
        workspace_id=workspace_id or context.workspace_id,
        ticket_id=ticket_id or context.ticket_id,
        agent_run_id=agent_run_id or context.agent_run_id,
        agent_tool_call_id=payload.agent_tool_call_id,
        status=approval_status,
        tool_name=payload.tool_name,
        tool_version=payload.tool_version,
        input_fingerprint=fingerprint,
        proposed_input=input_data,
        request_reason=payload.request_reason,
        expires_at=datetime.fromisoformat(payload.expires_at),
    )
    tool_call = SimpleNamespace(
        id=payload.agent_tool_call_id,
        workspace_id=workspace_id or context.workspace_id,
        ticket_id=ticket_id or context.ticket_id,
        agent_run_id=agent_run_id or context.agent_run_id,
        status=tool_status,
        tool_name=payload.tool_name,
        tool_version=payload.tool_version,
        input_fingerprint=fingerprint,
        safe_input=input_data,
    )
    return approval, tool_call


@pytest.mark.asyncio
async def test_rejected_request_plans_resume() -> None:
    context = _context()
    state, payload = _interrupted_state(context)
    approval, tool_call = _approved_pair(
        context,
        payload,
        approval_status=ApprovalRequestStatus.REJECTED,
        tool_status="rejected",
    )

    plan = await _planner(
        approval=approval,
        tool_call=tool_call,
    ).plan(
        context=context,
        checkpoint_values=state,
        checkpoint_interrupts=(SimpleNamespace(value=payload.to_interrupt_value()),),
    )

    assert isinstance(plan, ResumeGraphExecution)
    assert plan.decision_status is (ApprovalResumeDecisionStatus.REJECTED)
    assert build_approval_resume_value(plan) == {
        "approval_request_id": str(payload.approval_request_id),
        "agent_tool_call_id": str(payload.agent_tool_call_id),
        "decision_status": "rejected",
    }
    assert state["approval_status"] == "pending"


@pytest.mark.asyncio
async def test_expired_request_plans_resume() -> None:
    context = _context()
    state, payload = _interrupted_state(context)
    approval, tool_call = _approved_pair(
        context,
        payload,
        approval_status=ApprovalRequestStatus.EXPIRED,
        tool_status="expired",
    )

    plan = await _planner(
        approval=approval,
        tool_call=tool_call,
    ).plan(
        context=context,
        checkpoint_values=state,
        checkpoint_interrupts=(SimpleNamespace(value=payload.to_interrupt_value()),),
    )

    assert isinstance(plan, ResumeGraphExecution)
    assert plan.decision_status is (ApprovalResumeDecisionStatus.EXPIRED)
    assert build_approval_resume_value(plan) == {
        "approval_request_id": str(payload.approval_request_id),
        "agent_tool_call_id": str(payload.agent_tool_call_id),
        "decision_status": "expired",
    }
    assert state["approval_status"] == "pending"


@pytest.mark.asyncio
async def test_approval_ids_without_interrupt_fails_closed() -> None:
    context = _context()
    state, _payload = _interrupted_state(context)

    plan = await _planner().plan(
        context=context,
        checkpoint_values=state,
        checkpoint_interrupts=(),
    )

    assert isinstance(plan, IncompatibleGraphState)
    assert plan.error_code == "approval_state_without_interrupt"


@pytest.mark.asyncio
async def test_multiple_active_interrupts_fails_closed() -> None:
    context = _context()
    state, payload = _interrupted_state(context)
    interrupt = SimpleNamespace(value=payload.to_interrupt_value())

    plan = await _planner().plan(
        context=context,
        checkpoint_values=state,
        checkpoint_interrupts=(interrupt, interrupt),
    )

    assert isinstance(plan, IncompatibleGraphState)
    assert plan.error_code == "approval_interrupt_count_invalid"


@pytest.mark.asyncio
async def test_malformed_interrupt_payload_fails_closed() -> None:
    context = _context()
    state, _payload = _interrupted_state(context)

    plan = await _planner().plan(
        context=context,
        checkpoint_values=state,
        checkpoint_interrupts=(SimpleNamespace(value={"bad": True}),),
    )

    assert isinstance(plan, IncompatibleGraphState)
    assert plan.error_code == "approval_interrupt_payload_invalid"


@pytest.mark.asyncio
async def test_interrupt_state_mismatch_fails_closed() -> None:
    context = _context()
    state, payload = _interrupted_state(context)
    mismatched = payload.model_copy(
        update={"request_reason": "A different reason."},
    )

    plan = await _planner().plan(
        context=context,
        checkpoint_values=state,
        checkpoint_interrupts=(SimpleNamespace(value=mismatched.to_interrupt_value()),),
    )

    assert isinstance(plan, IncompatibleGraphState)
    assert plan.error_code == "approval_interrupt_state_mismatch"


@pytest.mark.asyncio
async def test_missing_approval_request_fails_closed() -> None:
    context = _context()
    state, payload = _interrupted_state(context)
    _approval, tool_call = _approved_pair(context, payload)

    plan = await _planner(
        approval=None,
        tool_call=tool_call,
    ).plan(
        context=context,
        checkpoint_values=state,
        checkpoint_interrupts=(SimpleNamespace(value=payload.to_interrupt_value()),),
    )

    assert isinstance(plan, IncompatibleGraphState)
    assert plan.error_code == "approval_request_not_found_for_resume"


@pytest.mark.asyncio
async def test_missing_tool_call_fails_closed() -> None:
    context = _context()
    state, payload = _interrupted_state(context)
    approval, _tool_call = _approved_pair(context, payload)

    plan = await _planner(
        approval=approval,
        tool_call=None,
    ).plan(
        context=context,
        checkpoint_values=state,
        checkpoint_interrupts=(SimpleNamespace(value=payload.to_interrupt_value()),),
    )

    assert isinstance(plan, IncompatibleGraphState)
    assert plan.error_code == "agent_tool_call_not_found_for_resume"


@pytest.mark.asyncio
async def test_wrong_workspace_fails_closed() -> None:
    context = _context()
    state, payload = _interrupted_state(context)
    approval, tool_call = _approved_pair(
        context,
        payload,
        workspace_id=uuid4(),
    )

    plan = await _planner(
        approval=approval,
        tool_call=tool_call,
    ).plan(
        context=context,
        checkpoint_values=state,
        checkpoint_interrupts=(SimpleNamespace(value=payload.to_interrupt_value()),),
    )

    assert isinstance(plan, IncompatibleGraphState)
    assert plan.error_code == "approval_durable_state_mismatch"


@pytest.mark.asyncio
async def test_wrong_ticket_fails_closed() -> None:
    context = _context()
    state, payload = _interrupted_state(context)
    approval, tool_call = _approved_pair(
        context,
        payload,
        ticket_id=uuid4(),
    )

    plan = await _planner(
        approval=approval,
        tool_call=tool_call,
    ).plan(
        context=context,
        checkpoint_values=state,
        checkpoint_interrupts=(SimpleNamespace(value=payload.to_interrupt_value()),),
    )

    assert isinstance(plan, IncompatibleGraphState)
    assert plan.error_code == "approval_durable_state_mismatch"


@pytest.mark.asyncio
async def test_wrong_agent_run_fails_closed() -> None:
    context = _context()
    state, payload = _interrupted_state(context)
    approval, tool_call = _approved_pair(
        context,
        payload,
        agent_run_id=uuid4(),
    )

    plan = await _planner(
        approval=approval,
        tool_call=tool_call,
    ).plan(
        context=context,
        checkpoint_values=state,
        checkpoint_interrupts=(SimpleNamespace(value=payload.to_interrupt_value()),),
    )

    assert isinstance(plan, IncompatibleGraphState)
    assert plan.error_code == "approval_durable_state_mismatch"


@pytest.mark.asyncio
async def test_wrong_fingerprint_fails_closed() -> None:
    context = _context()
    state, payload = _interrupted_state(context)
    approval, tool_call = _approved_pair(
        context,
        payload,
        fingerprint="b" * 64,
    )

    plan = await _planner(
        approval=approval,
        tool_call=tool_call,
    ).plan(
        context=context,
        checkpoint_values=state,
        checkpoint_interrupts=(SimpleNamespace(value=payload.to_interrupt_value()),),
    )

    assert isinstance(plan, IncompatibleGraphState)
    assert plan.error_code == "approval_durable_state_mismatch"


@pytest.mark.asyncio
async def test_wrong_safe_input_fails_closed() -> None:
    context = _context()
    state, payload = _interrupted_state(context)
    approval, tool_call = _approved_pair(
        context,
        payload,
        proposed_input={
            "target_queue": "security_operations",
            "reason": "A different escalation reason.",
        },
    )

    plan = await _planner(
        approval=approval,
        tool_call=tool_call,
    ).plan(
        context=context,
        checkpoint_values=state,
        checkpoint_interrupts=(SimpleNamespace(value=payload.to_interrupt_value()),),
    )

    assert isinstance(plan, IncompatibleGraphState)
    assert plan.error_code == "approval_durable_state_mismatch"


@pytest.mark.asyncio
async def test_checkpoint_ownership_mismatch_fails_closed() -> None:
    context = _context()
    other = HumanApprovedResumePlanningContext(
        workspace_id=uuid4(),
        ticket_id=context.ticket_id,
        agent_run_id=context.agent_run_id,
    )
    state, payload = _interrupted_state(context)

    plan = await _planner().plan(
        context=other,
        checkpoint_values=state,
        checkpoint_interrupts=(SimpleNamespace(value=payload.to_interrupt_value()),),
    )

    assert isinstance(plan, IncompatibleGraphState)
    assert plan.error_code == "human_approved_state_ownership_mismatch"


@pytest.mark.asyncio
async def test_incompatible_graph_state_fails_closed() -> None:
    context = _context()

    plan = await _planner().plan(
        context=context,
        checkpoint_values={"schema_version": "not-a-valid-schema"},
        checkpoint_interrupts=(),
    )

    assert isinstance(plan, IncompatibleGraphState)
    assert plan.error_code == "human_approved_graph_state_incompatible"


@pytest.mark.asyncio
async def test_completed_checkpoint_with_interrupt_fails_closed() -> None:
    context = _context()
    state = create_initial_human_approved_support_state(
        workspace_id=context.workspace_id,
        ticket_id=context.ticket_id,
        agent_run_id=context.agent_run_id,
    )
    state.update(
        {
            "run_context_loaded": True,
            "decision_kind": "terminal",
            "decision_invocation_id": str(uuid4()),
            "decision_summary": "Respond with available evidence.",
            "analysis_recommended_action": "respond",
            "analysis_evidence_sufficient": True,
            "analysis_requires_human_review": False,
            "recommendation_invocation_id": str(uuid4()),
            "recommendation_id": str(uuid4()),
        },
    )

    plan = await _planner().plan(
        context=context,
        checkpoint_values=state,
        checkpoint_interrupts=(SimpleNamespace(value={"unexpected": True}),),
    )

    assert isinstance(plan, IncompatibleGraphState)
    assert plan.error_code == "completed_graph_contains_interrupt"


@pytest.mark.asyncio
async def test_unsupported_approval_status_fails_closed() -> None:
    context = _context()
    state, payload = _interrupted_state(context)
    approval, tool_call = _approved_pair(context, payload)
    approval.status = "revoked"

    plan = await _planner(
        approval=approval,
        tool_call=tool_call,
    ).plan(
        context=context,
        checkpoint_values=state,
        checkpoint_interrupts=(SimpleNamespace(value=payload.to_interrupt_value()),),
    )

    assert isinstance(plan, IncompatibleGraphState)
    assert plan.error_code == "approval_status_not_resumable"


@pytest.mark.asyncio
async def test_checkpoint_approval_status_must_remain_pending() -> None:
    context = _context()
    state, payload = _interrupted_state(context)
    state["approval_status"] = "approved"
    approval, tool_call = _approved_pair(context, payload)

    plan = await _planner(
        approval=approval,
        tool_call=tool_call,
    ).plan(
        context=context,
        checkpoint_values=state,
        checkpoint_interrupts=(SimpleNamespace(value=payload.to_interrupt_value()),),
    )

    assert isinstance(plan, IncompatibleGraphState)
    assert plan.error_code == "approval_checkpoint_status_invalid"


@pytest.mark.asyncio
async def test_approved_resume_keeps_checkpoint_pending_before_resume() -> None:
    context = _context()
    state, payload = _interrupted_state(context)
    approval, tool_call = _approved_pair(context, payload)

    plan = await _planner(
        approval=approval,
        tool_call=tool_call,
    ).plan(
        context=context,
        checkpoint_values=state,
        checkpoint_interrupts=(SimpleNamespace(value=payload.to_interrupt_value()),),
    )

    assert isinstance(plan, ResumeGraphExecution)
    assert state["approval_status"] == "pending"
    assert approval.status is ApprovalRequestStatus.APPROVED
    assert set(build_approval_resume_value(plan)) == {
        "approval_request_id",
        "agent_tool_call_id",
        "decision_status",
    }
