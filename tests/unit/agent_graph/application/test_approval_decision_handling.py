"""Unit tests for approval decision handling."""

from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest
from pydantic import ValidationError

from supportops.agent_graph.application.approval_decision_handling import (
    ApprovalDecisionAction,
    ApprovalDecisionHandlingError,
    ApprovalDecisionResumePayload,
    handle_approval_decision,
)
from supportops.agent_graph.domain.resume_planning import (
    ApprovalResumeDecisionStatus,
)
from supportops.agent_tools.domain.audit import AgentToolCall
from supportops.modules.approvals.domain.models import ApprovalRequest

_NOW = datetime(2026, 8, 3, 19, 30, tzinfo=UTC)


def _records() -> tuple[AgentToolCall, ApprovalRequest]:
    tool_call = AgentToolCall.propose_for_approval(
        workspace_id=uuid4(),
        ticket_id=uuid4(),
        agent_run_id=uuid4(),
        proposed_by_agent_run_attempt_id=uuid4(),
        sequence=1,
        provider_tool_call_id="call-1",
        tool_name="escalate_ticket",
        tool_version=1,
        input_fingerprint="a" * 64,
        safe_input={
            "target_queue": "support_operations",
            "reason": "Operational review required.",
        },
        proposed_at=_NOW,
    )
    pending = ApprovalRequest.create_pending(
        tool_call=tool_call,
        requested_by_llm_invocation_id=uuid4(),
        request_reason="Operational review required.",
        expires_at=_NOW + timedelta(days=1),
        now=_NOW,
    )
    return tool_call, pending


def test_approved_maps_to_sensitive_execution() -> None:
    tool_call, pending = _records()
    approved = pending.approve(
        actor_reference="operator:alice",
        comment=None,
        request_id=uuid4(),
        correlation_id=uuid4(),
        decided_at=_NOW + timedelta(minutes=5),
    )
    payload = ApprovalDecisionResumePayload(
        approval_request_id=approved.id,
        agent_tool_call_id=tool_call.id,
        decision_status=ApprovalResumeDecisionStatus.APPROVED,
    )

    result = handle_approval_decision(
        payload=payload,
        approval_request=approved,
    )

    assert result.action is (ApprovalDecisionAction.EXECUTE_SENSITIVE_TOOL)


def test_rejected_maps_to_continue_without_execution() -> None:
    tool_call, pending = _records()
    rejected = pending.reject(
        actor_reference="operator:alice",
        comment="Do not escalate.",
        request_id=uuid4(),
        correlation_id=uuid4(),
        decided_at=_NOW + timedelta(minutes=5),
    )
    payload = ApprovalDecisionResumePayload(
        approval_request_id=rejected.id,
        agent_tool_call_id=tool_call.id,
        decision_status=ApprovalResumeDecisionStatus.REJECTED,
    )

    result = handle_approval_decision(
        payload=payload,
        approval_request=rejected,
    )

    assert result.action is (ApprovalDecisionAction.CONTINUE_WITHOUT_EXECUTION)


def test_pending_cannot_resume() -> None:
    tool_call, pending = _records()
    payload = ApprovalDecisionResumePayload(
        approval_request_id=pending.id,
        agent_tool_call_id=tool_call.id,
        decision_status=ApprovalResumeDecisionStatus.APPROVED,
    )

    with pytest.raises(
        ApprovalDecisionHandlingError,
        match="Pending",
    ):
        handle_approval_decision(
            payload=payload,
            approval_request=pending,
        )


def test_status_mismatch_fails_closed() -> None:
    tool_call, pending = _records()
    approved = pending.approve(
        actor_reference="operator:alice",
        comment=None,
        request_id=uuid4(),
        correlation_id=uuid4(),
        decided_at=_NOW + timedelta(minutes=5),
    )
    payload = ApprovalDecisionResumePayload(
        approval_request_id=approved.id,
        agent_tool_call_id=tool_call.id,
        decision_status=ApprovalResumeDecisionStatus.REJECTED,
    )

    with pytest.raises(
        ApprovalDecisionHandlingError,
        match="does not match",
    ):
        handle_approval_decision(
            payload=payload,
            approval_request=approved,
        )


def test_expired_maps_to_continue_without_execution() -> None:
    tool_call, pending = _records()
    expired = pending.expire(
        decided_at=_NOW + timedelta(days=1),
    )
    payload = ApprovalDecisionResumePayload(
        approval_request_id=expired.id,
        agent_tool_call_id=tool_call.id,
        decision_status=ApprovalResumeDecisionStatus.EXPIRED,
    )

    result = handle_approval_decision(
        payload=payload,
        approval_request=expired,
    )

    assert result.action is (ApprovalDecisionAction.CONTINUE_WITHOUT_EXECUTION)


def test_id_mismatch_fails_closed() -> None:
    tool_call, pending = _records()
    approved = pending.approve(
        actor_reference="operator:alice",
        comment=None,
        request_id=uuid4(),
        correlation_id=uuid4(),
        decided_at=_NOW + timedelta(minutes=5),
    )
    payload = ApprovalDecisionResumePayload(
        approval_request_id=uuid4(),
        agent_tool_call_id=tool_call.id,
        decision_status=ApprovalResumeDecisionStatus.APPROVED,
    )

    with pytest.raises(
        ApprovalDecisionHandlingError,
        match="different approval request",
    ):
        handle_approval_decision(
            payload=payload,
            approval_request=approved,
        )


def test_tool_call_id_mismatch_fails_closed() -> None:
    _tool_call, pending = _records()
    approved = pending.approve(
        actor_reference="operator:alice",
        comment=None,
        request_id=uuid4(),
        correlation_id=uuid4(),
        decided_at=_NOW + timedelta(minutes=5),
    )
    payload = ApprovalDecisionResumePayload(
        approval_request_id=approved.id,
        agent_tool_call_id=uuid4(),
        decision_status=ApprovalResumeDecisionStatus.APPROVED,
    )

    with pytest.raises(
        ApprovalDecisionHandlingError,
        match="different AgentToolCall",
    ):
        handle_approval_decision(
            payload=payload,
            approval_request=approved,
        )


def test_extra_payload_fields_rejected() -> None:
    with pytest.raises(ValidationError):
        ApprovalDecisionResumePayload.model_validate(
            {
                "approval_request_id": str(uuid4()),
                "agent_tool_call_id": str(uuid4()),
                "decision_status": "approved",
                "actor_reference": "operator:alice",
            },
        )
