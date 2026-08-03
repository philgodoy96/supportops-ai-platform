"""Unit tests for approval API schemas."""

from datetime import UTC, datetime, timedelta
from uuid import uuid4

from supportops.agent_tools.domain.audit import AgentToolCall
from supportops.modules.approvals.api.schemas import (
    ApprovalRequestResponse,
)
from supportops.modules.approvals.domain.models import ApprovalRequest


def test_response_maps_safe_approval_fields() -> None:
    now = datetime(2026, 8, 3, 21, 0, tzinfo=UTC)
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
        proposed_at=now,
    )
    approval = ApprovalRequest.create_pending(
        tool_call=tool_call,
        requested_by_llm_invocation_id=uuid4(),
        request_reason="Operational review required.",
        expires_at=now + timedelta(days=1),
        now=now,
    )

    response = ApprovalRequestResponse.from_domain(approval)

    assert response.id == approval.id
    assert response.workspace_id == approval.workspace_id
    assert response.proposed_input == dict(approval.proposed_input)
    assert response.status.value == "pending"
    assert response.decision_actor_reference is None
