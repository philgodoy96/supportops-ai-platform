"""Unit tests for approval-request persistence models."""

from datetime import UTC, datetime, timedelta
from typing import cast
from uuid import UUID

from sqlalchemy import Table
from sqlalchemy.orm import RelationshipProperty

from supportops.agent_tools.domain.audit import AgentToolCall
from supportops.infrastructure.postgresql.model_registry import (
    register_persistence_models,
)
from supportops.modules.approvals.domain.models import ApprovalRequest
from supportops.modules.approvals.infrastructure.models import (
    ApprovalRequestRecord,
)

APPROVAL_REQUEST_ID = UUID("11111111-1111-4111-8111-111111111111")
WORKSPACE_ID = UUID("22222222-2222-4222-8222-222222222222")
TICKET_ID = UUID("33333333-3333-4333-8333-333333333333")
AGENT_RUN_ID = UUID("44444444-4444-4444-8444-444444444444")
TOOL_CALL_ID = UUID("55555555-5555-4555-8555-555555555555")
ATTEMPT_ID = UUID("66666666-6666-4666-8666-666666666666")
INVOCATION_ID = UUID("77777777-7777-4777-8777-777777777777")
CREATED_AT = datetime(2026, 8, 2, 21, 0, tzinfo=UTC)
EXPIRES_AT = CREATED_AT + timedelta(hours=12)


def create_approval_request() -> ApprovalRequest:
    """Create one valid pending approval for model mapping tests."""

    tool_call = AgentToolCall.propose_for_approval(
        tool_call_id=TOOL_CALL_ID,
        workspace_id=WORKSPACE_ID,
        ticket_id=TICKET_ID,
        agent_run_id=AGENT_RUN_ID,
        proposed_by_agent_run_attempt_id=ATTEMPT_ID,
        sequence=1,
        provider_tool_call_id="provider-tool-call-1",
        tool_name="escalate_ticket",
        tool_version=1,
        input_fingerprint="a" * 64,
        safe_input={"reason_code": "policy_required"},
        proposed_at=CREATED_AT,
    )

    return ApprovalRequest.create_pending(
        tool_call=tool_call,
        requested_by_llm_invocation_id=INVOCATION_ID,
        request_reason="Requires human review.",
        expires_at=EXPIRES_AT,
        approval_request_id=APPROVAL_REQUEST_ID,
        now=CREATED_AT,
    )


def test_round_trips_approval_request_record() -> None:
    approval = create_approval_request()

    record = ApprovalRequestRecord.from_domain(approval)
    reconstructed = record.to_domain()

    assert reconstructed == approval
    assert record.proposed_input is not approval.proposed_input
    assert record.decision_actor_reference is None
    assert record.decision_comment is None
    assert record.decision_request_id is None
    assert record.decision_correlation_id is None
    assert record.decided_at is None


def test_approval_request_table_has_expected_constraints() -> None:
    register_persistence_models()

    table = cast(Table, ApprovalRequestRecord.__table__)
    constraint_names = {constraint.name for constraint in table.constraints}
    index_names = {index.name for index in table.indexes}
    column_names = {column.name for column in table.c}

    assert {
        "fk_approval_requests_workspace_ticket_agent_run",
        "fk_approval_requests_agent_tool_call",
        "fk_approval_requests_requesting_invocation",
        "uq_approval_requests_agent_tool_call",
        "uq_approval_requests_workspace_id",
        "ck_approval_requests_approval_request_status",
        "ck_approval_requests_approval_request_safety_level",
        "ck_approval_requests_approval_request_tool_name_format",
        "ck_approval_requests_approval_request_tool_version_positive",
        "ck_approval_requests_approval_request_input_fingerprint",
        "ck_approval_requests_approval_request_proposed_input_object",
        "ck_approval_requests_approval_request_proposed_input_size",
        "ck_approval_requests_approval_request_reason_format",
        "ck_approval_requests_approval_request_actor_format",
        "ck_approval_requests_approval_request_comment_format",
        "ck_approval_requests_approval_request_expiration_order",
        "ck_approval_requests_approval_request_update_order",
        "ck_approval_requests_approval_request_decision_order",
        "ck_approval_requests_approval_request_decision_state",
    }.issubset(constraint_names)

    assert {
        "ix_approval_requests_workspace_status_created_id",
        "ix_approval_requests_agent_run_status",
        "ix_approval_requests_pending_expiration",
    }.issubset(index_names)

    assert {
        "decision_actor_reference",
        "decision_comment",
        "decision_request_id",
        "decision_correlation_id",
        "decided_at",
    }.issubset(column_names)
    assert table.c.decision_actor_reference.nullable is True
    assert table.c.decision_comment.nullable is True
    assert table.c.decision_request_id.nullable is True
    assert table.c.decision_correlation_id.nullable is True
    assert table.c.decided_at.nullable is True

    relationship_properties = [
        attribute
        for attribute in ApprovalRequestRecord.__mapper__.attrs
        if isinstance(attribute, RelationshipProperty)
    ]
    assert relationship_properties == []
