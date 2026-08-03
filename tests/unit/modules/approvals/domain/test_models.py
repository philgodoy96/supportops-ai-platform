"""Unit tests for durable approval-request domain entities."""

from dataclasses import replace
from datetime import UTC, datetime, timedelta
from types import MappingProxyType
from uuid import UUID

import pytest
from pydantic import JsonValue

from supportops.agent_tools.domain.audit import (
    AgentToolCall,
    AgentToolCallStatus,
)
from supportops.agent_tools.domain.contracts import ToolSafetyLevel
from supportops.modules.approvals.domain.models import (
    APPROVAL_EXPIRATION_ACTOR_REFERENCE,
    APPROVAL_REQUEST_REASON_MAX_LENGTH,
    ApprovalRequest,
    ApprovalRequestStatus,
)

APPROVAL_REQUEST_ID = UUID("11111111-1111-4111-8111-111111111111")
WORKSPACE_ID = UUID("22222222-2222-4222-8222-222222222222")
TICKET_ID = UUID("33333333-3333-4333-8333-333333333333")
AGENT_RUN_ID = UUID("44444444-4444-4444-8444-444444444444")
TOOL_CALL_ID = UUID("55555555-5555-4555-8555-555555555555")
ATTEMPT_ID = UUID("66666666-6666-4666-8666-666666666666")
INVOCATION_ID = UUID("77777777-7777-4777-8777-777777777777")
DECISION_REQUEST_ID = UUID("88888888-8888-4888-8888-888888888888")
DECISION_CORRELATION_ID = UUID("99999999-9999-4999-8999-999999999999")
CREATED_AT = datetime(2026, 8, 2, 20, 0, tzinfo=UTC)
EXPIRES_AT = CREATED_AT + timedelta(hours=24)
DECIDED_AT = CREATED_AT + timedelta(minutes=10)
FINGERPRINT = "a" * 64


def create_pending_tool_call(
    *,
    safe_input: dict[str, JsonValue] | None = None,
    status: AgentToolCallStatus = AgentToolCallStatus.PENDING_APPROVAL,
    safety_level: ToolSafetyLevel = ToolSafetyLevel.SENSITIVE_WRITE,
    input_fingerprint: str = FINGERPRINT,
) -> AgentToolCall:
    """Create one sensitive proposal tool call for approval factories."""

    if safe_input is None:
        safe_input = {"reason_code": "policy_required"}

    if status is AgentToolCallStatus.PENDING_APPROVAL:
        return AgentToolCall.propose_for_approval(
            tool_call_id=TOOL_CALL_ID,
            workspace_id=WORKSPACE_ID,
            ticket_id=TICKET_ID,
            agent_run_id=AGENT_RUN_ID,
            proposed_by_agent_run_attempt_id=ATTEMPT_ID,
            sequence=1,
            provider_tool_call_id="provider-tool-call-1",
            tool_name="escalate_ticket",
            tool_version=1,
            input_fingerprint=input_fingerprint,
            safe_input=safe_input,
            proposed_at=CREATED_AT,
        )

    return AgentToolCall.create_terminal(
        tool_call_id=TOOL_CALL_ID,
        workspace_id=WORKSPACE_ID,
        ticket_id=TICKET_ID,
        agent_run_id=AGENT_RUN_ID,
        agent_run_attempt_id=ATTEMPT_ID,
        sequence=1,
        provider_tool_call_id="provider-tool-call-1",
        tool_name="search_knowledge",
        tool_version=1,
        safety_level=safety_level,
        status=status,
        input_fingerprint=input_fingerprint,
        safe_input=safe_input,
        safe_output={"result_count": 1},
        latency_ms=25,
        error_code=None,
        started_at=CREATED_AT,
        finished_at=CREATED_AT + timedelta(milliseconds=25),
    )


def create_pending_approval(
    *,
    tool_call: AgentToolCall | None = None,
    request_reason: str = "Requires human review before escalation.",
    expires_at: datetime = EXPIRES_AT,
    now: datetime = CREATED_AT,
    invocation_id: UUID = INVOCATION_ID,
    approval_request_id: UUID = APPROVAL_REQUEST_ID,
) -> ApprovalRequest:
    """Create one valid pending approval request."""

    return ApprovalRequest.create_pending(
        tool_call=tool_call or create_pending_tool_call(),
        requested_by_llm_invocation_id=invocation_id,
        request_reason=request_reason,
        expires_at=expires_at,
        approval_request_id=approval_request_id,
        now=now,
    )


def test_create_pending_copies_immutable_proposal_fields() -> None:
    tool_call = create_pending_tool_call()
    approval = create_pending_approval(tool_call=tool_call)

    assert approval.id == APPROVAL_REQUEST_ID
    assert approval.workspace_id == WORKSPACE_ID
    assert approval.ticket_id == TICKET_ID
    assert approval.agent_run_id == AGENT_RUN_ID
    assert approval.agent_tool_call_id == TOOL_CALL_ID
    assert approval.requested_by_llm_invocation_id == INVOCATION_ID
    assert approval.status is ApprovalRequestStatus.PENDING
    assert approval.tool_name == "escalate_ticket"
    assert approval.tool_version == 1
    assert approval.safety_level is ToolSafetyLevel.SENSITIVE_WRITE
    assert approval.input_fingerprint == FINGERPRINT
    assert dict(approval.proposed_input) == {"reason_code": "policy_required"}
    assert approval.request_reason == ("Requires human review before escalation.")
    assert approval.expires_at == EXPIRES_AT
    assert approval.decision_actor_reference is None
    assert approval.decision_comment is None
    assert approval.decision_request_id is None
    assert approval.decision_correlation_id is None
    assert approval.decided_at is None
    assert approval.created_at == CREATED_AT
    assert approval.updated_at == CREATED_AT
    assert approval.is_terminal is False


def test_create_pending_rejects_non_pending_tool_call() -> None:
    tool_call = create_pending_tool_call(
        status=AgentToolCallStatus.SUCCEEDED,
        safety_level=ToolSafetyLevel.READ_ONLY,
    )

    with pytest.raises(
        ValueError,
        match="pending_approval AgentToolCall",
    ):
        create_pending_approval(tool_call=tool_call)


def test_proposed_input_is_defensively_copied() -> None:
    safe_input: dict[str, JsonValue] = {
        "reason_code": "policy_required",
        "tags": ["priority"],
    }
    tool_call = create_pending_tool_call(safe_input=safe_input)
    approval = create_pending_approval(tool_call=tool_call)

    safe_input["reason_code"] = "mutated"
    cast_tags = safe_input["tags"]
    assert isinstance(cast_tags, list)
    cast_tags.append("extra")

    assert approval.proposed_input["reason_code"] == "policy_required"
    assert approval.proposed_input["tags"] == ["priority"]
    assert isinstance(approval.proposed_input, MappingProxyType)

    with pytest.raises(TypeError):
        approval.proposed_input["reason_code"] = "x"  # type: ignore[index]


def test_request_reason_is_stripped_and_bounded() -> None:
    approval = create_pending_approval(
        request_reason="  Needs review.  ",
    )

    assert approval.request_reason == "Needs review."

    with pytest.raises(ValueError, match="request_reason"):
        create_pending_approval(request_reason="   ")

    with pytest.raises(ValueError, match="request_reason"):
        create_pending_approval(
            request_reason="x" * (APPROVAL_REQUEST_REASON_MAX_LENGTH + 1),
        )


def test_expiration_must_follow_created_at() -> None:
    with pytest.raises(ValueError, match="expires_at"):
        create_pending_approval(expires_at=CREATED_AT)

    with pytest.raises(ValueError, match="expires_at"):
        create_pending_approval(
            expires_at=CREATED_AT - timedelta(seconds=1),
        )


def test_timestamps_must_be_utc_aware() -> None:
    with pytest.raises(ValueError, match="created_at"):
        create_pending_approval(now=datetime(2026, 8, 2, 20, 0))

    with pytest.raises(ValueError, match="expires_at"):
        create_pending_approval(
            expires_at=datetime(2026, 8, 3, 20, 0),
        )


def test_pending_decision_invariants() -> None:
    with pytest.raises(ValueError, match="decision fields"):
        replace(
            create_pending_approval(),
            decision_actor_reference="operator:alice",
        )


def test_approved_invariants() -> None:
    approved = replace(
        create_pending_approval(),
        status=ApprovalRequestStatus.APPROVED,
        decision_actor_reference="operator:alice",
        decision_request_id=DECISION_REQUEST_ID,
        decision_correlation_id=DECISION_CORRELATION_ID,
        decided_at=DECIDED_AT,
        updated_at=DECIDED_AT,
    )

    assert approved.is_terminal is True
    assert approved.decision_comment is None
    assert approved.decision_actor_reference == "operator:alice"
    assert approved.decision_request_id == DECISION_REQUEST_ID
    assert approved.decision_correlation_id == DECISION_CORRELATION_ID
    assert approved.decided_at == DECIDED_AT

    with pytest.raises(ValueError, match="decision_actor_reference"):
        replace(
            approved,
            decision_actor_reference=None,
        )

    with pytest.raises(ValueError, match="decision_request_id"):
        replace(
            approved,
            decision_request_id=None,
        )

    with pytest.raises(ValueError, match="decision_correlation_id"):
        replace(
            approved,
            decision_correlation_id=None,
        )

    with pytest.raises(ValueError, match="decided_at"):
        replace(
            approved,
            decided_at=None,
        )


def test_rejected_invariants() -> None:
    rejected = replace(
        create_pending_approval(),
        status=ApprovalRequestStatus.REJECTED,
        decision_actor_reference="operator:bob",
        decision_comment="Not warranted.",
        decision_request_id=DECISION_REQUEST_ID,
        decision_correlation_id=DECISION_CORRELATION_ID,
        decided_at=DECIDED_AT,
        updated_at=DECIDED_AT,
    )

    assert rejected.is_terminal is True
    assert rejected.decision_actor_reference == "operator:bob"
    assert rejected.decision_comment == "Not warranted."
    assert rejected.decision_request_id == DECISION_REQUEST_ID
    assert rejected.decision_correlation_id == DECISION_CORRELATION_ID
    assert rejected.decided_at == DECIDED_AT

    with pytest.raises(ValueError, match="decision_actor_reference"):
        replace(
            rejected,
            decision_actor_reference=None,
        )

    with pytest.raises(ValueError, match="decision_comment"):
        replace(
            rejected,
            decision_comment=None,
        )

    with pytest.raises(ValueError, match="decision_request_id"):
        replace(
            rejected,
            decision_request_id=None,
        )

    with pytest.raises(ValueError, match="decision_correlation_id"):
        replace(
            rejected,
            decision_correlation_id=None,
        )

    with pytest.raises(ValueError, match="decided_at"):
        replace(
            rejected,
            decided_at=None,
        )


def test_expired_invariants() -> None:
    expired = replace(
        create_pending_approval(),
        status=ApprovalRequestStatus.EXPIRED,
        decision_actor_reference=APPROVAL_EXPIRATION_ACTOR_REFERENCE,
        decided_at=DECIDED_AT,
        updated_at=DECIDED_AT,
    )

    assert expired.is_terminal is True
    assert expired.decision_actor_reference == (APPROVAL_EXPIRATION_ACTOR_REFERENCE)
    assert expired.decision_comment is None
    assert expired.decision_request_id is None
    assert expired.decision_correlation_id is None
    assert expired.decided_at == DECIDED_AT

    with pytest.raises(ValueError, match="system expiration actor"):
        replace(
            expired,
            decision_actor_reference="operator:alice",
        )

    with pytest.raises(ValueError, match="decision_comment"):
        replace(
            expired,
            decision_comment="expired",
        )

    with pytest.raises(ValueError, match="decision_request_id"):
        replace(
            expired,
            decision_request_id=DECISION_REQUEST_ID,
        )

    with pytest.raises(ValueError, match="decision_correlation_id"):
        replace(
            expired,
            decision_correlation_id=DECISION_CORRELATION_ID,
        )

    with pytest.raises(ValueError, match="decided_at"):
        replace(
            expired,
            decided_at=None,
        )


def _assert_proposal_fields_preserved(
    original: ApprovalRequest,
    decided: ApprovalRequest,
) -> None:
    assert decided.id == original.id
    assert decided.workspace_id == original.workspace_id
    assert decided.ticket_id == original.ticket_id
    assert decided.agent_run_id == original.agent_run_id
    assert decided.agent_tool_call_id == original.agent_tool_call_id
    assert decided.requested_by_llm_invocation_id == (original.requested_by_llm_invocation_id)
    assert decided.tool_name == original.tool_name
    assert decided.tool_version == original.tool_version
    assert decided.safety_level is original.safety_level
    assert decided.input_fingerprint == original.input_fingerprint
    assert decided.proposed_input == original.proposed_input
    assert isinstance(decided.proposed_input, MappingProxyType)
    assert decided.request_reason == original.request_reason
    assert decided.created_at == original.created_at
    assert decided.expires_at == original.expires_at


def test_approve_transition_persists_decision_fields() -> None:
    pending = create_pending_approval()
    approved = pending.approve(
        actor_reference="operator:alice",
        comment=None,
        request_id=DECISION_REQUEST_ID,
        correlation_id=DECISION_CORRELATION_ID,
        decided_at=DECIDED_AT,
    )

    assert approved.status is ApprovalRequestStatus.APPROVED
    assert approved.is_terminal is True
    assert approved.decision_actor_reference == "operator:alice"
    assert approved.decision_comment is None
    assert approved.decision_request_id == DECISION_REQUEST_ID
    assert approved.decision_correlation_id == DECISION_CORRELATION_ID
    assert approved.decided_at == DECIDED_AT
    assert approved.updated_at == DECIDED_AT
    _assert_proposal_fields_preserved(pending, approved)

    with pytest.raises(TypeError):
        approved.proposed_input["reason_code"] = "x"  # type: ignore[index]


def test_approve_transition_allows_optional_comment() -> None:
    approved = create_pending_approval().approve(
        actor_reference="operator:alice",
        comment="Reviewed and approved.",
        request_id=DECISION_REQUEST_ID,
        correlation_id=DECISION_CORRELATION_ID,
        decided_at=DECIDED_AT,
    )

    assert approved.decision_comment == "Reviewed and approved."


def test_reject_transition_requires_comment() -> None:
    pending = create_pending_approval()

    with pytest.raises(ValueError, match="comment"):
        pending.reject(
            actor_reference="operator:bob",
            comment="",
            request_id=DECISION_REQUEST_ID,
            correlation_id=DECISION_CORRELATION_ID,
            decided_at=DECIDED_AT,
        )

    rejected = pending.reject(
        actor_reference="operator:bob",
        comment="Not warranted.",
        request_id=DECISION_REQUEST_ID,
        correlation_id=DECISION_CORRELATION_ID,
        decided_at=DECIDED_AT,
    )

    assert rejected.status is ApprovalRequestStatus.REJECTED
    assert rejected.is_terminal is True
    assert rejected.decision_comment == "Not warranted."
    assert rejected.updated_at == DECIDED_AT
    _assert_proposal_fields_preserved(pending, rejected)

    with pytest.raises(TypeError):
        rejected.proposed_input["reason_code"] = "x"  # type: ignore[index]


def test_expire_transition_requires_overdue_request() -> None:
    pending = create_pending_approval()

    with pytest.raises(ValueError, match="expires_at"):
        pending.expire(decided_at=DECIDED_AT)

    with pytest.raises(ValueError, match="expires_at"):
        pending.expire(decided_at=EXPIRES_AT - timedelta(microseconds=1))

    expired = pending.expire(decided_at=EXPIRES_AT)

    assert expired.status is ApprovalRequestStatus.EXPIRED
    assert expired.is_terminal is True
    assert expired.decision_actor_reference == (APPROVAL_EXPIRATION_ACTOR_REFERENCE)
    assert expired.decision_comment is None
    assert expired.decision_request_id is None
    assert expired.decision_correlation_id is None
    assert expired.decided_at == EXPIRES_AT
    assert expired.updated_at == EXPIRES_AT
    _assert_proposal_fields_preserved(pending, expired)

    with pytest.raises(TypeError):
        expired.proposed_input["reason_code"] = "x"  # type: ignore[index]


def test_approve_and_reject_fail_at_expires_at() -> None:
    pending = create_pending_approval()

    with pytest.raises(ValueError, match="expires_at"):
        pending.approve(
            actor_reference="operator:alice",
            comment=None,
            request_id=DECISION_REQUEST_ID,
            correlation_id=DECISION_CORRELATION_ID,
            decided_at=EXPIRES_AT,
        )

    with pytest.raises(ValueError, match="expires_at"):
        pending.reject(
            actor_reference="operator:bob",
            comment="Too late.",
            request_id=DECISION_REQUEST_ID,
            correlation_id=DECISION_CORRELATION_ID,
            decided_at=EXPIRES_AT,
        )

    approved = pending.approve(
        actor_reference="operator:alice",
        comment=None,
        request_id=DECISION_REQUEST_ID,
        correlation_id=DECISION_CORRELATION_ID,
        decided_at=EXPIRES_AT - timedelta(microseconds=1),
    )
    assert approved.status is ApprovalRequestStatus.APPROVED


def test_decision_transitions_only_from_pending() -> None:
    approved = create_pending_approval().approve(
        actor_reference="operator:alice",
        comment=None,
        request_id=DECISION_REQUEST_ID,
        correlation_id=DECISION_CORRELATION_ID,
        decided_at=DECIDED_AT,
    )

    with pytest.raises(ValueError, match="pending"):
        approved.approve(
            actor_reference="operator:alice",
            comment=None,
            request_id=DECISION_REQUEST_ID,
            correlation_id=DECISION_CORRELATION_ID,
            decided_at=DECIDED_AT,
        )

    with pytest.raises(ValueError, match="pending"):
        approved.reject(
            actor_reference="operator:bob",
            comment="Too late.",
            request_id=DECISION_REQUEST_ID,
            correlation_id=DECISION_CORRELATION_ID,
            decided_at=DECIDED_AT,
        )

    with pytest.raises(ValueError, match="pending"):
        approved.expire(decided_at=EXPIRES_AT)

    rejected = create_pending_approval().reject(
        actor_reference="operator:bob",
        comment="Denied.",
        request_id=DECISION_REQUEST_ID,
        correlation_id=DECISION_CORRELATION_ID,
        decided_at=DECIDED_AT,
    )
    with pytest.raises(ValueError, match="pending"):
        rejected.expire(decided_at=EXPIRES_AT)

    expired = create_pending_approval().expire(decided_at=EXPIRES_AT)
    with pytest.raises(ValueError, match="pending"):
        expired.approve(
            actor_reference="operator:alice",
            comment=None,
            request_id=DECISION_REQUEST_ID,
            correlation_id=DECISION_CORRELATION_ID,
            decided_at=DECIDED_AT,
        )


def test_decision_timestamps_must_be_utc_aware() -> None:
    pending = create_pending_approval()
    naive = datetime(2026, 8, 2, 20, 10)

    with pytest.raises(ValueError, match="decided_at"):
        pending.approve(
            actor_reference="operator:alice",
            comment=None,
            request_id=DECISION_REQUEST_ID,
            correlation_id=DECISION_CORRELATION_ID,
            decided_at=naive,
        )

    with pytest.raises(ValueError, match="decided_at"):
        pending.reject(
            actor_reference="operator:bob",
            comment="Rejected.",
            request_id=DECISION_REQUEST_ID,
            correlation_id=DECISION_CORRELATION_ID,
            decided_at=naive,
        )

    with pytest.raises(ValueError, match="decided_at"):
        pending.expire(decided_at=datetime(2026, 8, 3, 20, 0))


def test_matches_pending_proposal_compares_immutable_identity() -> None:
    first = create_pending_approval()
    identical = create_pending_approval(
        approval_request_id=UUID("aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"),
        now=CREATED_AT + timedelta(seconds=5),
    )

    assert first.matches_pending_proposal(identical) is True

    different_invocation = create_pending_approval(
        invocation_id=UUID("bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb"),
    )
    assert first.matches_pending_proposal(different_invocation) is False

    different_fingerprint = create_pending_approval(
        tool_call=create_pending_tool_call(input_fingerprint="b" * 64),
    )
    assert first.matches_pending_proposal(different_fingerprint) is False

    different_input = create_pending_approval(
        tool_call=create_pending_tool_call(
            safe_input={"reason_code": "other"},
        ),
    )
    assert first.matches_pending_proposal(different_input) is False

    different_reason = create_pending_approval(
        request_reason="Different reason.",
    )
    assert first.matches_pending_proposal(different_reason) is False

    different_expiry = create_pending_approval(
        expires_at=EXPIRES_AT + timedelta(hours=1),
    )
    assert first.matches_pending_proposal(different_expiry) is False


def test_matches_pending_proposal_rejects_non_pending() -> None:
    pending = create_pending_approval()
    approved = replace(
        pending,
        status=ApprovalRequestStatus.APPROVED,
        decision_actor_reference="operator:alice",
        decision_request_id=DECISION_REQUEST_ID,
        decision_correlation_id=DECISION_CORRELATION_ID,
        decided_at=DECIDED_AT,
        updated_at=DECIDED_AT,
    )

    assert pending.matches_pending_proposal(approved) is False
    assert approved.matches_pending_proposal(pending) is False
