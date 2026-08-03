"""Unit tests for sensitive execution grants."""

from dataclasses import replace
from datetime import UTC, datetime, timedelta
from types import MappingProxyType
from uuid import uuid4

import pytest
from pydantic import JsonValue

from supportops.agent_tools.domain.audit import AgentToolCall
from supportops.agent_tools.domain.contracts import ToolSafetyLevel
from supportops.agent_tools.domain.grants import SensitiveExecutionGrant
from supportops.modules.approvals.domain.models import ApprovalRequest

_NOW = datetime(2026, 8, 3, 18, 0, tzinfo=UTC)


def _approved_records(
    *,
    safe_input: dict[str, JsonValue] | None = None,
) -> tuple[AgentToolCall, ApprovalRequest]:
    if safe_input is None:
        safe_input = {
            "target_queue": "engineering_support",
            "reason": "A product defect requires review.",
        }
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
        safe_input=safe_input,
        proposed_at=_NOW,
    )
    approval = ApprovalRequest.create_pending(
        tool_call=tool_call,
        requested_by_llm_invocation_id=uuid4(),
        request_reason="A product defect requires review.",
        expires_at=_NOW + timedelta(days=1),
        now=_NOW,
    ).approve(
        actor_reference="operator:alice",
        comment=None,
        request_id=uuid4(),
        correlation_id=uuid4(),
        decided_at=_NOW + timedelta(minutes=5),
    )
    return tool_call, approval


def test_grant_copies_approved_authorization() -> None:
    tool_call, approval = _approved_records()
    attempt_id = uuid4()

    grant = SensitiveExecutionGrant.create(
        approval_request=approval,
        tool_call=tool_call,
        executed_by_agent_run_attempt_id=attempt_id,
        created_at=_NOW + timedelta(minutes=6),
    )

    assert grant.approval_request_id == approval.id
    assert grant.agent_tool_call_id == tool_call.id
    assert grant.executed_by_agent_run_attempt_id == attempt_id
    assert grant.approved_at == approval.decided_at
    assert grant.decision_actor_reference == "operator:alice"
    assert dict(grant.granted_input) == dict(tool_call.safe_input)
    assert grant.decision_request_id == approval.decision_request_id
    assert grant.decision_correlation_id == approval.decision_correlation_id


def test_grant_rejects_pending_approval() -> None:
    tool_call, approved = _approved_records()
    pending = ApprovalRequest.create_pending(
        tool_call=tool_call,
        requested_by_llm_invocation_id=uuid4(),
        request_reason="A product defect requires review.",
        expires_at=_NOW + timedelta(days=1),
        now=_NOW,
    )

    with pytest.raises(ValueError, match="approved"):
        SensitiveExecutionGrant.create(
            approval_request=pending,
            tool_call=tool_call,
            executed_by_agent_run_attempt_id=uuid4(),
            created_at=_NOW + timedelta(minutes=1),
        )

    assert approved.status.value == "approved"


def test_grant_rejects_rejected_approval() -> None:
    tool_call, _ = _approved_records()
    rejected = ApprovalRequest.create_pending(
        tool_call=tool_call,
        requested_by_llm_invocation_id=uuid4(),
        request_reason="A product defect requires review.",
        expires_at=_NOW + timedelta(days=1),
        now=_NOW,
    ).reject(
        actor_reference="operator:bob",
        comment="Not warranted.",
        request_id=uuid4(),
        correlation_id=uuid4(),
        decided_at=_NOW + timedelta(minutes=5),
    )

    with pytest.raises(ValueError, match="approved"):
        SensitiveExecutionGrant.create(
            approval_request=rejected,
            tool_call=tool_call,
            executed_by_agent_run_attempt_id=uuid4(),
            created_at=_NOW + timedelta(minutes=6),
        )


def test_grant_rejects_expired_approval() -> None:
    tool_call, _ = _approved_records()
    expired = ApprovalRequest.create_pending(
        tool_call=tool_call,
        requested_by_llm_invocation_id=uuid4(),
        request_reason="A product defect requires review.",
        expires_at=_NOW + timedelta(days=1),
        now=_NOW,
    ).expire(decided_at=_NOW + timedelta(days=1, minutes=1))

    with pytest.raises(ValueError, match="approved"):
        SensitiveExecutionGrant.create(
            approval_request=expired,
            tool_call=tool_call,
            executed_by_agent_run_attempt_id=uuid4(),
            created_at=_NOW + timedelta(days=1, minutes=2),
        )


def test_grant_rejects_non_pending_tool_call() -> None:
    tool_call, approval = _approved_records()
    rejected_tool_call = tool_call.reject_for_approval(
        decided_at=_NOW + timedelta(minutes=6),
    )

    with pytest.raises(ValueError, match="pending_approval"):
        SensitiveExecutionGrant.create(
            approval_request=approval,
            tool_call=rejected_tool_call,
            executed_by_agent_run_attempt_id=uuid4(),
            created_at=_NOW + timedelta(minutes=7),
        )


def test_grant_rejects_read_only_tool_call() -> None:
    tool_call, approval = _approved_records()
    read_only = object.__new__(AgentToolCall)
    for field_name in tool_call.__slots__:
        object.__setattr__(
            read_only,
            field_name,
            getattr(tool_call, field_name),
        )
    object.__setattr__(
        read_only,
        "safety_level",
        ToolSafetyLevel.READ_ONLY,
    )

    with pytest.raises(ValueError, match="sensitive_write"):
        SensitiveExecutionGrant.create(
            approval_request=approval,
            tool_call=read_only,
            executed_by_agent_run_attempt_id=uuid4(),
            created_at=_NOW + timedelta(minutes=6),
        )


def test_grant_rejects_ownership_mismatch() -> None:
    tool_call, approval = _approved_records()
    mismatched = replace(tool_call, ticket_id=uuid4())

    with pytest.raises(ValueError, match="must match"):
        SensitiveExecutionGrant.create(
            approval_request=approval,
            tool_call=mismatched,
            executed_by_agent_run_attempt_id=uuid4(),
            created_at=_NOW + timedelta(minutes=6),
        )


def test_grant_rejects_fingerprint_and_input_mismatch() -> None:
    tool_call, approval = _approved_records()
    fingerprint_mismatch = replace(
        tool_call,
        input_fingerprint="b" * 64,
    )

    with pytest.raises(ValueError, match="must match"):
        SensitiveExecutionGrant.create(
            approval_request=approval,
            tool_call=fingerprint_mismatch,
            executed_by_agent_run_attempt_id=uuid4(),
            created_at=_NOW + timedelta(minutes=6),
        )

    input_mismatch = replace(
        tool_call,
        safe_input=MappingProxyType(
            {
                "target_queue": "engineering_support",
                "reason": "Different reason.",
            }
        ),
    )

    with pytest.raises(ValueError, match="must match"):
        SensitiveExecutionGrant.create(
            approval_request=approval,
            tool_call=input_mismatch,
            executed_by_agent_run_attempt_id=uuid4(),
            created_at=_NOW + timedelta(minutes=6),
        )


def test_grant_rejects_created_at_before_approved_at() -> None:
    tool_call, approval = _approved_records()
    assert approval.decided_at is not None

    with pytest.raises(ValueError, match="created_at"):
        SensitiveExecutionGrant.create(
            approval_request=approval,
            tool_call=tool_call,
            executed_by_agent_run_attempt_id=uuid4(),
            created_at=approval.decided_at - timedelta(seconds=1),
        )


def test_grant_requires_utc_timestamps() -> None:
    tool_call, approval = _approved_records()
    naive = datetime(2026, 8, 3, 18, 6)

    with pytest.raises(ValueError, match="UTC-aware"):
        SensitiveExecutionGrant.create(
            approval_request=approval,
            tool_call=tool_call,
            executed_by_agent_run_attempt_id=uuid4(),
            created_at=naive,
        )


def test_granted_input_is_defensively_copied() -> None:
    safe_input: dict[str, JsonValue] = {
        "target_queue": "engineering_support",
        "reason": "A product defect requires review.",
    }
    tool_call, approval = _approved_records(safe_input=safe_input)
    grant = SensitiveExecutionGrant.create(
        approval_request=approval,
        tool_call=tool_call,
        executed_by_agent_run_attempt_id=uuid4(),
        created_at=_NOW + timedelta(minutes=6),
    )

    original = dict(grant.granted_input)
    safe_input["reason"] = "Changed externally."

    assert dict(grant.granted_input) == original
    assert isinstance(grant.granted_input, MappingProxyType)

    with pytest.raises(TypeError):
        grant.granted_input["reason"] = "x"  # type: ignore[index]


def test_matching_authorization_ignores_generated_identity() -> None:
    tool_call, approval = _approved_records()
    first = SensitiveExecutionGrant.create(
        approval_request=approval,
        tool_call=tool_call,
        executed_by_agent_run_attempt_id=uuid4(),
        created_at=_NOW + timedelta(minutes=6),
    )
    second = replace(
        first,
        id=uuid4(),
        created_at=first.created_at + timedelta(seconds=1),
    )

    assert first.matches_authorization(second)


def test_different_execution_attempt_conflicts() -> None:
    tool_call, approval = _approved_records()
    first = SensitiveExecutionGrant.create(
        approval_request=approval,
        tool_call=tool_call,
        executed_by_agent_run_attempt_id=uuid4(),
        created_at=_NOW + timedelta(minutes=6),
    )
    second = replace(
        first,
        executed_by_agent_run_attempt_id=uuid4(),
    )

    assert not first.matches_authorization(second)


def test_different_actor_request_correlation_conflicts() -> None:
    tool_call, approval = _approved_records()
    first = SensitiveExecutionGrant.create(
        approval_request=approval,
        tool_call=tool_call,
        executed_by_agent_run_attempt_id=uuid4(),
        created_at=_NOW + timedelta(minutes=6),
    )

    assert not first.matches_authorization(
        replace(first, decision_actor_reference="operator:bob"),
    )
    assert not first.matches_authorization(
        replace(first, decision_request_id=uuid4()),
    )
    assert not first.matches_authorization(
        replace(first, decision_correlation_id=uuid4()),
    )
