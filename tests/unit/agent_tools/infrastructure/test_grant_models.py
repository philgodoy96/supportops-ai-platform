"""Unit tests for grant SQLAlchemy mapping."""

from datetime import UTC, datetime, timedelta
from typing import cast
from uuid import uuid4

from sqlalchemy import Table

from supportops.agent_tools.domain.audit import AgentToolCall
from supportops.agent_tools.domain.grants import SensitiveExecutionGrant
from supportops.agent_tools.infrastructure.grant_models import (
    SensitiveExecutionGrantRecord,
)
from supportops.modules.approvals.domain.models import ApprovalRequest


def test_record_round_trips_domain_grant() -> None:
    now = datetime(2026, 8, 3, 18, 0, tzinfo=UTC)
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
    ).approve(
        actor_reference="operator:alice",
        comment=None,
        request_id=uuid4(),
        correlation_id=uuid4(),
        decided_at=now + timedelta(minutes=5),
    )
    grant = SensitiveExecutionGrant.create(
        approval_request=approval,
        tool_call=tool_call,
        executed_by_agent_run_attempt_id=uuid4(),
        created_at=now + timedelta(minutes=6),
    )

    record = SensitiveExecutionGrantRecord.from_domain(grant)

    assert record.to_domain() == grant
    assert record.__tablename__ == "sensitive_execution_grants"


def test_record_exposes_exact_constraint_names() -> None:
    table = cast(Table, SensitiveExecutionGrantRecord.__table__)
    constraint_names = {
        constraint.name for constraint in table.constraints if constraint.name is not None
    }

    assert {
        "ck_sensitive_execution_grants_safety_level",
        "ck_sensitive_execution_grants_tool_name_format",
        "ck_sensitive_execution_grants_tool_version_positive",
        "ck_sensitive_execution_grants_input_fingerprint",
        "ck_sensitive_execution_grants_granted_input_object",
        "ck_sensitive_execution_grants_granted_input_size",
        "ck_sensitive_execution_grants_actor_format",
        "ck_sensitive_execution_grants_creation_order",
        "fk_sensitive_execution_grants_workspace_ticket_agent_run",
        "fk_sensitive_execution_grants_execution_attempt",
        "fk_sensitive_execution_grants_approval_request",
        "fk_sensitive_execution_grants_agent_tool_call",
        "uq_sensitive_execution_grants_approval_request",
        "uq_sensitive_execution_grants_agent_tool_call",
        "uq_sensitive_execution_grants_workspace_id",
        "pk_sensitive_execution_grants",
    }.issubset(constraint_names)

    index_names = {index.name for index in table.indexes}
    assert "ix_sensitive_execution_grants_workspace_created_id" in index_names
    assert "ix_sensitive_execution_grants_agent_run" in index_names
