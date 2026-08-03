"""Unit tests for ticket escalation SQLAlchemy mapping."""

from datetime import UTC, datetime, timedelta
from typing import cast
from uuid import uuid4

from sqlalchemy import Table

from supportops.agent_tools.domain.audit import AgentToolCall
from supportops.agent_tools.domain.grants import SensitiveExecutionGrant
from supportops.agent_tools.tools.escalate_ticket import (
    EscalateTicketInput,
)
from supportops.modules.approvals.domain.models import ApprovalRequest
from supportops.modules.tickets.domain.escalation import (
    TicketEscalation,
)
from supportops.modules.tickets.infrastructure.escalation_models import (
    TicketEscalationRecord,
)


def test_record_round_trips_domain_escalation() -> None:
    now = datetime(2026, 8, 3, 18, 30, tzinfo=UTC)
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
    escalation = TicketEscalation.create_from_grant(
        grant=grant,
        input_data=EscalateTicketInput.model_validate(
            dict(grant.granted_input),
        ),
        created_at=now + timedelta(minutes=7),
    )

    record = TicketEscalationRecord.from_domain(escalation)

    assert record.to_domain() == escalation
    assert record.__tablename__ == "ticket_escalations"


def test_record_exposes_exact_constraint_names() -> None:
    table = cast(Table, TicketEscalationRecord.__table__)
    constraint_names = {
        constraint.name for constraint in table.constraints if constraint.name is not None
    }

    assert {
        "ck_ticket_escalations_target_queue_format",
        "ck_ticket_escalations_reason_format",
        "fk_ticket_escalations_workspace_ticket",
        "fk_ticket_escalations_workspace_ticket_agent_run",
        "fk_ticket_escalations_execution_attempt",
        "fk_ticket_escalations_approval_request",
        "fk_ticket_escalations_agent_tool_call",
        "uq_ticket_escalations_approval_request",
        "uq_ticket_escalations_agent_tool_call",
        "uq_ticket_escalations_workspace_id",
        "pk_ticket_escalations",
    }.issubset(constraint_names)

    index_names = {index.name for index in table.indexes}
    assert "ix_ticket_escalations_workspace_created_id" in index_names
    assert "ix_ticket_escalations_ticket_created_id" in index_names
