"""Unit tests for ticket escalation API schemas."""

from datetime import UTC, datetime
from types import SimpleNamespace
from typing import cast
from uuid import uuid4

from supportops.modules.tickets.api.escalation_schemas import (
    TicketEscalationResponse,
)
from supportops.modules.tickets.domain.escalation import TicketEscalation


def test_response_maps_safe_escalation_fields() -> None:
    escalation = SimpleNamespace(
        id=uuid4(),
        workspace_id=uuid4(),
        ticket_id=uuid4(),
        agent_run_id=uuid4(),
        executed_by_agent_run_attempt_id=uuid4(),
        approval_request_id=uuid4(),
        agent_tool_call_id=uuid4(),
        target_queue="support_operations",
        reason="Operational review required.",
        created_at=datetime(2026, 8, 3, 22, 0, tzinfo=UTC),
    )

    response = TicketEscalationResponse.from_domain(
        cast(TicketEscalation, escalation),
    )

    assert response.id == escalation.id
    assert response.workspace_id == escalation.workspace_id
    assert response.ticket_id == escalation.ticket_id
    assert response.reason == escalation.reason
    assert "grant_id" not in response.model_fields
    assert "execution_output" not in response.model_fields
