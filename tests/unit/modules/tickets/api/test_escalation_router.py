"""Unit tests for ticket escalation inspection routes."""

from datetime import UTC, datetime
from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock
from uuid import UUID, uuid4

from fastapi.testclient import TestClient

from supportops.api.application import create_application
from supportops.modules.tickets.api.escalation_dependencies import (
    get_list_ticket_escalations,
    get_ticket_escalation,
)
from supportops.modules.tickets.application.escalation_queries import (
    TicketEscalationListPage,
    TicketEscalationPageCursor,
)
from supportops.modules.tickets.domain.escalation import TicketEscalation


def _escalation(
    *,
    workspace_id: UUID,
    escalation_id: UUID,
) -> SimpleNamespace:
    return SimpleNamespace(
        id=escalation_id,
        workspace_id=workspace_id,
        ticket_id=uuid4(),
        agent_run_id=uuid4(),
        executed_by_agent_run_attempt_id=uuid4(),
        approval_request_id=uuid4(),
        agent_tool_call_id=uuid4(),
        target_queue="support_operations",
        reason="Operational review required.",
        created_at=datetime(2026, 8, 3, 22, 0, tzinfo=UTC),
    )


def test_list_escalations_returns_page() -> None:
    app = create_application()
    workspace_id = uuid4()
    escalation = _escalation(
        workspace_id=workspace_id,
        escalation_id=uuid4(),
    )
    service = SimpleNamespace(
        execute=AsyncMock(
            return_value=TicketEscalationListPage(
                items=cast(
                    tuple[TicketEscalation, ...],
                    (escalation,),
                ),
                next_cursor=TicketEscalationPageCursor(
                    created_at=escalation.created_at,
                    ticket_escalation_id=escalation.id,
                ),
            ),
        ),
    )
    app.dependency_overrides[get_list_ticket_escalations] = lambda: service

    response = TestClient(app).get(
        (f"/api/v1/workspaces/{workspace_id}/ticket-escalations"),
        params={
            "ticket_id": str(escalation.ticket_id),
            "page_size": 1,
        },
    )

    assert response.status_code == 200
    assert response.json()["items"][0]["id"] == str(
        escalation.id,
    )
    assert response.json()["next_cursor"] is not None


def test_get_escalation_returns_workspace_detail() -> None:
    app = create_application()
    workspace_id = uuid4()
    escalation_id = uuid4()
    escalation = _escalation(
        workspace_id=workspace_id,
        escalation_id=escalation_id,
    )
    service = SimpleNamespace(
        execute=AsyncMock(return_value=escalation),
    )
    app.dependency_overrides[get_ticket_escalation] = lambda: service

    response = TestClient(app).get(
        (f"/api/v1/workspaces/{workspace_id}/ticket-escalations/{escalation_id}"),
    )

    assert response.status_code == 200
    assert response.json()["ticket_id"] == str(
        escalation.ticket_id,
    )
