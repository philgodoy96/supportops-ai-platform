"""Unit tests for support ticket API schemas."""

from datetime import UTC, datetime
from uuid import UUID

import pytest
from pydantic import ValidationError

from supportops.modules.agent_runs.domain.models import (
    DETERMINISTIC_BASELINE_WORKFLOW_VERSION,
    INITIAL_TICKET_PROCESSING_WORKFLOW_NAME,
    AgentRun,
    AgentRunStatus,
)
from supportops.modules.tickets.api.schemas import (
    TicketCreateRequest,
    TicketCreateResponse,
    TicketProcessingRunResponse,
    TicketResponse,
)
from supportops.modules.tickets.domain.models import Ticket

_TIMESTAMP = datetime(
    2026,
    7,
    31,
    12,
    0,
    tzinfo=UTC,
)
_TICKET_ID = UUID(
    "f84d7304-8171-4842-a111-c3dbda2ff79b",
)
_WORKSPACE_ID = UUID(
    "032c8c87-57cc-4d14-bfbd-04968b4e8cd4",
)
_INGESTION_REQUEST_ID = UUID(
    "725eec8a-c504-4071-ac96-c78cc907f26c",
)
_CORRELATION_ID = UUID(
    "1038c98e-62fd-45df-9839-138f7105cb78",
)
_AGENT_RUN_ID = UUID(
    "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
)


def _create_ticket() -> Ticket:
    return Ticket.create(
        ticket_id=_TICKET_ID,
        workspace_id=_WORKSPACE_ID,
        subject="Unable to access billing",
        description="The dashboard returns an error.",
        external_reference="SUP-1042",
        ingestion_request_id=_INGESTION_REQUEST_ID,
        correlation_id=_CORRELATION_ID,
        now=_TIMESTAMP,
    )


def _create_initial_agent_run(ticket: Ticket) -> AgentRun:
    return AgentRun.create_initial(
        agent_run_id=_AGENT_RUN_ID,
        workspace_id=ticket.workspace_id,
        ticket_id=ticket.id,
        ingestion_request_id=ticket.ingestion_request_id,
        correlation_id=ticket.correlation_id,
        max_attempts=3,
        now=_TIMESTAMP,
    )


def test_ticket_create_request_normalizes_content() -> None:
    request = TicketCreateRequest(
        subject="  Unable to access billing  ",
        description="  The dashboard returns an error.  ",
        external_reference="SUP-1042",
    )

    assert request.subject == "Unable to access billing"
    assert request.description == "The dashboard returns an error."
    assert request.external_reference == "SUP-1042"


@pytest.mark.parametrize(
    "payload",
    [
        {
            "subject": "   ",
            "description": "Valid description",
        },
        {
            "subject": "Valid subject",
            "description": "   ",
        },
        {
            "subject": "a" * 201,
            "description": "Valid description",
        },
        {
            "subject": "Valid subject",
            "description": "a" * 20_001,
        },
        {
            "subject": "Valid subject",
            "description": "Valid description",
            "external_reference": " SUP-1042 ",
        },
        {
            "subject": "Valid subject",
            "description": "Valid description",
            "status": "open",
        },
    ],
)
def test_ticket_create_request_rejects_invalid_payload(
    payload: dict[str, str],
) -> None:
    with pytest.raises(ValidationError):
        TicketCreateRequest.model_validate(payload)


def test_ticket_response_maps_domain_entity() -> None:
    ticket = _create_ticket()

    response = TicketResponse.from_domain(ticket)

    assert response.id == ticket.id
    assert response.workspace_id == ticket.workspace_id
    assert response.status == "open"
    assert response.ingestion_request_id == (ticket.ingestion_request_id)


def test_ticket_processing_run_response_maps_domain_entity() -> None:
    ticket = _create_ticket()
    agent_run = _create_initial_agent_run(ticket)

    response = TicketProcessingRunResponse.from_domain(agent_run)
    payload = response.model_dump()

    assert response.id == agent_run.id
    assert response.status == AgentRunStatus.QUEUED.value
    assert response.status == "queued"
    assert response.workflow_name == (INITIAL_TICKET_PROCESSING_WORKFLOW_NAME)
    assert response.workflow_name == "ticket-processing"
    assert response.workflow_version == (DETERMINISTIC_BASELINE_WORKFLOW_VERSION)
    assert response.workflow_version == "deterministic-baseline-v1"
    assert "attempt_count" not in payload
    assert "max_attempts" not in payload
    assert "lease_token" not in payload


def test_ticket_create_response_maps_domains() -> None:
    ticket = _create_ticket()
    agent_run = _create_initial_agent_run(ticket)

    response = TicketCreateResponse.from_domains(
        ticket=ticket,
        processing_run=agent_run,
    )
    processing_run_payload = response.processing_run.model_dump()

    assert response.ticket == TicketResponse.from_domain(ticket)
    assert response.ticket.ingestion_request_id == (ticket.ingestion_request_id)
    assert response.ticket.correlation_id == ticket.correlation_id
    assert response.processing_run == (TicketProcessingRunResponse.from_domain(agent_run))
    assert set(processing_run_payload) == {
        "id",
        "status",
        "workflow_name",
        "workflow_version",
    }
    assert "lease_owner" not in processing_run_payload
    assert "lease_token" not in processing_run_payload
    assert "lease_expires_at" not in processing_run_payload
