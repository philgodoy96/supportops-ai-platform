"""Unit tests for support ticket API schemas."""

from datetime import UTC, datetime
from uuid import UUID

import pytest
from pydantic import ValidationError

from supportops.modules.tickets.api.schemas import (
    TicketCreateRequest,
    TicketResponse,
)
from supportops.modules.tickets.domain.models import Ticket


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
    timestamp = datetime(
        2026,
        7,
        31,
        12,
        0,
        tzinfo=UTC,
    )
    ticket = Ticket.create(
        ticket_id=UUID(
            "f84d7304-8171-4842-a111-c3dbda2ff79b",
        ),
        workspace_id=UUID(
            "032c8c87-57cc-4d14-bfbd-04968b4e8cd4",
        ),
        subject="Unable to access billing",
        description="The dashboard returns an error.",
        external_reference="SUP-1042",
        ingestion_request_id=UUID(
            "725eec8a-c504-4071-ac96-c78cc907f26c",
        ),
        correlation_id=UUID(
            "1038c98e-62fd-45df-9839-138f7105cb78",
        ),
        now=timestamp,
    )

    response = TicketResponse.from_domain(ticket)

    assert response.id == ticket.id
    assert response.workspace_id == ticket.workspace_id
    assert response.status == "open"
    assert response.ingestion_request_id == (ticket.ingestion_request_id)
