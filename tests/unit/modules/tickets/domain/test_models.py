"""Unit tests for support ticket domain entities."""

from dataclasses import FrozenInstanceError
from datetime import UTC, datetime, timedelta, timezone
from re import escape
from uuid import UUID

import pytest

from supportops.modules.tickets.domain.models import (
    Ticket,
    TicketStatus,
)


def create_ticket(
    *,
    subject: str = "  Unable to access billing  ",
    description: str = "  The dashboard returns an access error.  ",
    external_reference: str | None = "SUP-1042",
    now: datetime | None = None,
) -> Ticket:
    return Ticket.create(
        workspace_id=UUID(
            "032c8c87-57cc-4d14-bfbd-04968b4e8cd4",
        ),
        subject=subject,
        description=description,
        external_reference=external_reference,
        ingestion_request_id=UUID(
            "725eec8a-c504-4071-ac96-c78cc907f26c",
        ),
        correlation_id=UUID(
            "1038c98e-62fd-45df-9839-138f7105cb78",
        ),
        now=now,
    )


def test_ticket_create_normalizes_content_and_assigns_open_status() -> None:
    now = datetime(2026, 7, 31, 12, 0, tzinfo=UTC)

    ticket = create_ticket(now=now)

    assert ticket.subject == "Unable to access billing"
    assert ticket.description == ("The dashboard returns an access error.")
    assert ticket.status is TicketStatus.OPEN
    assert ticket.external_reference == "SUP-1042"
    assert ticket.id.version == 4
    assert ticket.created_at == now
    assert ticket.updated_at == now


def test_ticket_create_allows_omitted_external_reference() -> None:
    ticket = create_ticket(external_reference=None)

    assert ticket.external_reference is None


@pytest.mark.parametrize(
    "subject",
    [
        "",
        "   ",
        "a" * 201,
    ],
)
def test_ticket_create_rejects_invalid_subject(
    subject: str,
) -> None:
    with pytest.raises(ValueError):
        create_ticket(subject=subject)


@pytest.mark.parametrize(
    "description",
    [
        "",
        "   ",
        "a" * 20_001,
    ],
)
def test_ticket_create_rejects_invalid_description(
    description: str,
) -> None:
    with pytest.raises(ValueError):
        create_ticket(description=description)


@pytest.mark.parametrize(
    "external_reference",
    [
        "",
        " SUP-1042",
        "SUP-1042 ",
        "a" * 129,
    ],
)
def test_ticket_create_rejects_invalid_external_reference(
    external_reference: str,
) -> None:
    with pytest.raises(ValueError):
        create_ticket(
            external_reference=external_reference,
        )


def test_ticket_workspace_ownership_is_immutable() -> None:
    ticket = create_ticket()

    with pytest.raises(FrozenInstanceError):
        ticket.workspace_id = UUID(  # type: ignore[misc]
            "4aefba3b-b57e-47d1-889e-bb28762fa1ed",
        )


def test_ticket_rejects_non_utc_timestamp() -> None:
    non_utc = datetime(
        2026,
        7,
        31,
        9,
        0,
        tzinfo=timezone(timedelta(hours=-3)),
    )

    with pytest.raises(
        ValueError,
        match=escape("created_at must be a UTC-aware timestamp."),
    ):
        Ticket(
            id=UUID("f84d7304-8171-4842-a111-c3dbda2ff79b"),
            workspace_id=UUID(
                "032c8c87-57cc-4d14-bfbd-04968b4e8cd4",
            ),
            subject="Unable to access billing",
            description="The dashboard returns an access error.",
            status=TicketStatus.OPEN,
            external_reference=None,
            ingestion_request_id=UUID(
                "725eec8a-c504-4071-ac96-c78cc907f26c",
            ),
            correlation_id=UUID(
                "1038c98e-62fd-45df-9839-138f7105cb78",
            ),
            created_at=non_utc,
            updated_at=non_utc,
        )
