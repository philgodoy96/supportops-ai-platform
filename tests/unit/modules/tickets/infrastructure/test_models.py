"""Unit tests for the support ticket persistence model."""

from datetime import UTC, datetime
from typing import cast
from uuid import UUID

from sqlalchemy import CheckConstraint, Table, UniqueConstraint

from supportops.modules.tickets.domain.models import (
    Ticket,
    TicketStatus,
)
from supportops.modules.tickets.infrastructure.models import (
    TicketRecord,
)


def create_ticket() -> Ticket:
    """Create a deterministic ticket entity for mapping tests."""

    timestamp = datetime(2026, 7, 31, 12, 0, tzinfo=UTC)

    return Ticket(
        id=UUID("f84d7304-8171-4842-a111-c3dbda2ff79b"),
        workspace_id=UUID(
            "032c8c87-57cc-4d14-bfbd-04968b4e8cd4",
        ),
        subject="Unable to access billing",
        description="The dashboard returns an access error.",
        status=TicketStatus.OPEN,
        external_reference="SUP-1042",
        ingestion_request_id=UUID(
            "725eec8a-c504-4071-ac96-c78cc907f26c",
        ),
        correlation_id=UUID(
            "1038c98e-62fd-45df-9839-138f7105cb78",
        ),
        created_at=timestamp,
        updated_at=timestamp,
    )


def test_ticket_record_round_trip_preserves_domain_values() -> None:
    ticket = create_ticket()

    record = TicketRecord.from_domain(ticket)

    assert record.to_domain() == ticket


def test_ticket_table_declares_expected_constraints() -> None:
    table = cast(Table, TicketRecord.__table__)
    constraint_names = {
        constraint.name
        for constraint in table.constraints
        if isinstance(
            constraint,
            (CheckConstraint, UniqueConstraint),
        )
    }

    assert {
        "uq_tickets_workspace_external_reference",
        "ck_tickets_ticket_subject_trimmed",
        "ck_tickets_ticket_subject_length",
        "ck_tickets_ticket_description_trimmed",
        "ck_tickets_ticket_description_length",
        "ck_tickets_ticket_external_reference_format",
        "ck_tickets_ticket_status",
        "ck_tickets_ticket_timestamp_order",
    }.issubset(constraint_names)


def test_ticket_table_declares_workspace_foreign_key() -> None:
    table = cast(Table, TicketRecord.__table__)
    foreign_keys = list(table.c.workspace_id.foreign_keys)

    assert len(foreign_keys) == 1
    assert foreign_keys[0].target_fullname == "workspaces.id"
    assert foreign_keys[0].ondelete == "RESTRICT"


def test_ticket_table_declares_listing_index() -> None:
    table = cast(Table, TicketRecord.__table__)
    index = next(
        index for index in table.indexes if index.name == "ix_tickets_workspace_created_id"
    )

    expressions = [str(expression) for expression in index.expressions]

    assert expressions[0] == "tickets.workspace_id"
    assert expressions[1] == "tickets.created_at DESC"
    assert expressions[2] == "tickets.id DESC"


def test_ticket_table_uses_nonnullable_ownership_and_trace_columns() -> None:
    table = cast(Table, TicketRecord.__table__)

    assert not table.c.workspace_id.nullable
    assert not table.c.ingestion_request_id.nullable
    assert not table.c.correlation_id.nullable
    assert table.c.external_reference.nullable
