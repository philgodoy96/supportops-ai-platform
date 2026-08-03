"""Unit tests for ticket escalation cursor encoding."""

from datetime import UTC, datetime
from uuid import uuid4

import pytest

from supportops.modules.tickets.api.escalation_pagination import (
    InvalidTicketEscalationPaginationCursor,
    decode_ticket_escalation_cursor,
    encode_ticket_escalation_cursor,
)
from supportops.modules.tickets.application.escalation_queries import (
    TicketEscalationPageCursor,
)


def test_cursor_round_trips() -> None:
    cursor = TicketEscalationPageCursor(
        created_at=datetime(2026, 8, 3, 22, 0, tzinfo=UTC),
        ticket_escalation_id=uuid4(),
    )

    assert (
        decode_ticket_escalation_cursor(
            encode_ticket_escalation_cursor(cursor),
        )
        == cursor
    )


def test_invalid_cursor_fails_closed() -> None:
    with pytest.raises(
        InvalidTicketEscalationPaginationCursor,
    ):
        decode_ticket_escalation_cursor("invalid")


def test_naive_cursor_fails_closed() -> None:
    import base64
    import json

    payload = {
        "version": 1,
        "created_at": "2026-08-03T22:00:00",
        "ticket_escalation_id": str(uuid4()),
    }
    encoded = base64.urlsafe_b64encode(
        json.dumps(payload).encode("utf-8"),
    ).decode("ascii")

    with pytest.raises(
        InvalidTicketEscalationPaginationCursor,
    ):
        decode_ticket_escalation_cursor(encoded)
