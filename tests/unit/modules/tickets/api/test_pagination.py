"""Unit tests for support ticket pagination cursors."""

import base64
import json
from datetime import UTC, datetime
from uuid import UUID

import pytest

from supportops.modules.tickets.api.pagination import (
    InvalidPaginationCursorError,
    decode_ticket_cursor,
    encode_ticket_cursor,
)


def test_ticket_cursor_round_trip_preserves_keyset_position() -> None:
    created_at = datetime(
        2026,
        7,
        31,
        12,
        30,
        45,
        123456,
        tzinfo=UTC,
    )
    ticket_id = UUID(
        "f84d7304-8171-4842-a111-c3dbda2ff79b",
    )

    cursor = encode_ticket_cursor(
        created_at=created_at,
        ticket_id=ticket_id,
    )
    position = decode_ticket_cursor(cursor)

    assert position.created_at == created_at
    assert position.ticket_id == ticket_id


@pytest.mark.parametrize(
    "cursor",
    [
        "",
        "not-base64",
        base64.urlsafe_b64encode(b"not-json").decode("ascii"),
        base64.urlsafe_b64encode(
            json.dumps(
                {
                    "version": 2,
                    "created_at": "2026-07-31T12:00:00+00:00",
                    "ticket_id": ("f84d7304-8171-4842-a111-c3dbda2ff79b"),
                }
            ).encode("utf-8")
        ).decode("ascii"),
        base64.urlsafe_b64encode(
            json.dumps(
                {
                    "version": 1,
                    "created_at": "not-a-timestamp",
                    "ticket_id": ("f84d7304-8171-4842-a111-c3dbda2ff79b"),
                }
            ).encode("utf-8")
        ).decode("ascii"),
    ],
)
def test_decode_ticket_cursor_rejects_invalid_values(
    cursor: str,
) -> None:
    with pytest.raises(
        InvalidPaginationCursorError,
        match=r"Pagination cursor is invalid\.",
    ):
        decode_ticket_cursor(cursor)


def test_encode_ticket_cursor_rejects_naive_timestamp() -> None:
    with pytest.raises(
        ValueError,
        match=r"Ticket cursor timestamp must be timezone-aware\.",
    ):
        encode_ticket_cursor(
            created_at=datetime(2026, 7, 31, 12, 0),
            ticket_id=UUID(
                "f84d7304-8171-4842-a111-c3dbda2ff79b",
            ),
        )
