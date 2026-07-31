"""Opaque cursor encoding for support ticket pagination."""

import base64
import binascii
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from uuid import UUID

_CURSOR_VERSION = 1


class InvalidPaginationCursorError(Exception):
    """Raised when a ticket pagination cursor cannot be trusted."""


@dataclass(frozen=True, slots=True)
class TicketPaginationPosition:
    """Decoded keyset position for a ticket list query."""

    created_at: datetime
    ticket_id: UUID


def encode_ticket_cursor(
    *,
    created_at: datetime,
    ticket_id: UUID,
) -> str:
    """Encode a ticket keyset position as an opaque URL-safe cursor."""

    if created_at.tzinfo is None or created_at.utcoffset() is None:
        raise ValueError("Ticket cursor timestamp must be timezone-aware.")

    normalized_created_at = created_at.astimezone(UTC)

    payload = {
        "version": _CURSOR_VERSION,
        "created_at": normalized_created_at.isoformat(),
        "ticket_id": str(ticket_id),
    }
    serialized = json.dumps(
        payload,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")

    return base64.urlsafe_b64encode(serialized).decode("ascii").rstrip("=")


def decode_ticket_cursor(
    cursor: str,
) -> TicketPaginationPosition:
    """Decode and validate an opaque ticket pagination cursor."""

    try:
        padding = "=" * (-len(cursor) % 4)
        decoded = base64.b64decode(
            cursor + padding,
            altchars=b"-_",
            validate=True,
        )
        payload = json.loads(decoded.decode("utf-8"))

        if not isinstance(payload, dict):
            raise ValueError("Cursor payload must be an object.")

        if set(payload) != {
            "version",
            "created_at",
            "ticket_id",
        }:
            raise ValueError("Cursor payload fields are invalid.")

        if payload["version"] != _CURSOR_VERSION:
            raise ValueError("Cursor version is unsupported.")

        created_at = datetime.fromisoformat(payload["created_at"])

        if created_at.tzinfo is None or created_at.utcoffset() is None:
            raise ValueError(
                "Cursor timestamp must be timezone-aware.",
            )

        ticket_id = UUID(payload["ticket_id"])
    except (
        binascii.Error,
        json.JSONDecodeError,
        TypeError,
        UnicodeDecodeError,
        ValueError,
    ) as error:
        raise InvalidPaginationCursorError(
            "Pagination cursor is invalid.",
        ) from error

    return TicketPaginationPosition(
        created_at=created_at.astimezone(UTC),
        ticket_id=ticket_id,
    )
