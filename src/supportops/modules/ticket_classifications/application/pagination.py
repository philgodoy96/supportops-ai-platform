"""Opaque cursor encoding for ticket-classification pagination."""

import base64
import binascii
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from uuid import UUID

_CURSOR_VERSION = 1


class InvalidClassificationPaginationCursorError(
    Exception,
):
    """Raised when a classification cursor cannot be trusted."""


@dataclass(frozen=True, slots=True)
class TicketClassificationPaginationPosition:
    """Decoded keyset position for a classification list query."""

    created_at: datetime
    classification_id: UUID


def encode_ticket_classification_cursor(
    *,
    created_at: datetime,
    classification_id: UUID,
) -> str:
    """Encode a classification keyset position as an opaque cursor."""

    if created_at.tzinfo is None or created_at.utcoffset() is None:
        raise ValueError(
            "Classification cursor timestamp must be timezone-aware.",
        )

    normalized_created_at = created_at.astimezone(UTC)
    payload = {
        "version": _CURSOR_VERSION,
        "created_at": normalized_created_at.isoformat(),
        "classification_id": str(classification_id),
    }
    serialized = json.dumps(
        payload,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")

    return base64.urlsafe_b64encode(serialized).decode("ascii").rstrip("=")


def decode_ticket_classification_cursor(
    cursor: str,
) -> TicketClassificationPaginationPosition:
    """Decode and validate an opaque classification cursor."""

    try:
        padding = "=" * (-len(cursor) % 4)
        decoded = base64.b64decode(
            cursor + padding,
            altchars=b"-_",
            validate=True,
        )
        payload = json.loads(decoded.decode("utf-8"))

        if not isinstance(payload, dict):
            raise ValueError(
                "Cursor payload must be an object.",
            )

        if set(payload) != {
            "version",
            "created_at",
            "classification_id",
        }:
            raise ValueError(
                "Cursor payload fields are invalid.",
            )

        if payload["version"] != _CURSOR_VERSION:
            raise ValueError(
                "Cursor version is unsupported.",
            )

        created_at = datetime.fromisoformat(
            payload["created_at"],
        )
        if created_at.tzinfo is None or created_at.utcoffset() is None:
            raise ValueError(
                "Cursor timestamp must be timezone-aware.",
            )

        classification_id = UUID(
            payload["classification_id"],
        )
    except (
        binascii.Error,
        json.JSONDecodeError,
        TypeError,
        UnicodeDecodeError,
        ValueError,
    ) as error:
        raise (
            InvalidClassificationPaginationCursorError(
                "Classification pagination cursor is invalid.",
            )
        ) from error

    return TicketClassificationPaginationPosition(
        created_at=created_at.astimezone(UTC),
        classification_id=classification_id,
    )
