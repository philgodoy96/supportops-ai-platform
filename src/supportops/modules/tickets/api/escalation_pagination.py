"""Opaque cursor encoding for ticket escalation inspection."""

import base64
import json
from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, ValidationError

from supportops.modules.tickets.application.escalation_queries import (
    TicketEscalationPageCursor,
)


class InvalidTicketEscalationPaginationCursor(ValueError):
    """Raised when an escalation cursor cannot be decoded."""


class _TicketEscalationCursorPayload(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    version: int
    created_at: datetime
    ticket_escalation_id: UUID


def encode_ticket_escalation_cursor(
    cursor: TicketEscalationPageCursor,
) -> str:
    """Encode an opaque versioned cursor."""

    payload = _TicketEscalationCursorPayload(
        version=1,
        created_at=cursor.created_at,
        ticket_escalation_id=cursor.ticket_escalation_id,
    )
    raw = payload.model_dump_json().encode("utf-8")
    return base64.urlsafe_b64encode(raw).decode("ascii")


def decode_ticket_escalation_cursor(
    encoded: str,
) -> TicketEscalationPageCursor:
    """Decode and validate one opaque cursor."""

    try:
        raw = base64.urlsafe_b64decode(
            encoded.encode("ascii"),
        )
        payload = _TicketEscalationCursorPayload.model_validate_json(
            raw,
        )
        if payload.version != 1:
            raise InvalidTicketEscalationPaginationCursor(
                "Ticket escalation pagination cursor version is unsupported.",
            )
        if payload.created_at.tzinfo is None:
            raise InvalidTicketEscalationPaginationCursor(
                "Ticket escalation pagination cursor requires timezone.",
            )
        return TicketEscalationPageCursor(
            created_at=payload.created_at,
            ticket_escalation_id=payload.ticket_escalation_id,
        )
    except (
        UnicodeEncodeError,
        ValueError,
        ValidationError,
        json.JSONDecodeError,
        TypeError,
    ) as exc:
        if isinstance(exc, InvalidTicketEscalationPaginationCursor):
            raise
        raise InvalidTicketEscalationPaginationCursor(
            "Ticket escalation pagination cursor is invalid.",
        ) from exc
