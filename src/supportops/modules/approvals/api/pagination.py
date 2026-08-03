"""Opaque cursor encoding for approval inspection."""

import base64
import json
from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, ValidationError

from supportops.modules.approvals.application.queries import (
    ApprovalRequestPageCursor,
)


class InvalidApprovalPaginationCursor(ValueError):
    """Raised when an approval cursor cannot be decoded."""


class _ApprovalCursorPayload(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    version: int
    created_at: datetime
    approval_request_id: UUID


def encode_approval_cursor(
    cursor: ApprovalRequestPageCursor,
) -> str:
    """Encode an opaque versioned cursor."""

    payload = _ApprovalCursorPayload(
        version=1,
        created_at=cursor.created_at,
        approval_request_id=cursor.approval_request_id,
    )
    raw = payload.model_dump_json().encode("utf-8")
    return base64.urlsafe_b64encode(raw).decode("ascii")


def decode_approval_cursor(
    encoded: str,
) -> ApprovalRequestPageCursor:
    """Decode and validate one opaque cursor."""

    try:
        raw = base64.urlsafe_b64decode(
            encoded.encode("ascii"),
        )
        payload = _ApprovalCursorPayload.model_validate_json(raw)
        if payload.version != 1:
            raise InvalidApprovalPaginationCursor(
                "Approval pagination cursor version is unsupported.",
            )
        return ApprovalRequestPageCursor(
            created_at=payload.created_at,
            approval_request_id=payload.approval_request_id,
        )
    except (
        UnicodeEncodeError,
        ValueError,
        ValidationError,
        json.JSONDecodeError,
        TypeError,
    ) as exc:
        if isinstance(exc, InvalidApprovalPaginationCursor):
            raise
        raise InvalidApprovalPaginationCursor(
            "Approval pagination cursor is invalid.",
        ) from exc
