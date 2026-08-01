"""Opaque keyset cursors for knowledge-document API listings."""

import base64
import binascii
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import cast
from uuid import UUID

_CURSOR_VERSION = 1
_DOCUMENT_CURSOR_KIND = "knowledge_document"
_DOCUMENT_VERSION_CURSOR_KIND = "knowledge_document_version"


class InvalidKnowledgePaginationCursorError(Exception):
    """Raised when a knowledge pagination cursor cannot be trusted."""


@dataclass(frozen=True, slots=True)
class DocumentPaginationPosition:
    """Decoded keyset position for a document listing."""

    created_at: datetime
    document_id: UUID


@dataclass(frozen=True, slots=True)
class DocumentVersionPaginationPosition:
    """Decoded keyset position for a document-version listing."""

    version_number: int
    document_version_id: UUID


def encode_document_cursor(
    *,
    created_at: datetime,
    document_id: UUID,
) -> str:
    """Encode a document keyset position as an opaque cursor."""

    if created_at.tzinfo is None or created_at.utcoffset() is None:
        raise ValueError("Document cursor timestamp must be timezone-aware.")

    return _encode_payload(
        {
            "version": _CURSOR_VERSION,
            "kind": _DOCUMENT_CURSOR_KIND,
            "created_at": created_at.astimezone(UTC).isoformat(),
            "document_id": str(document_id),
        }
    )


def decode_document_cursor(
    cursor: str,
) -> DocumentPaginationPosition:
    """Decode and validate a document-list cursor."""

    payload = _decode_payload(cursor)
    if set(payload) != {
        "version",
        "kind",
        "created_at",
        "document_id",
    }:
        raise InvalidKnowledgePaginationCursorError("Pagination cursor is invalid.")
    if payload["version"] != _CURSOR_VERSION or payload["kind"] != _DOCUMENT_CURSOR_KIND:
        raise InvalidKnowledgePaginationCursorError("Pagination cursor is invalid.")

    created_at_value = payload["created_at"]
    document_id_value = payload["document_id"]
    if not isinstance(created_at_value, str):
        raise InvalidKnowledgePaginationCursorError("Pagination cursor is invalid.")
    if not isinstance(document_id_value, str):
        raise InvalidKnowledgePaginationCursorError("Pagination cursor is invalid.")

    try:
        created_at = datetime.fromisoformat(created_at_value)
        document_id = UUID(document_id_value)
    except ValueError as error:
        raise InvalidKnowledgePaginationCursorError("Pagination cursor is invalid.") from error

    if created_at.tzinfo is None or created_at.utcoffset() is None:
        raise InvalidKnowledgePaginationCursorError("Pagination cursor is invalid.")

    return DocumentPaginationPosition(
        created_at=created_at.astimezone(UTC),
        document_id=document_id,
    )


def encode_document_version_cursor(
    *,
    version_number: int,
    document_version_id: UUID,
) -> str:
    """Encode a document-version keyset position."""

    if version_number <= 0:
        raise ValueError("Document version cursor number must be positive.")

    return _encode_payload(
        {
            "version": _CURSOR_VERSION,
            "kind": _DOCUMENT_VERSION_CURSOR_KIND,
            "version_number": version_number,
            "document_version_id": str(document_version_id),
        }
    )


def decode_document_version_cursor(
    cursor: str,
) -> DocumentVersionPaginationPosition:
    """Decode and validate a document-version-list cursor."""

    payload = _decode_payload(cursor)
    if set(payload) != {
        "version",
        "kind",
        "version_number",
        "document_version_id",
    }:
        raise InvalidKnowledgePaginationCursorError("Pagination cursor is invalid.")
    if payload["version"] != _CURSOR_VERSION or payload["kind"] != _DOCUMENT_VERSION_CURSOR_KIND:
        raise InvalidKnowledgePaginationCursorError("Pagination cursor is invalid.")

    version_number = payload["version_number"]
    document_version_id_value = payload["document_version_id"]
    if (
        not isinstance(version_number, int)
        or isinstance(version_number, bool)
        or version_number <= 0
    ):
        raise InvalidKnowledgePaginationCursorError("Pagination cursor is invalid.")
    if not isinstance(document_version_id_value, str):
        raise InvalidKnowledgePaginationCursorError("Pagination cursor is invalid.")

    try:
        document_version_id = UUID(document_version_id_value)
    except ValueError as error:
        raise InvalidKnowledgePaginationCursorError("Pagination cursor is invalid.") from error

    return DocumentVersionPaginationPosition(
        version_number=version_number,
        document_version_id=document_version_id,
    )


def _encode_payload(payload: dict[str, object]) -> str:
    serialized = json.dumps(
        payload,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return base64.urlsafe_b64encode(serialized).decode("ascii").rstrip("=")


def _decode_payload(cursor: str) -> dict[str, object]:
    try:
        padding = "=" * (-len(cursor) % 4)
        decoded = base64.b64decode(
            cursor + padding,
            altchars=b"-_",
            validate=True,
        )
        raw_payload: object = json.loads(decoded.decode("utf-8"))
    except (
        binascii.Error,
        json.JSONDecodeError,
        TypeError,
        UnicodeDecodeError,
        ValueError,
    ) as error:
        raise InvalidKnowledgePaginationCursorError("Pagination cursor is invalid.") from error

    if not isinstance(raw_payload, dict):
        raise InvalidKnowledgePaginationCursorError("Pagination cursor is invalid.")

    return cast(dict[str, object], raw_payload)
