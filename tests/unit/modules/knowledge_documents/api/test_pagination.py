"""Tests for knowledge-document opaque pagination cursors."""

import base64
import json
from datetime import UTC, datetime, timedelta, timezone
from uuid import UUID

import pytest

from supportops.modules.knowledge_documents.api.pagination import (
    InvalidKnowledgePaginationCursorError,
    decode_document_cursor,
    decode_document_version_cursor,
    encode_document_cursor,
    encode_document_version_cursor,
)

_DOCUMENT_ID = UUID("276046a2-28ec-4cb1-8bb6-a2ff70f9064b")
_VERSION_ID = UUID("09036916-84cf-4a58-bdf4-09bc52716ec5")
_TIMESTAMP = datetime(
    2026,
    8,
    1,
    22,
    0,
    tzinfo=UTC,
)


def encode_raw_payload(payload: object) -> str:
    """Encode an intentionally untrusted cursor fixture."""

    serialized = json.dumps(
        payload,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return base64.urlsafe_b64encode(serialized).decode("ascii").rstrip("=")


def test_document_cursor_round_trip_normalizes_timestamp_to_utc() -> None:
    source_timestamp = datetime(
        2026,
        8,
        1,
        19,
        0,
        tzinfo=timezone_offset(),
    )

    cursor = encode_document_cursor(
        created_at=source_timestamp,
        document_id=_DOCUMENT_ID,
    )
    position = decode_document_cursor(cursor)

    assert position.created_at == _TIMESTAMP
    assert position.document_id == _DOCUMENT_ID


def timezone_offset() -> timezone:
    """Return a fixed negative-three-hour timezone."""

    return timezone(-timedelta(hours=3))


def test_document_version_cursor_round_trip() -> None:
    cursor = encode_document_version_cursor(
        version_number=7,
        document_version_id=_VERSION_ID,
    )

    position = decode_document_version_cursor(cursor)

    assert position.version_number == 7
    assert position.document_version_id == _VERSION_ID


def test_document_cursor_cannot_be_used_as_version_cursor() -> None:
    cursor = encode_document_cursor(
        created_at=_TIMESTAMP,
        document_id=_DOCUMENT_ID,
    )

    with pytest.raises(
        InvalidKnowledgePaginationCursorError,
        match=r"Pagination cursor is invalid\.",
    ):
        decode_document_version_cursor(cursor)


def test_version_cursor_cannot_be_used_as_document_cursor() -> None:
    cursor = encode_document_version_cursor(
        version_number=1,
        document_version_id=_VERSION_ID,
    )

    with pytest.raises(
        InvalidKnowledgePaginationCursorError,
        match=r"Pagination cursor is invalid\.",
    ):
        decode_document_cursor(cursor)


@pytest.mark.parametrize(
    "cursor",
    [
        "not-a-valid-cursor",
        encode_raw_payload([]),
        encode_raw_payload(
            {
                "version": 1,
                "kind": "knowledge_document",
                "created_at": "not-a-timestamp",
                "document_id": str(_DOCUMENT_ID),
            }
        ),
        encode_raw_payload(
            {
                "version": 99,
                "kind": "knowledge_document",
                "created_at": _TIMESTAMP.isoformat(),
                "document_id": str(_DOCUMENT_ID),
            }
        ),
        encode_raw_payload(
            {
                "version": 1,
                "kind": "knowledge_document",
                "created_at": _TIMESTAMP.isoformat(),
                "document_id": "not-a-uuid",
            }
        ),
    ],
)
def test_document_cursor_rejects_untrusted_payloads(
    cursor: str,
) -> None:
    with pytest.raises(
        InvalidKnowledgePaginationCursorError,
        match=r"Pagination cursor is invalid\.",
    ):
        decode_document_cursor(cursor)


@pytest.mark.parametrize(
    "version_number",
    [0, -1],
)
def test_version_cursor_rejects_non_positive_number(
    version_number: int,
) -> None:
    with pytest.raises(
        ValueError,
        match="must be positive",
    ):
        encode_document_version_cursor(
            version_number=version_number,
            document_version_id=_VERSION_ID,
        )


def test_document_cursor_rejects_naive_timestamp() -> None:
    with pytest.raises(
        ValueError,
        match="must be timezone-aware",
    ):
        encode_document_cursor(
            created_at=datetime(2026, 8, 1, 22, 0),
            document_id=_DOCUMENT_ID,
        )
