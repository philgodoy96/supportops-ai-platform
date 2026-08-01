"""Unit tests for classification pagination cursors."""

import base64
import json
from datetime import UTC, datetime
from uuid import UUID

import pytest

from supportops.modules.ticket_classifications.application.pagination import (
    InvalidClassificationPaginationCursorError,
    decode_ticket_classification_cursor,
    encode_ticket_classification_cursor,
)

_CREATED_AT = datetime(
    2026,
    8,
    1,
    20,
    15,
    30,
    123456,
    tzinfo=UTC,
)
_CLASSIFICATION_ID = UUID(
    "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa",
)


def test_classification_cursor_round_trip_preserves_position() -> None:
    cursor = encode_ticket_classification_cursor(
        created_at=_CREATED_AT,
        classification_id=_CLASSIFICATION_ID,
    )

    position = decode_ticket_classification_cursor(
        cursor,
    )

    assert position.created_at == _CREATED_AT
    assert position.classification_id == (_CLASSIFICATION_ID)


def test_classification_cursor_normalizes_timestamp_to_utc() -> None:
    created_at = datetime.fromisoformat(
        "2026-08-01T17:15:30-03:00",
    )

    cursor = encode_ticket_classification_cursor(
        created_at=created_at,
        classification_id=_CLASSIFICATION_ID,
    )
    position = decode_ticket_classification_cursor(
        cursor,
    )

    assert position.created_at == _CREATED_AT.replace(
        microsecond=0,
    )


@pytest.mark.parametrize(
    "cursor",
    [
        "",
        "not-base64",
        base64.urlsafe_b64encode(
            b"not-json",
        ).decode("ascii"),
        base64.urlsafe_b64encode(
            json.dumps(
                {
                    "version": 2,
                    "created_at": ("2026-08-01T20:15:30+00:00"),
                    "classification_id": str(
                        _CLASSIFICATION_ID,
                    ),
                },
            ).encode("utf-8"),
        ).decode("ascii"),
        base64.urlsafe_b64encode(
            json.dumps(
                {
                    "version": 1,
                    "created_at": "not-a-timestamp",
                    "classification_id": str(
                        _CLASSIFICATION_ID,
                    ),
                },
            ).encode("utf-8"),
        ).decode("ascii"),
        base64.urlsafe_b64encode(
            json.dumps(
                {
                    "version": 1,
                    "created_at": ("2026-08-01T20:15:30+00:00"),
                    "classification_id": "not-a-uuid",
                },
            ).encode("utf-8"),
        ).decode("ascii"),
    ],
)
def test_decode_classification_cursor_rejects_invalid_values(
    cursor: str,
) -> None:
    with pytest.raises(
        InvalidClassificationPaginationCursorError,
        match=("Classification pagination cursor is invalid"),
    ):
        decode_ticket_classification_cursor(cursor)


def test_encode_classification_cursor_rejects_naive_timestamp() -> None:
    with pytest.raises(
        ValueError,
        match=("Classification cursor timestamp must be timezone-aware"),
    ):
        encode_ticket_classification_cursor(
            created_at=datetime(
                2026,
                8,
                1,
                20,
                15,
            ),
            classification_id=_CLASSIFICATION_ID,
        )
