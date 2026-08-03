"""Unit tests for approval cursor encoding."""

from datetime import UTC, datetime
from uuid import uuid4

import pytest

from supportops.modules.approvals.api.pagination import (
    InvalidApprovalPaginationCursor,
    decode_approval_cursor,
    encode_approval_cursor,
)
from supportops.modules.approvals.application.queries import (
    ApprovalRequestPageCursor,
)


def test_cursor_round_trips() -> None:
    cursor = ApprovalRequestPageCursor(
        created_at=datetime(2026, 8, 3, 21, 0, tzinfo=UTC),
        approval_request_id=uuid4(),
    )

    assert (
        decode_approval_cursor(
            encode_approval_cursor(cursor),
        )
        == cursor
    )


def test_invalid_cursor_fails_closed() -> None:
    with pytest.raises(InvalidApprovalPaginationCursor):
        decode_approval_cursor("not-a-valid-cursor")


def test_unknown_cursor_version_fails_closed() -> None:
    encoded = (
        "eyJ2ZXJzaW9uIjoyLCJjcmVhdGVkX2F0Ijoi"
        "MjAyNi0wOC0wM1QyMTowMDowMFoiLCJhcHBy"
        "b3ZhbF9yZXF1ZXN0X2lkIjoiMDAwMDAwMDAt"
        "MDAwMC0wMDAwLTAwMDAtMDAwMDAwMDAwMDAx"
        "In0="
    )

    with pytest.raises(InvalidApprovalPaginationCursor):
        decode_approval_cursor(encoded)
