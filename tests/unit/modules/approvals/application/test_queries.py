"""Unit tests for approval inspection queries."""

from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from supportops.modules.approvals.application.queries import (
    ApprovalRequestListPage,
    ApprovalRequestListQuery,
    ApprovalRequestNotFoundError,
    ApprovalRequestPageCursor,
    GetApprovalRequest,
    ListApprovalRequests,
)


@pytest.mark.asyncio
async def test_list_approval_requests_delegates_query() -> None:
    query = ApprovalRequestListQuery(
        workspace_id=uuid4(),
        page_size=20,
    )
    page = ApprovalRequestListPage(
        items=(),
        next_cursor=None,
    )
    repository = SimpleNamespace(
        list_page=AsyncMock(return_value=page),
    )

    result = await ListApprovalRequests(cast(Any, repository)).execute(query)

    assert result is page
    repository.list_page.assert_awaited_once_with(query)


@pytest.mark.asyncio
async def test_get_approval_request_returns_workspace_record() -> None:
    approval = SimpleNamespace(id=uuid4())
    repository = SimpleNamespace(
        get_by_id=AsyncMock(return_value=approval),
    )
    workspace_id = uuid4()

    result = await GetApprovalRequest(cast(Any, repository)).execute(
        workspace_id=workspace_id,
        approval_request_id=approval.id,
    )

    assert result is cast(Any, approval)
    repository.get_by_id.assert_awaited_once_with(
        workspace_id=workspace_id,
        approval_request_id=approval.id,
    )


@pytest.mark.asyncio
async def test_get_approval_request_hides_missing_record() -> None:
    repository = SimpleNamespace(
        get_by_id=AsyncMock(return_value=None),
    )

    with pytest.raises(ApprovalRequestNotFoundError):
        await GetApprovalRequest(cast(Any, repository)).execute(
            workspace_id=uuid4(),
            approval_request_id=uuid4(),
        )


def test_approval_page_size_minimum() -> None:
    with pytest.raises(ValueError, match="page_size"):
        ApprovalRequestListQuery(
            workspace_id=uuid4(),
            page_size=0,
        )


def test_approval_page_size_maximum() -> None:
    with pytest.raises(ValueError, match="page_size"):
        ApprovalRequestListQuery(
            workspace_id=uuid4(),
            page_size=101,
        )


def test_approval_workspace_id_must_be_uuid() -> None:
    with pytest.raises(TypeError, match="workspace_id"):
        ApprovalRequestListQuery(
            workspace_id="not-a-uuid",  # type: ignore[arg-type]
        )


def test_approval_cursor_rejects_naive_timestamp() -> None:
    with pytest.raises(ValueError, match="UTC-aware"):
        ApprovalRequestListQuery(
            workspace_id=uuid4(),
            cursor=ApprovalRequestPageCursor(
                created_at=datetime(2026, 8, 3, 18, 0),
                approval_request_id=uuid4(),
            ),
        )


def test_approval_cursor_accepts_utc_timestamp() -> None:
    cursor = ApprovalRequestPageCursor(
        created_at=datetime(2026, 8, 3, 18, 0, tzinfo=UTC),
        approval_request_id=uuid4(),
    )
    query = ApprovalRequestListQuery(
        workspace_id=uuid4(),
        cursor=cursor,
    )
    assert query.cursor is cursor
