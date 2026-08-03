"""Unit tests for ticket escalation inspection queries."""

from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from supportops.modules.tickets.application.escalation_queries import (
    GetTicketEscalation,
    ListTicketEscalations,
    TicketEscalationListPage,
    TicketEscalationListQuery,
    TicketEscalationNotFoundError,
    TicketEscalationPageCursor,
)


@pytest.mark.asyncio
async def test_list_ticket_escalations_delegates_query() -> None:
    query = TicketEscalationListQuery(
        workspace_id=uuid4(),
        ticket_id=uuid4(),
        page_size=20,
    )
    page = TicketEscalationListPage(
        items=(),
        next_cursor=None,
    )
    repository = SimpleNamespace(
        list_page=AsyncMock(return_value=page),
    )

    result = await ListTicketEscalations(cast(Any, repository)).execute(query)

    assert result is page
    repository.list_page.assert_awaited_once_with(query)


@pytest.mark.asyncio
async def test_get_ticket_escalation_returns_workspace_record() -> None:
    escalation = SimpleNamespace(id=uuid4())
    repository = SimpleNamespace(
        get_by_id=AsyncMock(return_value=escalation),
    )
    workspace_id = uuid4()

    result = await GetTicketEscalation(cast(Any, repository)).execute(
        workspace_id=workspace_id,
        escalation_id=escalation.id,
    )

    assert result is cast(Any, escalation)
    repository.get_by_id.assert_awaited_once_with(
        workspace_id=workspace_id,
        escalation_id=escalation.id,
    )


@pytest.mark.asyncio
async def test_get_ticket_escalation_hides_missing_record() -> None:
    repository = SimpleNamespace(
        get_by_id=AsyncMock(return_value=None),
    )

    with pytest.raises(TicketEscalationNotFoundError):
        await GetTicketEscalation(cast(Any, repository)).execute(
            workspace_id=uuid4(),
            escalation_id=uuid4(),
        )


def test_escalation_page_size_minimum() -> None:
    with pytest.raises(ValueError, match="page_size"):
        TicketEscalationListQuery(
            workspace_id=uuid4(),
            page_size=0,
        )


def test_escalation_page_size_maximum() -> None:
    with pytest.raises(ValueError, match="page_size"):
        TicketEscalationListQuery(
            workspace_id=uuid4(),
            page_size=101,
        )


def test_escalation_workspace_id_must_be_uuid() -> None:
    with pytest.raises(TypeError, match="workspace_id"):
        TicketEscalationListQuery(
            workspace_id="not-a-uuid",  # type: ignore[arg-type]
        )


def test_escalation_ticket_id_must_be_uuid() -> None:
    with pytest.raises(TypeError, match="ticket_id"):
        TicketEscalationListQuery(
            workspace_id=uuid4(),
            ticket_id="not-a-uuid",  # type: ignore[arg-type]
        )


def test_escalation_cursor_rejects_naive_timestamp() -> None:
    with pytest.raises(ValueError, match="UTC-aware"):
        TicketEscalationListQuery(
            workspace_id=uuid4(),
            cursor=TicketEscalationPageCursor(
                created_at=datetime(2026, 8, 3, 18, 0),
                ticket_escalation_id=uuid4(),
            ),
        )


def test_escalation_cursor_accepts_utc_timestamp() -> None:
    cursor = TicketEscalationPageCursor(
        created_at=datetime(2026, 8, 3, 18, 0, tzinfo=UTC),
        ticket_escalation_id=uuid4(),
    )
    query = TicketEscalationListQuery(
        workspace_id=uuid4(),
        cursor=cursor,
    )
    assert query.cursor is cursor
