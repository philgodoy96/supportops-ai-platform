"""Unit tests for support ticket application services."""

from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from uuid import UUID

import pytest

from supportops.modules.tickets.application.errors import (
    TicketExternalReferenceConflictApplicationError,
    TicketNotFoundError,
)
from supportops.modules.tickets.application.services import (
    CreateTicket,
    GetTicket,
    ListTickets,
)
from supportops.modules.tickets.domain.models import Ticket
from supportops.modules.tickets.domain.repositories import (
    TicketExternalReferenceConflictError,
)
from supportops.modules.workspaces.application.errors import (
    WorkspaceNotFoundError,
)
from supportops.modules.workspaces.domain.models import Workspace

_WORKSPACE_ID = UUID(
    "032c8c87-57cc-4d14-bfbd-04968b4e8cd4",
)
_OTHER_WORKSPACE_ID = UUID(
    "4aefba3b-b57e-47d1-889e-bb28762fa1ed",
)
_TICKET_ID = UUID(
    "f84d7304-8171-4842-a111-c3dbda2ff79b",
)
_REQUEST_ID = UUID(
    "725eec8a-c504-4071-ac96-c78cc907f26c",
)
_CORRELATION_ID = UUID(
    "1038c98e-62fd-45df-9839-138f7105cb78",
)
_TIMESTAMP = datetime(
    2026,
    7,
    31,
    12,
    0,
    tzinfo=UTC,
)


class FakeTransactionManager:
    """Record transaction completion and rollback."""

    def __init__(self) -> None:
        self.entered = False
        self.completed = False
        self.rolled_back = False

    @asynccontextmanager
    async def transaction(self) -> AsyncIterator[None]:
        self.entered = True

        try:
            yield
        except Exception:
            self.rolled_back = True
            raise
        else:
            self.completed = True


class FakeWorkspaceRepository:
    """In-memory workspace repository fake."""

    def __init__(self, *, workspace_exists: bool = True) -> None:
        self.workspace_exists = workspace_exists
        self.requested_workspace_id: UUID | None = None

    async def add(self, workspace: Workspace) -> None:
        raise AssertionError("add must not be called")

    async def get(
        self,
        workspace_id: UUID,
    ) -> Workspace | None:
        raise AssertionError("get must not be called")

    async def exists(self, workspace_id: UUID) -> bool:
        self.requested_workspace_id = workspace_id
        return self.workspace_exists


class FakeTicketRepository:
    """In-memory workspace-scoped ticket repository fake."""

    def __init__(self) -> None:
        self.tickets: dict[tuple[UUID, UUID], Ticket] = {}
        self.added_ticket: Ticket | None = None
        self.external_reference_conflict = False
        self.list_result: Sequence[Ticket] = ()
        self.list_arguments: (
            tuple[
                UUID,
                int,
                datetime | None,
                UUID | None,
            ]
            | None
        ) = None

    async def add(self, ticket: Ticket) -> None:
        if self.external_reference_conflict:
            raise TicketExternalReferenceConflictError(
                "duplicate external reference",
            )

        self.added_ticket = ticket
        self.tickets[(ticket.workspace_id, ticket.id)] = ticket

    async def get(
        self,
        workspace_id: UUID,
        ticket_id: UUID,
    ) -> Ticket | None:
        return self.tickets.get((workspace_id, ticket_id))

    async def list(
        self,
        workspace_id: UUID,
        *,
        limit: int,
        after_created_at: datetime | None = None,
        after_ticket_id: UUID | None = None,
    ) -> Sequence[Ticket]:
        self.list_arguments = (
            workspace_id,
            limit,
            after_created_at,
            after_ticket_id,
        )
        return self.list_result


def create_ticket(
    *,
    workspace_id: UUID = _WORKSPACE_ID,
) -> Ticket:
    """Create a deterministic ticket."""

    return Ticket.create(
        ticket_id=_TICKET_ID,
        workspace_id=workspace_id,
        subject="Unable to access billing",
        description="The dashboard returns an access error.",
        external_reference="SUP-1042",
        ingestion_request_id=_REQUEST_ID,
        correlation_id=_CORRELATION_ID,
        now=_TIMESTAMP,
    )


async def test_create_ticket_checks_workspace_and_persists_atomically() -> None:
    workspace_repository = FakeWorkspaceRepository()
    ticket_repository = FakeTicketRepository()
    transaction_manager = FakeTransactionManager()
    service = CreateTicket(
        workspace_repository=workspace_repository,
        ticket_repository=ticket_repository,
        transaction_manager=transaction_manager,
    )

    ticket = await service.execute(
        workspace_id=_WORKSPACE_ID,
        subject="  Unable to access billing  ",
        description="  The dashboard returns an access error.  ",
        external_reference="SUP-1042",
        ingestion_request_id=_REQUEST_ID,
        correlation_id=_CORRELATION_ID,
    )

    assert workspace_repository.requested_workspace_id == (_WORKSPACE_ID)
    assert ticket_repository.added_ticket == ticket
    assert transaction_manager.completed
    assert ticket.subject == "Unable to access billing"


async def test_create_ticket_raises_when_workspace_is_missing() -> None:
    workspace_repository = FakeWorkspaceRepository(
        workspace_exists=False,
    )
    ticket_repository = FakeTicketRepository()
    transaction_manager = FakeTransactionManager()
    service = CreateTicket(
        workspace_repository=workspace_repository,
        ticket_repository=ticket_repository,
        transaction_manager=transaction_manager,
    )

    with pytest.raises(
        WorkspaceNotFoundError,
        match=r"Workspace was not found\.",
    ):
        await service.execute(
            workspace_id=_WORKSPACE_ID,
            subject="Unable to access billing",
            description="The dashboard returns an access error.",
            ingestion_request_id=_REQUEST_ID,
            correlation_id=_CORRELATION_ID,
        )

    assert ticket_repository.added_ticket is None
    assert transaction_manager.rolled_back


async def test_create_ticket_translates_external_reference_conflict() -> None:
    workspace_repository = FakeWorkspaceRepository()
    ticket_repository = FakeTicketRepository()
    ticket_repository.external_reference_conflict = True
    transaction_manager = FakeTransactionManager()
    service = CreateTicket(
        workspace_repository=workspace_repository,
        ticket_repository=ticket_repository,
        transaction_manager=transaction_manager,
    )

    with pytest.raises(
        TicketExternalReferenceConflictApplicationError,
        match=r"Ticket external reference already exists in the workspace\.",
    ):
        await service.execute(
            workspace_id=_WORKSPACE_ID,
            subject="Unable to access billing",
            description="The dashboard returns an access error.",
            external_reference="SUP-1042",
            ingestion_request_id=_REQUEST_ID,
            correlation_id=_CORRELATION_ID,
        )

    assert transaction_manager.rolled_back


async def test_get_ticket_requires_matching_workspace() -> None:
    ticket = create_ticket()
    repository = FakeTicketRepository()
    repository.tickets[(ticket.workspace_id, ticket.id)] = ticket
    service = GetTicket(repository=repository)

    result = await service.execute(
        workspace_id=_WORKSPACE_ID,
        ticket_id=_TICKET_ID,
    )

    assert result == ticket

    with pytest.raises(
        TicketNotFoundError,
        match=r"Ticket was not found\.",
    ):
        await service.execute(
            workspace_id=_OTHER_WORKSPACE_ID,
            ticket_id=_TICKET_ID,
        )


async def test_list_tickets_checks_workspace_and_forwards_keyset() -> None:
    ticket = create_ticket()
    workspace_repository = FakeWorkspaceRepository()
    ticket_repository = FakeTicketRepository()
    ticket_repository.list_result = (ticket,)
    service = ListTickets(
        workspace_repository=workspace_repository,
        ticket_repository=ticket_repository,
    )

    result = await service.execute(
        workspace_id=_WORKSPACE_ID,
        limit=21,
        after_created_at=_TIMESTAMP,
        after_ticket_id=_TICKET_ID,
    )

    assert result == (ticket,)
    assert ticket_repository.list_arguments == (
        _WORKSPACE_ID,
        21,
        _TIMESTAMP,
        _TICKET_ID,
    )


async def test_list_tickets_raises_for_missing_workspace() -> None:
    workspace_repository = FakeWorkspaceRepository(
        workspace_exists=False,
    )
    ticket_repository = FakeTicketRepository()
    service = ListTickets(
        workspace_repository=workspace_repository,
        ticket_repository=ticket_repository,
    )

    with pytest.raises(
        WorkspaceNotFoundError,
        match=r"Workspace was not found\.",
    ):
        await service.execute(
            workspace_id=_WORKSPACE_ID,
            limit=20,
        )

    assert ticket_repository.list_arguments is None
