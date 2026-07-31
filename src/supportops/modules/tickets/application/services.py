"""Workspace-scoped support ticket application use cases."""

from collections.abc import Sequence
from datetime import datetime
from uuid import UUID

from supportops.core.transactions import TransactionManager
from supportops.modules.tickets.application.errors import (
    TicketExternalReferenceConflictApplicationError,
    TicketNotFoundError,
)
from supportops.modules.tickets.domain.models import Ticket
from supportops.modules.tickets.domain.repositories import (
    TicketExternalReferenceConflictError,
    TicketRepository,
)
from supportops.modules.workspaces.application.errors import (
    WorkspaceNotFoundError,
)
from supportops.modules.workspaces.domain.repositories import (
    WorkspaceRepository,
)


class CreateTicket:
    """Create a support ticket within one workspace."""

    def __init__(
        self,
        *,
        workspace_repository: WorkspaceRepository,
        ticket_repository: TicketRepository,
        transaction_manager: TransactionManager,
    ) -> None:
        self._workspace_repository = workspace_repository
        self._ticket_repository = ticket_repository
        self._transaction_manager = transaction_manager

    async def execute(
        self,
        *,
        workspace_id: UUID,
        subject: str,
        description: str,
        ingestion_request_id: UUID,
        correlation_id: UUID,
        external_reference: str | None = None,
    ) -> Ticket:
        """Create and persist a workspace-owned ticket atomically."""

        ticket = Ticket.create(
            workspace_id=workspace_id,
            subject=subject,
            description=description,
            external_reference=external_reference,
            ingestion_request_id=ingestion_request_id,
            correlation_id=correlation_id,
        )

        try:
            async with self._transaction_manager.transaction():
                if not await self._workspace_repository.exists(
                    workspace_id,
                ):
                    raise WorkspaceNotFoundError(
                        "Workspace was not found.",
                    )

                await self._ticket_repository.add(ticket)
        except TicketExternalReferenceConflictError as error:
            raise TicketExternalReferenceConflictApplicationError(
                "Ticket external reference already exists in the workspace.",
            ) from error

        return ticket


class GetTicket:
    """Retrieve a ticket through its workspace boundary."""

    def __init__(
        self,
        *,
        repository: TicketRepository,
    ) -> None:
        self._repository = repository

    async def execute(
        self,
        *,
        workspace_id: UUID,
        ticket_id: UUID,
    ) -> Ticket:
        """Return the scoped ticket or raise a stable not-found error."""

        ticket = await self._repository.get(
            workspace_id,
            ticket_id,
        )

        if ticket is None:
            raise TicketNotFoundError(
                "Ticket was not found.",
            )

        return ticket


class ListTickets:
    """List tickets belonging to one workspace."""

    def __init__(
        self,
        *,
        workspace_repository: WorkspaceRepository,
        ticket_repository: TicketRepository,
    ) -> None:
        self._workspace_repository = workspace_repository
        self._ticket_repository = ticket_repository

    async def execute(
        self,
        *,
        workspace_id: UUID,
        limit: int,
        after_created_at: datetime | None = None,
        after_ticket_id: UUID | None = None,
    ) -> Sequence[Ticket]:
        """Return one deterministic page of workspace tickets."""

        if not await self._workspace_repository.exists(workspace_id):
            raise WorkspaceNotFoundError(
                "Workspace was not found.",
            )

        return await self._ticket_repository.list(
            workspace_id,
            limit=limit,
            after_created_at=after_created_at,
            after_ticket_id=after_ticket_id,
        )
