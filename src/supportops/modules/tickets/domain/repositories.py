"""Support ticket repository contracts."""

from collections.abc import Sequence
from datetime import datetime
from typing import Protocol
from uuid import UUID

from supportops.modules.tickets.domain.models import Ticket


class TicketExternalReferenceConflictError(Exception):
    """Raised when an external reference already exists in one workspace."""


class TicketRepository(Protocol):
    """Workspace-scoped persistence operations for support tickets."""

    async def add(self, ticket: Ticket) -> None:
        """Add a ticket to the active transaction."""

        ...

    async def get(
        self,
        workspace_id: UUID,
        ticket_id: UUID,
    ) -> Ticket | None:
        """Return a ticket only through its workspace boundary."""

        ...

    async def list(
        self,
        workspace_id: UUID,
        *,
        limit: int,
        after_created_at: datetime | None = None,
        after_ticket_id: UUID | None = None,
    ) -> Sequence[Ticket]:
        """List workspace tickets in deterministic descending order."""

        ...
