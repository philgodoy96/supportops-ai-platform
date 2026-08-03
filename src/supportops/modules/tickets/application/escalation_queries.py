"""Workspace-scoped ticket escalation inspection queries."""

from uuid import UUID

from supportops.modules.tickets.domain.escalation import (
    TicketEscalation,
)
from supportops.modules.tickets.domain.escalation_repositories import (
    TicketEscalationListPage,
    TicketEscalationListQuery,
    TicketEscalationPageCursor,
    TicketEscalationRepository,
)

__all__ = [
    "GetTicketEscalation",
    "ListTicketEscalations",
    "TicketEscalationListPage",
    "TicketEscalationListQuery",
    "TicketEscalationNotFoundError",
    "TicketEscalationPageCursor",
]


class TicketEscalationNotFoundError(LookupError):
    """Raised when an escalation is absent from the workspace."""


class ListTicketEscalations:
    """List escalations without exposing persistence concerns."""

    def __init__(
        self,
        repository: TicketEscalationRepository,
    ) -> None:
        self._repository = repository

    async def execute(
        self,
        query: TicketEscalationListQuery,
    ) -> TicketEscalationListPage:
        """Return one stable escalation page."""

        return await self._repository.list_page(query)


class GetTicketEscalation:
    """Load one escalation through workspace-scoped nondisclosure."""

    def __init__(
        self,
        repository: TicketEscalationRepository,
    ) -> None:
        self._repository = repository

    async def execute(
        self,
        *,
        workspace_id: UUID,
        escalation_id: UUID,
    ) -> TicketEscalation:
        """Return one escalation or raise a nondisclosing not-found."""

        escalation = await self._repository.get_by_id(
            workspace_id=workspace_id,
            escalation_id=escalation_id,
        )
        if escalation is None:
            raise TicketEscalationNotFoundError(
                "Ticket escalation was not found.",
            )
        return escalation
