"""Dependency composition for ticket escalation inspection."""

from typing import Annotated

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from supportops.api.dependencies import get_postgresql_session
from supportops.modules.tickets.application.escalation_queries import (
    GetTicketEscalation,
    ListTicketEscalations,
)
from supportops.modules.tickets.infrastructure.escalation_repository import (
    SqlAlchemyTicketEscalationRepository,
)


def get_list_ticket_escalations(
    session: Annotated[
        AsyncSession,
        Depends(get_postgresql_session),
    ],
) -> ListTicketEscalations:
    """Build one session-scoped escalation list query."""

    repository = SqlAlchemyTicketEscalationRepository(session)
    return ListTicketEscalations(repository)


def get_ticket_escalation(
    session: Annotated[
        AsyncSession,
        Depends(get_postgresql_session),
    ],
) -> GetTicketEscalation:
    """Build one session-scoped escalation detail query."""

    repository = SqlAlchemyTicketEscalationRepository(session)
    return GetTicketEscalation(repository)
