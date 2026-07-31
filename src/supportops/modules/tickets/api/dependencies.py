"""FastAPI dependencies for support ticket use cases."""

from typing import Annotated

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from supportops.api.dependencies import get_postgresql_session
from supportops.infrastructure.postgresql.transaction import (
    SqlAlchemyTransactionManager,
)
from supportops.modules.tickets.application.services import (
    CreateTicket,
    GetTicket,
    ListTickets,
)
from supportops.modules.tickets.infrastructure.repository import (
    SqlAlchemyTicketRepository,
)
from supportops.modules.workspaces.infrastructure.repository import (
    SqlAlchemyWorkspaceRepository,
)

PostgresqlSessionDependency = Annotated[
    AsyncSession,
    Depends(get_postgresql_session),
]


def get_create_ticket(
    session: PostgresqlSessionDependency,
) -> CreateTicket:
    """Construct the create-ticket use case."""

    return CreateTicket(
        workspace_repository=SqlAlchemyWorkspaceRepository(session),
        ticket_repository=SqlAlchemyTicketRepository(session),
        transaction_manager=SqlAlchemyTransactionManager(session),
    )


def get_get_ticket(
    session: PostgresqlSessionDependency,
) -> GetTicket:
    """Construct the get-ticket use case."""

    return GetTicket(
        repository=SqlAlchemyTicketRepository(session),
    )


def get_list_tickets(
    session: PostgresqlSessionDependency,
) -> ListTickets:
    """Construct the list-tickets use case."""

    return ListTickets(
        workspace_repository=SqlAlchemyWorkspaceRepository(session),
        ticket_repository=SqlAlchemyTicketRepository(session),
    )


CreateTicketDependency = Annotated[
    CreateTicket,
    Depends(get_create_ticket),
]

GetTicketDependency = Annotated[
    GetTicket,
    Depends(get_get_ticket),
]

ListTicketsDependency = Annotated[
    ListTickets,
    Depends(get_list_tickets),
]
