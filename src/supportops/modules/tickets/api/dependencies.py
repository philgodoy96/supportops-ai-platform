"""FastAPI dependencies for support ticket use cases."""

from typing import Annotated

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from supportops.api.dependencies import (
    get_application_state,
    get_postgresql_session,
)
from supportops.api.state import ApplicationState
from supportops.application.ticket_intake import CreateTicketWithInitialRun
from supportops.infrastructure.postgresql.transaction import (
    SqlAlchemyTransactionManager,
)
from supportops.modules.agent_runs.infrastructure.repository import (
    SqlAlchemyAgentRunRepository,
)
from supportops.modules.tickets.application.services import (
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

ApplicationStateDependency = Annotated[
    ApplicationState,
    Depends(get_application_state),
]


def get_create_ticket(
    session: PostgresqlSessionDependency,
    state: ApplicationStateDependency,
) -> CreateTicketWithInitialRun:
    """Construct the create-ticket use case."""

    return CreateTicketWithInitialRun(
        workspace_repository=SqlAlchemyWorkspaceRepository(session),
        ticket_repository=SqlAlchemyTicketRepository(session),
        agent_run_repository=SqlAlchemyAgentRunRepository(session),
        transaction_manager=SqlAlchemyTransactionManager(session),
        workflow_version=(state.settings.ticket_processing_workflow_version),
        max_retryable_failures=(state.settings.worker_max_retryable_failures),
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
    CreateTicketWithInitialRun,
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
