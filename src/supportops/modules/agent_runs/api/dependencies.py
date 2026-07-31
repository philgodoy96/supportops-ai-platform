"""FastAPI dependencies for AgentRun inspection use cases."""

from typing import Annotated

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from supportops.api.dependencies import get_postgresql_session
from supportops.modules.agent_runs.application.services import (
    GetAgentRun,
    ListAgentRunAttempts,
)
from supportops.modules.agent_runs.infrastructure.repository import (
    SqlAlchemyAgentRunRepository,
)

PostgresqlSessionDependency = Annotated[
    AsyncSession,
    Depends(get_postgresql_session),
]


def get_get_agent_run(
    session: PostgresqlSessionDependency,
) -> GetAgentRun:
    """Construct the get-AgentRun use case."""

    return GetAgentRun(
        repository=SqlAlchemyAgentRunRepository(session),
    )


def get_list_agent_run_attempts(
    session: PostgresqlSessionDependency,
) -> ListAgentRunAttempts:
    """Construct the list-AgentRun-attempts use case."""

    return ListAgentRunAttempts(
        repository=SqlAlchemyAgentRunRepository(session),
    )


GetAgentRunDependency = Annotated[
    GetAgentRun,
    Depends(get_get_agent_run),
]

ListAgentRunAttemptsDependency = Annotated[
    ListAgentRunAttempts,
    Depends(get_list_agent_run_attempts),
]
