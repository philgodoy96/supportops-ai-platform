"""FastAPI dependencies for AgentRun inspection use cases."""

from typing import Annotated

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from supportops.api.dependencies import get_postgresql_session
from supportops.application.agent_run_inspection import (
    GetAgentRunInspection,
)
from supportops.modules.agent_runs.application.services import (
    ListAgentRunAttempts,
)
from supportops.modules.agent_runs.infrastructure.repository import (
    SqlAlchemyAgentRunRepository,
)
from supportops.modules.ticket_classifications.application.services import (
    ListAgentRunLLMInvocations,
)
from supportops.modules.ticket_classifications.infrastructure.repository import (
    SqlAlchemyTicketClassificationQueryRepository,
)

PostgresqlSessionDependency = Annotated[
    AsyncSession,
    Depends(get_postgresql_session),
]


def get_get_agent_run_inspection(
    session: PostgresqlSessionDependency,
) -> GetAgentRunInspection:
    """Construct the cross-module AgentRun inspection use case."""

    return GetAgentRunInspection(
        agent_run_repository=SqlAlchemyAgentRunRepository(
            session,
        ),
        classification_repository=(
            SqlAlchemyTicketClassificationQueryRepository(
                session,
            )
        ),
    )


def get_list_agent_run_attempts(
    session: PostgresqlSessionDependency,
) -> ListAgentRunAttempts:
    """Construct the list-AgentRun-attempts use case."""

    return ListAgentRunAttempts(
        repository=SqlAlchemyAgentRunRepository(session),
    )


def get_list_agent_run_llm_invocations(
    session: PostgresqlSessionDependency,
) -> ListAgentRunLLMInvocations:
    """Construct the scoped AgentRun invocation-history use case."""

    return ListAgentRunLLMInvocations(
        agent_run_repository=SqlAlchemyAgentRunRepository(
            session,
        ),
        classification_repository=(
            SqlAlchemyTicketClassificationQueryRepository(
                session,
            )
        ),
    )


GetAgentRunInspectionDependency = Annotated[
    GetAgentRunInspection,
    Depends(get_get_agent_run_inspection),
]

ListAgentRunAttemptsDependency = Annotated[
    ListAgentRunAttempts,
    Depends(get_list_agent_run_attempts),
]

ListAgentRunLLMInvocationsDependency = Annotated[
    ListAgentRunLLMInvocations,
    Depends(get_list_agent_run_llm_invocations),
]
