"""PostgreSQL repository for durable AgentRuns."""

from sqlalchemy.ext.asyncio import AsyncSession

from supportops.modules.agent_runs.domain.models import AgentRun
from supportops.modules.agent_runs.domain.repositories import (
    AgentRunRepository,
)
from supportops.modules.agent_runs.infrastructure.models import (
    AgentRunRecord,
)


class SqlAlchemyAgentRunRepository(AgentRunRepository):
    """Persist AgentRuns through an active SQLAlchemy transaction."""

    def __init__(
        self,
        session: AsyncSession,
    ) -> None:
        self._session = session

    async def add(
        self,
        agent_run: AgentRun,
    ) -> None:
        """Add and flush an AgentRun inside the active transaction."""

        self._session.add(
            AgentRunRecord.from_domain(agent_run),
        )
        await self._session.flush()
