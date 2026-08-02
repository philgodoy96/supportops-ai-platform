"""PostgreSQL queries for persisted controlled tool-call audits."""

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from supportops.agent_tools.application.queries import (
    AgentToolCallLookup,
    AgentToolCallQueryRepository,
)
from supportops.agent_tools.domain.audit import AgentToolCall
from supportops.agent_tools.infrastructure.models import (
    AgentToolCallRecord,
)


class SqlAlchemyAgentToolCallQueryRepository(AgentToolCallQueryRepository):
    """Read terminal tool-call audits through an active transaction."""

    def __init__(
        self,
        session: AsyncSession,
    ) -> None:
        self._session = session

    async def get_by_attempt_sequence(
        self,
        query: AgentToolCallLookup,
    ) -> AgentToolCall | None:
        """Return one workspace-scoped audit by attempt sequence."""

        statement = select(AgentToolCallRecord).where(
            AgentToolCallRecord.workspace_id == query.workspace_id,
            AgentToolCallRecord.ticket_id == query.ticket_id,
            AgentToolCallRecord.agent_run_id == query.agent_run_id,
            AgentToolCallRecord.agent_run_attempt_id == query.agent_run_attempt_id,
            AgentToolCallRecord.sequence == query.sequence,
        )
        result = await self._session.execute(statement)
        record = result.scalar_one_or_none()

        if record is None:
            return None

        return record.to_domain()
