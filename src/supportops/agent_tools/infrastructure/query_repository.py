"""PostgreSQL queries for persisted controlled tool-call lifecycle records."""

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from supportops.agent_tools.application.queries import (
    AgentToolCallLookup,
    AgentToolCallQueryRepository,
    SensitiveAgentToolCallLookup,
)
from supportops.agent_tools.domain.audit import AgentToolCall
from supportops.agent_tools.domain.contracts import ToolSafetyLevel
from supportops.agent_tools.infrastructure.models import (
    AgentToolCallRecord,
)


class SqlAlchemyAgentToolCallQueryRepository(AgentToolCallQueryRepository):
    """Read tool-call lifecycle records through an active transaction."""

    def __init__(
        self,
        session: AsyncSession,
    ) -> None:
        self._session = session

    async def get_by_proposal_attempt_sequence(
        self,
        query: AgentToolCallLookup,
    ) -> AgentToolCall | None:
        """Return one workspace-scoped audit by proposal-attempt sequence."""

        statement = select(AgentToolCallRecord).where(
            AgentToolCallRecord.workspace_id == query.workspace_id,
            AgentToolCallRecord.ticket_id == query.ticket_id,
            AgentToolCallRecord.agent_run_id == query.agent_run_id,
            AgentToolCallRecord.proposed_by_agent_run_attempt_id
            == query.proposed_by_agent_run_attempt_id,
            AgentToolCallRecord.sequence == query.sequence,
        )
        result = await self._session.execute(statement)
        record = result.scalar_one_or_none()

        if record is None:
            return None

        return record.to_domain()

    async def get_sensitive_by_identity(
        self,
        query: SensitiveAgentToolCallLookup,
    ) -> AgentToolCall | None:
        """Return one sensitive proposal scoped to workspace ownership."""

        statement = select(AgentToolCallRecord).where(
            AgentToolCallRecord.workspace_id == query.workspace_id,
            AgentToolCallRecord.ticket_id == query.ticket_id,
            AgentToolCallRecord.agent_run_id == query.agent_run_id,
            AgentToolCallRecord.tool_name == query.tool_name,
            AgentToolCallRecord.tool_version == query.tool_version,
            AgentToolCallRecord.input_fingerprint == (query.input_fingerprint),
            AgentToolCallRecord.safety_level == (ToolSafetyLevel.SENSITIVE_WRITE.value),
        )
        result = await self._session.execute(statement)
        record = result.scalar_one_or_none()

        if record is None:
            return None

        return record.to_domain()
