"""PostgreSQL queries for workflow LLM invocation history."""

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from supportops.modules.support_recommendations.application.invocation_queries import (
    AttemptLLMInvocationQuery,
    AttemptLLMInvocationQueryRepository,
)
from supportops.modules.ticket_classifications.domain.models import (
    LLMInvocation,
)
from supportops.modules.ticket_classifications.infrastructure.models import (
    LLMInvocationRecord,
)


class SqlAlchemyAttemptLLMInvocationQueryRepository(AttemptLLMInvocationQueryRepository):
    """Read attempt-scoped invocation history through an active transaction."""

    def __init__(
        self,
        session: AsyncSession,
    ) -> None:
        self._session = session

    async def list_by_attempt(
        self,
        query: AttemptLLMInvocationQuery,
    ) -> tuple[LLMInvocation, ...]:
        """Return exact ownership-scoped invocations in sequence order."""

        statement = (
            select(LLMInvocationRecord)
            .where(
                LLMInvocationRecord.workspace_id == query.workspace_id,
                LLMInvocationRecord.ticket_id == query.ticket_id,
                LLMInvocationRecord.agent_run_id == query.agent_run_id,
                LLMInvocationRecord.agent_run_attempt_id == query.agent_run_attempt_id,
            )
            .order_by(
                LLMInvocationRecord.invocation_sequence.asc(),
                LLMInvocationRecord.id.asc(),
            )
        )
        result = await self._session.execute(statement)

        return tuple(record.to_domain() for record in result.scalars().all())
