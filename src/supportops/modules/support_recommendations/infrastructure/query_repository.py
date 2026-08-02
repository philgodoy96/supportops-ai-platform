"""PostgreSQL queries for persisted support recommendations."""

from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from supportops.modules.support_recommendations.application.queries import (
    SupportRecommendationQueryRepository,
)
from supportops.modules.support_recommendations.domain.models import (
    SupportRecommendation,
)
from supportops.modules.support_recommendations.infrastructure.models import (
    SupportRecommendationRecord,
)


class SqlAlchemySupportRecommendationQueryRepository(SupportRecommendationQueryRepository):
    """Read recommendations through an application-owned transaction."""

    def __init__(
        self,
        session: AsyncSession,
    ) -> None:
        self._session = session

    async def get_by_agent_run_id(
        self,
        *,
        workspace_id: UUID,
        agent_run_id: UUID,
    ) -> SupportRecommendation | None:
        """Return the immutable recommendation for one AgentRun."""

        statement = select(SupportRecommendationRecord).where(
            SupportRecommendationRecord.workspace_id == workspace_id,
            SupportRecommendationRecord.agent_run_id == agent_run_id,
        )
        result = await self._session.execute(statement)
        record = result.scalar_one_or_none()

        if record is None:
            return None

        return record.to_domain()
