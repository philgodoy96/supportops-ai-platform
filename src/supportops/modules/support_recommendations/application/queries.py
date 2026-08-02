"""Read-only queries for persisted support recommendations."""

from typing import Protocol
from uuid import UUID

from supportops.modules.support_recommendations.domain.models import (
    SupportRecommendation,
)


class SupportRecommendationQueryRepository(Protocol):
    """Read immutable support recommendations by durable ownership."""

    async def get_by_agent_run_id(
        self,
        *,
        workspace_id: UUID,
        agent_run_id: UUID,
    ) -> SupportRecommendation | None:
        """Return one workspace-scoped AgentRun recommendation."""

        ...
