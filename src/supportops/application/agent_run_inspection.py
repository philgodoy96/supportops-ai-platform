"""Cross-module AgentRun inspection composition."""

from dataclasses import dataclass
from uuid import UUID

from supportops.modules.agent_runs.application.errors import (
    AgentRunNotFoundError,
)
from supportops.modules.agent_runs.domain.models import AgentRun
from supportops.modules.agent_runs.domain.repositories import (
    AgentRunQueryRepository,
)
from supportops.modules.ticket_classifications.domain.inspection import (
    AgentRunClassificationReference,
)
from supportops.modules.ticket_classifications.domain.repositories import (
    TicketClassificationQueryRepository,
)


@dataclass(frozen=True, slots=True)
class AgentRunInspection:
    """AgentRun state enriched with optional accepted classification state."""

    agent_run: AgentRun
    classification: AgentRunClassificationReference | None


class GetAgentRunInspection:
    """Retrieve one AgentRun and its optional classification reference."""

    def __init__(
        self,
        *,
        agent_run_repository: AgentRunQueryRepository,
        classification_repository: (TicketClassificationQueryRepository),
    ) -> None:
        self._agent_run_repository = agent_run_repository
        self._classification_repository = classification_repository

    async def execute(
        self,
        *,
        workspace_id: UUID,
        agent_run_id: UUID,
    ) -> AgentRunInspection:
        """Return scoped run state without coupling AgentRun to AI data."""

        agent_run = await self._agent_run_repository.get(
            workspace_id=workspace_id,
            agent_run_id=agent_run_id,
        )
        if agent_run is None:
            raise AgentRunNotFoundError(
                "AgentRun was not found.",
            )

        classification = await self._classification_repository.get_reference_by_agent_run_id(
            workspace_id=workspace_id,
            agent_run_id=agent_run.id,
        )

        return AgentRunInspection(
            agent_run=agent_run,
            classification=classification,
        )
