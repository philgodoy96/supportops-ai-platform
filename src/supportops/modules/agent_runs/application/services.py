"""Workspace-scoped AgentRun inspection application use cases."""

from collections.abc import Sequence
from uuid import UUID

from supportops.modules.agent_runs.application.errors import (
    AgentRunNotFoundError,
)
from supportops.modules.agent_runs.domain.models import (
    AgentRun,
    AgentRunAttempt,
)
from supportops.modules.agent_runs.domain.repositories import (
    AgentRunQueryRepository,
)


class GetAgentRun:
    """Retrieve an AgentRun through its workspace boundary."""

    def __init__(
        self,
        *,
        repository: AgentRunQueryRepository,
    ) -> None:
        self._repository = repository

    async def execute(
        self,
        *,
        workspace_id: UUID,
        agent_run_id: UUID,
    ) -> AgentRun:
        """Return the scoped AgentRun or raise a stable not-found error."""

        agent_run = await self._repository.get(
            workspace_id=workspace_id,
            agent_run_id=agent_run_id,
        )
        if agent_run is None:
            raise AgentRunNotFoundError(
                "AgentRun was not found.",
            )

        return agent_run


class ListAgentRunAttempts:
    """List the deterministic attempt history for one scoped AgentRun."""

    def __init__(
        self,
        *,
        repository: AgentRunQueryRepository,
    ) -> None:
        self._repository = repository

    async def execute(
        self,
        *,
        workspace_id: UUID,
        agent_run_id: UUID,
    ) -> Sequence[AgentRunAttempt]:
        """Return attempts after validating AgentRun workspace ownership."""

        agent_run = await self._repository.get(
            workspace_id=workspace_id,
            agent_run_id=agent_run_id,
        )
        if agent_run is None:
            raise AgentRunNotFoundError(
                "AgentRun was not found.",
            )

        return await self._repository.list_attempts(
            agent_run_id=agent_run.id,
        )
