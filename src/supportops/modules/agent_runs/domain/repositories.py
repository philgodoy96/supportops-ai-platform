"""AgentRun repository contracts."""

from typing import Protocol

from supportops.modules.agent_runs.domain.models import AgentRun


class AgentRunRepository(Protocol):
    """Persistence operations for durable AgentRuns."""

    async def add(self, agent_run: AgentRun) -> None:
        """Add an AgentRun to the active transaction."""

        ...
