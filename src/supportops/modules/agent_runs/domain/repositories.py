"""AgentRun repository contracts."""

from typing import Protocol

from supportops.modules.agent_runs.domain.claiming import (
    AgentRunClaim,
    ClaimAgentRunCommand,
)
from supportops.modules.agent_runs.domain.models import AgentRun


class AgentRunRepository(Protocol):
    """Persistence operations for durable AgentRuns."""

    async def add(self, agent_run: AgentRun) -> None:
        """Add an AgentRun to the active transaction."""

        ...

    async def claim_next_available(
        self,
        command: ClaimAgentRunCommand,
    ) -> AgentRunClaim | None:
        """Atomically claim the next eligible AgentRun, if available."""

        ...
