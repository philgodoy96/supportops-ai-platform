"""AgentRun repository contracts."""

from typing import Protocol

from supportops.modules.agent_runs.domain.claiming import (
    AgentRunClaim,
    ClaimAgentRunCommand,
)
from supportops.modules.agent_runs.domain.models import AgentRun
from supportops.modules.agent_runs.domain.recovery import (
    RecoverExpiredAgentRunCommand,
    RecoverExpiredAgentRunResult,
)
from supportops.modules.agent_runs.domain.transitions import (
    AgentRunTransitionResult,
    CompleteAgentRunCommand,
    FailAgentRunCommand,
)


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

    async def mark_succeeded(
        self,
        command: CompleteAgentRunCommand,
    ) -> AgentRunTransitionResult:
        """Persist a successful fenced AgentRun transition."""

        ...

    async def record_failure(
        self,
        command: FailAgentRunCommand,
    ) -> AgentRunTransitionResult:
        """Persist a fenced AgentRun failure transition."""

        ...

    async def recover_next_expired(
        self,
        command: RecoverExpiredAgentRunCommand,
    ) -> RecoverExpiredAgentRunResult | None:
        """Atomically recover the next expired running AgentRun, if available."""

        ...
