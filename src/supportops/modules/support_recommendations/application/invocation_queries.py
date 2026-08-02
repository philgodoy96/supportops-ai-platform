"""Read-only queries for durable workflow LLM invocations."""

from dataclasses import dataclass
from typing import Protocol
from uuid import UUID

from supportops.modules.ticket_classifications.domain.models import (
    LLMInvocation,
)


@dataclass(frozen=True, slots=True)
class AttemptLLMInvocationQuery:
    """Identify logical invocations for one exact AgentRun attempt."""

    workspace_id: UUID
    ticket_id: UUID
    agent_run_id: UUID
    agent_run_attempt_id: UUID

    def __post_init__(self) -> None:
        identifiers = (
            self.workspace_id,
            self.ticket_id,
            self.agent_run_id,
            self.agent_run_attempt_id,
        )

        if not all(isinstance(identifier, UUID) for identifier in identifiers):
            raise TypeError("Invocation query identifiers must be UUID values.")


class AttemptLLMInvocationQueryRepository(Protocol):
    """Read exact attempt-scoped logical invocation history."""

    async def list_by_attempt(
        self,
        query: AttemptLLMInvocationQuery,
    ) -> tuple[LLMInvocation, ...]:
        """Return invocations ordered by logical sequence."""

        ...
