"""Read-only query contracts for persisted controlled tool calls."""

from dataclasses import dataclass
from typing import Protocol
from uuid import UUID

from supportops.agent_tools.domain.audit import AgentToolCall


@dataclass(frozen=True, slots=True)
class AgentToolCallLookup:
    """Identify one terminal audit by exact ownership and sequence."""

    workspace_id: UUID
    ticket_id: UUID
    agent_run_id: UUID
    agent_run_attempt_id: UUID
    sequence: int

    def __post_init__(self) -> None:
        identifiers = (
            self.workspace_id,
            self.ticket_id,
            self.agent_run_id,
            self.agent_run_attempt_id,
        )

        if not all(isinstance(identifier, UUID) for identifier in identifiers):
            raise TypeError("Tool-call lookup identifiers must be UUID values.")

        if type(self.sequence) is not int:
            raise TypeError("sequence must be an integer.")

        if self.sequence <= 0:
            raise ValueError("sequence must be positive.")


class AgentToolCallQueryRepository(Protocol):
    """Read persisted terminal tool-call audits."""

    async def get_by_attempt_sequence(
        self,
        query: AgentToolCallLookup,
    ) -> AgentToolCall | None:
        """Return one exact audit without exposing cross-tenant state."""

        ...
