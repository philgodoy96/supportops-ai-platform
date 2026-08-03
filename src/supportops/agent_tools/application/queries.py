"""Read-only query contracts for persisted controlled tool calls."""

from dataclasses import dataclass
from typing import Protocol
from uuid import UUID

from supportops.agent_tools.domain.audit import AgentToolCall


@dataclass(frozen=True, slots=True)
class AgentToolCallLookup:
    """Identify one audit by proposal-attempt ownership and sequence."""

    workspace_id: UUID
    ticket_id: UUID
    agent_run_id: UUID
    proposed_by_agent_run_attempt_id: UUID
    sequence: int

    def __post_init__(self) -> None:
        identifiers = (
            self.workspace_id,
            self.ticket_id,
            self.agent_run_id,
            self.proposed_by_agent_run_attempt_id,
        )

        if not all(isinstance(identifier, UUID) for identifier in identifiers):
            raise TypeError("Tool-call lookup identifiers must be UUID values.")

        if type(self.sequence) is not int:
            raise TypeError("sequence must be an integer.")

        if self.sequence <= 0:
            raise ValueError("sequence must be positive.")


@dataclass(frozen=True, slots=True)
class SensitiveAgentToolCallLookup:
    """Identify one sensitive proposal by application-owned identity."""

    workspace_id: UUID
    ticket_id: UUID
    agent_run_id: UUID
    tool_name: str
    tool_version: int
    input_fingerprint: str

    def __post_init__(self) -> None:
        identifiers = (
            self.workspace_id,
            self.ticket_id,
            self.agent_run_id,
        )

        if not all(isinstance(identifier, UUID) for identifier in identifiers):
            raise TypeError("Sensitive tool-call lookup identifiers must be UUID values.")

        if not self.tool_name or self.tool_name != self.tool_name.strip():
            raise ValueError("tool_name is required.")

        if type(self.tool_version) is not int:
            raise TypeError("tool_version must be an integer.")

        if self.tool_version <= 0:
            raise ValueError("tool_version must be positive.")

        if len(self.input_fingerprint) != 64:
            raise ValueError("input_fingerprint must be a lowercase SHA-256 hash.")

        if any(character not in "0123456789abcdef" for character in self.input_fingerprint):
            raise ValueError("input_fingerprint must be a lowercase SHA-256 hash.")


class AgentToolCallQueryRepository(Protocol):
    """Read persisted tool-call lifecycle records."""

    async def get_by_proposal_attempt_sequence(
        self,
        query: AgentToolCallLookup,
    ) -> AgentToolCall | None:
        """Return one exact audit without exposing cross-tenant state."""

        ...

    async def get_sensitive_by_identity(
        self,
        query: SensitiveAgentToolCallLookup,
    ) -> AgentToolCall | None:
        """Return one sensitive proposal by scoped identity, if present."""

        ...
