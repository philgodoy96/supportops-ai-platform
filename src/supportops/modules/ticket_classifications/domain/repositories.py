"""Ticket-classification persistence contracts."""

from collections.abc import Sequence
from typing import Protocol
from uuid import UUID

from supportops.modules.ticket_classifications.domain.models import (
    LLMInvocation,
    TicketClassification,
)


class TicketClassificationRepository(Protocol):
    """Persistence operations for accepted ticket classifications."""

    async def add(
        self,
        classification: TicketClassification,
    ) -> None:
        """Add an accepted classification to the active transaction."""

        ...

    async def get_by_agent_run_id(
        self,
        *,
        workspace_id: UUID,
        agent_run_id: UUID,
    ) -> TicketClassification | None:
        """Return one workspace-scoped classification for an AgentRun."""

        ...


class LLMInvocationRepository(Protocol):
    """Persistence operations for logical LLM invocation history."""

    async def add_many(
        self,
        invocations: Sequence[LLMInvocation],
    ) -> None:
        """Add logical invocations to the active transaction."""

        ...
