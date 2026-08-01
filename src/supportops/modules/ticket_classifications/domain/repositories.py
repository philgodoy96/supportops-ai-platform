"""Ticket-classification persistence contracts."""

from collections.abc import Sequence
from datetime import datetime
from typing import Protocol
from uuid import UUID

from supportops.modules.ticket_classifications.domain.inspection import (
    AgentRunClassificationReference,
    LLMInvocationInspection,
)
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


class TicketClassificationQueryRepository(Protocol):
    """Read-only persistence contract for classification inspection."""

    async def get(
        self,
        *,
        workspace_id: UUID,
        classification_id: UUID,
    ) -> TicketClassification | None:
        """Return one classification only through its workspace boundary."""

        ...

    async def get_by_agent_run_id(
        self,
        *,
        workspace_id: UUID,
        agent_run_id: UUID,
    ) -> TicketClassification | None:
        """Return one accepted classification for a scoped AgentRun."""

        ...

    async def get_reference_by_agent_run_id(
        self,
        *,
        workspace_id: UUID,
        agent_run_id: UUID,
    ) -> AgentRunClassificationReference | None:
        """Return a lightweight classification reference for one AgentRun."""

        ...

    async def list_by_ticket(
        self,
        *,
        workspace_id: UUID,
        ticket_id: UUID,
        limit: int,
        after_created_at: datetime | None = None,
        after_classification_id: UUID | None = None,
    ) -> Sequence[TicketClassification]:
        """List ticket classifications in deterministic descending order."""

        ...

    async def list_invocations_by_agent_run(
        self,
        *,
        workspace_id: UUID,
        agent_run_id: UUID,
    ) -> Sequence[LLMInvocationInspection]:
        """List safe invocation projections in attempt and sequence order."""

        ...
