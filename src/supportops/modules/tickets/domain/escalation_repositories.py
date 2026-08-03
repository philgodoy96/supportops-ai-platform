"""Persistence contracts for immutable ticket escalations."""

from enum import StrEnum
from typing import Protocol
from uuid import UUID

from supportops.modules.tickets.domain.escalation import (
    TicketEscalation,
)


class TicketEscalationPersistenceResult(StrEnum):
    """Outcome of idempotent escalation persistence."""

    APPLIED = "applied"
    ALREADY_RECORDED = "already_recorded"


class TicketEscalationConsistencyError(RuntimeError):
    """Raised when replay conflicts with an existing escalation."""


class TicketEscalationRepository(Protocol):
    """Application-owned persistence boundary for escalations."""

    async def persist(
        self,
        escalation: TicketEscalation,
    ) -> TicketEscalationPersistenceResult:
        """Persist or reuse one immutable escalation."""

        ...

    async def get_by_id(
        self,
        *,
        workspace_id: UUID,
        escalation_id: UUID,
    ) -> TicketEscalation | None:
        """Return one workspace-scoped escalation."""

        ...

    async def get_by_approval_request_id(
        self,
        *,
        workspace_id: UUID,
        approval_request_id: UUID,
    ) -> TicketEscalation | None:
        """Return the escalation for one approval."""

        ...

    async def get_by_agent_tool_call_id(
        self,
        *,
        workspace_id: UUID,
        agent_tool_call_id: UUID,
    ) -> TicketEscalation | None:
        """Return the escalation for one tool call."""

        ...
