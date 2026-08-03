"""Persistence contracts for immutable ticket escalations."""

from dataclasses import dataclass
from datetime import datetime, timedelta
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


@dataclass(frozen=True, slots=True)
class TicketEscalationPageCursor:
    """Stable keyset cursor for escalation listing."""

    created_at: datetime
    ticket_escalation_id: UUID

    def __post_init__(self) -> None:
        if self.created_at.tzinfo is None or self.created_at.utcoffset() != timedelta(
            0,
        ):
            raise ValueError("created_at must be a UTC-aware timestamp.")
        if not isinstance(self.ticket_escalation_id, UUID):
            raise TypeError("ticket_escalation_id must be a UUID.")


@dataclass(frozen=True, slots=True)
class TicketEscalationListQuery:
    """Workspace-scoped escalation list criteria."""

    workspace_id: UUID
    ticket_id: UUID | None = None
    cursor: TicketEscalationPageCursor | None = None
    page_size: int = 20

    def __post_init__(self) -> None:
        if not isinstance(self.workspace_id, UUID):
            raise TypeError("workspace_id must be a UUID.")
        if self.ticket_id is not None and not isinstance(
            self.ticket_id,
            UUID,
        ):
            raise TypeError("ticket_id must be a UUID.")
        if self.page_size < 1 or self.page_size > 100:
            raise ValueError("page_size must be between 1 and 100.")


@dataclass(frozen=True, slots=True)
class TicketEscalationListPage:
    """One ordered page of ticket escalations."""

    items: tuple[TicketEscalation, ...]
    next_cursor: TicketEscalationPageCursor | None


class TicketEscalationRepository(Protocol):
    """Application-owned persistence boundary for escalations."""

    async def persist(
        self,
        escalation: TicketEscalation,
    ) -> TicketEscalationPersistenceResult:
        """Persist or reuse one immutable escalation."""

        ...

    async def list_page(
        self,
        query: TicketEscalationListQuery,
    ) -> TicketEscalationListPage:
        """Return one workspace-scoped escalation page."""

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
