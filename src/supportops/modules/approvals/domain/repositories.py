"""Application-owned persistence contracts for approval requests."""

from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import StrEnum
from typing import Protocol
from uuid import UUID

from supportops.modules.approvals.domain.models import (
    ApprovalRequest,
    ApprovalRequestStatus,
)


class ApprovalRequestPersistenceResult(StrEnum):
    """Outcome of idempotent pending approval persistence."""

    APPLIED = "applied"
    ALREADY_RECORDED = "already_recorded"


class ApprovalRequestConsistencyError(RuntimeError):
    """Raised when a replay conflicts with an existing approval request."""


@dataclass(frozen=True, slots=True)
class ApprovalRequestPageCursor:
    """Stable keyset cursor for approval request listing."""

    created_at: datetime
    approval_request_id: UUID

    def __post_init__(self) -> None:
        if self.created_at.tzinfo is None or self.created_at.utcoffset() != timedelta(
            0,
        ):
            raise ValueError("created_at must be a UTC-aware timestamp.")
        if not isinstance(self.approval_request_id, UUID):
            raise TypeError("approval_request_id must be a UUID.")


@dataclass(frozen=True, slots=True)
class ApprovalRequestListQuery:
    """Workspace-scoped approval list criteria."""

    workspace_id: UUID
    status: ApprovalRequestStatus | None = None
    cursor: ApprovalRequestPageCursor | None = None
    page_size: int = 20

    def __post_init__(self) -> None:
        if not isinstance(self.workspace_id, UUID):
            raise TypeError("workspace_id must be a UUID.")
        if self.page_size < 1 or self.page_size > 100:
            raise ValueError("page_size must be between 1 and 100.")


@dataclass(frozen=True, slots=True)
class ApprovalRequestListPage:
    """One ordered page of approval requests."""

    items: tuple[ApprovalRequest, ...]
    next_cursor: ApprovalRequestPageCursor | None


class ApprovalRequestRepository(Protocol):
    """Persistence operations for durable approval requests."""

    async def persist_pending(
        self,
        approval_request: ApprovalRequest,
    ) -> ApprovalRequestPersistenceResult:
        """Persist one pending approval request idempotently."""

        ...

    async def list_page(
        self,
        query: ApprovalRequestListQuery,
    ) -> ApprovalRequestListPage:
        """Return one workspace-scoped approval page."""

        ...

    async def get_by_id(
        self,
        *,
        workspace_id: UUID,
        approval_request_id: UUID,
    ) -> ApprovalRequest | None:
        """Return one workspace-scoped approval request by ID."""

        ...

    async def get_by_agent_tool_call_id(
        self,
        *,
        workspace_id: UUID,
        agent_tool_call_id: UUID,
    ) -> ApprovalRequest | None:
        """Return one workspace-scoped approval for a tool call."""

        ...

    async def list_by_agent_run(
        self,
        *,
        workspace_id: UUID,
        agent_run_id: UUID,
    ) -> tuple[ApprovalRequest, ...]:
        """Return workspace-scoped approvals for one AgentRun."""

        ...

    async def get_by_id_for_update(
        self,
        *,
        workspace_id: UUID,
        approval_request_id: UUID,
    ) -> ApprovalRequest | None:
        """Lock and return one workspace-scoped approval request by ID."""

        ...

    async def get_next_expired_pending_for_update(
        self,
        *,
        now: datetime,
    ) -> ApprovalRequest | None:
        """Lock the next overdue pending approval, if available."""

        ...

    async def save(
        self,
        approval_request: ApprovalRequest,
    ) -> None:
        """Persist one terminal approval decision without committing."""

        ...
