"""Application-owned persistence contracts for approval requests."""

from datetime import datetime
from enum import StrEnum
from typing import Protocol
from uuid import UUID

from supportops.modules.approvals.domain.models import ApprovalRequest


class ApprovalRequestPersistenceResult(StrEnum):
    """Outcome of idempotent pending approval persistence."""

    APPLIED = "applied"
    ALREADY_RECORDED = "already_recorded"


class ApprovalRequestConsistencyError(RuntimeError):
    """Raised when a replay conflicts with an existing approval request."""


class ApprovalRequestRepository(Protocol):
    """Persistence operations for durable approval requests."""

    async def persist_pending(
        self,
        approval_request: ApprovalRequest,
    ) -> ApprovalRequestPersistenceResult:
        """Persist one pending approval request idempotently."""

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
