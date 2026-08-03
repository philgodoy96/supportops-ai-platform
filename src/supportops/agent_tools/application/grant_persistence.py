"""Application persistence contract for sensitive execution grants."""

from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol
from uuid import UUID

from supportops.agent_tools.domain.grants import (
    SensitiveExecutionGrant,
)


class SensitiveExecutionGrantPersistenceResult(StrEnum):
    """Outcome of idempotent grant persistence."""

    APPLIED = "applied"
    ALREADY_RECORDED = "already_recorded"


class SensitiveExecutionGrantConsistencyError(RuntimeError):
    """Raised when replay conflicts with existing authorization."""


class SensitiveExecutionGrantRepository(Protocol):
    """Application-owned persistence boundary for grants."""

    async def persist(
        self,
        grant: SensitiveExecutionGrant,
    ) -> SensitiveExecutionGrantPersistenceResult:
        """Persist or reuse one immutable authorization."""

        ...

    async def get_by_id(
        self,
        *,
        workspace_id: UUID,
        grant_id: UUID,
    ) -> SensitiveExecutionGrant | None:
        """Return one workspace-scoped grant."""

        ...

    async def get_by_approval_request_id(
        self,
        *,
        workspace_id: UUID,
        approval_request_id: UUID,
    ) -> SensitiveExecutionGrant | None:
        """Return the grant for one approval request."""

        ...

    async def get_by_agent_tool_call_id(
        self,
        *,
        workspace_id: UUID,
        agent_tool_call_id: UUID,
    ) -> SensitiveExecutionGrant | None:
        """Return the grant for one tool call."""

        ...


@dataclass(frozen=True, slots=True)
class PersistSensitiveExecutionGrantCommand:
    """Inputs required to create one execution authorization."""

    grant: SensitiveExecutionGrant
