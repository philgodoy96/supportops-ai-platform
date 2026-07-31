"""Workspace repository contracts."""

from typing import Protocol
from uuid import UUID

from supportops.modules.workspaces.domain.models import Workspace


class WorkspaceSlugConflictError(Exception):
    """Raised when a workspace slug violates its uniqueness invariant."""


class WorkspaceRepository(Protocol):
    """Persistence operations required by workspace use cases."""

    async def add(self, workspace: Workspace) -> None:
        """Add a workspace to the active transaction."""

        ...

    async def get(self, workspace_id: UUID) -> Workspace | None:
        """Return a workspace by identifier."""

        ...

    async def exists(self, workspace_id: UUID) -> bool:
        """Return whether a workspace exists."""

        ...
