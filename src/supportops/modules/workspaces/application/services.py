"""Workspace application use cases."""

from uuid import UUID

from supportops.core.transactions import TransactionManager
from supportops.modules.workspaces.application.errors import (
    WorkspaceNotFoundError,
    WorkspaceSlugConflictApplicationError,
)
from supportops.modules.workspaces.domain.models import Workspace
from supportops.modules.workspaces.domain.repositories import (
    WorkspaceRepository,
    WorkspaceSlugConflictError,
)


class CreateWorkspace:
    """Create a workspace atomically."""

    def __init__(
        self,
        *,
        repository: WorkspaceRepository,
        transaction_manager: TransactionManager,
    ) -> None:
        self._repository = repository
        self._transaction_manager = transaction_manager

    async def execute(
        self,
        *,
        name: str,
        slug: str,
    ) -> Workspace:
        """Create and persist a workspace."""

        workspace = Workspace.create(
            name=name,
            slug=slug,
        )

        try:
            async with self._transaction_manager.transaction():
                await self._repository.add(workspace)
        except WorkspaceSlugConflictError as error:
            raise WorkspaceSlugConflictApplicationError(
                "Workspace slug is already in use.",
            ) from error

        return workspace


class GetWorkspace:
    """Retrieve one workspace by identifier."""

    def __init__(
        self,
        *,
        repository: WorkspaceRepository,
    ) -> None:
        self._repository = repository

    async def execute(
        self,
        *,
        workspace_id: UUID,
    ) -> Workspace:
        """Return the workspace or raise a stable not-found error."""

        workspace = await self._repository.get(workspace_id)

        if workspace is None:
            raise WorkspaceNotFoundError(
                "Workspace was not found.",
            )

        return workspace
