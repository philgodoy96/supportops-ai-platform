"""Unit tests for workspace application services."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from uuid import UUID

import pytest

from supportops.modules.workspaces.application.errors import (
    WorkspaceNotFoundError,
    WorkspaceSlugConflictApplicationError,
)
from supportops.modules.workspaces.application.services import (
    CreateWorkspace,
    GetWorkspace,
)
from supportops.modules.workspaces.domain.models import Workspace
from supportops.modules.workspaces.domain.repositories import (
    WorkspaceSlugConflictError,
)


class FakeTransactionManager:
    """Record transaction entry and completion."""

    def __init__(self) -> None:
        self.entered = False
        self.completed = False
        self.rolled_back = False

    @asynccontextmanager
    async def transaction(self) -> AsyncIterator[None]:
        self.entered = True

        try:
            yield
        except Exception:
            self.rolled_back = True
            raise
        else:
            self.completed = True


class FakeWorkspaceRepository:
    """In-memory workspace repository fake."""

    def __init__(self) -> None:
        self.workspaces: dict[UUID, Workspace] = {}
        self.added_workspace: Workspace | None = None
        self.slug_conflict = False

    async def add(self, workspace: Workspace) -> None:
        if self.slug_conflict:
            raise WorkspaceSlugConflictError(
                "duplicate workspace slug",
            )

        self.added_workspace = workspace
        self.workspaces[workspace.id] = workspace

    async def get(
        self,
        workspace_id: UUID,
    ) -> Workspace | None:
        return self.workspaces.get(workspace_id)

    async def exists(self, workspace_id: UUID) -> bool:
        return workspace_id in self.workspaces


def create_workspace() -> Workspace:
    """Create a deterministic workspace."""

    timestamp = datetime(2026, 7, 31, 12, 0, tzinfo=UTC)

    return Workspace(
        id=UUID("50617047-0dbd-4704-b44f-c22da75b0595"),
        name="Platform Support",
        slug="platform-support",
        created_at=timestamp,
        updated_at=timestamp,
    )


async def test_create_workspace_persists_inside_transaction() -> None:
    repository = FakeWorkspaceRepository()
    transaction_manager = FakeTransactionManager()
    service = CreateWorkspace(
        repository=repository,
        transaction_manager=transaction_manager,
    )

    workspace = await service.execute(
        name="  Platform Support  ",
        slug="platform-support",
    )

    assert transaction_manager.entered
    assert transaction_manager.completed
    assert not transaction_manager.rolled_back
    assert repository.added_workspace == workspace
    assert workspace.name == "Platform Support"


async def test_create_workspace_translates_slug_conflict() -> None:
    repository = FakeWorkspaceRepository()
    repository.slug_conflict = True
    transaction_manager = FakeTransactionManager()
    service = CreateWorkspace(
        repository=repository,
        transaction_manager=transaction_manager,
    )

    with pytest.raises(
        WorkspaceSlugConflictApplicationError,
        match=r"Workspace slug is already in use\.",
    ):
        await service.execute(
            name="Platform Support",
            slug="platform-support",
        )

    assert transaction_manager.entered
    assert transaction_manager.rolled_back
    assert not transaction_manager.completed


async def test_get_workspace_returns_existing_workspace() -> None:
    workspace = create_workspace()
    repository = FakeWorkspaceRepository()
    repository.workspaces[workspace.id] = workspace
    service = GetWorkspace(repository=repository)

    result = await service.execute(
        workspace_id=workspace.id,
    )

    assert result == workspace


async def test_get_workspace_raises_not_found() -> None:
    service = GetWorkspace(
        repository=FakeWorkspaceRepository(),
    )

    with pytest.raises(
        WorkspaceNotFoundError,
        match=r"Workspace was not found\.",
    ):
        await service.execute(
            workspace_id=UUID(
                "ee9f2d68-38c5-4b4a-af4b-f6970f7f29fb",
            ),
        )
