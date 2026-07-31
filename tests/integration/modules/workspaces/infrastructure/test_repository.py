"""Integration tests for the PostgreSQL workspace repository."""

from datetime import UTC, datetime
from uuid import UUID

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from supportops.infrastructure.postgresql.transaction import (
    SqlAlchemyTransactionManager,
)
from supportops.modules.workspaces.domain.models import Workspace
from supportops.modules.workspaces.domain.repositories import (
    WorkspaceSlugConflictError,
)
from supportops.modules.workspaces.infrastructure.repository import (
    SqlAlchemyWorkspaceRepository,
)

pytestmark = pytest.mark.integration

DEFAULT_WORKSPACE_ID = UUID(
    "50617047-0dbd-4704-b44f-c22da75b0595",
)


def create_workspace(
    *,
    workspace_id: UUID = DEFAULT_WORKSPACE_ID,
    name: str = "Platform Support",
    slug: str = "platform-support",
) -> Workspace:
    """Create a deterministic workspace for repository tests."""

    timestamp = datetime(
        2026,
        7,
        31,
        12,
        0,
        tzinfo=UTC,
    )

    return Workspace(
        id=workspace_id,
        name=name,
        slug=slug,
        created_at=timestamp,
        updated_at=timestamp,
    )


async def test_repository_adds_and_retrieves_workspace(
    postgresql_session: AsyncSession,
    clean_business_tables: None,
) -> None:
    repository = SqlAlchemyWorkspaceRepository(
        postgresql_session,
    )
    transaction_manager = SqlAlchemyTransactionManager(
        postgresql_session,
    )
    workspace = create_workspace()

    async with transaction_manager.transaction():
        await repository.add(workspace)

    persisted_workspace = await repository.get(
        workspace.id,
    )

    assert persisted_workspace == workspace
    assert await repository.exists(workspace.id)
    assert not await repository.exists(
        UUID("ee9f2d68-38c5-4b4a-af4b-f6970f7f29fb"),
    )


async def test_repository_returns_none_for_missing_workspace(
    postgresql_session: AsyncSession,
    clean_business_tables: None,
) -> None:
    repository = SqlAlchemyWorkspaceRepository(
        postgresql_session,
    )

    workspace = await repository.get(
        UUID("ee9f2d68-38c5-4b4a-af4b-f6970f7f29fb"),
    )

    assert workspace is None


async def test_repository_translates_duplicate_slug_constraint(
    postgresql_session: AsyncSession,
    clean_business_tables: None,
) -> None:
    repository = SqlAlchemyWorkspaceRepository(
        postgresql_session,
    )
    transaction_manager = SqlAlchemyTransactionManager(
        postgresql_session,
    )
    first_workspace = create_workspace()
    duplicate_workspace = create_workspace(
        workspace_id=UUID(
            "f3c9e8cd-ce6a-4436-89d9-ad6bf447dc23",
        ),
        name="Escalation Support",
    )

    async with transaction_manager.transaction():
        await repository.add(first_workspace)

    with pytest.raises(
        WorkspaceSlugConflictError,
        match=r"Workspace slug already exists\.",
    ):
        async with transaction_manager.transaction():
            await repository.add(duplicate_workspace)

    assert (
        await repository.get(
            duplicate_workspace.id,
        )
        is None
    )
    assert (
        await repository.get(
            first_workspace.id,
        )
        == first_workspace
    )


async def test_transaction_rolls_back_workspace_after_failure(
    postgresql_session: AsyncSession,
    clean_business_tables: None,
) -> None:
    repository = SqlAlchemyWorkspaceRepository(
        postgresql_session,
    )
    transaction_manager = SqlAlchemyTransactionManager(
        postgresql_session,
    )
    workspace = create_workspace()

    with pytest.raises(
        RuntimeError,
        match="controlled transaction failure",
    ):
        async with transaction_manager.transaction():
            await repository.add(workspace)
            raise RuntimeError(
                "controlled transaction failure",
            )

    assert await repository.get(workspace.id) is None
