"""FastAPI dependencies for workspace use cases."""

from typing import Annotated

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from supportops.api.dependencies import get_postgresql_session
from supportops.infrastructure.postgresql.transaction import (
    SqlAlchemyTransactionManager,
)
from supportops.modules.workspaces.application.services import (
    CreateWorkspace,
    GetWorkspace,
)
from supportops.modules.workspaces.infrastructure.repository import (
    SqlAlchemyWorkspaceRepository,
)

PostgresqlSessionDependency = Annotated[
    AsyncSession,
    Depends(get_postgresql_session),
]


def get_create_workspace(
    session: PostgresqlSessionDependency,
) -> CreateWorkspace:
    """Construct the create-workspace use case."""

    repository = SqlAlchemyWorkspaceRepository(session)

    return CreateWorkspace(
        repository=repository,
        transaction_manager=SqlAlchemyTransactionManager(session),
    )


def get_get_workspace(
    session: PostgresqlSessionDependency,
) -> GetWorkspace:
    """Construct the get-workspace use case."""

    return GetWorkspace(
        repository=SqlAlchemyWorkspaceRepository(session),
    )


CreateWorkspaceDependency = Annotated[
    CreateWorkspace,
    Depends(get_create_workspace),
]

GetWorkspaceDependency = Annotated[
    GetWorkspace,
    Depends(get_get_workspace),
]
