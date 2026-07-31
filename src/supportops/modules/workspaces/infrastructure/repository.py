"""PostgreSQL repository for workspaces."""

from uuid import UUID

from sqlalchemy import exists, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from supportops.infrastructure.postgresql.errors import (
    get_constraint_name,
)
from supportops.modules.workspaces.domain.models import Workspace
from supportops.modules.workspaces.domain.repositories import (
    WorkspaceRepository,
    WorkspaceSlugConflictError,
)
from supportops.modules.workspaces.infrastructure.models import (
    WorkspaceRecord,
)

_WORKSPACE_SLUG_CONSTRAINT = "uq_workspaces_slug"


class SqlAlchemyWorkspaceRepository(WorkspaceRepository):
    """Persist workspace entities using an active SQLAlchemy session."""

    def __init__(
        self,
        session: AsyncSession,
    ) -> None:
        self._session = session

    async def add(
        self,
        workspace: Workspace,
    ) -> None:
        """Add and flush a workspace inside the active transaction."""

        self._session.add(
            WorkspaceRecord.from_domain(workspace),
        )

        try:
            await self._session.flush()
        except IntegrityError as error:
            if get_constraint_name(error) == _WORKSPACE_SLUG_CONSTRAINT:
                raise WorkspaceSlugConflictError(
                    "Workspace slug already exists.",
                ) from error

            raise

    async def get(
        self,
        workspace_id: UUID,
    ) -> Workspace | None:
        """Return a workspace by identifier."""

        statement = select(WorkspaceRecord).where(
            WorkspaceRecord.id == workspace_id,
        )
        result = await self._session.execute(statement)
        record = result.scalar_one_or_none()

        if record is None:
            return None

        return record.to_domain()

    async def exists(
        self,
        workspace_id: UUID,
    ) -> bool:
        """Return whether a workspace exists."""

        statement = select(
            exists().where(
                WorkspaceRecord.id == workspace_id,
            )
        )
        result = await self._session.execute(statement)

        return bool(result.scalar_one())
