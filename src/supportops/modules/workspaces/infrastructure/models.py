"""SQLAlchemy persistence model for workspaces."""

from datetime import datetime
from uuid import UUID

from sqlalchemy import CheckConstraint, DateTime, String, Uuid
from sqlalchemy.orm import Mapped, mapped_column

from supportops.infrastructure.postgresql.base import Base
from supportops.modules.workspaces.domain.models import Workspace


class WorkspaceRecord(Base):
    """Persisted workspace record."""

    __tablename__ = "workspaces"
    __table_args__ = (
        CheckConstraint(
            "name = btrim(name)",
            name="workspace_name_trimmed",
        ),
        CheckConstraint(
            "char_length(name) BETWEEN 1 AND 120",
            name="workspace_name_length",
        ),
        CheckConstraint(
            "char_length(slug) BETWEEN 3 AND 63",
            name="workspace_slug_length",
        ),
        CheckConstraint(
            "slug ~ '^[a-z0-9]+(?:-[a-z0-9]+)*$'",
            name="workspace_slug_format",
        ),
        CheckConstraint(
            "updated_at >= created_at",
            name="workspace_timestamp_order",
        ),
    )

    id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True),
        primary_key=True,
    )
    name: Mapped[str] = mapped_column(
        String(120),
        nullable=False,
    )
    slug: Mapped[str] = mapped_column(
        String(63),
        nullable=False,
        unique=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
    )

    @classmethod
    def from_domain(
        cls,
        workspace: Workspace,
    ) -> "WorkspaceRecord":
        """Create a persistence record from a workspace entity."""

        return cls(
            id=workspace.id,
            name=workspace.name,
            slug=workspace.slug,
            created_at=workspace.created_at,
            updated_at=workspace.updated_at,
        )

    def to_domain(self) -> Workspace:
        """Map the persistence record to a workspace entity."""

        return Workspace(
            id=self.id,
            name=self.name,
            slug=self.slug,
            created_at=self.created_at,
            updated_at=self.updated_at,
        )
