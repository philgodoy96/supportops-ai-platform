"""Workspace HTTP request and response schemas."""

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator

from supportops.modules.workspaces.domain.models import Workspace


class WorkspaceCreateRequest(BaseModel):
    """Payload accepted when creating a workspace."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(
        min_length=1,
        max_length=120,
    )
    slug: str = Field(
        min_length=3,
        max_length=63,
        pattern=r"^[a-z0-9]+(?:-[a-z0-9]+)*$",
    )

    @field_validator("name", mode="before")
    @classmethod
    def normalize_name(cls, value: object) -> object:
        """Trim surrounding whitespace from workspace names."""

        if isinstance(value, str):
            return value.strip()

        return value

    @field_validator("slug")
    @classmethod
    def reject_noncanonical_slug(
        cls,
        value: str,
    ) -> str:
        """Reject rather than transform noncanonical slugs."""

        if value != value.strip():
            raise ValueError(
                "Workspace slug must not contain surrounding whitespace.",
            )

        return value


class WorkspaceResponse(BaseModel):
    """Stable workspace representation returned by the API."""

    id: UUID
    name: str
    slug: str
    created_at: datetime
    updated_at: datetime

    @classmethod
    def from_domain(
        cls,
        workspace: Workspace,
    ) -> "WorkspaceResponse":
        """Create an API response from a domain entity."""

        return cls(
            id=workspace.id,
            name=workspace.name,
            slug=workspace.slug,
            created_at=workspace.created_at,
            updated_at=workspace.updated_at,
        )
