"""Unit tests for workspace API schemas."""

from datetime import UTC, datetime
from uuid import UUID

import pytest
from pydantic import ValidationError

from supportops.modules.workspaces.api.schemas import (
    WorkspaceCreateRequest,
    WorkspaceResponse,
)
from supportops.modules.workspaces.domain.models import Workspace


def test_workspace_create_request_normalizes_name() -> None:
    request = WorkspaceCreateRequest(
        name="  Platform Support  ",
        slug="platform-support",
    )

    assert request.name == "Platform Support"
    assert request.slug == "platform-support"


@pytest.mark.parametrize(
    "name",
    [
        "",
        "   ",
        "a" * 121,
    ],
)
def test_workspace_create_request_rejects_invalid_name(
    name: str,
) -> None:
    with pytest.raises(ValidationError):
        WorkspaceCreateRequest(
            name=name,
            slug="platform-support",
        )


@pytest.mark.parametrize(
    "slug",
    [
        "ab",
        "a" * 64,
        "Platform-Support",
        "platform_support",
        " platform-support ",
        "-platform-support",
        "platform-support-",
        "platform--support",
    ],
)
def test_workspace_create_request_rejects_invalid_slug(
    slug: str,
) -> None:
    with pytest.raises(ValidationError):
        WorkspaceCreateRequest(
            name="Platform Support",
            slug=slug,
        )


def test_workspace_create_request_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        WorkspaceCreateRequest.model_validate(
            {
                "name": "Platform Support",
                "slug": "platform-support",
                "unexpected": "value",
            }
        )


def test_workspace_response_maps_domain_entity() -> None:
    timestamp = datetime(
        2026,
        7,
        31,
        12,
        0,
        tzinfo=UTC,
    )
    workspace = Workspace(
        id=UUID("50617047-0dbd-4704-b44f-c22da75b0595"),
        name="Platform Support",
        slug="platform-support",
        created_at=timestamp,
        updated_at=timestamp,
    )

    response = WorkspaceResponse.from_domain(workspace)

    assert response.model_dump() == {
        "id": workspace.id,
        "name": workspace.name,
        "slug": workspace.slug,
        "created_at": timestamp,
        "updated_at": timestamp,
    }
