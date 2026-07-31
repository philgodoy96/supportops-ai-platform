"""Workspace HTTP routes."""

from uuid import UUID

from fastapi import APIRouter, status

from supportops.modules.workspaces.api.dependencies import (
    CreateWorkspaceDependency,
    GetWorkspaceDependency,
)
from supportops.modules.workspaces.api.schemas import (
    WorkspaceCreateRequest,
    WorkspaceResponse,
)

router = APIRouter(
    prefix="/workspaces",
    tags=["workspaces"],
)


@router.post(
    "",
    response_model=WorkspaceResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_workspace(
    request: WorkspaceCreateRequest,
    service: CreateWorkspaceDependency,
) -> WorkspaceResponse:
    """Create a workspace."""

    workspace = await service.execute(
        name=request.name,
        slug=request.slug,
    )

    return WorkspaceResponse.from_domain(workspace)


@router.get(
    "/{workspace_id}",
    response_model=WorkspaceResponse,
)
async def get_workspace(
    workspace_id: UUID,
    service: GetWorkspaceDependency,
) -> WorkspaceResponse:
    """Retrieve one workspace."""

    workspace = await service.execute(
        workspace_id=workspace_id,
    )

    return WorkspaceResponse.from_domain(workspace)
