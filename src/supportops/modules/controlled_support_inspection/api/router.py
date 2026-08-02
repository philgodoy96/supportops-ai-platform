"""Workspace-scoped controlled-support inspection route."""

from uuid import UUID

from fastapi import APIRouter

from supportops.modules.controlled_support_inspection.api.dependencies import (
    GetControlledSupportInspectionDependency,
)
from supportops.modules.controlled_support_inspection.api.schemas import (
    ControlledSupportInspectionResponse,
)

router = APIRouter(
    prefix=("/workspaces/{workspace_id}/tickets/{ticket_id}/agent-runs"),
    tags=["controlled-support-inspection"],
)


@router.get(
    "/{agent_run_id}/inspection",
    response_model=ControlledSupportInspectionResponse,
)
async def get_controlled_support_inspection(
    workspace_id: UUID,
    ticket_id: UUID,
    agent_run_id: UUID,
    service: GetControlledSupportInspectionDependency,
) -> ControlledSupportInspectionResponse:
    """Retrieve one bounded controlled-support inspection."""

    inspection = await service.execute(
        workspace_id=workspace_id,
        ticket_id=ticket_id,
        agent_run_id=agent_run_id,
    )

    return ControlledSupportInspectionResponse.from_domain(inspection)
