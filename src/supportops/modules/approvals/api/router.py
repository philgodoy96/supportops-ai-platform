"""Workspace-scoped approval inspection endpoints."""

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, Query

from supportops.modules.approvals.api.dependencies import (
    get_approval_request,
    get_list_approval_requests,
)
from supportops.modules.approvals.api.pagination import (
    decode_approval_cursor,
    encode_approval_cursor,
)
from supportops.modules.approvals.api.schemas import (
    ApprovalRequestListResponse,
    ApprovalRequestResponse,
)
from supportops.modules.approvals.application.queries import (
    ApprovalRequestListQuery,
    GetApprovalRequest,
    ListApprovalRequests,
)
from supportops.modules.approvals.domain.models import (
    ApprovalRequestStatus,
)

router = APIRouter(
    prefix="/workspaces/{workspace_id}/approvals",
    tags=["approvals"],
)


@router.get(
    "",
    response_model=ApprovalRequestListResponse,
)
async def list_approval_requests(
    workspace_id: UUID,
    service: Annotated[
        ListApprovalRequests,
        Depends(get_list_approval_requests),
    ],
    status: Annotated[
        ApprovalRequestStatus | None,
        Query(),
    ] = None,
    cursor: Annotated[str | None, Query()] = None,
    page_size: Annotated[
        int,
        Query(ge=1, le=100),
    ] = 20,
) -> ApprovalRequestListResponse:
    """List approval requests in stable descending order."""

    page = await service.execute(
        ApprovalRequestListQuery(
            workspace_id=workspace_id,
            status=status,
            cursor=(None if cursor is None else decode_approval_cursor(cursor)),
            page_size=page_size,
        ),
    )
    return ApprovalRequestListResponse(
        items=tuple(ApprovalRequestResponse.from_domain(item) for item in page.items),
        next_cursor=(
            None if page.next_cursor is None else encode_approval_cursor(page.next_cursor)
        ),
    )


@router.get(
    "/{approval_request_id}",
    response_model=ApprovalRequestResponse,
)
async def get_approval_request_detail(
    workspace_id: UUID,
    approval_request_id: UUID,
    service: Annotated[
        GetApprovalRequest,
        Depends(get_approval_request),
    ],
) -> ApprovalRequestResponse:
    """Return one workspace-scoped approval request."""

    approval = await service.execute(
        workspace_id=workspace_id,
        approval_request_id=approval_request_id,
    )
    return ApprovalRequestResponse.from_domain(approval)
