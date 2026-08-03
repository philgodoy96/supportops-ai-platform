"""Workspace-scoped approval inspection endpoints."""

from datetime import UTC, datetime
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, Query

from supportops.core.request_context import (
    RequestContext,
    get_request_context,
)
from supportops.modules.approvals.api.dependencies import (
    get_approval_request,
    get_decide_approval_request,
    get_list_approval_requests,
)
from supportops.modules.approvals.api.pagination import (
    decode_approval_cursor,
    encode_approval_cursor,
)
from supportops.modules.approvals.api.schemas import (
    ApprovalDecisionResponse,
    ApprovalRequestListResponse,
    ApprovalRequestResponse,
    ApproveApprovalRequestBody,
    RejectApprovalRequestBody,
)
from supportops.modules.approvals.application.models import (
    ApproveApprovalRequestCommand,
    RejectApprovalRequestCommand,
)
from supportops.modules.approvals.application.queries import (
    ApprovalRequestListQuery,
    GetApprovalRequest,
    ListApprovalRequests,
)
from supportops.modules.approvals.application.services import (
    DecideApprovalRequest,
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


@router.post(
    "/{approval_request_id}/approve",
    response_model=ApprovalDecisionResponse,
)
async def approve_approval_request(
    workspace_id: UUID,
    approval_request_id: UUID,
    body: ApproveApprovalRequestBody,
    service: Annotated[
        DecideApprovalRequest,
        Depends(get_decide_approval_request),
    ],
    request_context: Annotated[
        RequestContext | None,
        Depends(get_request_context),
    ],
) -> ApprovalDecisionResponse:
    """Approve one pending approval request and requeue its AgentRun."""

    if request_context is None:
        raise RuntimeError("HTTP request context is unavailable.")

    result = await service.approve(
        ApproveApprovalRequestCommand(
            workspace_id=workspace_id,
            approval_request_id=approval_request_id,
            actor_reference=body.actor_reference,
            comment=body.comment,
            request_id=body.decision_request_id,
            correlation_id=request_context.correlation_id,
            decided_at=datetime.now(UTC),
        ),
    )
    return ApprovalDecisionResponse.from_result(result)


@router.post(
    "/{approval_request_id}/reject",
    response_model=ApprovalDecisionResponse,
)
async def reject_approval_request(
    workspace_id: UUID,
    approval_request_id: UUID,
    body: RejectApprovalRequestBody,
    service: Annotated[
        DecideApprovalRequest,
        Depends(get_decide_approval_request),
    ],
    request_context: Annotated[
        RequestContext | None,
        Depends(get_request_context),
    ],
) -> ApprovalDecisionResponse:
    """Reject one pending approval request and requeue its AgentRun."""

    if request_context is None:
        raise RuntimeError("HTTP request context is unavailable.")

    result = await service.reject(
        RejectApprovalRequestCommand(
            workspace_id=workspace_id,
            approval_request_id=approval_request_id,
            actor_reference=body.actor_reference,
            comment=body.comment,
            request_id=body.decision_request_id,
            correlation_id=request_context.correlation_id,
            decided_at=datetime.now(UTC),
        ),
    )
    return ApprovalDecisionResponse.from_result(result)
