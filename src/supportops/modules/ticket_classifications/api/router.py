"""Workspace-scoped ticket-classification inspection routes."""

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Query

from supportops.modules.ticket_classifications.api.dependencies import (
    GetTicketClassificationDependency,
    ListTicketClassificationsDependency,
)
from supportops.modules.ticket_classifications.api.schemas import (
    TicketClassificationListResponse,
    TicketClassificationResponse,
)
from supportops.modules.ticket_classifications.application.pagination import (
    decode_ticket_classification_cursor,
    encode_ticket_classification_cursor,
)

_DEFAULT_PAGE_SIZE = 20
_MAX_PAGE_SIZE = 100

router = APIRouter(
    prefix="/workspaces/{workspace_id}",
    tags=["ticket-classifications"],
)


@router.get(
    "/ticket-classifications/{classification_id}",
    response_model=TicketClassificationResponse,
)
async def get_ticket_classification(
    workspace_id: UUID,
    classification_id: UUID,
    service: GetTicketClassificationDependency,
) -> TicketClassificationResponse:
    """Retrieve one accepted classification through its workspace boundary."""

    classification = await service.execute(
        workspace_id=workspace_id,
        classification_id=classification_id,
    )

    return TicketClassificationResponse.from_domain(
        classification,
    )


@router.get(
    "/tickets/{ticket_id}/classifications",
    response_model=TicketClassificationListResponse,
)
async def list_ticket_classifications(
    workspace_id: UUID,
    ticket_id: UUID,
    service: ListTicketClassificationsDependency,
    page_size: Annotated[
        int,
        Query(
            ge=1,
            le=_MAX_PAGE_SIZE,
        ),
    ] = _DEFAULT_PAGE_SIZE,
    cursor: str | None = None,
) -> TicketClassificationListResponse:
    """List one deterministic page of classifications for a ticket."""

    position = (
        decode_ticket_classification_cursor(
            cursor,
        )
        if cursor is not None
        else None
    )
    classifications = await service.execute(
        workspace_id=workspace_id,
        ticket_id=ticket_id,
        limit=page_size + 1,
        after_created_at=(position.created_at if position is not None else None),
        after_classification_id=(position.classification_id if position is not None else None),
    )

    has_next_page = len(classifications) > page_size
    page = list(
        classifications[:page_size],
    )

    next_cursor = None
    if has_next_page and page:
        last_classification = page[-1]
        next_cursor = encode_ticket_classification_cursor(
            created_at=last_classification.created_at,
            classification_id=last_classification.id,
        )

    return TicketClassificationListResponse(
        items=[
            TicketClassificationResponse.from_domain(
                classification,
            )
            for classification in page
        ],
        next_cursor=next_cursor,
    )
