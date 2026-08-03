"""Workspace-scoped ticket escalation inspection endpoints."""

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, Query

from supportops.modules.tickets.api.escalation_dependencies import (
    get_list_ticket_escalations,
    get_ticket_escalation,
)
from supportops.modules.tickets.api.escalation_pagination import (
    decode_ticket_escalation_cursor,
    encode_ticket_escalation_cursor,
)
from supportops.modules.tickets.api.escalation_schemas import (
    TicketEscalationListResponse,
    TicketEscalationResponse,
)
from supportops.modules.tickets.application.escalation_queries import (
    GetTicketEscalation,
    ListTicketEscalations,
    TicketEscalationListQuery,
)

router = APIRouter(
    prefix="/workspaces/{workspace_id}/ticket-escalations",
    tags=["ticket-escalations"],
)


@router.get(
    "",
    response_model=TicketEscalationListResponse,
)
async def list_ticket_escalations(
    workspace_id: UUID,
    service: Annotated[
        ListTicketEscalations,
        Depends(get_list_ticket_escalations),
    ],
    ticket_id: Annotated[UUID | None, Query()] = None,
    cursor: Annotated[str | None, Query()] = None,
    page_size: Annotated[
        int,
        Query(ge=1, le=100),
    ] = 20,
) -> TicketEscalationListResponse:
    """List immutable escalations in stable descending order."""

    page = await service.execute(
        TicketEscalationListQuery(
            workspace_id=workspace_id,
            ticket_id=ticket_id,
            cursor=(None if cursor is None else decode_ticket_escalation_cursor(cursor)),
            page_size=page_size,
        ),
    )
    return TicketEscalationListResponse(
        items=tuple(TicketEscalationResponse.from_domain(item) for item in page.items),
        next_cursor=(
            None
            if page.next_cursor is None
            else encode_ticket_escalation_cursor(
                page.next_cursor,
            )
        ),
    )


@router.get(
    "/{ticket_escalation_id}",
    response_model=TicketEscalationResponse,
)
async def get_ticket_escalation_detail(
    workspace_id: UUID,
    ticket_escalation_id: UUID,
    service: Annotated[
        GetTicketEscalation,
        Depends(get_ticket_escalation),
    ],
) -> TicketEscalationResponse:
    """Return one workspace-scoped immutable escalation."""

    escalation = await service.execute(
        workspace_id=workspace_id,
        escalation_id=ticket_escalation_id,
    )
    return TicketEscalationResponse.from_domain(escalation)
