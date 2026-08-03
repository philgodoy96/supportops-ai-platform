"""Workspace-scoped support ticket HTTP routes."""

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Query, status

from supportops.core.request_context import get_request_context
from supportops.modules.tickets.api.dependencies import (
    CreateTicketDependency,
    GetTicketDependency,
    ListTicketsDependency,
)
from supportops.modules.tickets.api.escalation_router import (
    router as ticket_escalations_router,
)
from supportops.modules.tickets.api.pagination import (
    decode_ticket_cursor,
    encode_ticket_cursor,
)
from supportops.modules.tickets.api.schemas import (
    TicketCreateRequest,
    TicketListResponse,
    TicketResponse,
)

_DEFAULT_PAGE_SIZE = 20
_MAX_PAGE_SIZE = 100

tickets_router = APIRouter(
    prefix="/workspaces/{workspace_id}/tickets",
    tags=["tickets"],
)
router = APIRouter()
router.include_router(tickets_router)
router.include_router(ticket_escalations_router)


@tickets_router.post(
    "",
    response_model=TicketResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_ticket(
    workspace_id: UUID,
    request: TicketCreateRequest,
    service: CreateTicketDependency,
) -> TicketResponse:
    """Create a support ticket inside one workspace."""

    context = get_request_context()

    if context is None:
        raise RuntimeError(
            "HTTP request context is unavailable.",
        )

    result = await service.execute(
        workspace_id=workspace_id,
        subject=request.subject,
        description=request.description,
        external_reference=request.external_reference,
        ingestion_request_id=context.request_id,
        correlation_id=context.correlation_id,
    )

    return TicketResponse.from_domain(result.ticket)


@tickets_router.get(
    "/{ticket_id}",
    response_model=TicketResponse,
)
async def get_ticket(
    workspace_id: UUID,
    ticket_id: UUID,
    service: GetTicketDependency,
) -> TicketResponse:
    """Retrieve a ticket only through its workspace boundary."""

    ticket = await service.execute(
        workspace_id=workspace_id,
        ticket_id=ticket_id,
    )

    return TicketResponse.from_domain(ticket)


@tickets_router.get(
    "",
    response_model=TicketListResponse,
)
async def list_tickets(
    workspace_id: UUID,
    service: ListTicketsDependency,
    page_size: Annotated[
        int,
        Query(
            ge=1,
            le=_MAX_PAGE_SIZE,
        ),
    ] = _DEFAULT_PAGE_SIZE,
    cursor: str | None = None,
) -> TicketListResponse:
    """List one deterministic page of workspace tickets."""

    position = decode_ticket_cursor(cursor) if cursor is not None else None

    tickets = await service.execute(
        workspace_id=workspace_id,
        limit=page_size + 1,
        after_created_at=(position.created_at if position is not None else None),
        after_ticket_id=(position.ticket_id if position is not None else None),
    )

    has_next_page = len(tickets) > page_size
    page = list(tickets[:page_size])

    next_cursor = None

    if has_next_page and page:
        last_ticket = page[-1]
        next_cursor = encode_ticket_cursor(
            created_at=last_ticket.created_at,
            ticket_id=last_ticket.id,
        )

    return TicketListResponse(
        items=[TicketResponse.from_domain(ticket) for ticket in page],
        next_cursor=next_cursor,
    )
