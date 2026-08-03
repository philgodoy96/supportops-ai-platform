"""Dependency composition for approval inspection endpoints."""

from typing import Annotated

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from supportops.api.dependencies import get_postgresql_session
from supportops.modules.approvals.application.queries import (
    GetApprovalRequest,
    ListApprovalRequests,
)
from supportops.modules.approvals.infrastructure.repository import (
    SqlAlchemyApprovalRequestRepository,
)


def get_list_approval_requests(
    session: Annotated[
        AsyncSession,
        Depends(get_postgresql_session),
    ],
) -> ListApprovalRequests:
    """Build one session-scoped approval list query."""

    repository = SqlAlchemyApprovalRequestRepository(session)
    return ListApprovalRequests(repository)


def get_approval_request(
    session: Annotated[
        AsyncSession,
        Depends(get_postgresql_session),
    ],
) -> GetApprovalRequest:
    """Build one session-scoped approval detail query."""

    repository = SqlAlchemyApprovalRequestRepository(session)
    return GetApprovalRequest(repository)
