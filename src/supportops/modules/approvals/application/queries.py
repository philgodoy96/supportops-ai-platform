"""Workspace-scoped approval inspection queries."""

from uuid import UUID

from supportops.modules.approvals.domain.models import (
    ApprovalRequest,
)
from supportops.modules.approvals.domain.repositories import (
    ApprovalRequestListPage,
    ApprovalRequestListQuery,
    ApprovalRequestPageCursor,
    ApprovalRequestRepository,
)

__all__ = [
    "ApprovalRequestListPage",
    "ApprovalRequestListQuery",
    "ApprovalRequestNotFoundError",
    "ApprovalRequestPageCursor",
    "GetApprovalRequest",
    "ListApprovalRequests",
]


class ApprovalRequestNotFoundError(LookupError):
    """Raised when an approval is absent from the workspace."""


class ListApprovalRequests:
    """List approvals without exposing persistence concerns."""

    def __init__(
        self,
        repository: ApprovalRequestRepository,
    ) -> None:
        self._repository = repository

    async def execute(
        self,
        query: ApprovalRequestListQuery,
    ) -> ApprovalRequestListPage:
        """Return one stable approval page."""

        return await self._repository.list_page(query)


class GetApprovalRequest:
    """Load one approval through workspace-scoped nondisclosure."""

    def __init__(
        self,
        repository: ApprovalRequestRepository,
    ) -> None:
        self._repository = repository

    async def execute(
        self,
        *,
        workspace_id: UUID,
        approval_request_id: UUID,
    ) -> ApprovalRequest:
        """Return one approval or raise a nondisclosing not-found."""

        approval = await self._repository.get_by_id(
            workspace_id=workspace_id,
            approval_request_id=approval_request_id,
        )
        if approval is None:
            raise ApprovalRequestNotFoundError(
                "Approval request was not found.",
            )
        return approval
