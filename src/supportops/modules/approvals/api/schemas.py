"""HTTP schemas for approval request inspection."""

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, JsonValue

from supportops.modules.approvals.domain.models import (
    ApprovalRequest,
    ApprovalRequestStatus,
)


class ApprovalRequestResponse(BaseModel):
    """Workspace-scoped approval request representation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: UUID
    workspace_id: UUID
    ticket_id: UUID
    agent_run_id: UUID
    agent_tool_call_id: UUID
    requested_by_llm_invocation_id: UUID
    status: ApprovalRequestStatus
    tool_name: str
    tool_version: int
    input_fingerprint: str
    proposed_input: dict[str, JsonValue]
    request_reason: str
    expires_at: datetime
    decision_actor_reference: str | None
    decision_comment: str | None
    decision_request_id: UUID | None
    decision_correlation_id: UUID | None
    decided_at: datetime | None
    created_at: datetime
    updated_at: datetime

    @classmethod
    def from_domain(
        cls,
        approval: ApprovalRequest,
    ) -> "ApprovalRequestResponse":
        """Create a safe API response from the domain entity."""

        return cls(
            id=approval.id,
            workspace_id=approval.workspace_id,
            ticket_id=approval.ticket_id,
            agent_run_id=approval.agent_run_id,
            agent_tool_call_id=approval.agent_tool_call_id,
            requested_by_llm_invocation_id=(approval.requested_by_llm_invocation_id),
            status=approval.status,
            tool_name=approval.tool_name,
            tool_version=approval.tool_version,
            input_fingerprint=approval.input_fingerprint,
            proposed_input=dict(approval.proposed_input),
            request_reason=approval.request_reason,
            expires_at=approval.expires_at,
            decision_actor_reference=(approval.decision_actor_reference),
            decision_comment=approval.decision_comment,
            decision_request_id=approval.decision_request_id,
            decision_correlation_id=(approval.decision_correlation_id),
            decided_at=approval.decided_at,
            created_at=approval.created_at,
            updated_at=approval.updated_at,
        )


class ApprovalRequestListResponse(BaseModel):
    """One keyset-paginated approval request page."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    items: tuple[ApprovalRequestResponse, ...]
    next_cursor: str | None = None


class ApprovalRequestListQueryParams(BaseModel):
    """Validated approval list query parameters."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    status: ApprovalRequestStatus | None = None
    cursor: str | None = None
    page_size: int = Field(default=20, ge=1, le=100)
