"""HTTP schemas for approval request inspection."""

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, JsonValue, field_validator

from supportops.modules.approvals.application.models import (
    ApprovalDecisionResult,
)
from supportops.modules.approvals.domain.models import (
    APPROVAL_DECISION_ACTOR_MAX_LENGTH,
    APPROVAL_DECISION_COMMENT_MAX_LENGTH,
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


class ApproveApprovalRequestBody(BaseModel):
    """Asserted actor metadata for approving one request."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    actor_reference: str = Field(
        min_length=1,
        max_length=APPROVAL_DECISION_ACTOR_MAX_LENGTH,
    )
    decision_request_id: UUID
    comment: str | None = Field(
        default=None,
        min_length=1,
        max_length=APPROVAL_DECISION_COMMENT_MAX_LENGTH,
    )

    @field_validator("actor_reference")
    @classmethod
    def reject_noncanonical_actor_reference(
        cls,
        value: str,
    ) -> str:
        """Reject surrounding whitespace rather than normalize silently."""

        if value != value.strip():
            raise ValueError(
                "actor_reference must not contain surrounding whitespace.",
            )
        return value

    @field_validator("comment")
    @classmethod
    def reject_noncanonical_comment(
        cls,
        value: str | None,
    ) -> str | None:
        """Reject surrounding whitespace on optional approve comments."""

        if value is not None and value != value.strip():
            raise ValueError(
                "comment must not contain surrounding whitespace.",
            )
        return value


class RejectApprovalRequestBody(BaseModel):
    """Asserted actor metadata for rejecting one request."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    actor_reference: str = Field(
        min_length=1,
        max_length=APPROVAL_DECISION_ACTOR_MAX_LENGTH,
    )
    decision_request_id: UUID
    comment: str = Field(
        min_length=1,
        max_length=APPROVAL_DECISION_COMMENT_MAX_LENGTH,
    )

    @field_validator("actor_reference", "comment")
    @classmethod
    def reject_noncanonical_text(
        cls,
        value: str,
    ) -> str:
        """Reject surrounding whitespace rather than normalize silently."""

        if value != value.strip():
            raise ValueError(
                "Value must not contain surrounding whitespace.",
            )
        return value


class ApprovalDecisionResponse(BaseModel):
    """Terminal approval decision returned by command endpoints."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    approval_request_id: UUID
    workspace_id: UUID
    agent_run_id: UUID
    status: ApprovalRequestStatus
    decision_actor_reference: str | None
    decision_comment: str | None
    decision_request_id: UUID | None
    decision_correlation_id: UUID | None
    decided_at: datetime | None
    idempotent: bool

    @classmethod
    def from_result(
        cls,
        result: ApprovalDecisionResult,
    ) -> "ApprovalDecisionResponse":
        """Map a decision result to the public HTTP envelope."""

        approval = result.approval_request
        return cls(
            approval_request_id=approval.id,
            workspace_id=approval.workspace_id,
            agent_run_id=approval.agent_run_id,
            status=approval.status,
            decision_actor_reference=(approval.decision_actor_reference),
            decision_comment=approval.decision_comment,
            decision_request_id=approval.decision_request_id,
            decision_correlation_id=(approval.decision_correlation_id),
            decided_at=approval.decided_at,
            idempotent=result.idempotent,
        )
