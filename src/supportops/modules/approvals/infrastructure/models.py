"""SQLAlchemy persistence model for application-owned approval requests."""

from datetime import datetime
from uuid import UUID

from pydantic import JsonValue
from sqlalchemy import (
    CheckConstraint,
    DateTime,
    ForeignKeyConstraint,
    Index,
    Integer,
    String,
    UniqueConstraint,
    Uuid,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from supportops.agent_tools.domain.audit import AGENT_TOOL_CALL_NAME_MAX_LENGTH
from supportops.agent_tools.domain.contracts import ToolSafetyLevel
from supportops.infrastructure.postgresql.base import Base
from supportops.modules.approvals.domain.models import (
    APPROVAL_DECISION_ACTOR_MAX_LENGTH,
    APPROVAL_DECISION_COMMENT_MAX_LENGTH,
    APPROVAL_PROPOSED_INPUT_MAX_BYTES,
    APPROVAL_REQUEST_REASON_MAX_LENGTH,
    ApprovalRequest,
    ApprovalRequestStatus,
)

_INPUT_FINGERPRINT_LENGTH = 64
_STATUS_MAX_LENGTH = 32
_SAFETY_LEVEL_MAX_LENGTH = 32

_APPROVAL_STATUS_SQL_VALUES = ", ".join(f"'{member.value}'" for member in ApprovalRequestStatus)


class ApprovalRequestRecord(Base):
    """Persisted application-owned approval-request state."""

    __tablename__ = "approval_requests"

    id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True),
        primary_key=True,
    )
    workspace_id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True),
        nullable=False,
    )
    ticket_id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True),
        nullable=False,
    )
    agent_run_id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True),
        nullable=False,
    )
    agent_tool_call_id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True),
        nullable=False,
    )
    requested_by_llm_invocation_id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True),
        nullable=False,
    )
    status: Mapped[str] = mapped_column(
        String(_STATUS_MAX_LENGTH),
        nullable=False,
    )
    tool_name: Mapped[str] = mapped_column(
        String(AGENT_TOOL_CALL_NAME_MAX_LENGTH),
        nullable=False,
    )
    tool_version: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )
    safety_level: Mapped[str] = mapped_column(
        String(_SAFETY_LEVEL_MAX_LENGTH),
        nullable=False,
    )
    input_fingerprint: Mapped[str] = mapped_column(
        String(_INPUT_FINGERPRINT_LENGTH),
        nullable=False,
    )
    proposed_input: Mapped[dict[str, JsonValue]] = mapped_column(
        JSONB,
        nullable=False,
    )
    request_reason: Mapped[str] = mapped_column(
        String(APPROVAL_REQUEST_REASON_MAX_LENGTH),
        nullable=False,
    )
    expires_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
    )
    decision_actor_reference: Mapped[str | None] = mapped_column(
        String(APPROVAL_DECISION_ACTOR_MAX_LENGTH),
        nullable=True,
    )
    decision_comment: Mapped[str | None] = mapped_column(
        String(APPROVAL_DECISION_COMMENT_MAX_LENGTH),
        nullable=True,
    )
    decision_request_id: Mapped[UUID | None] = mapped_column(
        Uuid(as_uuid=True),
        nullable=True,
    )
    decision_correlation_id: Mapped[UUID | None] = mapped_column(
        Uuid(as_uuid=True),
        nullable=True,
    )
    decided_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
    )

    __table_args__ = (
        ForeignKeyConstraint(
            [
                "workspace_id",
                "ticket_id",
                "agent_run_id",
            ],
            [
                "agent_runs.workspace_id",
                "agent_runs.ticket_id",
                "agent_runs.id",
            ],
            name=("fk_approval_requests_workspace_ticket_agent_run"),
            ondelete="CASCADE",
        ),
        ForeignKeyConstraint(
            ["agent_tool_call_id"],
            ["agent_tool_calls.id"],
            name="fk_approval_requests_agent_tool_call",
            ondelete="CASCADE",
        ),
        ForeignKeyConstraint(
            [
                "agent_run_id",
                "requested_by_llm_invocation_id",
            ],
            [
                "llm_invocations.agent_run_id",
                "llm_invocations.id",
            ],
            name="fk_approval_requests_requesting_invocation",
            ondelete="RESTRICT",
        ),
        UniqueConstraint(
            "agent_tool_call_id",
            name="uq_approval_requests_agent_tool_call",
        ),
        UniqueConstraint(
            "workspace_id",
            "id",
            name="uq_approval_requests_workspace_id",
        ),
        CheckConstraint(
            f"status IN ({_APPROVAL_STATUS_SQL_VALUES})",
            name="approval_request_status",
        ),
        CheckConstraint(
            "safety_level = 'sensitive_write'",
            name="approval_request_safety_level",
        ),
        CheckConstraint(
            (
                "tool_name = btrim(tool_name) "
                "AND char_length(tool_name) BETWEEN 1 AND 64 "
                "AND tool_name ~ '^[a-z][a-z0-9_]*$'"
            ),
            name="approval_request_tool_name_format",
        ),
        CheckConstraint(
            "tool_version >= 1",
            name="approval_request_tool_version_positive",
        ),
        CheckConstraint(
            "input_fingerprint ~ '^[0-9a-f]{64}$'",
            name="approval_request_input_fingerprint",
        ),
        CheckConstraint(
            "jsonb_typeof(proposed_input) = 'object'",
            name="approval_request_proposed_input_object",
        ),
        CheckConstraint(
            (f"octet_length(proposed_input::text) <= {APPROVAL_PROPOSED_INPUT_MAX_BYTES}"),
            name="approval_request_proposed_input_size",
        ),
        CheckConstraint(
            (
                "request_reason = btrim(request_reason) "
                "AND char_length(request_reason) BETWEEN 1 AND 1000"
            ),
            name="approval_request_reason_format",
        ),
        CheckConstraint(
            (
                "decision_actor_reference IS NULL OR ("
                "decision_actor_reference = "
                "btrim(decision_actor_reference) "
                "AND char_length(decision_actor_reference) "
                "BETWEEN 1 AND 255"
                ")"
            ),
            name="approval_request_actor_format",
        ),
        CheckConstraint(
            (
                "decision_comment IS NULL OR ("
                "decision_comment = btrim(decision_comment) "
                "AND char_length(decision_comment) "
                "BETWEEN 1 AND 2000"
                ")"
            ),
            name="approval_request_comment_format",
        ),
        CheckConstraint(
            "expires_at > created_at",
            name="approval_request_expiration_order",
        ),
        CheckConstraint(
            "updated_at >= created_at",
            name="approval_request_update_order",
        ),
        CheckConstraint(
            "decided_at IS NULL OR decided_at >= created_at",
            name="approval_request_decision_order",
        ),
        CheckConstraint(
            (
                "("
                "status = 'pending' "
                "AND decision_actor_reference IS NULL "
                "AND decision_comment IS NULL "
                "AND decision_request_id IS NULL "
                "AND decision_correlation_id IS NULL "
                "AND decided_at IS NULL"
                ") OR ("
                "status = 'approved' "
                "AND decision_actor_reference IS NOT NULL "
                "AND decision_request_id IS NOT NULL "
                "AND decision_correlation_id IS NOT NULL "
                "AND decided_at IS NOT NULL"
                ") OR ("
                "status = 'rejected' "
                "AND decision_actor_reference IS NOT NULL "
                "AND decision_comment IS NOT NULL "
                "AND decision_request_id IS NOT NULL "
                "AND decision_correlation_id IS NOT NULL "
                "AND decided_at IS NOT NULL"
                ") OR ("
                "status = 'expired' "
                "AND decision_actor_reference = "
                "'system:approval-expiration' "
                "AND decision_comment IS NULL "
                "AND decision_request_id IS NULL "
                "AND decision_correlation_id IS NULL "
                "AND decided_at IS NOT NULL"
                ")"
            ),
            name="approval_request_decision_state",
        ),
        Index(
            "ix_approval_requests_workspace_status_created_id",
            "workspace_id",
            "status",
            created_at.desc(),
            id.desc(),
        ),
        Index(
            "ix_approval_requests_agent_run_status",
            "agent_run_id",
            "status",
        ),
        Index(
            "ix_approval_requests_pending_expiration",
            "expires_at",
            "id",
            postgresql_where=text("status = 'pending'"),
        ),
    )

    @classmethod
    def from_domain(
        cls,
        approval_request: ApprovalRequest,
    ) -> "ApprovalRequestRecord":
        """Create a persistence record from an approval-request entity."""

        return cls(
            id=approval_request.id,
            workspace_id=approval_request.workspace_id,
            ticket_id=approval_request.ticket_id,
            agent_run_id=approval_request.agent_run_id,
            agent_tool_call_id=approval_request.agent_tool_call_id,
            requested_by_llm_invocation_id=(approval_request.requested_by_llm_invocation_id),
            status=approval_request.status.value,
            tool_name=approval_request.tool_name,
            tool_version=approval_request.tool_version,
            safety_level=approval_request.safety_level.value,
            input_fingerprint=approval_request.input_fingerprint,
            proposed_input=dict(approval_request.proposed_input),
            request_reason=approval_request.request_reason,
            expires_at=approval_request.expires_at,
            decision_actor_reference=(approval_request.decision_actor_reference),
            decision_comment=approval_request.decision_comment,
            decision_request_id=approval_request.decision_request_id,
            decision_correlation_id=(approval_request.decision_correlation_id),
            decided_at=approval_request.decided_at,
            created_at=approval_request.created_at,
            updated_at=approval_request.updated_at,
        )

    def to_domain(self) -> ApprovalRequest:
        """Map the persistence record to an approval-request entity."""

        return ApprovalRequest(
            id=self.id,
            workspace_id=self.workspace_id,
            ticket_id=self.ticket_id,
            agent_run_id=self.agent_run_id,
            agent_tool_call_id=self.agent_tool_call_id,
            requested_by_llm_invocation_id=(self.requested_by_llm_invocation_id),
            status=ApprovalRequestStatus(self.status),
            tool_name=self.tool_name,
            tool_version=self.tool_version,
            safety_level=ToolSafetyLevel(self.safety_level),
            input_fingerprint=self.input_fingerprint,
            proposed_input=self.proposed_input,
            request_reason=self.request_reason,
            expires_at=self.expires_at,
            decision_actor_reference=self.decision_actor_reference,
            decision_comment=self.decision_comment,
            decision_request_id=self.decision_request_id,
            decision_correlation_id=self.decision_correlation_id,
            decided_at=self.decided_at,
            created_at=self.created_at,
            updated_at=self.updated_at,
        )
