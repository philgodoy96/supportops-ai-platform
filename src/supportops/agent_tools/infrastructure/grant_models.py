"""SQLAlchemy persistence model for sensitive execution grants."""

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
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from supportops.agent_tools.domain.contracts import ToolSafetyLevel
from supportops.agent_tools.domain.grants import SensitiveExecutionGrant
from supportops.infrastructure.postgresql.base import Base


class SensitiveExecutionGrantRecord(Base):
    """Application-owned immutable authorization record."""

    __tablename__ = "sensitive_execution_grants"

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
    executed_by_agent_run_attempt_id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True),
        nullable=False,
    )
    approval_request_id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True),
        nullable=False,
    )
    agent_tool_call_id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True),
        nullable=False,
    )
    tool_name: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
    )
    tool_version: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )
    safety_level: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
    )
    input_fingerprint: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
    )
    granted_input: Mapped[dict[str, JsonValue]] = mapped_column(
        JSONB,
        nullable=False,
    )
    decision_actor_reference: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )
    decision_request_id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True),
        nullable=False,
    )
    decision_correlation_id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True),
        nullable=False,
    )
    approved_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
    )

    __table_args__ = (
        CheckConstraint(
            "safety_level = 'sensitive_write'",
            name="safety_level",
        ),
        CheckConstraint(
            (
                "tool_name = btrim(tool_name) "
                "AND char_length(tool_name) BETWEEN 1 AND 64 "
                "AND tool_name ~ '^[a-z][a-z0-9_]*$'"
            ),
            name="tool_name_format",
        ),
        CheckConstraint(
            "tool_version >= 1",
            name="tool_version_positive",
        ),
        CheckConstraint(
            "input_fingerprint ~ '^[0-9a-f]{64}$'",
            name="input_fingerprint",
        ),
        CheckConstraint(
            "jsonb_typeof(granted_input) = 'object'",
            name="granted_input_object",
        ),
        CheckConstraint(
            "octet_length(granted_input::text) <= 8192",
            name="granted_input_size",
        ),
        CheckConstraint(
            (
                "decision_actor_reference = "
                "btrim(decision_actor_reference) "
                "AND char_length(decision_actor_reference) "
                "BETWEEN 1 AND 255"
            ),
            name="actor_format",
        ),
        CheckConstraint(
            "created_at >= approved_at",
            name="creation_order",
        ),
        ForeignKeyConstraint(
            ["workspace_id", "ticket_id", "agent_run_id"],
            [
                "agent_runs.workspace_id",
                "agent_runs.ticket_id",
                "agent_runs.id",
            ],
            name=("fk_sensitive_execution_grants_workspace_ticket_agent_run"),
            ondelete="CASCADE",
        ),
        ForeignKeyConstraint(
            [
                "agent_run_id",
                "executed_by_agent_run_attempt_id",
            ],
            [
                "agent_run_attempts.agent_run_id",
                "agent_run_attempts.id",
            ],
            name=("fk_sensitive_execution_grants_execution_attempt"),
            ondelete="RESTRICT",
        ),
        ForeignKeyConstraint(
            ["approval_request_id"],
            ["approval_requests.id"],
            name=("fk_sensitive_execution_grants_approval_request"),
            ondelete="RESTRICT",
        ),
        ForeignKeyConstraint(
            ["agent_tool_call_id"],
            ["agent_tool_calls.id"],
            name=("fk_sensitive_execution_grants_agent_tool_call"),
            ondelete="RESTRICT",
        ),
        UniqueConstraint(
            "approval_request_id",
            name=("uq_sensitive_execution_grants_approval_request"),
        ),
        UniqueConstraint(
            "agent_tool_call_id",
            name=("uq_sensitive_execution_grants_agent_tool_call"),
        ),
        UniqueConstraint(
            "workspace_id",
            "id",
            name=("uq_sensitive_execution_grants_workspace_id"),
        ),
        Index(
            "ix_sensitive_execution_grants_workspace_created_id",
            "workspace_id",
            created_at.desc(),
            id.desc(),
        ),
        Index(
            "ix_sensitive_execution_grants_agent_run",
            "agent_run_id",
            "created_at",
            "id",
        ),
    )

    @classmethod
    def from_domain(
        cls,
        grant: SensitiveExecutionGrant,
    ) -> "SensitiveExecutionGrantRecord":
        """Create one persistence record from the domain."""

        return cls(
            id=grant.id,
            workspace_id=grant.workspace_id,
            ticket_id=grant.ticket_id,
            agent_run_id=grant.agent_run_id,
            executed_by_agent_run_attempt_id=(grant.executed_by_agent_run_attempt_id),
            approval_request_id=grant.approval_request_id,
            agent_tool_call_id=grant.agent_tool_call_id,
            tool_name=grant.tool_name,
            tool_version=grant.tool_version,
            safety_level=grant.safety_level.value,
            input_fingerprint=grant.input_fingerprint,
            granted_input=dict(grant.granted_input),
            decision_actor_reference=(grant.decision_actor_reference),
            decision_request_id=grant.decision_request_id,
            decision_correlation_id=(grant.decision_correlation_id),
            approved_at=grant.approved_at,
            created_at=grant.created_at,
        )

    def to_domain(self) -> SensitiveExecutionGrant:
        """Reconstruct one immutable domain grant."""

        return SensitiveExecutionGrant(
            id=self.id,
            workspace_id=self.workspace_id,
            ticket_id=self.ticket_id,
            agent_run_id=self.agent_run_id,
            executed_by_agent_run_attempt_id=(self.executed_by_agent_run_attempt_id),
            approval_request_id=self.approval_request_id,
            agent_tool_call_id=self.agent_tool_call_id,
            tool_name=self.tool_name,
            tool_version=self.tool_version,
            safety_level=ToolSafetyLevel(self.safety_level),
            input_fingerprint=self.input_fingerprint,
            granted_input=dict(self.granted_input),
            decision_actor_reference=(self.decision_actor_reference),
            decision_request_id=self.decision_request_id,
            decision_correlation_id=(self.decision_correlation_id),
            approved_at=self.approved_at,
            created_at=self.created_at,
        )
