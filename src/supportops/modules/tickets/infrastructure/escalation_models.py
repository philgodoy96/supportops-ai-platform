"""SQLAlchemy model for immutable ticket escalations."""

from datetime import datetime
from uuid import UUID

from sqlalchemy import (
    CheckConstraint,
    DateTime,
    ForeignKeyConstraint,
    Index,
    String,
    UniqueConstraint,
    Uuid,
)
from sqlalchemy.orm import Mapped, mapped_column

from supportops.agent_tools.tools.escalate_ticket import (
    TicketEscalationTargetQueue,
)
from supportops.infrastructure.postgresql.base import Base
from supportops.modules.tickets.domain.escalation import (
    TicketEscalation,
)


class TicketEscalationRecord(Base):
    """Persistence record for one immutable internal escalation."""

    __tablename__ = "ticket_escalations"

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
    target_queue: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
    )
    reason: Mapped[str] = mapped_column(
        String(1000),
        nullable=False,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
    )

    __table_args__ = (
        CheckConstraint(
            (
                "target_queue = btrim(target_queue) "
                "AND char_length(target_queue) BETWEEN 1 AND 64 "
                "AND target_queue ~ '^[a-z][a-z0-9_]*$'"
            ),
            name="target_queue_format",
        ),
        CheckConstraint(
            ("reason = btrim(reason) AND char_length(reason) BETWEEN 1 AND 1000"),
            name="reason_format",
        ),
        ForeignKeyConstraint(
            ["workspace_id", "ticket_id"],
            ["tickets.workspace_id", "tickets.id"],
            name="fk_ticket_escalations_workspace_ticket",
            ondelete="CASCADE",
        ),
        ForeignKeyConstraint(
            ["workspace_id", "ticket_id", "agent_run_id"],
            [
                "agent_runs.workspace_id",
                "agent_runs.ticket_id",
                "agent_runs.id",
            ],
            name=("fk_ticket_escalations_workspace_ticket_agent_run"),
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
            name="fk_ticket_escalations_execution_attempt",
            ondelete="RESTRICT",
        ),
        ForeignKeyConstraint(
            ["approval_request_id"],
            ["approval_requests.id"],
            name="fk_ticket_escalations_approval_request",
            ondelete="RESTRICT",
        ),
        ForeignKeyConstraint(
            ["agent_tool_call_id"],
            ["agent_tool_calls.id"],
            name="fk_ticket_escalations_agent_tool_call",
            ondelete="RESTRICT",
        ),
        UniqueConstraint(
            "approval_request_id",
            name="uq_ticket_escalations_approval_request",
        ),
        UniqueConstraint(
            "agent_tool_call_id",
            name="uq_ticket_escalations_agent_tool_call",
        ),
        UniqueConstraint(
            "workspace_id",
            "id",
            name="uq_ticket_escalations_workspace_id",
        ),
        Index(
            "ix_ticket_escalations_workspace_created_id",
            "workspace_id",
            created_at.desc(),
            id.desc(),
        ),
        Index(
            "ix_ticket_escalations_ticket_created_id",
            "workspace_id",
            "ticket_id",
            created_at.desc(),
            id.desc(),
        ),
    )

    @classmethod
    def from_domain(
        cls,
        escalation: TicketEscalation,
    ) -> "TicketEscalationRecord":
        """Create a persistence record from the domain."""

        return cls(
            id=escalation.id,
            workspace_id=escalation.workspace_id,
            ticket_id=escalation.ticket_id,
            agent_run_id=escalation.agent_run_id,
            executed_by_agent_run_attempt_id=(escalation.executed_by_agent_run_attempt_id),
            approval_request_id=escalation.approval_request_id,
            agent_tool_call_id=escalation.agent_tool_call_id,
            target_queue=escalation.target_queue.value,
            reason=escalation.reason,
            created_at=escalation.created_at,
        )

    def to_domain(self) -> TicketEscalation:
        """Reconstruct the immutable domain record."""

        return TicketEscalation(
            id=self.id,
            workspace_id=self.workspace_id,
            ticket_id=self.ticket_id,
            agent_run_id=self.agent_run_id,
            executed_by_agent_run_attempt_id=(self.executed_by_agent_run_attempt_id),
            approval_request_id=self.approval_request_id,
            agent_tool_call_id=self.agent_tool_call_id,
            target_queue=TicketEscalationTargetQueue(
                self.target_queue,
            ),
            reason=self.reason,
            created_at=self.created_at,
        )
