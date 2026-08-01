"""SQLAlchemy persistence models for durable AgentRun processing."""

from datetime import datetime
from uuid import UUID

from sqlalchemy import (
    CheckConstraint,
    DateTime,
    ForeignKey,
    ForeignKeyConstraint,
    Index,
    Integer,
    String,
    UniqueConstraint,
    Uuid,
    text,
)
from sqlalchemy.orm import Mapped, mapped_column

from supportops.infrastructure.postgresql.base import Base
from supportops.modules.agent_runs.domain.models import (
    AGENT_RUN_ATTEMPT_WORKER_ID_MAX_LENGTH,
    AGENT_RUN_ERROR_CODE_MAX_LENGTH,
    AGENT_RUN_ERROR_SUMMARY_MAX_LENGTH,
    AGENT_RUN_LEASE_OWNER_MAX_LENGTH,
    AGENT_RUN_TRIGGER_KEY_MAX_LENGTH,
    AGENT_RUN_WORKFLOW_NAME_MAX_LENGTH,
    AGENT_RUN_WORKFLOW_VERSION_MAX_LENGTH,
    AgentRun,
    AgentRunAttempt,
    AgentRunAttemptOutcome,
    AgentRunStatus,
)


class AgentRunRecord(Base):
    """Persisted durable execution state for a support ticket."""

    __tablename__ = "agent_runs"

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
    workflow_name: Mapped[str] = mapped_column(
        String(AGENT_RUN_WORKFLOW_NAME_MAX_LENGTH),
        nullable=False,
    )
    workflow_version: Mapped[str] = mapped_column(
        String(AGENT_RUN_WORKFLOW_VERSION_MAX_LENGTH),
        nullable=False,
    )
    trigger_key: Mapped[str] = mapped_column(
        String(AGENT_RUN_TRIGGER_KEY_MAX_LENGTH),
        nullable=False,
    )
    status: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
    )
    available_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
    )
    attempt_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )
    max_attempts: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )
    lease_owner: Mapped[str | None] = mapped_column(
        String(AGENT_RUN_LEASE_OWNER_MAX_LENGTH),
        nullable=True,
    )
    lease_token: Mapped[UUID | None] = mapped_column(
        Uuid(as_uuid=True),
        nullable=True,
    )
    lease_expires_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    first_started_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    completed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    last_error_code: Mapped[str | None] = mapped_column(
        String(AGENT_RUN_ERROR_CODE_MAX_LENGTH),
        nullable=True,
    )
    last_error_summary: Mapped[str | None] = mapped_column(
        String(AGENT_RUN_ERROR_SUMMARY_MAX_LENGTH),
        nullable=True,
    )
    ingestion_request_id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True),
        nullable=False,
    )
    correlation_id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True),
        nullable=False,
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
            ["workspace_id", "ticket_id"],
            ["tickets.workspace_id", "tickets.id"],
            name="fk_agent_runs_workspace_ticket",
            ondelete="RESTRICT",
        ),
        UniqueConstraint(
            "workspace_id",
            "ticket_id",
            "id",
            name="uq_agent_runs_workspace_ticket_id",
        ),
        UniqueConstraint(
            "ticket_id",
            "trigger_key",
            name="uq_agent_runs_ticket_trigger",
        ),
        CheckConstraint(
            (
                "workflow_name = btrim(workflow_name) "
                "AND char_length(workflow_name) BETWEEN 1 AND 64"
            ),
            name="agent_run_workflow_name_format",
        ),
        CheckConstraint(
            (
                "workflow_version = btrim(workflow_version) "
                "AND char_length(workflow_version) BETWEEN 1 AND 64"
            ),
            name="agent_run_workflow_version_format",
        ),
        CheckConstraint(
            ("trigger_key = btrim(trigger_key) AND char_length(trigger_key) BETWEEN 1 AND 64"),
            name="agent_run_trigger_key_format",
        ),
        CheckConstraint(
            ("status IN ('queued', 'running', 'retry_scheduled', 'succeeded', 'failed')"),
            name="agent_run_status",
        ),
        CheckConstraint(
            "attempt_count >= 0",
            name="agent_run_attempt_count_non_negative",
        ),
        CheckConstraint(
            "max_attempts >= 1",
            name="agent_run_max_attempts_positive",
        ),
        CheckConstraint(
            "attempt_count <= max_attempts",
            name="agent_run_attempt_limit",
        ),
        CheckConstraint(
            (
                "("
                "attempt_count = 0 "
                "AND first_started_at IS NULL"
                ") OR ("
                "attempt_count > 0 "
                "AND first_started_at IS NOT NULL"
                ")"
            ),
            name="agent_run_started_attempt_state",
        ),
        CheckConstraint(
            (
                "("
                "lease_owner IS NULL "
                "AND lease_token IS NULL "
                "AND lease_expires_at IS NULL"
                ") OR ("
                "lease_owner IS NOT NULL "
                "AND lease_token IS NOT NULL "
                "AND lease_expires_at IS NOT NULL"
                ")"
            ),
            name="agent_run_lease_fields_complete",
        ),
        CheckConstraint(
            (
                "("
                "status = 'running' "
                "AND lease_owner IS NOT NULL "
                "AND lease_token IS NOT NULL "
                "AND lease_expires_at IS NOT NULL"
                ") OR ("
                "status <> 'running' "
                "AND lease_owner IS NULL "
                "AND lease_token IS NULL "
                "AND lease_expires_at IS NULL"
                ")"
            ),
            name="agent_run_lease_state",
        ),
        CheckConstraint(
            ("lease_expires_at IS NULL OR lease_expires_at > first_started_at"),
            name="agent_run_lease_expiration_order",
        ),
        CheckConstraint(
            (
                "("
                "status IN ('succeeded', 'failed') "
                "AND completed_at IS NOT NULL"
                ") OR ("
                "status NOT IN ('succeeded', 'failed') "
                "AND completed_at IS NULL"
                ")"
            ),
            name="agent_run_completion_state",
        ),
        CheckConstraint(
            (
                "("
                "last_error_code IS NULL "
                "AND last_error_summary IS NULL"
                ") OR ("
                "last_error_code IS NOT NULL "
                "AND last_error_summary IS NOT NULL"
                ")"
            ),
            name="agent_run_error_fields_complete",
        ),
        CheckConstraint(
            (
                "last_error_code IS NULL OR ("
                "last_error_code = btrim(last_error_code) "
                "AND char_length(last_error_code) BETWEEN 1 AND 64"
                ")"
            ),
            name="agent_run_error_code_format",
        ),
        CheckConstraint(
            (
                "last_error_summary IS NULL OR ("
                "last_error_summary = btrim(last_error_summary) "
                "AND char_length(last_error_summary) BETWEEN 1 AND 512"
                ")"
            ),
            name="agent_run_error_summary_format",
        ),
        CheckConstraint(
            ("status NOT IN ('queued', 'succeeded') OR last_error_code IS NULL"),
            name="agent_run_success_error_state",
        ),
        CheckConstraint(
            ("status NOT IN ('retry_scheduled', 'failed') OR last_error_code IS NOT NULL"),
            name="agent_run_failure_error_state",
        ),
        CheckConstraint(
            "updated_at >= created_at",
            name="agent_run_timestamp_order",
        ),
        CheckConstraint(
            (
                "available_at >= created_at "
                "AND ("
                "first_started_at IS NULL "
                "OR first_started_at >= created_at"
                ") "
                "AND ("
                "completed_at IS NULL "
                "OR completed_at >= created_at"
                ")"
            ),
            name="agent_run_lifecycle_timestamp_order",
        ),
        CheckConstraint(
            (
                "lease_owner IS NULL OR ("
                "lease_owner = btrim(lease_owner) "
                "AND char_length(lease_owner) BETWEEN 1 AND 128"
                ")"
            ),
            name="agent_run_lease_owner_format",
        ),
        Index(
            "ix_agent_runs_available_claim",
            "available_at",
            "created_at",
            "id",
            postgresql_where=text("status IN ('queued', 'retry_scheduled')"),
        ),
        Index(
            "ix_agent_runs_expired_lease",
            "lease_expires_at",
            "created_at",
            "id",
            postgresql_where=text("status = 'running'"),
        ),
        Index(
            "ix_agent_runs_workspace_ticket_created_id",
            "workspace_id",
            "ticket_id",
            created_at.desc(),
            id.desc(),
        ),
    )

    @classmethod
    def from_domain(
        cls,
        agent_run: AgentRun,
    ) -> "AgentRunRecord":
        """Create a persistence record from an AgentRun entity."""

        return cls(
            id=agent_run.id,
            workspace_id=agent_run.workspace_id,
            ticket_id=agent_run.ticket_id,
            workflow_name=agent_run.workflow_name,
            workflow_version=agent_run.workflow_version,
            trigger_key=agent_run.trigger_key,
            status=agent_run.status.value,
            available_at=agent_run.available_at,
            attempt_count=agent_run.attempt_count,
            max_attempts=agent_run.max_attempts,
            lease_owner=agent_run.lease_owner,
            lease_token=agent_run.lease_token,
            lease_expires_at=agent_run.lease_expires_at,
            first_started_at=agent_run.first_started_at,
            completed_at=agent_run.completed_at,
            last_error_code=agent_run.last_error_code,
            last_error_summary=agent_run.last_error_summary,
            ingestion_request_id=agent_run.ingestion_request_id,
            correlation_id=agent_run.correlation_id,
            created_at=agent_run.created_at,
            updated_at=agent_run.updated_at,
        )

    def to_domain(self) -> AgentRun:
        """Map the persistence record to an AgentRun entity."""

        return AgentRun(
            id=self.id,
            workspace_id=self.workspace_id,
            ticket_id=self.ticket_id,
            workflow_name=self.workflow_name,
            workflow_version=self.workflow_version,
            trigger_key=self.trigger_key,
            status=AgentRunStatus(self.status),
            available_at=self.available_at,
            attempt_count=self.attempt_count,
            max_attempts=self.max_attempts,
            lease_owner=self.lease_owner,
            lease_token=self.lease_token,
            lease_expires_at=self.lease_expires_at,
            first_started_at=self.first_started_at,
            completed_at=self.completed_at,
            last_error_code=self.last_error_code,
            last_error_summary=self.last_error_summary,
            ingestion_request_id=self.ingestion_request_id,
            correlation_id=self.correlation_id,
            created_at=self.created_at,
            updated_at=self.updated_at,
        )


class AgentRunAttemptRecord(Base):
    """Persisted history for one claimed AgentRun attempt."""

    __tablename__ = "agent_run_attempts"

    id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True),
        primary_key=True,
    )
    agent_run_id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey(
            "agent_runs.id",
            ondelete="CASCADE",
        ),
        nullable=False,
    )
    attempt_number: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )
    worker_id: Mapped[str] = mapped_column(
        String(AGENT_RUN_ATTEMPT_WORKER_ID_MAX_LENGTH),
        nullable=False,
    )
    lease_token: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True),
        nullable=False,
    )
    execution_request_id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True),
        nullable=False,
    )
    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
    )
    finished_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    outcome: Mapped[str | None] = mapped_column(
        String(32),
        nullable=True,
    )
    error_code: Mapped[str | None] = mapped_column(
        String(AGENT_RUN_ERROR_CODE_MAX_LENGTH),
        nullable=True,
    )
    error_summary: Mapped[str | None] = mapped_column(
        String(AGENT_RUN_ERROR_SUMMARY_MAX_LENGTH),
        nullable=True,
    )

    __table_args__ = (
        UniqueConstraint(
            "agent_run_id",
            "id",
            name="uq_agent_run_attempts_run_id",
        ),
        UniqueConstraint(
            "agent_run_id",
            "attempt_number",
            name="uq_agent_run_attempts_run_number",
        ),
        CheckConstraint(
            "attempt_number >= 1",
            name="agent_run_attempt_number_positive",
        ),
        CheckConstraint(
            ("worker_id = btrim(worker_id) AND char_length(worker_id) BETWEEN 1 AND 128"),
            name="agent_run_attempt_worker_id_format",
        ),
        CheckConstraint(
            (
                "outcome IS NULL OR outcome IN ("
                "'succeeded', "
                "'retryable_failure', "
                "'terminal_failure', "
                "'timed_out', "
                "'lease_expired'"
                ")"
            ),
            name="agent_run_attempt_outcome",
        ),
        CheckConstraint(
            (
                "("
                "finished_at IS NULL "
                "AND outcome IS NULL"
                ") OR ("
                "finished_at IS NOT NULL "
                "AND outcome IS NOT NULL"
                ")"
            ),
            name="agent_run_attempt_completion_state",
        ),
        CheckConstraint(
            ("finished_at IS NULL OR finished_at >= started_at"),
            name="agent_run_attempt_timestamp_order",
        ),
        CheckConstraint(
            (
                "("
                "error_code IS NULL "
                "AND error_summary IS NULL"
                ") OR ("
                "error_code IS NOT NULL "
                "AND error_summary IS NOT NULL"
                ")"
            ),
            name="agent_run_attempt_error_fields_complete",
        ),
        CheckConstraint(
            (
                "error_code IS NULL OR ("
                "error_code = btrim(error_code) "
                "AND char_length(error_code) BETWEEN 1 AND 64"
                ")"
            ),
            name="agent_run_attempt_error_code_format",
        ),
        CheckConstraint(
            (
                "error_summary IS NULL OR ("
                "error_summary = btrim(error_summary) "
                "AND char_length(error_summary) BETWEEN 1 AND 512"
                ")"
            ),
            name="agent_run_attempt_error_summary_format",
        ),
        CheckConstraint(
            ("outcome <> 'succeeded' OR error_code IS NULL"),
            name="agent_run_attempt_success_error_state",
        ),
        CheckConstraint(
            ("outcome IS NULL OR outcome = 'succeeded' OR error_code IS NOT NULL"),
            name="agent_run_attempt_failure_error_state",
        ),
        CheckConstraint(
            "outcome IS NOT NULL OR error_code IS NULL",
            name="agent_run_attempt_active_error_state",
        ),
    )

    @classmethod
    def from_domain(
        cls,
        attempt: AgentRunAttempt,
    ) -> "AgentRunAttemptRecord":
        """Create a persistence record from an attempt entity."""

        return cls(
            id=attempt.id,
            agent_run_id=attempt.agent_run_id,
            attempt_number=attempt.attempt_number,
            worker_id=attempt.worker_id,
            lease_token=attempt.lease_token,
            execution_request_id=attempt.execution_request_id,
            started_at=attempt.started_at,
            finished_at=attempt.finished_at,
            outcome=(attempt.outcome.value if attempt.outcome is not None else None),
            error_code=attempt.error_code,
            error_summary=attempt.error_summary,
        )

    def to_domain(self) -> AgentRunAttempt:
        """Map the persistence record to an attempt entity."""

        return AgentRunAttempt(
            id=self.id,
            agent_run_id=self.agent_run_id,
            attempt_number=self.attempt_number,
            worker_id=self.worker_id,
            lease_token=self.lease_token,
            execution_request_id=self.execution_request_id,
            started_at=self.started_at,
            finished_at=self.finished_at,
            outcome=(AgentRunAttemptOutcome(self.outcome) if self.outcome is not None else None),
            error_code=self.error_code,
            error_summary=self.error_summary,
        )
