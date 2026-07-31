"""create agent run tables

Revision ID: a2c4e6f8091b
Revises: 6d9f0a2b4c31
Create Date: 2026-07-31 17:20:00+00:00

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "a2c4e6f8091b"
down_revision: str | Sequence[str] | None = "6d9f0a2b4c31"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Apply the migration."""

    op.create_unique_constraint(
        "uq_tickets_workspace_id",
        "tickets",
        ["workspace_id", "id"],
    )

    op.create_table(
        "agent_runs",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("workspace_id", sa.Uuid(), nullable=False),
        sa.Column("ticket_id", sa.Uuid(), nullable=False),
        sa.Column(
            "workflow_name",
            sa.String(length=64),
            nullable=False,
        ),
        sa.Column(
            "workflow_version",
            sa.String(length=64),
            nullable=False,
        ),
        sa.Column(
            "trigger_key",
            sa.String(length=64),
            nullable=False,
        ),
        sa.Column(
            "status",
            sa.String(length=32),
            nullable=False,
        ),
        sa.Column(
            "available_at",
            sa.DateTime(timezone=True),
            nullable=False,
        ),
        sa.Column(
            "attempt_count",
            sa.Integer(),
            nullable=False,
        ),
        sa.Column(
            "max_attempts",
            sa.Integer(),
            nullable=False,
        ),
        sa.Column(
            "lease_owner",
            sa.String(length=128),
            nullable=True,
        ),
        sa.Column(
            "lease_token",
            sa.Uuid(),
            nullable=True,
        ),
        sa.Column(
            "lease_expires_at",
            sa.DateTime(timezone=True),
            nullable=True,
        ),
        sa.Column(
            "first_started_at",
            sa.DateTime(timezone=True),
            nullable=True,
        ),
        sa.Column(
            "completed_at",
            sa.DateTime(timezone=True),
            nullable=True,
        ),
        sa.Column(
            "last_error_code",
            sa.String(length=64),
            nullable=True,
        ),
        sa.Column(
            "last_error_summary",
            sa.String(length=512),
            nullable=True,
        ),
        sa.Column(
            "ingestion_request_id",
            sa.Uuid(),
            nullable=False,
        ),
        sa.Column(
            "correlation_id",
            sa.Uuid(),
            nullable=False,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
        ),
        sa.CheckConstraint(
            (
                "workflow_name = btrim(workflow_name) "
                "AND char_length(workflow_name) BETWEEN 1 AND 64"
            ),
            name=op.f(
                "ck_agent_runs_agent_run_workflow_name_format",
            ),
        ),
        sa.CheckConstraint(
            (
                "workflow_version = btrim(workflow_version) "
                "AND char_length(workflow_version) BETWEEN 1 AND 64"
            ),
            name=op.f(
                "ck_agent_runs_agent_run_workflow_version_format",
            ),
        ),
        sa.CheckConstraint(
            ("trigger_key = btrim(trigger_key) AND char_length(trigger_key) BETWEEN 1 AND 64"),
            name=op.f(
                "ck_agent_runs_agent_run_trigger_key_format",
            ),
        ),
        sa.CheckConstraint(
            ("status IN ('queued', 'running', 'retry_scheduled', 'succeeded', 'failed')"),
            name=op.f("ck_agent_runs_agent_run_status"),
        ),
        sa.CheckConstraint(
            "attempt_count >= 0",
            name=op.f(
                "ck_agent_runs_agent_run_attempt_count_non_negative",
            ),
        ),
        sa.CheckConstraint(
            "max_attempts >= 1",
            name=op.f(
                "ck_agent_runs_agent_run_max_attempts_positive",
            ),
        ),
        sa.CheckConstraint(
            "attempt_count <= max_attempts",
            name=op.f(
                "ck_agent_runs_agent_run_attempt_limit",
            ),
        ),
        sa.CheckConstraint(
            (
                "("
                "attempt_count = 0 "
                "AND first_started_at IS NULL"
                ") OR ("
                "attempt_count > 0 "
                "AND first_started_at IS NOT NULL"
                ")"
            ),
            name=op.f(
                "ck_agent_runs_agent_run_started_attempt_state",
            ),
        ),
        sa.CheckConstraint(
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
            name=op.f(
                "ck_agent_runs_agent_run_lease_fields_complete",
            ),
        ),
        sa.CheckConstraint(
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
            name=op.f(
                "ck_agent_runs_agent_run_lease_state",
            ),
        ),
        sa.CheckConstraint(
            ("lease_expires_at IS NULL OR lease_expires_at > first_started_at"),
            name=op.f(
                "ck_agent_runs_agent_run_lease_expiration_order",
            ),
        ),
        sa.CheckConstraint(
            (
                "("
                "status IN ('succeeded', 'failed') "
                "AND completed_at IS NOT NULL"
                ") OR ("
                "status NOT IN ('succeeded', 'failed') "
                "AND completed_at IS NULL"
                ")"
            ),
            name=op.f(
                "ck_agent_runs_agent_run_completion_state",
            ),
        ),
        sa.CheckConstraint(
            (
                "("
                "last_error_code IS NULL "
                "AND last_error_summary IS NULL"
                ") OR ("
                "last_error_code IS NOT NULL "
                "AND last_error_summary IS NOT NULL"
                ")"
            ),
            name=op.f(
                "ck_agent_runs_agent_run_error_fields_complete",
            ),
        ),
        sa.CheckConstraint(
            (
                "last_error_code IS NULL OR ("
                "last_error_code = btrim(last_error_code) "
                "AND char_length(last_error_code) BETWEEN 1 AND 64"
                ")"
            ),
            name=op.f(
                "ck_agent_runs_agent_run_error_code_format",
            ),
        ),
        sa.CheckConstraint(
            (
                "last_error_summary IS NULL OR ("
                "last_error_summary = btrim(last_error_summary) "
                "AND char_length(last_error_summary) BETWEEN 1 AND 512"
                ")"
            ),
            name=op.f(
                "ck_agent_runs_agent_run_error_summary_format",
            ),
        ),
        sa.CheckConstraint(
            ("status NOT IN ('queued', 'succeeded') OR last_error_code IS NULL"),
            name=op.f(
                "ck_agent_runs_agent_run_success_error_state",
            ),
        ),
        sa.CheckConstraint(
            ("status NOT IN ('retry_scheduled', 'failed') OR last_error_code IS NOT NULL"),
            name=op.f(
                "ck_agent_runs_agent_run_failure_error_state",
            ),
        ),
        sa.CheckConstraint(
            "updated_at >= created_at",
            name=op.f(
                "ck_agent_runs_agent_run_timestamp_order",
            ),
        ),
        sa.CheckConstraint(
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
            name=op.f(
                "ck_agent_runs_agent_run_lifecycle_timestamp_order",
            ),
        ),
        sa.CheckConstraint(
            (
                "lease_owner IS NULL OR ("
                "lease_owner = btrim(lease_owner) "
                "AND char_length(lease_owner) BETWEEN 1 AND 128"
                ")"
            ),
            name=op.f(
                "ck_agent_runs_agent_run_lease_owner_format",
            ),
        ),
        sa.ForeignKeyConstraint(
            ["workspace_id", "ticket_id"],
            ["tickets.workspace_id", "tickets.id"],
            name="fk_agent_runs_workspace_ticket",
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint(
            "id",
            name=op.f("pk_agent_runs"),
        ),
        sa.UniqueConstraint(
            "ticket_id",
            "trigger_key",
            name="uq_agent_runs_ticket_trigger",
        ),
    )

    op.create_index(
        "ix_agent_runs_available_claim",
        "agent_runs",
        [
            "available_at",
            "created_at",
            "id",
        ],
        unique=False,
        postgresql_where=sa.text("status IN ('queued', 'retry_scheduled')"),
    )
    op.create_index(
        "ix_agent_runs_expired_lease",
        "agent_runs",
        [
            "lease_expires_at",
            "created_at",
            "id",
        ],
        unique=False,
        postgresql_where=sa.text("status = 'running'"),
    )
    op.create_index(
        "ix_agent_runs_workspace_ticket_created_id",
        "agent_runs",
        [
            "workspace_id",
            "ticket_id",
            sa.literal_column("created_at DESC"),
            sa.literal_column("id DESC"),
        ],
        unique=False,
    )

    op.create_table(
        "agent_run_attempts",
        sa.Column(
            "id",
            sa.Uuid(),
            nullable=False,
        ),
        sa.Column(
            "agent_run_id",
            sa.Uuid(),
            nullable=False,
        ),
        sa.Column(
            "attempt_number",
            sa.Integer(),
            nullable=False,
        ),
        sa.Column(
            "worker_id",
            sa.String(length=128),
            nullable=False,
        ),
        sa.Column(
            "lease_token",
            sa.Uuid(),
            nullable=False,
        ),
        sa.Column(
            "execution_request_id",
            sa.Uuid(),
            nullable=False,
        ),
        sa.Column(
            "started_at",
            sa.DateTime(timezone=True),
            nullable=False,
        ),
        sa.Column(
            "finished_at",
            sa.DateTime(timezone=True),
            nullable=True,
        ),
        sa.Column(
            "outcome",
            sa.String(length=32),
            nullable=True,
        ),
        sa.Column(
            "error_code",
            sa.String(length=64),
            nullable=True,
        ),
        sa.Column(
            "error_summary",
            sa.String(length=512),
            nullable=True,
        ),
        sa.CheckConstraint(
            "attempt_number >= 1",
            name=op.f(
                "ck_agent_run_attempts_agent_run_attempt_number_positive",
            ),
        ),
        sa.CheckConstraint(
            ("worker_id = btrim(worker_id) AND char_length(worker_id) BETWEEN 1 AND 128"),
            name=op.f(
                "ck_agent_run_attempts_agent_run_attempt_worker_id_format",
            ),
        ),
        sa.CheckConstraint(
            (
                "outcome IS NULL OR outcome IN ("
                "'succeeded', "
                "'retryable_failure', "
                "'terminal_failure', "
                "'timed_out', "
                "'lease_expired'"
                ")"
            ),
            name=op.f(
                "ck_agent_run_attempts_agent_run_attempt_outcome",
            ),
        ),
        sa.CheckConstraint(
            (
                "("
                "finished_at IS NULL "
                "AND outcome IS NULL"
                ") OR ("
                "finished_at IS NOT NULL "
                "AND outcome IS NOT NULL"
                ")"
            ),
            name=op.f(
                "ck_agent_run_attempts_agent_run_attempt_completion_state",
            ),
        ),
        sa.CheckConstraint(
            ("finished_at IS NULL OR finished_at >= started_at"),
            name=op.f(
                "ck_agent_run_attempts_agent_run_attempt_timestamp_order",
            ),
        ),
        sa.CheckConstraint(
            (
                "("
                "error_code IS NULL "
                "AND error_summary IS NULL"
                ") OR ("
                "error_code IS NOT NULL "
                "AND error_summary IS NOT NULL"
                ")"
            ),
            name=op.f(
                "ck_agent_run_attempts_agent_run_attempt_error_fields_complete",
            ),
        ),
        sa.CheckConstraint(
            (
                "error_code IS NULL OR ("
                "error_code = btrim(error_code) "
                "AND char_length(error_code) BETWEEN 1 AND 64"
                ")"
            ),
            name=op.f(
                "ck_agent_run_attempts_agent_run_attempt_error_code_format",
            ),
        ),
        sa.CheckConstraint(
            (
                "error_summary IS NULL OR ("
                "error_summary = btrim(error_summary) "
                "AND char_length(error_summary) BETWEEN 1 AND 512"
                ")"
            ),
            name=op.f(
                "ck_agent_run_attempts_agent_run_attempt_error_summary_format",
            ),
        ),
        sa.CheckConstraint(
            ("outcome <> 'succeeded' OR error_code IS NULL"),
            name=op.f(
                "ck_agent_run_attempts_agent_run_attempt_success_error_state",
            ),
        ),
        sa.CheckConstraint(
            ("outcome IS NULL OR outcome = 'succeeded' OR error_code IS NOT NULL"),
            name=op.f(
                "ck_agent_run_attempts_agent_run_attempt_failure_error_state",
            ),
        ),
        sa.CheckConstraint(
            "outcome IS NOT NULL OR error_code IS NULL",
            name=op.f(
                "ck_agent_run_attempts_agent_run_attempt_active_error_state",
            ),
        ),
        sa.ForeignKeyConstraint(
            ["agent_run_id"],
            ["agent_runs.id"],
            name=op.f(
                "fk_agent_run_attempts_agent_run_id_agent_runs",
            ),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint(
            "id",
            name=op.f("pk_agent_run_attempts"),
        ),
        sa.UniqueConstraint(
            "agent_run_id",
            "attempt_number",
            name="uq_agent_run_attempts_run_number",
        ),
    )


def downgrade() -> None:
    """Revert the migration."""

    op.drop_table("agent_run_attempts")

    op.drop_index(
        "ix_agent_runs_workspace_ticket_created_id",
        table_name="agent_runs",
    )
    op.drop_index(
        "ix_agent_runs_expired_lease",
        table_name="agent_runs",
        postgresql_where=sa.text("status = 'running'"),
    )
    op.drop_index(
        "ix_agent_runs_available_claim",
        table_name="agent_runs",
        postgresql_where=sa.text("status IN ('queued', 'retry_scheduled')"),
    )

    op.drop_table("agent_runs")

    op.drop_constraint(
        "uq_tickets_workspace_id",
        "tickets",
        type_="unique",
    )
