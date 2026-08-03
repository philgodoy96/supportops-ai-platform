"""add AgentRun waiting-for-approval lifecycle

Revision ID: b7c4d2e9a1f6
Revises: f3a9c1d7e5b2
Create Date: 2026-08-02 22:00:00+00:00
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "b7c4d2e9a1f6"
down_revision: str | Sequence[str] | None = "f3a9c1d7e5b2"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Add the lease-free human-approval waiting lifecycle."""
    op.drop_constraint(
        op.f("ck_agent_runs_agent_run_status"),
        "agent_runs",
        type_="check",
    )
    op.drop_constraint(
        op.f("ck_agent_runs_agent_run_lifecycle_timestamp_order"),
        "agent_runs",
        type_="check",
    )
    op.drop_constraint(
        op.f("ck_agent_runs_agent_run_success_error_state"),
        "agent_runs",
        type_="check",
    )
    op.drop_constraint(
        op.f("ck_agent_run_attempts_agent_run_attempt_outcome"),
        "agent_run_attempts",
        type_="check",
    )
    op.drop_constraint(
        op.f("ck_agent_run_attempts_agent_run_attempt_failure_error_state"),
        "agent_run_attempts",
        type_="check",
    )
    op.drop_constraint(
        op.f("ck_agent_run_attempts_agent_run_attempt_success_error_state"),
        "agent_run_attempts",
        type_="check",
    )

    op.alter_column(
        "agent_runs",
        "available_at",
        existing_type=sa.DateTime(timezone=True),
        existing_nullable=False,
        nullable=True,
    )

    op.create_check_constraint(
        op.f("ck_agent_runs_agent_run_status"),
        "agent_runs",
        (
            "status IN ("
            "'queued', "
            "'running', "
            "'retry_scheduled', "
            "'waiting_for_approval', "
            "'succeeded', "
            "'failed'"
            ")"
        ),
    )
    op.create_check_constraint(
        op.f("ck_agent_runs_agent_run_lifecycle_timestamp_order"),
        "agent_runs",
        (
            "(available_at IS NULL OR available_at >= created_at) "
            "AND (first_started_at IS NULL OR first_started_at >= created_at) "
            "AND (completed_at IS NULL OR completed_at >= created_at) "
            "AND updated_at >= created_at"
        ),
    )
    op.create_check_constraint(
        op.f("ck_agent_runs_agent_run_available_at_state"),
        "agent_runs",
        (
            "(status = 'waiting_for_approval' AND available_at IS NULL) "
            "OR "
            "(status <> 'waiting_for_approval' AND available_at IS NOT NULL)"
        ),
    )
    op.create_check_constraint(
        op.f("ck_agent_runs_agent_run_success_error_state"),
        "agent_runs",
        (
            "status NOT IN ('queued', 'waiting_for_approval', 'succeeded') "
            "OR "
            "(last_error_code IS NULL AND last_error_summary IS NULL)"
        ),
    )

    op.create_check_constraint(
        op.f("ck_agent_run_attempts_agent_run_attempt_outcome"),
        "agent_run_attempts",
        (
            "outcome IS NULL OR outcome IN ("
            "'succeeded', "
            "'awaiting_approval', "
            "'retryable_failure', "
            "'terminal_failure', "
            "'timed_out', "
            "'lease_expired'"
            ")"
        ),
    )
    op.create_check_constraint(
        op.f("ck_agent_run_attempts_agent_run_attempt_failure_error_state"),
        "agent_run_attempts",
        (
            "outcome IS NULL "
            "OR outcome IN ('succeeded', 'awaiting_approval') "
            "OR (error_code IS NOT NULL AND error_summary IS NOT NULL)"
        ),
    )
    op.create_check_constraint(
        op.f("ck_agent_run_attempts_agent_run_attempt_success_error_state"),
        "agent_run_attempts",
        (
            "outcome NOT IN ('succeeded', 'awaiting_approval') "
            "OR (error_code IS NULL AND error_summary IS NULL)"
        ),
    )


def downgrade() -> None:
    """Restore the pre-approval AgentRun lifecycle."""
    op.execute(
        sa.text(
            """
            DO $$
            BEGIN
                IF EXISTS (
                    SELECT 1
                    FROM agent_runs
                    WHERE status = 'waiting_for_approval'
                ) THEN
                    RAISE EXCEPTION
                        'Cannot downgrade while AgentRuns are waiting for approval';
                END IF;

                IF EXISTS (
                    SELECT 1
                    FROM agent_run_attempts
                    WHERE outcome = 'awaiting_approval'
                ) THEN
                    RAISE EXCEPTION
                        'Cannot downgrade while awaiting-approval attempts exist';
                END IF;
            END
            $$;
            """
        )
    )

    op.drop_constraint(
        op.f("ck_agent_run_attempts_agent_run_attempt_success_error_state"),
        "agent_run_attempts",
        type_="check",
    )
    op.drop_constraint(
        op.f("ck_agent_run_attempts_agent_run_attempt_failure_error_state"),
        "agent_run_attempts",
        type_="check",
    )
    op.drop_constraint(
        op.f("ck_agent_run_attempts_agent_run_attempt_outcome"),
        "agent_run_attempts",
        type_="check",
    )
    op.drop_constraint(
        op.f("ck_agent_runs_agent_run_success_error_state"),
        "agent_runs",
        type_="check",
    )
    op.drop_constraint(
        op.f("ck_agent_runs_agent_run_available_at_state"),
        "agent_runs",
        type_="check",
    )
    op.drop_constraint(
        op.f("ck_agent_runs_agent_run_lifecycle_timestamp_order"),
        "agent_runs",
        type_="check",
    )
    op.drop_constraint(
        op.f("ck_agent_runs_agent_run_status"),
        "agent_runs",
        type_="check",
    )

    op.alter_column(
        "agent_runs",
        "available_at",
        existing_type=sa.DateTime(timezone=True),
        existing_nullable=True,
        nullable=False,
    )

    op.create_check_constraint(
        op.f("ck_agent_runs_agent_run_status"),
        "agent_runs",
        ("status IN ('queued', 'running', 'retry_scheduled', 'succeeded', 'failed')"),
    )
    op.create_check_constraint(
        op.f("ck_agent_runs_agent_run_lifecycle_timestamp_order"),
        "agent_runs",
        (
            "available_at >= created_at "
            "AND (first_started_at IS NULL OR first_started_at >= created_at) "
            "AND (completed_at IS NULL OR completed_at >= created_at) "
            "AND updated_at >= created_at"
        ),
    )
    op.create_check_constraint(
        op.f("ck_agent_runs_agent_run_success_error_state"),
        "agent_runs",
        (
            "status NOT IN ('queued', 'succeeded') "
            "OR "
            "(last_error_code IS NULL AND last_error_summary IS NULL)"
        ),
    )
    op.create_check_constraint(
        op.f("ck_agent_run_attempts_agent_run_attempt_outcome"),
        "agent_run_attempts",
        (
            "outcome IS NULL OR outcome IN ("
            "'succeeded', "
            "'retryable_failure', "
            "'terminal_failure', "
            "'timed_out', "
            "'lease_expired'"
            ")"
        ),
    )
    op.create_check_constraint(
        op.f("ck_agent_run_attempts_agent_run_attempt_failure_error_state"),
        "agent_run_attempts",
        (
            "outcome IS NULL "
            "OR outcome = 'succeeded' "
            "OR (error_code IS NOT NULL AND error_summary IS NOT NULL)"
        ),
    )
    op.create_check_constraint(
        op.f("ck_agent_run_attempts_agent_run_attempt_success_error_state"),
        "agent_run_attempts",
        ("outcome <> 'succeeded' OR (error_code IS NULL AND error_summary IS NULL)"),
    )
