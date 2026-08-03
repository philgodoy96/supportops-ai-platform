"""separate retryable failure budget

Revision ID: f3a9c1d7e5b2
Revises: e8b7c6d5a4f3
Create Date: 2026-08-02 21:45:00+00:00
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "f3a9c1d7e5b2"
down_revision: str | Sequence[str] | None = "e8b7c6d5a4f3"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Separate total execution attempts from retryable failure capacity."""
    op.drop_constraint(
        op.f("ck_agent_runs_agent_run_attempt_limit"),
        "agent_runs",
        type_="check",
    )
    op.drop_constraint(
        op.f("ck_agent_runs_agent_run_max_attempts_positive"),
        "agent_runs",
        type_="check",
    )

    op.alter_column(
        "agent_runs",
        "max_attempts",
        existing_type=sa.Integer(),
        existing_nullable=False,
        new_column_name="max_retryable_failures",
    )
    op.add_column(
        "agent_runs",
        sa.Column(
            "retryable_failure_count",
            sa.Integer(),
            nullable=True,
        ),
    )

    op.execute(
        sa.text(
            """
            UPDATE agent_runs AS runs
            SET retryable_failure_count = COALESCE(
                (
                    SELECT COUNT(*)::integer
                    FROM agent_run_attempts AS attempts
                    WHERE attempts.agent_run_id = runs.id
                      AND attempts.outcome IN (
                          'retryable_failure',
                          'timed_out',
                          'lease_expired'
                      )
                ),
                0
            )
            """
        )
    )

    op.alter_column(
        "agent_runs",
        "retryable_failure_count",
        existing_type=sa.Integer(),
        nullable=False,
    )

    op.create_check_constraint(
        op.f("ck_agent_runs_agent_run_max_retryable_failures_positive"),
        "agent_runs",
        "max_retryable_failures >= 1",
    )
    op.create_check_constraint(
        op.f("ck_agent_runs_agent_run_retryable_failure_count_non_negative"),
        "agent_runs",
        "retryable_failure_count >= 0",
    )
    op.create_check_constraint(
        op.f("ck_agent_runs_agent_run_retryable_failure_limit"),
        "agent_runs",
        "retryable_failure_count <= max_retryable_failures",
    )


def downgrade() -> None:
    """Restore the previous attempt-budget representation."""
    op.drop_constraint(
        op.f("ck_agent_runs_agent_run_retryable_failure_limit"),
        "agent_runs",
        type_="check",
    )
    op.drop_constraint(
        op.f("ck_agent_runs_agent_run_retryable_failure_count_non_negative"),
        "agent_runs",
        type_="check",
    )
    op.drop_constraint(
        op.f("ck_agent_runs_agent_run_max_retryable_failures_positive"),
        "agent_runs",
        type_="check",
    )

    op.execute(
        sa.text(
            """
            UPDATE agent_runs
            SET max_retryable_failures = GREATEST(
                max_retryable_failures,
                attempt_count
            )
            """
        )
    )

    op.drop_column(
        "agent_runs",
        "retryable_failure_count",
    )
    op.alter_column(
        "agent_runs",
        "max_retryable_failures",
        existing_type=sa.Integer(),
        existing_nullable=False,
        new_column_name="max_attempts",
    )

    op.create_check_constraint(
        op.f("ck_agent_runs_agent_run_max_attempts_positive"),
        "agent_runs",
        "max_attempts >= 1",
    )
    op.create_check_constraint(
        op.f("ck_agent_runs_agent_run_attempt_limit"),
        "agent_runs",
        "attempt_count <= max_attempts",
    )
