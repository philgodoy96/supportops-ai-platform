"""evolve tool calls for approval-backed execution

Revision ID: c9e2f4a7b6d1
Revises: b7c4d2e9a1f6
Create Date: 2026-08-02 22:20:00+00:00
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "c9e2f4a7b6d1"
down_revision: str | Sequence[str] | None = "b7c4d2e9a1f6"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Allow tool calls to span proposal and execution attempts."""
    op.drop_constraint(
        "fk_agent_tool_calls_agent_run_attempt",
        "agent_tool_calls",
        type_="foreignkey",
    )
    op.drop_constraint(
        "uq_agent_tool_calls_attempt_sequence",
        "agent_tool_calls",
        type_="unique",
    )
    op.drop_constraint(
        "uq_agent_tool_calls_attempt_provider_call",
        "agent_tool_calls",
        type_="unique",
    )
    op.drop_constraint(
        op.f("ck_agent_tool_calls_agent_tool_call_status"),
        "agent_tool_calls",
        type_="check",
    )
    op.drop_constraint(
        op.f("ck_agent_tool_calls_agent_tool_call_terminal_outcome"),
        "agent_tool_calls",
        type_="check",
    )
    op.drop_constraint(
        op.f("ck_agent_tool_calls_agent_tool_call_latency_non_negative"),
        "agent_tool_calls",
        type_="check",
    )
    op.drop_constraint(
        op.f("ck_agent_tool_calls_agent_tool_call_timestamp_order"),
        "agent_tool_calls",
        type_="check",
    )

    op.alter_column(
        "agent_tool_calls",
        "agent_run_attempt_id",
        existing_type=sa.Uuid(),
        existing_nullable=False,
        new_column_name="proposed_by_agent_run_attempt_id",
    )
    op.alter_column(
        "agent_tool_calls",
        "started_at",
        existing_type=sa.DateTime(timezone=True),
        existing_nullable=False,
        new_column_name="execution_started_at",
    )
    op.add_column(
        "agent_tool_calls",
        sa.Column(
            "executed_by_agent_run_attempt_id",
            sa.Uuid(),
            nullable=True,
        ),
    )
    op.add_column(
        "agent_tool_calls",
        sa.Column(
            "proposed_at",
            sa.DateTime(timezone=True),
            nullable=True,
        ),
    )

    op.execute(
        sa.text(
            """
            UPDATE agent_tool_calls
            SET
                executed_by_agent_run_attempt_id =
                    proposed_by_agent_run_attempt_id,
                proposed_at = execution_started_at
            """
        )
    )

    op.alter_column(
        "agent_tool_calls",
        "proposed_at",
        existing_type=sa.DateTime(timezone=True),
        nullable=False,
    )
    op.alter_column(
        "agent_tool_calls",
        "execution_started_at",
        existing_type=sa.DateTime(timezone=True),
        existing_nullable=False,
        nullable=True,
    )
    op.alter_column(
        "agent_tool_calls",
        "finished_at",
        existing_type=sa.DateTime(timezone=True),
        existing_nullable=False,
        nullable=True,
    )
    op.alter_column(
        "agent_tool_calls",
        "latency_ms",
        existing_type=sa.BigInteger(),
        existing_nullable=False,
        nullable=True,
    )

    op.create_foreign_key(
        "fk_agent_tool_calls_proposed_by_attempt",
        "agent_tool_calls",
        "agent_run_attempts",
        [
            "agent_run_id",
            "proposed_by_agent_run_attempt_id",
        ],
        [
            "agent_run_id",
            "id",
        ],
        ondelete="CASCADE",
    )
    op.create_foreign_key(
        "fk_agent_tool_calls_executed_by_attempt",
        "agent_tool_calls",
        "agent_run_attempts",
        [
            "agent_run_id",
            "executed_by_agent_run_attempt_id",
        ],
        [
            "agent_run_id",
            "id",
        ],
        ondelete="RESTRICT",
    )
    op.create_unique_constraint(
        "uq_agent_tool_calls_proposal_attempt_sequence",
        "agent_tool_calls",
        [
            "proposed_by_agent_run_attempt_id",
            "sequence",
        ],
    )
    op.create_unique_constraint(
        "uq_agent_tool_calls_proposal_attempt_provider_call",
        "agent_tool_calls",
        [
            "proposed_by_agent_run_attempt_id",
            "provider_tool_call_id",
        ],
    )
    op.create_index(
        "uq_agent_tool_calls_sensitive_proposal_identity",
        "agent_tool_calls",
        [
            "agent_run_id",
            "tool_name",
            "tool_version",
            "input_fingerprint",
        ],
        unique=True,
        postgresql_where=sa.text("safety_level = 'sensitive_write'"),
    )
    op.create_check_constraint(
        op.f("ck_agent_tool_calls_agent_tool_call_status"),
        "agent_tool_calls",
        (
            "status IN ("
            "'pending_approval', "
            "'succeeded', "
            "'failed', "
            "'timed_out', "
            "'rejected', "
            "'expired'"
            ")"
        ),
    )
    op.create_check_constraint(
        op.f("ck_agent_tool_calls_agent_tool_call_latency_non_negative"),
        "agent_tool_calls",
        "latency_ms IS NULL OR latency_ms >= 0",
    )
    op.create_check_constraint(
        op.f("ck_agent_tool_calls_agent_tool_call_timestamp_order"),
        "agent_tool_calls",
        (
            "(execution_started_at IS NULL "
            "OR execution_started_at >= proposed_at) "
            "AND (finished_at IS NULL OR finished_at >= proposed_at) "
            "AND (execution_started_at IS NULL "
            "OR finished_at IS NULL "
            "OR finished_at >= execution_started_at)"
        ),
    )
    op.create_check_constraint(
        op.f("ck_agent_tool_calls_agent_tool_call_sensitive_pending_state"),
        "agent_tool_calls",
        ("status <> 'pending_approval' OR safety_level = 'sensitive_write'"),
    )
    op.create_check_constraint(
        op.f("ck_agent_tool_calls_agent_tool_call_lifecycle_state"),
        "agent_tool_calls",
        (
            "("
            "status = 'pending_approval' "
            "AND executed_by_agent_run_attempt_id IS NULL "
            "AND safe_output IS NULL "
            "AND latency_ms IS NULL "
            "AND error_code IS NULL "
            "AND execution_started_at IS NULL "
            "AND finished_at IS NULL"
            ") OR ("
            "status = 'succeeded' "
            "AND executed_by_agent_run_attempt_id IS NOT NULL "
            "AND safe_output IS NOT NULL "
            "AND latency_ms IS NOT NULL "
            "AND error_code IS NULL "
            "AND execution_started_at IS NOT NULL "
            "AND finished_at IS NOT NULL"
            ") OR ("
            "status IN ('failed', 'timed_out') "
            "AND executed_by_agent_run_attempt_id IS NOT NULL "
            "AND safe_output IS NULL "
            "AND latency_ms IS NOT NULL "
            "AND error_code IS NOT NULL "
            "AND execution_started_at IS NOT NULL "
            "AND finished_at IS NOT NULL"
            ") OR ("
            "status = 'rejected' "
            "AND safe_output IS NULL "
            "AND finished_at IS NOT NULL "
            "AND ("
            "("
            "executed_by_agent_run_attempt_id IS NULL "
            "AND safety_level = 'sensitive_write' "
            "AND latency_ms IS NULL "
            "AND error_code IS NULL "
            "AND execution_started_at IS NULL"
            ") OR ("
            "executed_by_agent_run_attempt_id IS NOT NULL "
            "AND latency_ms IS NOT NULL "
            "AND error_code IS NOT NULL "
            "AND execution_started_at IS NOT NULL"
            ")"
            ")"
            ") OR ("
            "status = 'expired' "
            "AND safety_level = 'sensitive_write' "
            "AND executed_by_agent_run_attempt_id IS NULL "
            "AND safe_output IS NULL "
            "AND latency_ms IS NULL "
            "AND error_code IS NULL "
            "AND execution_started_at IS NULL "
            "AND finished_at IS NOT NULL"
            ")"
        ),
    )


def downgrade() -> None:
    """Restore terminal, single-attempt tool-call audits."""
    op.execute(
        sa.text(
            """
            DO $$
            BEGIN
                IF EXISTS (
                    SELECT 1
                    FROM agent_tool_calls
                    WHERE
                        status IN ('pending_approval', 'expired')
                        OR (
                            status = 'rejected'
                            AND executed_by_agent_run_attempt_id IS NULL
                        )
                        OR executed_by_agent_run_attempt_id IS NULL
                        OR (
                            executed_by_agent_run_attempt_id
                            <> proposed_by_agent_run_attempt_id
                        )
                        OR execution_started_at IS NULL
                        OR finished_at IS NULL
                        OR latency_ms IS NULL
                ) THEN
                    RAISE EXCEPTION
                        'Cannot downgrade approval-aware tool-call records';
                END IF;
            END
            $$;
            """
        )
    )

    op.drop_constraint(
        op.f("ck_agent_tool_calls_agent_tool_call_lifecycle_state"),
        "agent_tool_calls",
        type_="check",
    )
    op.drop_constraint(
        op.f("ck_agent_tool_calls_agent_tool_call_sensitive_pending_state"),
        "agent_tool_calls",
        type_="check",
    )
    op.drop_constraint(
        op.f("ck_agent_tool_calls_agent_tool_call_timestamp_order"),
        "agent_tool_calls",
        type_="check",
    )
    op.drop_constraint(
        op.f("ck_agent_tool_calls_agent_tool_call_latency_non_negative"),
        "agent_tool_calls",
        type_="check",
    )
    op.drop_constraint(
        op.f("ck_agent_tool_calls_agent_tool_call_status"),
        "agent_tool_calls",
        type_="check",
    )
    op.drop_index(
        "uq_agent_tool_calls_sensitive_proposal_identity",
        table_name="agent_tool_calls",
    )
    op.drop_constraint(
        "uq_agent_tool_calls_proposal_attempt_provider_call",
        "agent_tool_calls",
        type_="unique",
    )
    op.drop_constraint(
        "uq_agent_tool_calls_proposal_attempt_sequence",
        "agent_tool_calls",
        type_="unique",
    )
    op.drop_constraint(
        "fk_agent_tool_calls_executed_by_attempt",
        "agent_tool_calls",
        type_="foreignkey",
    )
    op.drop_constraint(
        "fk_agent_tool_calls_proposed_by_attempt",
        "agent_tool_calls",
        type_="foreignkey",
    )

    op.alter_column(
        "agent_tool_calls",
        "latency_ms",
        existing_type=sa.BigInteger(),
        existing_nullable=True,
        nullable=False,
    )
    op.alter_column(
        "agent_tool_calls",
        "finished_at",
        existing_type=sa.DateTime(timezone=True),
        existing_nullable=True,
        nullable=False,
    )
    op.alter_column(
        "agent_tool_calls",
        "execution_started_at",
        existing_type=sa.DateTime(timezone=True),
        existing_nullable=True,
        nullable=False,
    )

    op.drop_column(
        "agent_tool_calls",
        "proposed_at",
    )
    op.drop_column(
        "agent_tool_calls",
        "executed_by_agent_run_attempt_id",
    )
    op.alter_column(
        "agent_tool_calls",
        "execution_started_at",
        existing_type=sa.DateTime(timezone=True),
        existing_nullable=False,
        new_column_name="started_at",
    )
    op.alter_column(
        "agent_tool_calls",
        "proposed_by_agent_run_attempt_id",
        existing_type=sa.Uuid(),
        existing_nullable=False,
        new_column_name="agent_run_attempt_id",
    )

    op.create_foreign_key(
        "fk_agent_tool_calls_agent_run_attempt",
        "agent_tool_calls",
        "agent_run_attempts",
        [
            "agent_run_id",
            "agent_run_attempt_id",
        ],
        [
            "agent_run_id",
            "id",
        ],
        ondelete="CASCADE",
    )
    op.create_unique_constraint(
        "uq_agent_tool_calls_attempt_sequence",
        "agent_tool_calls",
        [
            "agent_run_attempt_id",
            "sequence",
        ],
    )
    op.create_unique_constraint(
        "uq_agent_tool_calls_attempt_provider_call",
        "agent_tool_calls",
        [
            "agent_run_attempt_id",
            "provider_tool_call_id",
        ],
    )
    op.create_check_constraint(
        op.f("ck_agent_tool_calls_agent_tool_call_status"),
        "agent_tool_calls",
        ("status IN ('succeeded', 'failed', 'timed_out', 'rejected')"),
    )
    op.create_check_constraint(
        op.f("ck_agent_tool_calls_agent_tool_call_latency_non_negative"),
        "agent_tool_calls",
        "latency_ms >= 0",
    )
    op.create_check_constraint(
        op.f("ck_agent_tool_calls_agent_tool_call_terminal_outcome"),
        "agent_tool_calls",
        (
            "("
            "status = 'succeeded' "
            "AND safe_output IS NOT NULL "
            "AND error_code IS NULL"
            ") OR ("
            "status IN ('failed', 'timed_out', 'rejected') "
            "AND safe_output IS NULL "
            "AND error_code IS NOT NULL"
            ")"
        ),
    )
    op.create_check_constraint(
        op.f("ck_agent_tool_calls_agent_tool_call_timestamp_order"),
        "agent_tool_calls",
        "finished_at >= started_at",
    )
