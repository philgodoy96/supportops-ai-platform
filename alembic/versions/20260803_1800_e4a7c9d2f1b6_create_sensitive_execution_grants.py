"""create sensitive execution grants

Revision ID: e4a7c9d2f1b6
Revises: d6f1a8c3e5b7
Create Date: 2026-08-03 18:00:00+00:00
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "e4a7c9d2f1b6"
down_revision: str | Sequence[str] | None = "d6f1a8c3e5b7"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Create immutable grants for approved sensitive execution."""

    op.create_table(
        "sensitive_execution_grants",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("workspace_id", sa.Uuid(), nullable=False),
        sa.Column("ticket_id", sa.Uuid(), nullable=False),
        sa.Column("agent_run_id", sa.Uuid(), nullable=False),
        sa.Column(
            "executed_by_agent_run_attempt_id",
            sa.Uuid(),
            nullable=False,
        ),
        sa.Column("approval_request_id", sa.Uuid(), nullable=False),
        sa.Column("agent_tool_call_id", sa.Uuid(), nullable=False),
        sa.Column("tool_name", sa.String(length=64), nullable=False),
        sa.Column("tool_version", sa.Integer(), nullable=False),
        sa.Column("safety_level", sa.String(length=32), nullable=False),
        sa.Column(
            "input_fingerprint",
            sa.String(length=64),
            nullable=False,
        ),
        sa.Column(
            "granted_input",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
        ),
        sa.Column(
            "decision_actor_reference",
            sa.String(length=255),
            nullable=False,
        ),
        sa.Column("decision_request_id", sa.Uuid(), nullable=False),
        sa.Column(
            "decision_correlation_id",
            sa.Uuid(),
            nullable=False,
        ),
        sa.Column(
            "approved_at",
            sa.DateTime(timezone=True),
            nullable=False,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
        ),
        sa.CheckConstraint(
            "safety_level = 'sensitive_write'",
            name=op.f("ck_sensitive_execution_grants_safety_level"),
        ),
        sa.CheckConstraint(
            (
                "tool_name = btrim(tool_name) "
                "AND char_length(tool_name) BETWEEN 1 AND 64 "
                "AND tool_name ~ '^[a-z][a-z0-9_]*$'"
            ),
            name=op.f("ck_sensitive_execution_grants_tool_name_format"),
        ),
        sa.CheckConstraint(
            "tool_version >= 1",
            name=op.f("ck_sensitive_execution_grants_tool_version_positive"),
        ),
        sa.CheckConstraint(
            "input_fingerprint ~ '^[0-9a-f]{64}$'",
            name=op.f("ck_sensitive_execution_grants_input_fingerprint"),
        ),
        sa.CheckConstraint(
            "jsonb_typeof(granted_input) = 'object'",
            name=op.f("ck_sensitive_execution_grants_granted_input_object"),
        ),
        sa.CheckConstraint(
            "octet_length(granted_input::text) <= 8192",
            name=op.f("ck_sensitive_execution_grants_granted_input_size"),
        ),
        sa.CheckConstraint(
            (
                "decision_actor_reference = "
                "btrim(decision_actor_reference) "
                "AND char_length(decision_actor_reference) "
                "BETWEEN 1 AND 255"
            ),
            name=op.f("ck_sensitive_execution_grants_actor_format"),
        ),
        sa.CheckConstraint(
            "created_at >= approved_at",
            name=op.f("ck_sensitive_execution_grants_creation_order"),
        ),
        sa.ForeignKeyConstraint(
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
            name=("fk_sensitive_execution_grants_workspace_ticket_agent_run"),
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
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
        sa.ForeignKeyConstraint(
            ["approval_request_id"],
            ["approval_requests.id"],
            name=("fk_sensitive_execution_grants_approval_request"),
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["agent_tool_call_id"],
            ["agent_tool_calls.id"],
            name=("fk_sensitive_execution_grants_agent_tool_call"),
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint(
            "id",
            name=op.f("pk_sensitive_execution_grants"),
        ),
        sa.UniqueConstraint(
            "approval_request_id",
            name=("uq_sensitive_execution_grants_approval_request"),
        ),
        sa.UniqueConstraint(
            "agent_tool_call_id",
            name=("uq_sensitive_execution_grants_agent_tool_call"),
        ),
        sa.UniqueConstraint(
            "workspace_id",
            "id",
            name=("uq_sensitive_execution_grants_workspace_id"),
        ),
    )

    op.create_index(
        "ix_sensitive_execution_grants_workspace_created_id",
        "sensitive_execution_grants",
        [
            "workspace_id",
            sa.literal_column("created_at DESC"),
            sa.literal_column("id DESC"),
        ],
        unique=False,
    )
    op.create_index(
        "ix_sensitive_execution_grants_agent_run",
        "sensitive_execution_grants",
        ["agent_run_id", "created_at", "id"],
        unique=False,
    )


def downgrade() -> None:
    """Drop immutable sensitive execution grants."""

    op.drop_index(
        "ix_sensitive_execution_grants_agent_run",
        table_name="sensitive_execution_grants",
    )
    op.drop_index(
        "ix_sensitive_execution_grants_workspace_created_id",
        table_name="sensitive_execution_grants",
    )
    op.drop_table("sensitive_execution_grants")
