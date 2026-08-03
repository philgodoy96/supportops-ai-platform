"""create ticket escalations

Revision ID: b8d3f6a1c9e4
Revises: e4a7c9d2f1b6
Create Date: 2026-08-03 18:30:00+00:00
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "b8d3f6a1c9e4"
down_revision: str | Sequence[str] | None = "e4a7c9d2f1b6"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Create immutable ticket-escalation records."""

    op.create_table(
        "ticket_escalations",
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
        sa.Column(
            "target_queue",
            sa.String(length=64),
            nullable=False,
        ),
        sa.Column(
            "reason",
            sa.String(length=1000),
            nullable=False,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
        ),
        sa.CheckConstraint(
            (
                "target_queue = btrim(target_queue) "
                "AND char_length(target_queue) BETWEEN 1 AND 64 "
                "AND target_queue ~ '^[a-z][a-z0-9_]*$'"
            ),
            name=op.f("ck_ticket_escalations_target_queue_format"),
        ),
        sa.CheckConstraint(
            ("reason = btrim(reason) AND char_length(reason) BETWEEN 1 AND 1000"),
            name=op.f("ck_ticket_escalations_reason_format"),
        ),
        sa.ForeignKeyConstraint(
            ["workspace_id", "ticket_id"],
            [
                "tickets.workspace_id",
                "tickets.id",
            ],
            name=("fk_ticket_escalations_workspace_ticket"),
            ondelete="CASCADE",
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
            name=("fk_ticket_escalations_workspace_ticket_agent_run"),
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
            name=("fk_ticket_escalations_execution_attempt"),
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["approval_request_id"],
            ["approval_requests.id"],
            name=("fk_ticket_escalations_approval_request"),
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["agent_tool_call_id"],
            ["agent_tool_calls.id"],
            name=("fk_ticket_escalations_agent_tool_call"),
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint(
            "id",
            name=op.f("pk_ticket_escalations"),
        ),
        sa.UniqueConstraint(
            "approval_request_id",
            name=("uq_ticket_escalations_approval_request"),
        ),
        sa.UniqueConstraint(
            "agent_tool_call_id",
            name=("uq_ticket_escalations_agent_tool_call"),
        ),
        sa.UniqueConstraint(
            "workspace_id",
            "id",
            name="uq_ticket_escalations_workspace_id",
        ),
    )

    op.create_index(
        "ix_ticket_escalations_workspace_created_id",
        "ticket_escalations",
        [
            "workspace_id",
            sa.literal_column("created_at DESC"),
            sa.literal_column("id DESC"),
        ],
        unique=False,
    )
    op.create_index(
        "ix_ticket_escalations_ticket_created_id",
        "ticket_escalations",
        [
            "workspace_id",
            "ticket_id",
            sa.literal_column("created_at DESC"),
            sa.literal_column("id DESC"),
        ],
        unique=False,
    )


def downgrade() -> None:
    """Drop immutable ticket-escalation records."""

    op.drop_index(
        "ix_ticket_escalations_ticket_created_id",
        table_name="ticket_escalations",
    )
    op.drop_index(
        "ix_ticket_escalations_workspace_created_id",
        table_name="ticket_escalations",
    )
    op.drop_table("ticket_escalations")
