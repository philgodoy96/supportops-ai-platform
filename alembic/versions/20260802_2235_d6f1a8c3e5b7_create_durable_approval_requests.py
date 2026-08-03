"""create durable approval requests

Revision ID: d6f1a8c3e5b7
Revises: c9e2f4a7b6d1
Create Date: 2026-08-02 22:35:00+00:00
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "d6f1a8c3e5b7"
down_revision: str | Sequence[str] | None = "c9e2f4a7b6d1"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Create application-owned approval-request persistence."""
    op.create_table(
        "approval_requests",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("workspace_id", sa.Uuid(), nullable=False),
        sa.Column("ticket_id", sa.Uuid(), nullable=False),
        sa.Column("agent_run_id", sa.Uuid(), nullable=False),
        sa.Column("agent_tool_call_id", sa.Uuid(), nullable=False),
        sa.Column(
            "requested_by_llm_invocation_id",
            sa.Uuid(),
            nullable=False,
        ),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("tool_name", sa.String(length=64), nullable=False),
        sa.Column("tool_version", sa.Integer(), nullable=False),
        sa.Column("safety_level", sa.String(length=32), nullable=False),
        sa.Column(
            "input_fingerprint",
            sa.String(length=64),
            nullable=False,
        ),
        sa.Column(
            "proposed_input",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
        ),
        sa.Column(
            "request_reason",
            sa.String(length=1000),
            nullable=False,
        ),
        sa.Column(
            "expires_at",
            sa.DateTime(timezone=True),
            nullable=False,
        ),
        sa.Column(
            "decision_actor_reference",
            sa.String(length=255),
            nullable=True,
        ),
        sa.Column(
            "decision_comment",
            sa.String(length=2000),
            nullable=True,
        ),
        sa.Column(
            "decision_request_id",
            sa.Uuid(),
            nullable=True,
        ),
        sa.Column(
            "decision_correlation_id",
            sa.Uuid(),
            nullable=True,
        ),
        sa.Column(
            "decided_at",
            sa.DateTime(timezone=True),
            nullable=True,
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
            ("status IN ('pending', 'approved', 'rejected', 'expired')"),
            name=op.f("ck_approval_requests_approval_request_status"),
        ),
        sa.CheckConstraint(
            "safety_level = 'sensitive_write'",
            name=op.f("ck_approval_requests_approval_request_safety_level"),
        ),
        sa.CheckConstraint(
            (
                "tool_name = btrim(tool_name) "
                "AND char_length(tool_name) BETWEEN 1 AND 64 "
                "AND tool_name ~ '^[a-z][a-z0-9_]*$'"
            ),
            name=op.f("ck_approval_requests_approval_request_tool_name_format"),
        ),
        sa.CheckConstraint(
            "tool_version >= 1",
            name=op.f("ck_approval_requests_approval_request_tool_version_positive"),
        ),
        sa.CheckConstraint(
            "input_fingerprint ~ '^[0-9a-f]{64}$'",
            name=op.f("ck_approval_requests_approval_request_input_fingerprint"),
        ),
        sa.CheckConstraint(
            "jsonb_typeof(proposed_input) = 'object'",
            name=op.f("ck_approval_requests_approval_request_proposed_input_object"),
        ),
        sa.CheckConstraint(
            "octet_length(proposed_input::text) <= 8192",
            name=op.f("ck_approval_requests_approval_request_proposed_input_size"),
        ),
        sa.CheckConstraint(
            (
                "request_reason = btrim(request_reason) "
                "AND char_length(request_reason) BETWEEN 1 AND 1000"
            ),
            name=op.f("ck_approval_requests_approval_request_reason_format"),
        ),
        sa.CheckConstraint(
            (
                "decision_actor_reference IS NULL OR ("
                "decision_actor_reference = "
                "btrim(decision_actor_reference) "
                "AND char_length(decision_actor_reference) "
                "BETWEEN 1 AND 255"
                ")"
            ),
            name=op.f("ck_approval_requests_approval_request_actor_format"),
        ),
        sa.CheckConstraint(
            (
                "decision_comment IS NULL OR ("
                "decision_comment = btrim(decision_comment) "
                "AND char_length(decision_comment) "
                "BETWEEN 1 AND 2000"
                ")"
            ),
            name=op.f("ck_approval_requests_approval_request_comment_format"),
        ),
        sa.CheckConstraint(
            "expires_at > created_at",
            name=op.f("ck_approval_requests_approval_request_expiration_order"),
        ),
        sa.CheckConstraint(
            "updated_at >= created_at",
            name=op.f("ck_approval_requests_approval_request_update_order"),
        ),
        sa.CheckConstraint(
            "decided_at IS NULL OR decided_at >= created_at",
            name=op.f("ck_approval_requests_approval_request_decision_order"),
        ),
        sa.CheckConstraint(
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
            name=op.f("ck_approval_requests_approval_request_decision_state"),
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
            name=("fk_approval_requests_workspace_ticket_agent_run"),
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["agent_tool_call_id"],
            ["agent_tool_calls.id"],
            name="fk_approval_requests_agent_tool_call",
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
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
        sa.PrimaryKeyConstraint(
            "id",
            name=op.f("pk_approval_requests"),
        ),
        sa.UniqueConstraint(
            "agent_tool_call_id",
            name="uq_approval_requests_agent_tool_call",
        ),
        sa.UniqueConstraint(
            "workspace_id",
            "id",
            name="uq_approval_requests_workspace_id",
        ),
    )

    op.create_index(
        "ix_approval_requests_workspace_status_created_id",
        "approval_requests",
        [
            "workspace_id",
            "status",
            sa.literal_column("created_at DESC"),
            sa.literal_column("id DESC"),
        ],
        unique=False,
    )
    op.create_index(
        "ix_approval_requests_agent_run_status",
        "approval_requests",
        [
            "agent_run_id",
            "status",
        ],
        unique=False,
    )
    op.create_index(
        "ix_approval_requests_pending_expiration",
        "approval_requests",
        [
            "expires_at",
            "id",
        ],
        unique=False,
        postgresql_where=sa.text("status = 'pending'"),
    )


def downgrade() -> None:
    """Drop durable approval-request persistence."""
    op.drop_index(
        "ix_approval_requests_pending_expiration",
        table_name="approval_requests",
    )
    op.drop_index(
        "ix_approval_requests_agent_run_status",
        table_name="approval_requests",
    )
    op.drop_index(
        "ix_approval_requests_workspace_status_created_id",
        table_name="approval_requests",
    )
    op.drop_table("approval_requests")
