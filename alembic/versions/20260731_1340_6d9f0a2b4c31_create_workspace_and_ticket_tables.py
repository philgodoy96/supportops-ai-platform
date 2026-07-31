"""create workspace and ticket tables

Revision ID: 6d9f0a2b4c31
Revises:
Create Date: 2026-07-31 13:40:34.120235+00:00
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "6d9f0a2b4c31"
down_revision: str | Sequence[str] | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Apply the migration."""
    op.create_table(
        "workspaces",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("name", sa.String(length=120), nullable=False),
        sa.Column("slug", sa.String(length=63), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint(
            "name = btrim(name)",
            name=op.f("ck_workspaces_workspace_name_trimmed"),
        ),
        sa.CheckConstraint(
            "char_length(name) BETWEEN 1 AND 120",
            name=op.f("ck_workspaces_workspace_name_length"),
        ),
        sa.CheckConstraint(
            "char_length(slug) BETWEEN 3 AND 63",
            name=op.f("ck_workspaces_workspace_slug_length"),
        ),
        sa.CheckConstraint(
            "slug ~ '^[a-z0-9]+(?:-[a-z0-9]+)*$'",
            name=op.f("ck_workspaces_workspace_slug_format"),
        ),
        sa.CheckConstraint(
            "updated_at >= created_at",
            name=op.f("ck_workspaces_workspace_timestamp_order"),
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_workspaces")),
        sa.UniqueConstraint("slug", name=op.f("uq_workspaces_slug")),
    )
    op.create_table(
        "tickets",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("workspace_id", sa.Uuid(), nullable=False),
        sa.Column("subject", sa.String(length=200), nullable=False),
        sa.Column("description", sa.String(length=20000), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("external_reference", sa.String(length=128), nullable=True),
        sa.Column("ingestion_request_id", sa.Uuid(), nullable=False),
        sa.Column("correlation_id", sa.Uuid(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint(
            "subject = btrim(subject)",
            name=op.f("ck_tickets_ticket_subject_trimmed"),
        ),
        sa.CheckConstraint(
            "char_length(subject) BETWEEN 1 AND 200",
            name=op.f("ck_tickets_ticket_subject_length"),
        ),
        sa.CheckConstraint(
            "description = btrim(description)",
            name=op.f("ck_tickets_ticket_description_trimmed"),
        ),
        sa.CheckConstraint(
            "char_length(description) BETWEEN 1 AND 20000",
            name=op.f("ck_tickets_ticket_description_length"),
        ),
        sa.CheckConstraint(
            (
                "external_reference IS NULL OR "
                "("
                "external_reference = btrim(external_reference) "
                "AND char_length(external_reference) BETWEEN 1 AND 128"
                ")"
            ),
            name=op.f("ck_tickets_ticket_external_reference_format"),
        ),
        sa.CheckConstraint(
            "status IN ('open')",
            name=op.f("ck_tickets_ticket_status"),
        ),
        sa.CheckConstraint(
            "updated_at >= created_at",
            name=op.f("ck_tickets_ticket_timestamp_order"),
        ),
        sa.ForeignKeyConstraint(
            ["workspace_id"],
            ["workspaces.id"],
            name=op.f("fk_tickets_workspace_id_workspaces"),
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_tickets")),
        sa.UniqueConstraint(
            "workspace_id",
            "external_reference",
            name="uq_tickets_workspace_external_reference",
        ),
    )
    op.create_index(
        "ix_tickets_workspace_created_id",
        "tickets",
        [
            "workspace_id",
            sa.literal_column("created_at DESC"),
            sa.literal_column("id DESC"),
        ],
        unique=False,
    )


def downgrade() -> None:
    """Revert the migration."""
    op.drop_index("ix_tickets_workspace_created_id", table_name="tickets")
    op.drop_table("tickets")
    op.drop_table("workspaces")
