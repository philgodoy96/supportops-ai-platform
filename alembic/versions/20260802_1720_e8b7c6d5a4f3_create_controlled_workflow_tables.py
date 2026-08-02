"""create controlled workflow tables

Revision ID: e8b7c6d5a4f3
Revises: d4e8f2a6c901
Create Date: 2026-08-02 17:20:00+00:00
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "e8b7c6d5a4f3"
down_revision: str | Sequence[str] | None = "d4e8f2a6c901"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Apply the migration."""

    op.create_unique_constraint(
        "uq_ticket_classifications_run_id",
        "ticket_classifications",
        [
            "agent_run_id",
            "id",
        ],
    )
    op.create_unique_constraint(
        ("uq_knowledge_document_chunks_workspace_document_version_id"),
        "knowledge_document_chunks",
        [
            "workspace_id",
            "document_id",
            "document_version_id",
            "id",
        ],
    )

    op.create_table(
        "agent_tool_calls",
        sa.Column(
            "id",
            sa.Uuid(),
            nullable=False,
        ),
        sa.Column(
            "workspace_id",
            sa.Uuid(),
            nullable=False,
        ),
        sa.Column(
            "ticket_id",
            sa.Uuid(),
            nullable=False,
        ),
        sa.Column(
            "agent_run_id",
            sa.Uuid(),
            nullable=False,
        ),
        sa.Column(
            "agent_run_attempt_id",
            sa.Uuid(),
            nullable=False,
        ),
        sa.Column(
            "sequence",
            sa.Integer(),
            nullable=False,
        ),
        sa.Column(
            "provider_tool_call_id",
            sa.String(length=255),
            nullable=True,
        ),
        sa.Column(
            "tool_name",
            sa.String(length=64),
            nullable=False,
        ),
        sa.Column(
            "tool_version",
            sa.Integer(),
            nullable=False,
        ),
        sa.Column(
            "safety_level",
            sa.String(length=32),
            nullable=False,
        ),
        sa.Column(
            "status",
            sa.String(length=32),
            nullable=False,
        ),
        sa.Column(
            "input_fingerprint",
            sa.String(length=64),
            nullable=False,
        ),
        sa.Column(
            "safe_input",
            postgresql.JSONB(
                astext_type=sa.Text(),
            ),
            nullable=False,
        ),
        sa.Column(
            "safe_output",
            postgresql.JSONB(
                astext_type=sa.Text(),
            ),
            nullable=True,
        ),
        sa.Column(
            "latency_ms",
            sa.BigInteger(),
            nullable=False,
        ),
        sa.Column(
            "error_code",
            sa.String(length=128),
            nullable=True,
        ),
        sa.Column(
            "started_at",
            sa.DateTime(timezone=True),
            nullable=False,
        ),
        sa.Column(
            "finished_at",
            sa.DateTime(timezone=True),
            nullable=False,
        ),
        sa.CheckConstraint(
            "sequence >= 1",
            name=op.f("ck_agent_tool_calls_agent_tool_call_sequence_positive"),
        ),
        sa.CheckConstraint(
            (
                "provider_tool_call_id IS NULL OR ("
                "provider_tool_call_id = "
                "btrim(provider_tool_call_id) "
                "AND char_length(provider_tool_call_id) "
                "BETWEEN 1 AND 255"
                ")"
            ),
            name=op.f("ck_agent_tool_calls_agent_tool_call_provider_call_id_format"),
        ),
        sa.CheckConstraint(
            (
                "tool_name = btrim(tool_name) "
                "AND char_length(tool_name) BETWEEN 1 AND 64 "
                "AND tool_name ~ '^[a-z][a-z0-9_]*$'"
            ),
            name=op.f("ck_agent_tool_calls_agent_tool_call_tool_name_format"),
        ),
        sa.CheckConstraint(
            "tool_version >= 1",
            name=op.f("ck_agent_tool_calls_agent_tool_call_tool_version_positive"),
        ),
        sa.CheckConstraint(
            ("safety_level IN ('read_only', 'sensitive_write', 'external_side_effect')"),
            name=op.f("ck_agent_tool_calls_agent_tool_call_safety_level"),
        ),
        sa.CheckConstraint(
            ("status IN ('succeeded', 'failed', 'timed_out', 'rejected')"),
            name=op.f("ck_agent_tool_calls_agent_tool_call_status"),
        ),
        sa.CheckConstraint(
            "input_fingerprint ~ '^[0-9a-f]{64}$'",
            name=op.f("ck_agent_tool_calls_agent_tool_call_input_fingerprint"),
        ),
        sa.CheckConstraint(
            "jsonb_typeof(safe_input) = 'object'",
            name=op.f("ck_agent_tool_calls_agent_tool_call_safe_input_object"),
        ),
        sa.CheckConstraint(
            "octet_length(safe_input::text) <= 8192",
            name=op.f("ck_agent_tool_calls_agent_tool_call_safe_input_size"),
        ),
        sa.CheckConstraint(
            ("safe_output IS NULL OR jsonb_typeof(safe_output) = 'object'"),
            name=op.f("ck_agent_tool_calls_agent_tool_call_safe_output_object"),
        ),
        sa.CheckConstraint(
            ("safe_output IS NULL OR octet_length(safe_output::text) <= 32768"),
            name=op.f("ck_agent_tool_calls_agent_tool_call_safe_output_size"),
        ),
        sa.CheckConstraint(
            "latency_ms >= 0",
            name=op.f("ck_agent_tool_calls_agent_tool_call_latency_non_negative"),
        ),
        sa.CheckConstraint(
            (
                "error_code IS NULL OR ("
                "error_code = btrim(error_code) "
                "AND char_length(error_code) BETWEEN 1 AND 128 "
                "AND error_code ~ '^[a-z][a-z0-9_]*$'"
                ")"
            ),
            name=op.f("ck_agent_tool_calls_agent_tool_call_error_code_format"),
        ),
        sa.CheckConstraint(
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
            name=op.f("ck_agent_tool_calls_agent_tool_call_terminal_outcome"),
        ),
        sa.CheckConstraint(
            "finished_at >= started_at",
            name=op.f("ck_agent_tool_calls_agent_tool_call_timestamp_order"),
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
            name=("fk_agent_tool_calls_workspace_ticket_agent_run"),
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            [
                "agent_run_id",
                "agent_run_attempt_id",
            ],
            [
                "agent_run_attempts.agent_run_id",
                "agent_run_attempts.id",
            ],
            name="fk_agent_tool_calls_agent_run_attempt",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint(
            "id",
            name=op.f("pk_agent_tool_calls"),
        ),
        sa.UniqueConstraint(
            "agent_run_attempt_id",
            "sequence",
            name="uq_agent_tool_calls_attempt_sequence",
        ),
        sa.UniqueConstraint(
            "agent_run_attempt_id",
            "provider_tool_call_id",
            name=("uq_agent_tool_calls_attempt_provider_call"),
        ),
    )
    op.create_index(
        "ix_agent_tool_calls_workspace_run_sequence",
        "agent_tool_calls",
        [
            "workspace_id",
            "agent_run_id",
            "sequence",
        ],
        unique=False,
    )

    op.create_table(
        "support_recommendations",
        sa.Column(
            "id",
            sa.Uuid(),
            nullable=False,
        ),
        sa.Column(
            "workspace_id",
            sa.Uuid(),
            nullable=False,
        ),
        sa.Column(
            "ticket_id",
            sa.Uuid(),
            nullable=False,
        ),
        sa.Column(
            "agent_run_id",
            sa.Uuid(),
            nullable=False,
        ),
        sa.Column(
            "classification_id",
            sa.Uuid(),
            nullable=False,
        ),
        sa.Column(
            "accepted_llm_invocation_id",
            sa.Uuid(),
            nullable=False,
        ),
        sa.Column(
            "recommended_action",
            sa.String(length=32),
            nullable=False,
        ),
        sa.Column(
            "response_text",
            sa.Text(),
            nullable=False,
        ),
        sa.Column(
            "requires_human_review",
            sa.Boolean(),
            nullable=False,
        ),
        sa.Column(
            "decision_summary",
            sa.String(length=500),
            nullable=False,
        ),
        sa.Column(
            "schema_version",
            sa.String(length=128),
            nullable=False,
        ),
        sa.Column(
            "prompt_id",
            sa.String(length=128),
            nullable=False,
        ),
        sa.Column(
            "prompt_version",
            sa.Integer(),
            nullable=False,
        ),
        sa.Column(
            "prompt_content_hash",
            sa.String(length=64),
            nullable=False,
        ),
        sa.Column(
            "provider",
            sa.String(length=64),
            nullable=False,
        ),
        sa.Column(
            "model",
            sa.String(length=128),
            nullable=False,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
        ),
        sa.CheckConstraint(
            (
                "recommended_action IN ("
                "'respond', "
                "'request_more_information', "
                "'recommend_escalation'"
                ")"
            ),
            name=op.f("ck_support_recommendations_support_recommendation_action"),
        ),
        sa.CheckConstraint(
            (
                "response_text = btrim(response_text) "
                "AND char_length(response_text) "
                "BETWEEN 1 AND 4000"
            ),
            name=op.f("ck_support_recommendations_support_recommendation_response_format"),
        ),
        sa.CheckConstraint(
            (
                "decision_summary = btrim(decision_summary) "
                "AND char_length(decision_summary) "
                "BETWEEN 1 AND 500"
            ),
            name=op.f("ck_support_recommendations_support_recommendation_summary_format"),
        ),
        sa.CheckConstraint(
            ("schema_version = 'support-recommendation-v1'"),
            name=op.f("ck_support_recommendations_support_recommendation_schema_version"),
        ),
        sa.CheckConstraint(
            ("prompt_id = btrim(prompt_id) AND char_length(prompt_id) BETWEEN 1 AND 128"),
            name=op.f("ck_support_recommendations_support_recommendation_prompt_id_format"),
        ),
        sa.CheckConstraint(
            "prompt_version >= 1",
            name=op.f("ck_support_recommendations_support_recommendation_prompt_version_positive"),
        ),
        sa.CheckConstraint(
            "prompt_content_hash ~ '^[0-9a-f]{64}$'",
            name=op.f("ck_support_recommendations_support_recommendation_prompt_content_hash"),
        ),
        sa.CheckConstraint(
            ("provider = btrim(provider) AND char_length(provider) BETWEEN 1 AND 64"),
            name=op.f("ck_support_recommendations_support_recommendation_provider_format"),
        ),
        sa.CheckConstraint(
            ("model = btrim(model) AND char_length(model) BETWEEN 1 AND 128"),
            name=op.f("ck_support_recommendations_support_recommendation_model_format"),
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
            name=("fk_support_recommendations_workspace_ticket_agent_run"),
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            [
                "agent_run_id",
                "classification_id",
            ],
            [
                "ticket_classifications.agent_run_id",
                "ticket_classifications.id",
            ],
            name=("fk_support_recommendations_agent_run_classification"),
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            [
                "agent_run_id",
                "accepted_llm_invocation_id",
            ],
            [
                "llm_invocations.agent_run_id",
                "llm_invocations.id",
            ],
            name=("fk_support_recommendations_accepted_invocation"),
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint(
            "id",
            name=op.f("pk_support_recommendations"),
        ),
        sa.UniqueConstraint(
            "agent_run_id",
            name="uq_support_recommendations_agent_run",
        ),
        sa.UniqueConstraint(
            "accepted_llm_invocation_id",
            name=("uq_support_recommendations_accepted_invocation"),
        ),
        sa.UniqueConstraint(
            "workspace_id",
            "id",
            name="uq_support_recommendations_workspace_id",
        ),
    )
    op.create_index(
        ("ix_support_recommendations_workspace_ticket_created_id"),
        "support_recommendations",
        [
            "workspace_id",
            "ticket_id",
            sa.literal_column("created_at DESC"),
            sa.literal_column("id DESC"),
        ],
        unique=False,
    )

    op.create_table(
        "support_recommendation_citations",
        sa.Column(
            "id",
            sa.Uuid(),
            nullable=False,
        ),
        sa.Column(
            "workspace_id",
            sa.Uuid(),
            nullable=False,
        ),
        sa.Column(
            "support_recommendation_id",
            sa.Uuid(),
            nullable=False,
        ),
        sa.Column(
            "ordinal",
            sa.Integer(),
            nullable=False,
        ),
        sa.Column(
            "document_id",
            sa.Uuid(),
            nullable=False,
        ),
        sa.Column(
            "document_version_id",
            sa.Uuid(),
            nullable=False,
        ),
        sa.Column(
            "chunk_id",
            sa.Uuid(),
            nullable=False,
        ),
        sa.Column(
            "retrieval_query_id",
            sa.Uuid(),
            nullable=False,
        ),
        sa.Column(
            "retrieval_rank",
            sa.Integer(),
            nullable=False,
        ),
        sa.Column(
            "retrieval_score",
            sa.Float(),
            nullable=False,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
        ),
        sa.CheckConstraint(
            "ordinal >= 1",
            name=op.f(
                "ck_support_recommendation_citations_"
                "support_recommendation_citation_"
                "ordinal_positive"
            ),
        ),
        sa.CheckConstraint(
            "retrieval_rank >= 0",
            name=op.f(
                "ck_support_recommendation_citations_"
                "support_recommendation_citation_"
                "rank_non_negative"
            ),
        ),
        sa.CheckConstraint(
            (
                "retrieval_score <> "
                "'NaN'::double precision "
                "AND retrieval_score <> "
                "'Infinity'::double precision "
                "AND retrieval_score <> "
                "'-Infinity'::double precision"
            ),
            name=op.f(
                "ck_support_recommendation_citations_support_recommendation_citation_score_finite"
            ),
        ),
        sa.ForeignKeyConstraint(
            [
                "workspace_id",
                "support_recommendation_id",
            ],
            [
                "support_recommendations.workspace_id",
                "support_recommendations.id",
            ],
            name=("fk_support_recommendation_citations_workspace_recommendation"),
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            [
                "workspace_id",
                "document_id",
                "document_version_id",
                "chunk_id",
            ],
            [
                "knowledge_document_chunks.workspace_id",
                "knowledge_document_chunks.document_id",
                ("knowledge_document_chunks.document_version_id"),
                "knowledge_document_chunks.id",
            ],
            name=("fk_support_recommendation_citations_authoritative_chunk"),
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint(
            "id",
            name=op.f("pk_support_recommendation_citations"),
        ),
        sa.UniqueConstraint(
            "support_recommendation_id",
            "ordinal",
            name=("uq_support_recommendation_citations_recommendation_ordinal"),
        ),
        sa.UniqueConstraint(
            "support_recommendation_id",
            "chunk_id",
            name=("uq_support_recommendation_citations_recommendation_chunk"),
        ),
    )
    op.create_index(
        ("ix_support_recommendation_citations_recommendation_ordinal"),
        "support_recommendation_citations",
        [
            "support_recommendation_id",
            "ordinal",
        ],
        unique=False,
    )


def downgrade() -> None:
    """Revert the migration."""

    op.drop_index(
        ("ix_support_recommendation_citations_recommendation_ordinal"),
        table_name="support_recommendation_citations",
    )
    op.drop_table("support_recommendation_citations")

    op.drop_index(
        ("ix_support_recommendations_workspace_ticket_created_id"),
        table_name="support_recommendations",
    )
    op.drop_table("support_recommendations")

    op.drop_index(
        "ix_agent_tool_calls_workspace_run_sequence",
        table_name="agent_tool_calls",
    )
    op.drop_table("agent_tool_calls")

    op.drop_constraint(
        ("uq_knowledge_document_chunks_workspace_document_version_id"),
        "knowledge_document_chunks",
        type_="unique",
    )
    op.drop_constraint(
        "uq_ticket_classifications_run_id",
        "ticket_classifications",
        type_="unique",
    )
