"""create classification tables

Revision ID: b7d3f1e5a9c2
Revises: a2c4e6f8091b
Create Date: 2026-08-01 17:40:00+00:00

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "b7d3f1e5a9c2"
down_revision: str | Sequence[str] | None = "a2c4e6f8091b"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Apply the migration."""

    op.create_unique_constraint(
        "uq_agent_runs_workspace_ticket_id",
        "agent_runs",
        [
            "workspace_id",
            "ticket_id",
            "id",
        ],
    )
    op.create_unique_constraint(
        "uq_agent_run_attempts_run_id",
        "agent_run_attempts",
        [
            "agent_run_id",
            "id",
        ],
    )

    op.create_table(
        "llm_invocations",
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
            "invocation_sequence",
            sa.Integer(),
            nullable=False,
        ),
        sa.Column(
            "status",
            sa.String(length=32),
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
            "provider_request_id",
            sa.String(length=255),
            nullable=True,
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
            "schema_version",
            sa.String(length=128),
            nullable=False,
        ),
        sa.Column(
            "input_tokens",
            sa.BigInteger(),
            nullable=True,
        ),
        sa.Column(
            "cached_input_tokens",
            sa.BigInteger(),
            nullable=True,
        ),
        sa.Column(
            "output_tokens",
            sa.BigInteger(),
            nullable=True,
        ),
        sa.Column(
            "reasoning_tokens",
            sa.BigInteger(),
            nullable=True,
        ),
        sa.Column(
            "total_tokens",
            sa.BigInteger(),
            nullable=True,
        ),
        sa.Column(
            "pricing_catalog_version",
            sa.String(length=128),
            nullable=False,
        ),
        sa.Column(
            "pricing_found",
            sa.Boolean(),
            nullable=False,
        ),
        sa.Column(
            "estimated_input_cost_usd",
            sa.Numeric(
                precision=20,
                scale=12,
            ),
            nullable=True,
        ),
        sa.Column(
            "estimated_cached_input_cost_usd",
            sa.Numeric(
                precision=20,
                scale=12,
            ),
            nullable=True,
        ),
        sa.Column(
            "estimated_output_cost_usd",
            sa.Numeric(
                precision=20,
                scale=12,
            ),
            nullable=True,
        ),
        sa.Column(
            "estimated_total_cost_usd",
            sa.Numeric(
                precision=20,
                scale=12,
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
            sa.String(length=64),
            nullable=True,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
        ),
        sa.CheckConstraint(
            "invocation_sequence >= 1",
            name=op.f("ck_llm_invocations_llm_invocation_sequence_positive"),
        ),
        sa.CheckConstraint(
            (
                "status IN ("
                "'succeeded', "
                "'refused', "
                "'incomplete', "
                "'validation_failed', "
                "'provider_failed', "
                "'timed_out'"
                ")"
            ),
            name=op.f("ck_llm_invocations_llm_invocation_status"),
        ),
        sa.CheckConstraint(
            ("provider = btrim(provider) AND char_length(provider) BETWEEN 1 AND 64"),
            name=op.f("ck_llm_invocations_llm_invocation_provider_format"),
        ),
        sa.CheckConstraint(
            ("model = btrim(model) AND char_length(model) BETWEEN 1 AND 128"),
            name=op.f("ck_llm_invocations_llm_invocation_model_format"),
        ),
        sa.CheckConstraint(
            (
                "provider_request_id IS NULL OR ("
                "provider_request_id = btrim(provider_request_id) "
                "AND char_length(provider_request_id) "
                "BETWEEN 1 AND 255"
                ")"
            ),
            name=op.f("ck_llm_invocations_llm_invocation_provider_request_id_format"),
        ),
        sa.CheckConstraint(
            ("prompt_id = btrim(prompt_id) AND char_length(prompt_id) BETWEEN 1 AND 128"),
            name=op.f("ck_llm_invocations_llm_invocation_prompt_id_format"),
        ),
        sa.CheckConstraint(
            "prompt_version >= 1",
            name=op.f("ck_llm_invocations_llm_invocation_prompt_version_positive"),
        ),
        sa.CheckConstraint(
            "prompt_content_hash ~ '^[0-9a-f]{64}$'",
            name=op.f("ck_llm_invocations_llm_invocation_prompt_content_hash"),
        ),
        sa.CheckConstraint(
            (
                "schema_version = btrim(schema_version) "
                "AND char_length(schema_version) BETWEEN 1 AND 128"
            ),
            name=op.f("ck_llm_invocations_llm_invocation_schema_version_format"),
        ),
        sa.CheckConstraint(
            (
                "pricing_catalog_version "
                "= btrim(pricing_catalog_version) "
                "AND char_length(pricing_catalog_version) "
                "BETWEEN 1 AND 128"
            ),
            name=op.f("ck_llm_invocations_llm_invocation_pricing_catalog_version_format"),
        ),
        sa.CheckConstraint(
            (
                "(input_tokens IS NULL OR input_tokens >= 0) "
                "AND ("
                "cached_input_tokens IS NULL "
                "OR cached_input_tokens >= 0"
                ") "
                "AND ("
                "output_tokens IS NULL "
                "OR output_tokens >= 0"
                ") "
                "AND ("
                "reasoning_tokens IS NULL "
                "OR reasoning_tokens >= 0"
                ") "
                "AND ("
                "total_tokens IS NULL "
                "OR total_tokens >= 0"
                ")"
            ),
            name=op.f("ck_llm_invocations_llm_invocation_tokens_non_negative"),
        ),
        sa.CheckConstraint(
            (
                "cached_input_tokens IS NULL "
                "OR input_tokens IS NULL "
                "OR cached_input_tokens <= input_tokens"
            ),
            name=op.f("ck_llm_invocations_llm_invocation_cached_input_limit"),
        ),
        sa.CheckConstraint(
            (
                "reasoning_tokens IS NULL "
                "OR output_tokens IS NULL "
                "OR reasoning_tokens <= output_tokens"
            ),
            name=op.f("ck_llm_invocations_llm_invocation_reasoning_token_limit"),
        ),
        sa.CheckConstraint(
            (
                "input_tokens IS NULL "
                "OR output_tokens IS NULL "
                "OR total_tokens IS NULL "
                "OR total_tokens = input_tokens + output_tokens"
            ),
            name=op.f("ck_llm_invocations_llm_invocation_total_token_consistency"),
        ),
        sa.CheckConstraint(
            (
                "("
                "estimated_input_cost_usd IS NULL "
                "OR estimated_input_cost_usd >= 0"
                ") AND ("
                "estimated_cached_input_cost_usd IS NULL "
                "OR estimated_cached_input_cost_usd >= 0"
                ") AND ("
                "estimated_output_cost_usd IS NULL "
                "OR estimated_output_cost_usd >= 0"
                ") AND ("
                "estimated_total_cost_usd IS NULL "
                "OR estimated_total_cost_usd >= 0"
                ")"
            ),
            name=op.f("ck_llm_invocations_llm_invocation_costs_non_negative"),
        ),
        sa.CheckConstraint(
            (
                "("
                "pricing_found = false "
                "AND estimated_input_cost_usd IS NULL "
                "AND estimated_cached_input_cost_usd IS NULL "
                "AND estimated_output_cost_usd IS NULL "
                "AND estimated_total_cost_usd IS NULL"
                ") OR ("
                "pricing_found = true "
                "AND ("
                "("
                "estimated_input_cost_usd IS NOT NULL "
                "AND estimated_cached_input_cost_usd IS NOT NULL "
                "AND estimated_output_cost_usd IS NOT NULL "
                "AND estimated_total_cost_usd = "
                "estimated_input_cost_usd "
                "+ estimated_cached_input_cost_usd "
                "+ estimated_output_cost_usd"
                ") OR ("
                "("
                "estimated_input_cost_usd IS NULL "
                "OR estimated_cached_input_cost_usd IS NULL "
                "OR estimated_output_cost_usd IS NULL"
                ") "
                "AND estimated_total_cost_usd IS NULL"
                ")"
                ")"
                ")"
            ),
            name=op.f("ck_llm_invocations_llm_invocation_pricing_state"),
        ),
        sa.CheckConstraint(
            "latency_ms >= 0",
            name=op.f("ck_llm_invocations_llm_invocation_latency_non_negative"),
        ),
        sa.CheckConstraint(
            (
                "error_code IS NULL OR error_code IN ("
                "'llm_timeout', "
                "'llm_rate_limited', "
                "'llm_authentication_failed', "
                "'llm_quota_exhausted', "
                "'llm_invalid_request', "
                "'llm_provider_unavailable', "
                "'llm_refusal', "
                "'llm_incomplete_response', "
                "'llm_output_validation_failed', "
                "'llm_unexpected_provider_failure'"
                ")"
            ),
            name=op.f("ck_llm_invocations_llm_invocation_error_code"),
        ),
        sa.CheckConstraint(
            (
                "("
                "status = 'succeeded' "
                "AND error_code IS NULL"
                ") OR ("
                "status <> 'succeeded' "
                "AND error_code IS NOT NULL"
                ")"
            ),
            name=op.f("ck_llm_invocations_llm_invocation_error_state"),
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
            name=("fk_llm_invocations_workspace_ticket_agent_run"),
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
            name="fk_llm_invocations_agent_run_attempt",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint(
            "id",
            name=op.f("pk_llm_invocations"),
        ),
        sa.UniqueConstraint(
            "agent_run_attempt_id",
            "invocation_sequence",
            name="uq_llm_invocations_attempt_sequence",
        ),
        sa.UniqueConstraint(
            "agent_run_id",
            "id",
            name="uq_llm_invocations_run_id",
        ),
    )
    op.create_index(
        "ix_llm_invocations_workspace_run_created_id",
        "llm_invocations",
        [
            "workspace_id",
            "agent_run_id",
            sa.literal_column("created_at DESC"),
            sa.literal_column("id DESC"),
        ],
        unique=False,
    )

    op.create_table(
        "ticket_classifications",
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
            "accepted_llm_invocation_id",
            sa.Uuid(),
            nullable=False,
        ),
        sa.Column(
            "category",
            sa.String(length=32),
            nullable=False,
        ),
        sa.Column(
            "intent",
            sa.String(length=32),
            nullable=False,
        ),
        sa.Column(
            "urgency",
            sa.String(length=32),
            nullable=False,
        ),
        sa.Column(
            "sentiment",
            sa.String(length=32),
            nullable=False,
        ),
        sa.Column(
            "requires_human_review",
            sa.Boolean(),
            nullable=False,
        ),
        sa.Column(
            "summary",
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
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
        ),
        sa.CheckConstraint(
            (
                "category IN ("
                "'account_access', "
                "'service_incident', "
                "'billing', "
                "'product_bug', "
                "'how_to', "
                "'security', "
                "'feature_request', "
                "'other'"
                ")"
            ),
            name=op.f("ck_ticket_classifications_ticket_classification_category"),
        ),
        sa.CheckConstraint(
            (
                "intent IN ("
                "'request_access', "
                "'report_incident', "
                "'report_problem', "
                "'ask_question', "
                "'request_change', "
                "'provide_feedback', "
                "'other'"
                ")"
            ),
            name=op.f("ck_ticket_classifications_ticket_classification_intent"),
        ),
        sa.CheckConstraint(
            ("urgency IN ('low', 'normal', 'high', 'critical')"),
            name=op.f("ck_ticket_classifications_ticket_classification_urgency"),
        ),
        sa.CheckConstraint(
            ("sentiment IN ('negative', 'neutral', 'positive', 'mixed')"),
            name=op.f("ck_ticket_classifications_ticket_classification_sentiment"),
        ),
        sa.CheckConstraint(
            ("summary = btrim(summary) AND char_length(summary) BETWEEN 1 AND 500"),
            name=op.f("ck_ticket_classifications_ticket_classification_summary_format"),
        ),
        sa.CheckConstraint(
            ("schema_version = 'ticket-classification-v1'"),
            name=op.f("ck_ticket_classifications_ticket_classification_schema_version"),
        ),
        sa.CheckConstraint(
            ("prompt_id = btrim(prompt_id) AND char_length(prompt_id) BETWEEN 1 AND 128"),
            name=op.f("ck_ticket_classifications_ticket_classification_prompt_id_format"),
        ),
        sa.CheckConstraint(
            "prompt_version >= 1",
            name=op.f("ck_ticket_classifications_ticket_classification_prompt_version_positive"),
        ),
        sa.CheckConstraint(
            "prompt_content_hash ~ '^[0-9a-f]{64}$'",
            name=op.f("ck_ticket_classifications_ticket_classification_prompt_content_hash"),
        ),
        sa.CheckConstraint(
            ("provider = btrim(provider) AND char_length(provider) BETWEEN 1 AND 64"),
            name=op.f("ck_ticket_classifications_ticket_classification_provider_format"),
        ),
        sa.CheckConstraint(
            ("model = btrim(model) AND char_length(model) BETWEEN 1 AND 128"),
            name=op.f("ck_ticket_classifications_ticket_classification_model_format"),
        ),
        sa.CheckConstraint(
            "updated_at = created_at",
            name=op.f("ck_ticket_classifications_ticket_classification_immutable_timestamp"),
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
            name=("fk_ticket_classifications_workspace_ticket_agent_run"),
            ondelete="CASCADE",
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
            name=("fk_ticket_classifications_accepted_invocation"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint(
            "id",
            name=op.f("pk_ticket_classifications"),
        ),
        sa.UniqueConstraint(
            "agent_run_id",
            name="uq_ticket_classifications_agent_run",
        ),
        sa.UniqueConstraint(
            "accepted_llm_invocation_id",
            name=("uq_ticket_classifications_accepted_invocation"),
        ),
    )
    op.create_index(
        ("ix_ticket_classifications_workspace_ticket_created_id"),
        "ticket_classifications",
        [
            "workspace_id",
            "ticket_id",
            sa.literal_column("created_at DESC"),
            sa.literal_column("id DESC"),
        ],
        unique=False,
    )


def downgrade() -> None:
    """Revert the migration."""

    op.drop_index(
        ("ix_ticket_classifications_workspace_ticket_created_id"),
        table_name="ticket_classifications",
    )
    op.drop_table("ticket_classifications")

    op.drop_index(
        "ix_llm_invocations_workspace_run_created_id",
        table_name="llm_invocations",
    )
    op.drop_table("llm_invocations")

    op.drop_constraint(
        "uq_agent_run_attempts_run_id",
        "agent_run_attempts",
        type_="unique",
    )
    op.drop_constraint(
        "uq_agent_runs_workspace_ticket_id",
        "agent_runs",
        type_="unique",
    )
