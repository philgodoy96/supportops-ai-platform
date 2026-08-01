"""SQLAlchemy models for classifications and logical LLM invocations."""

from datetime import datetime
from decimal import Decimal
from typing import cast
from uuid import UUID

from sqlalchemy import (
    BigInteger,
    Boolean,
    CheckConstraint,
    DateTime,
    ForeignKeyConstraint,
    Index,
    Integer,
    Numeric,
    String,
    UniqueConstraint,
    Uuid,
)
from sqlalchemy.orm import Mapped, mapped_column

from supportops.ai.gateway.errors import LLMErrorCode
from supportops.ai.gateway.results import LLMInvocationStatus
from supportops.ai.schemas.ticket_classification import (
    TICKET_CLASSIFICATION_SCHEMA_VERSION,
    TicketCategory,
    TicketClassificationSchemaVersion,
    TicketIntent,
    TicketSentiment,
    TicketUrgency,
)
from supportops.infrastructure.postgresql.base import Base
from supportops.modules.ticket_classifications.domain.models import (
    LLM_INVOCATION_MONETARY_SCALE,
    LLM_INVOCATION_PRICING_CATALOG_VERSION_MAX_LENGTH,
    LLM_INVOCATION_PROVIDER_REQUEST_ID_MAX_LENGTH,
    TICKET_CLASSIFICATION_MODEL_MAX_LENGTH,
    TICKET_CLASSIFICATION_PROMPT_ID_MAX_LENGTH,
    TICKET_CLASSIFICATION_PROVIDER_MAX_LENGTH,
    TICKET_CLASSIFICATION_SCHEMA_VERSION_MAX_LENGTH,
    TICKET_CLASSIFICATION_SUMMARY_MAX_LENGTH,
    LLMInvocation,
    TicketClassification,
)

_LLM_INVOCATION_COST_PRECISION = 20
_PROMPT_CONTENT_HASH_LENGTH = 64
_ERROR_CODE_MAX_LENGTH = 64
_INVOCATION_STATUS_MAX_LENGTH = 32
_TAXONOMY_VALUE_MAX_LENGTH = 32

_CATEGORY_SQL_VALUES = ", ".join(f"'{member.value}'" for member in TicketCategory)
_INTENT_SQL_VALUES = ", ".join(f"'{member.value}'" for member in TicketIntent)
_URGENCY_SQL_VALUES = ", ".join(f"'{member.value}'" for member in TicketUrgency)
_SENTIMENT_SQL_VALUES = ", ".join(f"'{member.value}'" for member in TicketSentiment)
_INVOCATION_STATUS_SQL_VALUES = ", ".join(f"'{member.value}'" for member in LLMInvocationStatus)
_ERROR_CODE_SQL_VALUES = ", ".join(f"'{member.value}'" for member in LLMErrorCode)


class LLMInvocationRecord(Base):
    """Persisted metadata for one logical LLM provider invocation."""

    __tablename__ = "llm_invocations"

    id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True),
        primary_key=True,
    )
    workspace_id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True),
        nullable=False,
    )
    ticket_id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True),
        nullable=False,
    )
    agent_run_id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True),
        nullable=False,
    )
    agent_run_attempt_id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True),
        nullable=False,
    )
    invocation_sequence: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )
    status: Mapped[str] = mapped_column(
        String(_INVOCATION_STATUS_MAX_LENGTH),
        nullable=False,
    )
    provider: Mapped[str] = mapped_column(
        String(TICKET_CLASSIFICATION_PROVIDER_MAX_LENGTH),
        nullable=False,
    )
    model: Mapped[str] = mapped_column(
        String(TICKET_CLASSIFICATION_MODEL_MAX_LENGTH),
        nullable=False,
    )
    provider_request_id: Mapped[str | None] = mapped_column(
        String(LLM_INVOCATION_PROVIDER_REQUEST_ID_MAX_LENGTH),
        nullable=True,
    )
    prompt_id: Mapped[str] = mapped_column(
        String(TICKET_CLASSIFICATION_PROMPT_ID_MAX_LENGTH),
        nullable=False,
    )
    prompt_version: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )
    prompt_content_hash: Mapped[str] = mapped_column(
        String(_PROMPT_CONTENT_HASH_LENGTH),
        nullable=False,
    )
    schema_version: Mapped[str] = mapped_column(
        String(TICKET_CLASSIFICATION_SCHEMA_VERSION_MAX_LENGTH),
        nullable=False,
    )
    input_tokens: Mapped[int | None] = mapped_column(
        BigInteger,
        nullable=True,
    )
    cached_input_tokens: Mapped[int | None] = mapped_column(
        BigInteger,
        nullable=True,
    )
    output_tokens: Mapped[int | None] = mapped_column(
        BigInteger,
        nullable=True,
    )
    reasoning_tokens: Mapped[int | None] = mapped_column(
        BigInteger,
        nullable=True,
    )
    total_tokens: Mapped[int | None] = mapped_column(
        BigInteger,
        nullable=True,
    )
    pricing_catalog_version: Mapped[str] = mapped_column(
        String(LLM_INVOCATION_PRICING_CATALOG_VERSION_MAX_LENGTH),
        nullable=False,
    )
    pricing_found: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
    )
    estimated_input_cost_usd: Mapped[Decimal | None] = mapped_column(
        Numeric(
            _LLM_INVOCATION_COST_PRECISION,
            LLM_INVOCATION_MONETARY_SCALE,
            asdecimal=True,
        ),
        nullable=True,
    )
    estimated_cached_input_cost_usd: Mapped[Decimal | None] = mapped_column(
        Numeric(
            _LLM_INVOCATION_COST_PRECISION,
            LLM_INVOCATION_MONETARY_SCALE,
            asdecimal=True,
        ),
        nullable=True,
    )
    estimated_output_cost_usd: Mapped[Decimal | None] = mapped_column(
        Numeric(
            _LLM_INVOCATION_COST_PRECISION,
            LLM_INVOCATION_MONETARY_SCALE,
            asdecimal=True,
        ),
        nullable=True,
    )
    estimated_total_cost_usd: Mapped[Decimal | None] = mapped_column(
        Numeric(
            _LLM_INVOCATION_COST_PRECISION,
            LLM_INVOCATION_MONETARY_SCALE,
            asdecimal=True,
        ),
        nullable=True,
    )
    latency_ms: Mapped[int] = mapped_column(
        BigInteger,
        nullable=False,
    )
    error_code: Mapped[str | None] = mapped_column(
        String(_ERROR_CODE_MAX_LENGTH),
        nullable=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
    )

    __table_args__ = (
        ForeignKeyConstraint(
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
        ForeignKeyConstraint(
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
        UniqueConstraint(
            "agent_run_attempt_id",
            "invocation_sequence",
            name="uq_llm_invocations_attempt_sequence",
        ),
        UniqueConstraint(
            "agent_run_id",
            "id",
            name="uq_llm_invocations_run_id",
        ),
        CheckConstraint(
            "invocation_sequence >= 1",
            name="llm_invocation_sequence_positive",
        ),
        CheckConstraint(
            f"status IN ({_INVOCATION_STATUS_SQL_VALUES})",
            name="llm_invocation_status",
        ),
        CheckConstraint(
            ("provider = btrim(provider) AND char_length(provider) BETWEEN 1 AND 64"),
            name="llm_invocation_provider_format",
        ),
        CheckConstraint(
            ("model = btrim(model) AND char_length(model) BETWEEN 1 AND 128"),
            name="llm_invocation_model_format",
        ),
        CheckConstraint(
            (
                "provider_request_id IS NULL OR ("
                "provider_request_id = btrim(provider_request_id) "
                "AND char_length(provider_request_id) "
                "BETWEEN 1 AND 255"
                ")"
            ),
            name="llm_invocation_provider_request_id_format",
        ),
        CheckConstraint(
            ("prompt_id = btrim(prompt_id) AND char_length(prompt_id) BETWEEN 1 AND 128"),
            name="llm_invocation_prompt_id_format",
        ),
        CheckConstraint(
            "prompt_version >= 1",
            name="llm_invocation_prompt_version_positive",
        ),
        CheckConstraint(
            ("prompt_content_hash ~ '^[0-9a-f]{64}$'"),
            name="llm_invocation_prompt_content_hash",
        ),
        CheckConstraint(
            (
                "schema_version = btrim(schema_version) "
                "AND char_length(schema_version) BETWEEN 1 AND 128"
            ),
            name="llm_invocation_schema_version_format",
        ),
        CheckConstraint(
            (
                "pricing_catalog_version "
                "= btrim(pricing_catalog_version) "
                "AND char_length(pricing_catalog_version) "
                "BETWEEN 1 AND 128"
            ),
            name="llm_invocation_pricing_catalog_version_format",
        ),
        CheckConstraint(
            (
                "(input_tokens IS NULL OR input_tokens >= 0) "
                "AND ("
                "cached_input_tokens IS NULL "
                "OR cached_input_tokens >= 0"
                ") "
                "AND ("
                "output_tokens IS NULL OR output_tokens >= 0"
                ") "
                "AND ("
                "reasoning_tokens IS NULL "
                "OR reasoning_tokens >= 0"
                ") "
                "AND ("
                "total_tokens IS NULL OR total_tokens >= 0"
                ")"
            ),
            name="llm_invocation_tokens_non_negative",
        ),
        CheckConstraint(
            (
                "cached_input_tokens IS NULL "
                "OR input_tokens IS NULL "
                "OR cached_input_tokens <= input_tokens"
            ),
            name="llm_invocation_cached_input_limit",
        ),
        CheckConstraint(
            (
                "reasoning_tokens IS NULL "
                "OR output_tokens IS NULL "
                "OR reasoning_tokens <= output_tokens"
            ),
            name="llm_invocation_reasoning_token_limit",
        ),
        CheckConstraint(
            (
                "input_tokens IS NULL "
                "OR output_tokens IS NULL "
                "OR total_tokens IS NULL "
                "OR total_tokens = input_tokens + output_tokens"
            ),
            name="llm_invocation_total_token_consistency",
        ),
        CheckConstraint(
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
            name="llm_invocation_costs_non_negative",
        ),
        CheckConstraint(
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
            name="llm_invocation_pricing_state",
        ),
        CheckConstraint(
            "latency_ms >= 0",
            name="llm_invocation_latency_non_negative",
        ),
        CheckConstraint(
            (f"error_code IS NULL OR error_code IN ({_ERROR_CODE_SQL_VALUES})"),
            name="llm_invocation_error_code",
        ),
        CheckConstraint(
            (
                "("
                "status = 'succeeded' "
                "AND error_code IS NULL"
                ") OR ("
                "status <> 'succeeded' "
                "AND error_code IS NOT NULL"
                ")"
            ),
            name="llm_invocation_error_state",
        ),
        Index(
            "ix_llm_invocations_workspace_run_created_id",
            "workspace_id",
            "agent_run_id",
            created_at.desc(),
            id.desc(),
        ),
    )

    @classmethod
    def from_domain(
        cls,
        invocation: LLMInvocation,
    ) -> "LLMInvocationRecord":
        """Create a persistence record from a logical invocation."""

        return cls(
            id=invocation.id,
            workspace_id=invocation.workspace_id,
            ticket_id=invocation.ticket_id,
            agent_run_id=invocation.agent_run_id,
            agent_run_attempt_id=(invocation.agent_run_attempt_id),
            invocation_sequence=invocation.invocation_sequence,
            status=invocation.status.value,
            provider=invocation.provider,
            model=invocation.model,
            provider_request_id=(invocation.provider_request_id),
            prompt_id=invocation.prompt_id,
            prompt_version=invocation.prompt_version,
            prompt_content_hash=(invocation.prompt_content_hash),
            schema_version=invocation.schema_version,
            input_tokens=invocation.input_tokens,
            cached_input_tokens=(invocation.cached_input_tokens),
            output_tokens=invocation.output_tokens,
            reasoning_tokens=invocation.reasoning_tokens,
            total_tokens=invocation.total_tokens,
            pricing_catalog_version=(invocation.pricing_catalog_version),
            pricing_found=invocation.pricing_found,
            estimated_input_cost_usd=(invocation.estimated_input_cost_usd),
            estimated_cached_input_cost_usd=(invocation.estimated_cached_input_cost_usd),
            estimated_output_cost_usd=(invocation.estimated_output_cost_usd),
            estimated_total_cost_usd=(invocation.estimated_total_cost_usd),
            latency_ms=invocation.latency_ms,
            error_code=(invocation.error_code.value if invocation.error_code is not None else None),
            created_at=invocation.created_at,
        )

    def to_domain(self) -> LLMInvocation:
        """Map the persistence record to a logical invocation entity."""

        return LLMInvocation(
            id=self.id,
            workspace_id=self.workspace_id,
            ticket_id=self.ticket_id,
            agent_run_id=self.agent_run_id,
            agent_run_attempt_id=self.agent_run_attempt_id,
            invocation_sequence=self.invocation_sequence,
            status=LLMInvocationStatus(self.status),
            provider=self.provider,
            model=self.model,
            provider_request_id=self.provider_request_id,
            prompt_id=self.prompt_id,
            prompt_version=self.prompt_version,
            prompt_content_hash=self.prompt_content_hash,
            schema_version=self.schema_version,
            input_tokens=self.input_tokens,
            cached_input_tokens=self.cached_input_tokens,
            output_tokens=self.output_tokens,
            reasoning_tokens=self.reasoning_tokens,
            total_tokens=self.total_tokens,
            pricing_catalog_version=(self.pricing_catalog_version),
            pricing_found=self.pricing_found,
            estimated_input_cost_usd=(self.estimated_input_cost_usd),
            estimated_cached_input_cost_usd=(self.estimated_cached_input_cost_usd),
            estimated_output_cost_usd=(self.estimated_output_cost_usd),
            estimated_total_cost_usd=(self.estimated_total_cost_usd),
            latency_ms=self.latency_ms,
            error_code=(LLMErrorCode(self.error_code) if self.error_code is not None else None),
            created_at=self.created_at,
        )


class TicketClassificationRecord(Base):
    """Persisted accepted structured classification for one AgentRun."""

    __tablename__ = "ticket_classifications"

    id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True),
        primary_key=True,
    )
    workspace_id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True),
        nullable=False,
    )
    ticket_id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True),
        nullable=False,
    )
    agent_run_id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True),
        nullable=False,
    )
    accepted_llm_invocation_id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True),
        nullable=False,
    )
    category: Mapped[str] = mapped_column(
        String(_TAXONOMY_VALUE_MAX_LENGTH),
        nullable=False,
    )
    intent: Mapped[str] = mapped_column(
        String(_TAXONOMY_VALUE_MAX_LENGTH),
        nullable=False,
    )
    urgency: Mapped[str] = mapped_column(
        String(_TAXONOMY_VALUE_MAX_LENGTH),
        nullable=False,
    )
    sentiment: Mapped[str] = mapped_column(
        String(_TAXONOMY_VALUE_MAX_LENGTH),
        nullable=False,
    )
    requires_human_review: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
    )
    summary: Mapped[str] = mapped_column(
        String(TICKET_CLASSIFICATION_SUMMARY_MAX_LENGTH),
        nullable=False,
    )
    schema_version: Mapped[str] = mapped_column(
        String(TICKET_CLASSIFICATION_SCHEMA_VERSION_MAX_LENGTH),
        nullable=False,
    )
    prompt_id: Mapped[str] = mapped_column(
        String(TICKET_CLASSIFICATION_PROMPT_ID_MAX_LENGTH),
        nullable=False,
    )
    prompt_version: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )
    prompt_content_hash: Mapped[str] = mapped_column(
        String(_PROMPT_CONTENT_HASH_LENGTH),
        nullable=False,
    )
    provider: Mapped[str] = mapped_column(
        String(TICKET_CLASSIFICATION_PROVIDER_MAX_LENGTH),
        nullable=False,
    )
    model: Mapped[str] = mapped_column(
        String(TICKET_CLASSIFICATION_MODEL_MAX_LENGTH),
        nullable=False,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
    )

    __table_args__ = (
        ForeignKeyConstraint(
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
        ForeignKeyConstraint(
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
        UniqueConstraint(
            "agent_run_id",
            name="uq_ticket_classifications_agent_run",
        ),
        UniqueConstraint(
            "accepted_llm_invocation_id",
            name=("uq_ticket_classifications_accepted_invocation"),
        ),
        CheckConstraint(
            f"category IN ({_CATEGORY_SQL_VALUES})",
            name="ticket_classification_category",
        ),
        CheckConstraint(
            f"intent IN ({_INTENT_SQL_VALUES})",
            name="ticket_classification_intent",
        ),
        CheckConstraint(
            f"urgency IN ({_URGENCY_SQL_VALUES})",
            name="ticket_classification_urgency",
        ),
        CheckConstraint(
            f"sentiment IN ({_SENTIMENT_SQL_VALUES})",
            name="ticket_classification_sentiment",
        ),
        CheckConstraint(
            ("summary = btrim(summary) AND char_length(summary) BETWEEN 1 AND 500"),
            name="ticket_classification_summary_format",
        ),
        CheckConstraint(
            (f"schema_version = '{TICKET_CLASSIFICATION_SCHEMA_VERSION}'"),
            name="ticket_classification_schema_version",
        ),
        CheckConstraint(
            ("prompt_id = btrim(prompt_id) AND char_length(prompt_id) BETWEEN 1 AND 128"),
            name="ticket_classification_prompt_id_format",
        ),
        CheckConstraint(
            "prompt_version >= 1",
            name="ticket_classification_prompt_version_positive",
        ),
        CheckConstraint(
            ("prompt_content_hash ~ '^[0-9a-f]{64}$'"),
            name="ticket_classification_prompt_content_hash",
        ),
        CheckConstraint(
            ("provider = btrim(provider) AND char_length(provider) BETWEEN 1 AND 64"),
            name="ticket_classification_provider_format",
        ),
        CheckConstraint(
            ("model = btrim(model) AND char_length(model) BETWEEN 1 AND 128"),
            name="ticket_classification_model_format",
        ),
        CheckConstraint(
            "updated_at = created_at",
            name="ticket_classification_immutable_timestamp",
        ),
        Index(
            "ix_ticket_classifications_workspace_ticket_created_id",
            "workspace_id",
            "ticket_id",
            created_at.desc(),
            id.desc(),
        ),
    )

    @classmethod
    def from_domain(
        cls,
        classification: TicketClassification,
    ) -> "TicketClassificationRecord":
        """Create a persistence record from an accepted classification."""

        return cls(
            id=classification.id,
            workspace_id=classification.workspace_id,
            ticket_id=classification.ticket_id,
            agent_run_id=classification.agent_run_id,
            accepted_llm_invocation_id=(classification.accepted_llm_invocation_id),
            category=classification.category.value,
            intent=classification.intent.value,
            urgency=classification.urgency.value,
            sentiment=classification.sentiment.value,
            requires_human_review=(classification.requires_human_review),
            summary=classification.summary,
            schema_version=classification.schema_version,
            prompt_id=classification.prompt_id,
            prompt_version=classification.prompt_version,
            prompt_content_hash=(classification.prompt_content_hash),
            provider=classification.provider,
            model=classification.model,
            created_at=classification.created_at,
            updated_at=classification.updated_at,
        )

    def to_domain(self) -> TicketClassification:
        """Map the persistence record to an accepted classification."""

        return TicketClassification(
            id=self.id,
            workspace_id=self.workspace_id,
            ticket_id=self.ticket_id,
            agent_run_id=self.agent_run_id,
            accepted_llm_invocation_id=(self.accepted_llm_invocation_id),
            category=TicketCategory(self.category),
            intent=TicketIntent(self.intent),
            urgency=TicketUrgency(self.urgency),
            sentiment=TicketSentiment(self.sentiment),
            requires_human_review=(self.requires_human_review),
            summary=self.summary,
            schema_version=cast(
                TicketClassificationSchemaVersion,
                self.schema_version,
            ),
            prompt_id=self.prompt_id,
            prompt_version=self.prompt_version,
            prompt_content_hash=self.prompt_content_hash,
            provider=self.provider,
            model=self.model,
            created_at=self.created_at,
            updated_at=self.updated_at,
        )
