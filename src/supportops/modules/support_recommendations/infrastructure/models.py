"""SQLAlchemy models for grounded recommendations and citations."""

from datetime import datetime
from uuid import UUID

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    DateTime,
    Float,
    ForeignKeyConstraint,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    Uuid,
)
from sqlalchemy.orm import Mapped, mapped_column

from supportops.infrastructure.postgresql.base import Base
from supportops.modules.support_recommendations.domain.models import (
    SUPPORT_RECOMMENDATION_DECISION_SUMMARY_MAX_LENGTH,
    SUPPORT_RECOMMENDATION_MODEL_MAX_LENGTH,
    SUPPORT_RECOMMENDATION_PROMPT_ID_MAX_LENGTH,
    SUPPORT_RECOMMENDATION_PROVIDER_MAX_LENGTH,
    SUPPORT_RECOMMENDATION_RESPONSE_MAX_LENGTH,
    SUPPORT_RECOMMENDATION_SCHEMA_VERSION,
    SUPPORT_RECOMMENDATION_SCHEMA_VERSION_MAX_LENGTH,
    SupportRecommendation,
    SupportRecommendationAction,
    SupportRecommendationCitation,
)

_PROMPT_CONTENT_HASH_LENGTH = 64
_ACTION_MAX_LENGTH = 32

_RECOMMENDATION_ACTION_SQL_VALUES = ", ".join(
    f"'{member.value}'" for member in SupportRecommendationAction
)


class SupportRecommendationRecord(Base):
    """Persisted immutable recommendation accepted for an AgentRun."""

    __tablename__ = "support_recommendations"

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
    classification_id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True),
        nullable=False,
    )
    accepted_llm_invocation_id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True),
        nullable=False,
    )
    recommended_action: Mapped[str] = mapped_column(
        String(_ACTION_MAX_LENGTH),
        nullable=False,
    )
    response_text: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )
    requires_human_review: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
    )
    decision_summary: Mapped[str] = mapped_column(
        String(SUPPORT_RECOMMENDATION_DECISION_SUMMARY_MAX_LENGTH),
        nullable=False,
    )
    schema_version: Mapped[str] = mapped_column(
        String(SUPPORT_RECOMMENDATION_SCHEMA_VERSION_MAX_LENGTH),
        nullable=False,
    )
    prompt_id: Mapped[str] = mapped_column(
        String(SUPPORT_RECOMMENDATION_PROMPT_ID_MAX_LENGTH),
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
        String(SUPPORT_RECOMMENDATION_PROVIDER_MAX_LENGTH),
        nullable=False,
    )
    model: Mapped[str] = mapped_column(
        String(SUPPORT_RECOMMENDATION_MODEL_MAX_LENGTH),
        nullable=False,
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
            name=("fk_support_recommendations_workspace_ticket_agent_run"),
            ondelete="CASCADE",
        ),
        ForeignKeyConstraint(
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
        ForeignKeyConstraint(
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
        UniqueConstraint(
            "agent_run_id",
            name="uq_support_recommendations_agent_run",
        ),
        UniqueConstraint(
            "accepted_llm_invocation_id",
            name=("uq_support_recommendations_accepted_invocation"),
        ),
        UniqueConstraint(
            "workspace_id",
            "id",
            name="uq_support_recommendations_workspace_id",
        ),
        CheckConstraint(
            (f"recommended_action IN ({_RECOMMENDATION_ACTION_SQL_VALUES})"),
            name="support_recommendation_action",
        ),
        CheckConstraint(
            (
                "response_text = btrim(response_text) "
                "AND char_length(response_text) "
                f"BETWEEN 1 AND "
                f"{SUPPORT_RECOMMENDATION_RESPONSE_MAX_LENGTH}"
            ),
            name="support_recommendation_response_format",
        ),
        CheckConstraint(
            (
                "decision_summary = btrim(decision_summary) "
                "AND char_length(decision_summary) "
                f"BETWEEN 1 AND "
                f"{SUPPORT_RECOMMENDATION_DECISION_SUMMARY_MAX_LENGTH}"
            ),
            name="support_recommendation_summary_format",
        ),
        CheckConstraint(
            (f"schema_version = '{SUPPORT_RECOMMENDATION_SCHEMA_VERSION}'"),
            name="support_recommendation_schema_version",
        ),
        CheckConstraint(
            ("prompt_id = btrim(prompt_id) AND char_length(prompt_id) BETWEEN 1 AND 128"),
            name="support_recommendation_prompt_id_format",
        ),
        CheckConstraint(
            "prompt_version >= 1",
            name="support_recommendation_prompt_version_positive",
        ),
        CheckConstraint(
            "prompt_content_hash ~ '^[0-9a-f]{64}$'",
            name="support_recommendation_prompt_content_hash",
        ),
        CheckConstraint(
            ("provider = btrim(provider) AND char_length(provider) BETWEEN 1 AND 64"),
            name="support_recommendation_provider_format",
        ),
        CheckConstraint(
            ("model = btrim(model) AND char_length(model) BETWEEN 1 AND 128"),
            name="support_recommendation_model_format",
        ),
        Index(
            "ix_support_recommendations_workspace_ticket_created_id",
            "workspace_id",
            "ticket_id",
            created_at.desc(),
            id.desc(),
        ),
    )

    @classmethod
    def from_domain(
        cls,
        recommendation: SupportRecommendation,
    ) -> "SupportRecommendationRecord":
        """Create a persistence record from an accepted recommendation."""

        return cls(
            id=recommendation.id,
            workspace_id=recommendation.workspace_id,
            ticket_id=recommendation.ticket_id,
            agent_run_id=recommendation.agent_run_id,
            classification_id=recommendation.classification_id,
            accepted_llm_invocation_id=(recommendation.accepted_llm_invocation_id),
            recommended_action=(recommendation.recommended_action.value),
            response_text=recommendation.response_text,
            requires_human_review=(recommendation.requires_human_review),
            decision_summary=recommendation.decision_summary,
            schema_version=recommendation.schema_version,
            prompt_id=recommendation.prompt_id,
            prompt_version=recommendation.prompt_version,
            prompt_content_hash=(recommendation.prompt_content_hash),
            provider=recommendation.provider,
            model=recommendation.model,
            created_at=recommendation.created_at,
        )

    def to_domain(self) -> SupportRecommendation:
        """Map the persistence record to an accepted recommendation."""

        return SupportRecommendation(
            id=self.id,
            workspace_id=self.workspace_id,
            ticket_id=self.ticket_id,
            agent_run_id=self.agent_run_id,
            classification_id=self.classification_id,
            accepted_llm_invocation_id=(self.accepted_llm_invocation_id),
            recommended_action=SupportRecommendationAction(self.recommended_action),
            response_text=self.response_text,
            requires_human_review=self.requires_human_review,
            decision_summary=self.decision_summary,
            schema_version=self.schema_version,
            prompt_id=self.prompt_id,
            prompt_version=self.prompt_version,
            prompt_content_hash=self.prompt_content_hash,
            provider=self.provider,
            model=self.model,
            created_at=self.created_at,
        )


class SupportRecommendationCitationRecord(Base):
    """Persisted ordered reference to authoritative knowledge evidence."""

    __tablename__ = "support_recommendation_citations"

    id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True),
        primary_key=True,
    )
    workspace_id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True),
        nullable=False,
    )
    support_recommendation_id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True),
        nullable=False,
    )
    ordinal: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )
    document_id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True),
        nullable=False,
    )
    document_version_id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True),
        nullable=False,
    )
    chunk_id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True),
        nullable=False,
    )
    retrieval_query_id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True),
        nullable=False,
    )
    retrieval_rank: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )
    retrieval_score: Mapped[float] = mapped_column(
        Float,
        nullable=False,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
    )

    __table_args__ = (
        ForeignKeyConstraint(
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
        ForeignKeyConstraint(
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
        UniqueConstraint(
            "support_recommendation_id",
            "ordinal",
            name=("uq_support_recommendation_citations_recommendation_ordinal"),
        ),
        UniqueConstraint(
            "support_recommendation_id",
            "chunk_id",
            name=("uq_support_recommendation_citations_recommendation_chunk"),
        ),
        CheckConstraint(
            "ordinal >= 1",
            name=("support_recommendation_citation_ordinal_positive"),
        ),
        CheckConstraint(
            "retrieval_rank >= 0",
            name=("support_recommendation_citation_rank_non_negative"),
        ),
        CheckConstraint(
            (
                "retrieval_score <> 'NaN'::double precision "
                "AND retrieval_score <> "
                "'Infinity'::double precision "
                "AND retrieval_score <> "
                "'-Infinity'::double precision"
            ),
            name=("support_recommendation_citation_score_finite"),
        ),
        Index(
            "ix_support_recommendation_citations_recommendation_ordinal",
            "support_recommendation_id",
            "ordinal",
        ),
    )

    @classmethod
    def from_domain(
        cls,
        citation: SupportRecommendationCitation,
    ) -> "SupportRecommendationCitationRecord":
        """Create a persistence record from an ordered citation."""

        return cls(
            id=citation.id,
            workspace_id=citation.workspace_id,
            support_recommendation_id=(citation.support_recommendation_id),
            ordinal=citation.ordinal,
            document_id=citation.document_id,
            document_version_id=citation.document_version_id,
            chunk_id=citation.chunk_id,
            retrieval_query_id=citation.retrieval_query_id,
            retrieval_rank=citation.retrieval_rank,
            retrieval_score=citation.retrieval_score,
            created_at=citation.created_at,
        )

    def to_domain(self) -> SupportRecommendationCitation:
        """Map the persistence record to an ordered citation."""

        return SupportRecommendationCitation(
            id=self.id,
            workspace_id=self.workspace_id,
            support_recommendation_id=(self.support_recommendation_id),
            ordinal=self.ordinal,
            document_id=self.document_id,
            document_version_id=self.document_version_id,
            chunk_id=self.chunk_id,
            retrieval_query_id=self.retrieval_query_id,
            retrieval_rank=self.retrieval_rank,
            retrieval_score=self.retrieval_score,
            created_at=self.created_at,
        )
