"""Immutable grounded support-recommendation domain entities."""

import math
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from uuid import UUID, uuid4

SUPPORT_RECOMMENDATION_SCHEMA_VERSION = "support-recommendation-v1"

SUPPORT_RECOMMENDATION_RESPONSE_MAX_LENGTH = 4_000
SUPPORT_RECOMMENDATION_DECISION_SUMMARY_MAX_LENGTH = 500
SUPPORT_RECOMMENDATION_PROMPT_ID_MAX_LENGTH = 128
SUPPORT_RECOMMENDATION_SCHEMA_VERSION_MAX_LENGTH = 128
SUPPORT_RECOMMENDATION_PROVIDER_MAX_LENGTH = 64
SUPPORT_RECOMMENDATION_MODEL_MAX_LENGTH = 128

_PROMPT_CONTENT_HASH_LENGTH = 64
_LOWERCASE_HEXADECIMAL_CHARACTERS = frozenset("0123456789abcdef")


class SupportRecommendationAction(StrEnum):
    """Non-executable outcomes proposed by the support workflow."""

    RESPOND = "respond"
    REQUEST_MORE_INFORMATION = "request_more_information"
    RECOMMEND_ESCALATION = "recommend_escalation"


@dataclass(frozen=True, slots=True)
class SupportRecommendation:
    """One immutable accepted recommendation for an AgentRun."""

    id: UUID
    workspace_id: UUID
    ticket_id: UUID
    agent_run_id: UUID
    classification_id: UUID
    accepted_llm_invocation_id: UUID
    recommended_action: SupportRecommendationAction
    response_text: str
    requires_human_review: bool
    decision_summary: str
    schema_version: str
    prompt_id: str
    prompt_version: int
    prompt_content_hash: str
    provider: str
    model: str
    created_at: datetime

    def __post_init__(self) -> None:
        if not isinstance(
            self.recommended_action,
            SupportRecommendationAction,
        ):
            raise ValueError("recommended_action must use the supported recommendation taxonomy.")

        _validate_bounded_text(
            self.response_text,
            field_name="response_text",
            maximum_length=(SUPPORT_RECOMMENDATION_RESPONSE_MAX_LENGTH),
        )

        if type(self.requires_human_review) is not bool:
            raise ValueError("requires_human_review must be a boolean.")

        _validate_bounded_text(
            self.decision_summary,
            field_name="decision_summary",
            maximum_length=(SUPPORT_RECOMMENDATION_DECISION_SUMMARY_MAX_LENGTH),
        )

        if self.schema_version != SUPPORT_RECOMMENDATION_SCHEMA_VERSION:
            raise ValueError(f"schema_version must be {SUPPORT_RECOMMENDATION_SCHEMA_VERSION}.")

        _validate_bounded_identifier(
            self.schema_version,
            field_name="schema_version",
            maximum_length=(SUPPORT_RECOMMENDATION_SCHEMA_VERSION_MAX_LENGTH),
        )
        _validate_bounded_identifier(
            self.prompt_id,
            field_name="prompt_id",
            maximum_length=(SUPPORT_RECOMMENDATION_PROMPT_ID_MAX_LENGTH),
        )

        if self.prompt_version <= 0:
            raise ValueError("prompt_version must be positive.")

        _validate_sha256_hash(
            self.prompt_content_hash,
            field_name="prompt_content_hash",
        )
        _validate_bounded_identifier(
            self.provider,
            field_name="provider",
            maximum_length=(SUPPORT_RECOMMENDATION_PROVIDER_MAX_LENGTH),
        )
        _validate_bounded_identifier(
            self.model,
            field_name="model",
            maximum_length=(SUPPORT_RECOMMENDATION_MODEL_MAX_LENGTH),
        )
        _validate_utc_timestamp(
            self.created_at,
            field_name="created_at",
        )

    @classmethod
    def create(
        cls,
        *,
        workspace_id: UUID,
        ticket_id: UUID,
        agent_run_id: UUID,
        classification_id: UUID,
        accepted_llm_invocation_id: UUID,
        recommended_action: SupportRecommendationAction,
        response_text: str,
        requires_human_review: bool,
        decision_summary: str,
        prompt_id: str,
        prompt_version: int,
        prompt_content_hash: str,
        provider: str,
        model: str,
        recommendation_id: UUID | None = None,
        now: datetime | None = None,
    ) -> "SupportRecommendation":
        """Create one immutable accepted support recommendation."""

        return cls(
            id=recommendation_id or uuid4(),
            workspace_id=workspace_id,
            ticket_id=ticket_id,
            agent_run_id=agent_run_id,
            classification_id=classification_id,
            accepted_llm_invocation_id=(accepted_llm_invocation_id),
            recommended_action=recommended_action,
            response_text=response_text.strip(),
            requires_human_review=requires_human_review,
            decision_summary=decision_summary.strip(),
            schema_version=(SUPPORT_RECOMMENDATION_SCHEMA_VERSION),
            prompt_id=prompt_id,
            prompt_version=prompt_version,
            prompt_content_hash=prompt_content_hash,
            provider=provider,
            model=model,
            created_at=now or datetime.now(UTC),
        )


@dataclass(frozen=True, slots=True)
class SupportRecommendationCitation:
    """One ordered citation to authoritative retrieved evidence."""

    id: UUID
    workspace_id: UUID
    support_recommendation_id: UUID
    ordinal: int
    document_id: UUID
    document_version_id: UUID
    chunk_id: UUID
    retrieval_query_id: UUID
    retrieval_rank: int
    retrieval_score: float
    created_at: datetime

    def __post_init__(self) -> None:
        if self.ordinal <= 0:
            raise ValueError("ordinal must be positive.")

        if self.retrieval_rank < 0:
            raise ValueError("retrieval_rank must be non-negative.")

        if type(self.retrieval_score) is not float:
            raise TypeError("retrieval_score must be a float.")

        if not math.isfinite(self.retrieval_score):
            raise ValueError("retrieval_score must be finite.")

        _validate_utc_timestamp(
            self.created_at,
            field_name="created_at",
        )

    @classmethod
    def create(
        cls,
        *,
        workspace_id: UUID,
        support_recommendation_id: UUID,
        ordinal: int,
        document_id: UUID,
        document_version_id: UUID,
        chunk_id: UUID,
        retrieval_query_id: UUID,
        retrieval_rank: int,
        retrieval_score: float,
        citation_id: UUID | None = None,
        now: datetime | None = None,
    ) -> "SupportRecommendationCitation":
        """Create one immutable ordered recommendation citation."""

        return cls(
            id=citation_id or uuid4(),
            workspace_id=workspace_id,
            support_recommendation_id=(support_recommendation_id),
            ordinal=ordinal,
            document_id=document_id,
            document_version_id=document_version_id,
            chunk_id=chunk_id,
            retrieval_query_id=retrieval_query_id,
            retrieval_rank=retrieval_rank,
            retrieval_score=retrieval_score,
            created_at=now or datetime.now(UTC),
        )


def _validate_sha256_hash(
    value: str,
    *,
    field_name: str,
) -> None:
    if len(value) != _PROMPT_CONTENT_HASH_LENGTH:
        raise ValueError(f"{field_name} must be a lowercase SHA-256 hash.")

    if any(character not in _LOWERCASE_HEXADECIMAL_CHARACTERS for character in value):
        raise ValueError(f"{field_name} must be a lowercase SHA-256 hash.")


def _validate_bounded_identifier(
    value: str,
    *,
    field_name: str,
    maximum_length: int,
) -> None:
    if not value:
        raise ValueError(f"{field_name} is required.")

    if value != value.strip():
        raise ValueError(f"{field_name} must not contain surrounding whitespace.")

    if len(value) > maximum_length:
        raise ValueError(f"{field_name} exceeds the maximum length.")


def _validate_bounded_text(
    value: str,
    *,
    field_name: str,
    maximum_length: int,
) -> None:
    _validate_bounded_identifier(
        value,
        field_name=field_name,
        maximum_length=maximum_length,
    )


def _validate_utc_timestamp(
    value: datetime,
    *,
    field_name: str,
) -> None:
    if value.tzinfo is None or value.utcoffset() != timedelta(0):
        raise ValueError(f"{field_name} must be a UTC-aware timestamp.")
