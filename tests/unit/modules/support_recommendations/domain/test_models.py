"""Unit tests for grounded support-recommendation entities."""

from datetime import UTC, datetime
from typing import Any, cast
from uuid import UUID

import pytest

from supportops.modules.support_recommendations.domain.models import (
    SUPPORT_RECOMMENDATION_DECISION_SUMMARY_MAX_LENGTH,
    SUPPORT_RECOMMENDATION_RESPONSE_MAX_LENGTH,
    SUPPORT_RECOMMENDATION_SCHEMA_VERSION,
    SupportRecommendation,
    SupportRecommendationAction,
    SupportRecommendationCitation,
)

RECOMMENDATION_ID = UUID("11111111-1111-4111-8111-111111111111")
CITATION_ID = UUID("22222222-2222-4222-8222-222222222222")
WORKSPACE_ID = UUID("33333333-3333-4333-8333-333333333333")
TICKET_ID = UUID("44444444-4444-4444-8444-444444444444")
AGENT_RUN_ID = UUID("55555555-5555-4555-8555-555555555555")
CLASSIFICATION_ID = UUID("66666666-6666-4666-8666-666666666666")
INVOCATION_ID = UUID("77777777-7777-4777-8777-777777777777")
DOCUMENT_ID = UUID("88888888-8888-4888-8888-888888888888")
DOCUMENT_VERSION_ID = UUID("99999999-9999-4999-8999-999999999999")
CHUNK_ID = UUID("aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa")
RETRIEVAL_QUERY_ID = UUID("bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb")
CREATED_AT = datetime(
    2026,
    8,
    2,
    15,
    30,
    tzinfo=UTC,
)
PROMPT_HASH = "a" * 64


def create_recommendation(
    *,
    recommended_action: SupportRecommendationAction = (SupportRecommendationAction.RESPOND),
    response_text: str = ("Follow the documented access-reset procedure."),
    requires_human_review: bool = False,
    decision_summary: str = ("The active runbook contains a matching procedure."),
    prompt_content_hash: str = PROMPT_HASH,
    now: datetime = CREATED_AT,
) -> SupportRecommendation:
    """Create one valid recommendation."""

    return SupportRecommendation.create(
        recommendation_id=RECOMMENDATION_ID,
        workspace_id=WORKSPACE_ID,
        ticket_id=TICKET_ID,
        agent_run_id=AGENT_RUN_ID,
        classification_id=CLASSIFICATION_ID,
        accepted_llm_invocation_id=INVOCATION_ID,
        recommended_action=recommended_action,
        response_text=response_text,
        requires_human_review=requires_human_review,
        decision_summary=decision_summary,
        prompt_id="support-recommendation-draft",
        prompt_version=1,
        prompt_content_hash=prompt_content_hash,
        provider="mock",
        model="mock-support-model-v1",
        now=now,
    )


def test_creates_immutable_grounded_recommendation() -> None:
    recommendation = create_recommendation(
        response_text=("  Follow the documented reset procedure.  "),
        decision_summary=("  Relevant runbook evidence is available.  "),
    )

    assert recommendation.id == RECOMMENDATION_ID
    assert recommendation.workspace_id == WORKSPACE_ID
    assert recommendation.ticket_id == TICKET_ID
    assert recommendation.agent_run_id == AGENT_RUN_ID
    assert recommendation.classification_id == CLASSIFICATION_ID
    assert recommendation.accepted_llm_invocation_id == INVOCATION_ID
    assert recommendation.recommended_action is SupportRecommendationAction.RESPOND
    assert recommendation.response_text == ("Follow the documented reset procedure.")
    assert recommendation.decision_summary == ("Relevant runbook evidence is available.")
    assert recommendation.schema_version == (SUPPORT_RECOMMENDATION_SCHEMA_VERSION)
    assert recommendation.created_at == CREATED_AT

    with pytest.raises(AttributeError):
        recommendation.response_text = "Changed"  # type: ignore[misc]


@pytest.mark.parametrize(
    "action",
    list(SupportRecommendationAction),
)
def test_supports_only_non_executable_actions(
    action: SupportRecommendationAction,
) -> None:
    recommendation = create_recommendation(
        recommended_action=action,
    )

    assert recommendation.recommended_action is action


def test_rejects_string_instead_of_action_enum() -> None:
    with pytest.raises(
        ValueError,
        match="supported recommendation taxonomy",
    ):
        create_recommendation(
            recommended_action="respond",  # type: ignore[arg-type]
        )


def test_rejects_non_boolean_human_review() -> None:
    with pytest.raises(
        ValueError,
        match="must be a boolean",
    ):
        create_recommendation(
            requires_human_review=1,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    ("response_text", "expected_message"),
    [
        (
            "",
            "response_text is required",
        ),
        (
            "x" * (SUPPORT_RECOMMENDATION_RESPONSE_MAX_LENGTH + 1),
            "response_text exceeds",
        ),
    ],
)
def test_validates_response_text(
    response_text: str,
    expected_message: str,
) -> None:
    with pytest.raises(
        ValueError,
        match=expected_message,
    ):
        create_recommendation(
            response_text=response_text,
        )


def test_validates_decision_summary_bound() -> None:
    with pytest.raises(
        ValueError,
        match="decision_summary exceeds",
    ):
        create_recommendation(
            decision_summary=("x" * (SUPPORT_RECOMMENDATION_DECISION_SUMMARY_MAX_LENGTH + 1)),
        )


@pytest.mark.parametrize(
    "prompt_hash",
    [
        "",
        "a" * 63,
        "A" * 64,
        "z" * 64,
    ],
)
def test_rejects_invalid_prompt_hash(
    prompt_hash: str,
) -> None:
    with pytest.raises(
        ValueError,
        match="lowercase SHA-256",
    ):
        create_recommendation(
            prompt_content_hash=prompt_hash,
        )


def test_rejects_non_utc_recommendation_timestamp() -> None:
    with pytest.raises(
        ValueError,
        match="UTC-aware",
    ):
        create_recommendation(
            now=CREATED_AT.replace(tzinfo=None),
        )


def test_creates_exact_historical_citation() -> None:
    citation = SupportRecommendationCitation.create(
        citation_id=CITATION_ID,
        workspace_id=WORKSPACE_ID,
        support_recommendation_id=(RECOMMENDATION_ID),
        ordinal=1,
        document_id=DOCUMENT_ID,
        document_version_id=DOCUMENT_VERSION_ID,
        chunk_id=CHUNK_ID,
        retrieval_query_id=RETRIEVAL_QUERY_ID,
        retrieval_rank=0,
        retrieval_score=0.875,
        now=CREATED_AT,
    )

    assert citation.id == CITATION_ID
    assert citation.workspace_id == WORKSPACE_ID
    assert citation.support_recommendation_id == RECOMMENDATION_ID
    assert citation.ordinal == 1
    assert citation.document_id == DOCUMENT_ID
    assert citation.document_version_id == DOCUMENT_VERSION_ID
    assert citation.chunk_id == CHUNK_ID
    assert citation.retrieval_query_id == RETRIEVAL_QUERY_ID
    assert citation.retrieval_rank == 0
    assert citation.retrieval_score == 0.875
    assert citation.created_at == CREATED_AT

    with pytest.raises(AttributeError):
        citation.ordinal = 2  # type: ignore[misc]


@pytest.mark.parametrize(
    "ordinal",
    [
        0,
        -1,
    ],
)
def test_rejects_non_positive_citation_ordinal(
    ordinal: int,
) -> None:
    with pytest.raises(
        ValueError,
        match="ordinal must be positive",
    ):
        SupportRecommendationCitation.create(
            workspace_id=WORKSPACE_ID,
            support_recommendation_id=(RECOMMENDATION_ID),
            ordinal=ordinal,
            document_id=DOCUMENT_ID,
            document_version_id=DOCUMENT_VERSION_ID,
            chunk_id=CHUNK_ID,
            retrieval_query_id=RETRIEVAL_QUERY_ID,
            retrieval_rank=0,
            retrieval_score=0.875,
            now=CREATED_AT,
        )


def test_rejects_negative_retrieval_rank() -> None:
    with pytest.raises(
        ValueError,
        match="retrieval_rank must be non-negative",
    ):
        SupportRecommendationCitation.create(
            workspace_id=WORKSPACE_ID,
            support_recommendation_id=(RECOMMENDATION_ID),
            ordinal=1,
            document_id=DOCUMENT_ID,
            document_version_id=DOCUMENT_VERSION_ID,
            chunk_id=CHUNK_ID,
            retrieval_query_id=RETRIEVAL_QUERY_ID,
            retrieval_rank=-1,
            retrieval_score=0.875,
            now=CREATED_AT,
        )


@pytest.mark.parametrize(
    "retrieval_score",
    [
        float("nan"),
        float("inf"),
        float("-inf"),
    ],
)
def test_rejects_non_finite_retrieval_score(
    retrieval_score: float,
) -> None:
    with pytest.raises(
        ValueError,
        match="retrieval_score must be finite",
    ):
        SupportRecommendationCitation.create(
            workspace_id=WORKSPACE_ID,
            support_recommendation_id=(RECOMMENDATION_ID),
            ordinal=1,
            document_id=DOCUMENT_ID,
            document_version_id=DOCUMENT_VERSION_ID,
            chunk_id=CHUNK_ID,
            retrieval_query_id=RETRIEVAL_QUERY_ID,
            retrieval_rank=0,
            retrieval_score=retrieval_score,
            now=CREATED_AT,
        )


def test_rejects_non_float_retrieval_score() -> None:
    with pytest.raises(
        TypeError,
        match="retrieval_score must be a float",
    ):
        SupportRecommendationCitation.create(
            workspace_id=WORKSPACE_ID,
            support_recommendation_id=(RECOMMENDATION_ID),
            ordinal=1,
            document_id=DOCUMENT_ID,
            document_version_id=DOCUMENT_VERSION_ID,
            chunk_id=CHUNK_ID,
            retrieval_query_id=RETRIEVAL_QUERY_ID,
            retrieval_rank=0,
            retrieval_score=cast(Any, 1),
            now=CREATED_AT,
        )


def test_rejects_non_utc_citation_timestamp() -> None:
    with pytest.raises(
        ValueError,
        match="UTC-aware",
    ):
        SupportRecommendationCitation.create(
            workspace_id=WORKSPACE_ID,
            support_recommendation_id=(RECOMMENDATION_ID),
            ordinal=1,
            document_id=DOCUMENT_ID,
            document_version_id=DOCUMENT_VERSION_ID,
            chunk_id=CHUNK_ID,
            retrieval_query_id=RETRIEVAL_QUERY_ID,
            retrieval_rank=0,
            retrieval_score=0.875,
            now=CREATED_AT.replace(tzinfo=None),
        )
