"""Unit tests for recommendation persistence models."""

from datetime import UTC, datetime
from typing import cast
from uuid import UUID

from sqlalchemy import Table

from supportops.infrastructure.postgresql.model_registry import (
    register_persistence_models,
)
from supportops.modules.knowledge_documents.infrastructure.models import (
    DocumentChunkRecord,
)
from supportops.modules.support_recommendations.domain.models import (
    SupportRecommendation,
    SupportRecommendationAction,
    SupportRecommendationCitation,
)
from supportops.modules.support_recommendations.infrastructure.models import (
    SupportRecommendationCitationRecord,
    SupportRecommendationRecord,
)
from supportops.modules.ticket_classifications.infrastructure.models import (
    TicketClassificationRecord,
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
    16,
    30,
    tzinfo=UTC,
)


def create_recommendation() -> SupportRecommendation:
    """Create one accepted recommendation."""

    return SupportRecommendation.create(
        recommendation_id=RECOMMENDATION_ID,
        workspace_id=WORKSPACE_ID,
        ticket_id=TICKET_ID,
        agent_run_id=AGENT_RUN_ID,
        classification_id=CLASSIFICATION_ID,
        accepted_llm_invocation_id=INVOCATION_ID,
        recommended_action=(SupportRecommendationAction.RESPOND),
        response_text=("Follow the documented access-reset procedure."),
        requires_human_review=False,
        decision_summary=("The active runbook contains matching evidence."),
        prompt_id="support-recommendation-draft",
        prompt_version=1,
        prompt_content_hash="a" * 64,
        provider="mock",
        model="mock-support-model-v1",
        now=CREATED_AT,
    )


def create_citation() -> SupportRecommendationCitation:
    """Create one exact ordered citation."""

    return SupportRecommendationCitation.create(
        citation_id=CITATION_ID,
        workspace_id=WORKSPACE_ID,
        support_recommendation_id=RECOMMENDATION_ID,
        ordinal=1,
        document_id=DOCUMENT_ID,
        document_version_id=DOCUMENT_VERSION_ID,
        chunk_id=CHUNK_ID,
        retrieval_query_id=RETRIEVAL_QUERY_ID,
        retrieval_rank=0,
        retrieval_score=0.875,
        now=CREATED_AT,
    )


def test_round_trips_recommendation_record() -> None:
    recommendation = create_recommendation()

    record = SupportRecommendationRecord.from_domain(recommendation)

    assert record.to_domain() == recommendation


def test_round_trips_citation_record() -> None:
    citation = create_citation()

    record = SupportRecommendationCitationRecord.from_domain(citation)

    assert record.to_domain() == citation


def test_recommendation_table_has_expected_constraints() -> None:
    register_persistence_models()

    table = cast(Table, SupportRecommendationRecord.__table__)
    constraint_names = {constraint.name for constraint in table.constraints}
    index_names = {index.name for index in table.indexes}

    assert {
        ("fk_support_recommendations_workspace_ticket_agent_run"),
        ("fk_support_recommendations_agent_run_classification"),
        ("fk_support_recommendations_accepted_invocation"),
        "uq_support_recommendations_agent_run",
        ("uq_support_recommendations_accepted_invocation"),
        "uq_support_recommendations_workspace_id",
        ("ck_support_recommendations_support_recommendation_action"),
        ("ck_support_recommendations_support_recommendation_response_format"),
        ("ck_support_recommendations_support_recommendation_summary_format"),
        ("ck_support_recommendations_support_recommendation_schema_version"),
        ("ck_support_recommendations_support_recommendation_prompt_id_format"),
        ("ck_support_recommendations_support_recommendation_prompt_version_positive"),
        ("ck_support_recommendations_support_recommendation_prompt_content_hash"),
        ("ck_support_recommendations_support_recommendation_provider_format"),
        ("ck_support_recommendations_support_recommendation_model_format"),
    }.issubset(constraint_names)

    assert "ix_support_recommendations_workspace_ticket_created_id" in index_names


def test_citation_table_has_expected_constraints() -> None:
    register_persistence_models()

    table = cast(Table, SupportRecommendationCitationRecord.__table__)
    constraint_names = {constraint.name for constraint in table.constraints}
    index_names = {index.name for index in table.indexes}

    assert {
        ("fk_support_recommendation_citations_workspace_recommendation"),
        ("fk_support_recommendation_citations_authoritative_chunk"),
        ("uq_support_recommendation_citations_recommendation_ordinal"),
        ("uq_support_recommendation_citations_recommendation_chunk"),
        ("ck_support_recommendation_citations_support_recommendation_citation_ordinal_positive"),
        ("ck_support_recommendation_citations_support_recommendation_citation_rank_non_negative"),
        ("ck_support_recommendation_citations_support_recommendation_citation_score_finite"),
    }.issubset(constraint_names)

    assert "ix_support_recommendation_citations_recommendation_ordinal" in index_names


def test_existing_tables_expose_composite_fk_identities() -> None:
    register_persistence_models()

    classification_table = cast(Table, TicketClassificationRecord.__table__)
    chunk_table = cast(Table, DocumentChunkRecord.__table__)
    classification_constraints = {
        constraint.name for constraint in classification_table.constraints
    }
    chunk_constraints = {constraint.name for constraint in chunk_table.constraints}

    assert "uq_ticket_classifications_run_id" in classification_constraints
    assert "uq_knowledge_document_chunks_workspace_document_version_id" in chunk_constraints
