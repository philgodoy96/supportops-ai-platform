"""Unit tests for recommendation persistence contracts."""

from dataclasses import FrozenInstanceError, replace
from datetime import UTC, datetime, timedelta
from uuid import UUID

import pytest

from supportops.ai.gateway.errors import LLMErrorCode
from supportops.ai.gateway.results import LLMInvocationStatus
from supportops.ai.pricing.catalog import (
    PRICING_CATALOG_VERSION,
)
from supportops.modules.support_recommendations.application.persistence import (
    PersistSupportRecommendationCommand,
)
from supportops.modules.support_recommendations.domain.models import (
    SupportRecommendation,
    SupportRecommendationAction,
    SupportRecommendationCitation,
)
from supportops.modules.ticket_classifications.domain.models import (
    LLMInvocation,
)

_WORKSPACE_ID = UUID("10000000-0000-4000-8000-000000000001")
_TICKET_ID = UUID("20000000-0000-4000-8000-000000000002")
_AGENT_RUN_ID = UUID("30000000-0000-4000-8000-000000000003")
_ATTEMPT_ID = UUID("40000000-0000-4000-8000-000000000004")
_LEASE_TOKEN = UUID("50000000-0000-4000-8000-000000000005")
_CLASSIFICATION_ID = UUID("60000000-0000-4000-8000-000000000006")
_DECISION_INVOCATION_ID = UUID("70000000-0000-4000-8000-000000000007")
_DRAFT_INVOCATION_ID = UUID("80000000-0000-4000-8000-000000000008")
_RECOMMENDATION_ID = UUID("90000000-0000-4000-8000-000000000009")
_FIRST_CITATION_ID = UUID("a0000000-0000-4000-8000-000000000010")
_SECOND_CITATION_ID = UUID("b0000000-0000-4000-8000-000000000011")
_FIRST_DOCUMENT_ID = UUID("c0000000-0000-4000-8000-000000000012")
_SECOND_DOCUMENT_ID = UUID("d0000000-0000-4000-8000-000000000013")
_FIRST_VERSION_ID = UUID("e0000000-0000-4000-8000-000000000014")
_SECOND_VERSION_ID = UUID("f0000000-0000-4000-8000-000000000015")
_FIRST_CHUNK_ID = UUID("11000000-0000-4000-8000-000000000016")
_SECOND_CHUNK_ID = UUID("12000000-0000-4000-8000-000000000017")
_RETRIEVAL_QUERY_ID = UUID("13000000-0000-4000-8000-000000000018")

_BASE_TIMESTAMP = datetime(
    2026,
    8,
    2,
    18,
    0,
    tzinfo=UTC,
)
_DECISION_CREATED_AT = _BASE_TIMESTAMP + timedelta(seconds=1)
_DRAFT_CREATED_AT = _BASE_TIMESTAMP + timedelta(seconds=2)
_RECOMMENDATION_CREATED_AT = _BASE_TIMESTAMP + timedelta(seconds=3)
_PERSISTED_AT = _BASE_TIMESTAMP + timedelta(seconds=4)

_DRAFT_PROMPT_ID = "support-recommendation-draft"
_DRAFT_PROMPT_HASH = "b" * 64
_DRAFT_SCHEMA_VERSION = "support-recommendation-v1"
_PROVIDER = "mock"
_MODEL = "mock-support-model-v1"


def _invocation(
    *,
    invocation_id: UUID,
    sequence: int,
    created_at: datetime,
    prompt_id: str,
    prompt_content_hash: str,
    schema_version: str,
    status: LLMInvocationStatus = (LLMInvocationStatus.SUCCEEDED),
    error_code: LLMErrorCode | None = None,
) -> LLMInvocation:
    return LLMInvocation.create(
        invocation_id=invocation_id,
        workspace_id=_WORKSPACE_ID,
        ticket_id=_TICKET_ID,
        agent_run_id=_AGENT_RUN_ID,
        agent_run_attempt_id=_ATTEMPT_ID,
        invocation_sequence=sequence,
        status=status,
        provider=_PROVIDER,
        model=_MODEL,
        provider_request_id=f"mock-request-{sequence}",
        prompt_id=prompt_id,
        prompt_version=1,
        prompt_content_hash=prompt_content_hash,
        schema_version=schema_version,
        input_tokens=20,
        cached_input_tokens=0,
        output_tokens=10,
        reasoning_tokens=0,
        total_tokens=30,
        pricing_catalog_version=PRICING_CATALOG_VERSION,
        pricing_found=False,
        estimated_input_cost_usd=None,
        estimated_cached_input_cost_usd=None,
        estimated_output_cost_usd=None,
        estimated_total_cost_usd=None,
        latency_ms=25,
        error_code=error_code,
        now=created_at,
    )


def _decision_invocation() -> LLMInvocation:
    return _invocation(
        invocation_id=_DECISION_INVOCATION_ID,
        sequence=1,
        created_at=_DECISION_CREATED_AT,
        prompt_id="support-action-decision",
        prompt_content_hash="a" * 64,
        schema_version="support-action-decision-v1",
    )


def _draft_invocation(
    *,
    status: LLMInvocationStatus = (LLMInvocationStatus.SUCCEEDED),
    error_code: LLMErrorCode | None = None,
) -> LLMInvocation:
    return _invocation(
        invocation_id=_DRAFT_INVOCATION_ID,
        sequence=2,
        created_at=_DRAFT_CREATED_AT,
        prompt_id=_DRAFT_PROMPT_ID,
        prompt_content_hash=_DRAFT_PROMPT_HASH,
        schema_version=_DRAFT_SCHEMA_VERSION,
        status=status,
        error_code=error_code,
    )


def _recommendation() -> SupportRecommendation:
    return SupportRecommendation.create(
        recommendation_id=_RECOMMENDATION_ID,
        workspace_id=_WORKSPACE_ID,
        ticket_id=_TICKET_ID,
        agent_run_id=_AGENT_RUN_ID,
        classification_id=_CLASSIFICATION_ID,
        accepted_llm_invocation_id=_DRAFT_INVOCATION_ID,
        recommended_action=(SupportRecommendationAction.RESPOND),
        response_text=("Follow the documented account-access recovery procedure."),
        requires_human_review=False,
        decision_summary=("The active runbook contains matching recovery instructions."),
        prompt_id=_DRAFT_PROMPT_ID,
        prompt_version=1,
        prompt_content_hash=_DRAFT_PROMPT_HASH,
        provider=_PROVIDER,
        model=_MODEL,
        now=_RECOMMENDATION_CREATED_AT,
    )


def _citation(
    *,
    citation_id: UUID,
    ordinal: int,
    document_id: UUID,
    document_version_id: UUID,
    chunk_id: UUID,
) -> SupportRecommendationCitation:
    return SupportRecommendationCitation.create(
        citation_id=citation_id,
        workspace_id=_WORKSPACE_ID,
        support_recommendation_id=_RECOMMENDATION_ID,
        ordinal=ordinal,
        document_id=document_id,
        document_version_id=document_version_id,
        chunk_id=chunk_id,
        retrieval_query_id=_RETRIEVAL_QUERY_ID,
        retrieval_rank=ordinal - 1,
        retrieval_score=0.9 - (ordinal * 0.1),
        now=_RECOMMENDATION_CREATED_AT,
    )


def _citations() -> tuple[
    SupportRecommendationCitation,
    ...,
]:
    return (
        _citation(
            citation_id=_FIRST_CITATION_ID,
            ordinal=1,
            document_id=_FIRST_DOCUMENT_ID,
            document_version_id=_FIRST_VERSION_ID,
            chunk_id=_FIRST_CHUNK_ID,
        ),
        _citation(
            citation_id=_SECOND_CITATION_ID,
            ordinal=2,
            document_id=_SECOND_DOCUMENT_ID,
            document_version_id=_SECOND_VERSION_ID,
            chunk_id=_SECOND_CHUNK_ID,
        ),
    )


def _command(
    *,
    invocations: tuple[LLMInvocation, ...] | None = None,
    recommendation: SupportRecommendation | None = None,
    citations: (tuple[SupportRecommendationCitation, ...] | None) = None,
    persisted_at: datetime = _PERSISTED_AT,
) -> PersistSupportRecommendationCommand:
    return PersistSupportRecommendationCommand(
        workspace_id=_WORKSPACE_ID,
        ticket_id=_TICKET_ID,
        agent_run_id=_AGENT_RUN_ID,
        agent_run_attempt_id=_ATTEMPT_ID,
        lease_token=_LEASE_TOKEN,
        persisted_at=persisted_at,
        invocations=(
            (
                _decision_invocation(),
                _draft_invocation(),
            )
            if invocations is None
            else invocations
        ),
        recommendation=(_recommendation() if recommendation is None else recommendation),
        citations=(_citations() if citations is None else citations),
    )


def test_accepts_consistent_recommendation_aggregate() -> None:
    command = _command()

    assert len(command.invocations) == 2
    assert command.invocations[-1].id == (
        command.recommendation.accepted_llm_invocation_id
        if command.recommendation is not None
        else None
    )
    assert tuple(citation.ordinal for citation in command.citations) == (
        1,
        2,
    )


def test_accepts_invocation_only_failure_recording() -> None:
    invocation = _invocation(
        invocation_id=_DECISION_INVOCATION_ID,
        sequence=1,
        created_at=_DECISION_CREATED_AT,
        prompt_id="support-action-decision",
        prompt_content_hash="a" * 64,
        schema_version="support-action-decision-v1",
        status=LLMInvocationStatus.TIMED_OUT,
        error_code=LLMErrorCode.TIMEOUT,
    )

    command = PersistSupportRecommendationCommand(
        workspace_id=_WORKSPACE_ID,
        ticket_id=_TICKET_ID,
        agent_run_id=_AGENT_RUN_ID,
        agent_run_attempt_id=_ATTEMPT_ID,
        lease_token=_LEASE_TOKEN,
        persisted_at=_PERSISTED_AT,
        invocations=(invocation,),
        recommendation=None,
        citations=(),
    )

    assert command.recommendation is None
    assert command.citations == ()


def test_requires_at_least_one_invocation() -> None:
    with pytest.raises(
        ValueError,
        match="At least one LLM invocation",
    ):
        _command(
            invocations=(),
        )


def test_requires_contiguous_ordered_sequences() -> None:
    second = replace(
        _draft_invocation(),
        invocation_sequence=3,
    )

    with pytest.raises(
        ValueError,
        match="contiguous, ordered",
    ):
        _command(
            invocations=(
                _decision_invocation(),
                second,
            )
        )


def test_requires_unique_invocation_identifiers() -> None:
    duplicate = replace(
        _draft_invocation(),
        id=_DECISION_INVOCATION_ID,
    )

    with pytest.raises(
        ValueError,
        match="identifiers must be unique",
    ):
        _command(
            invocations=(
                _decision_invocation(),
                duplicate,
            )
        )


@pytest.mark.parametrize(
    ("field_name", "expected_message"),
    [
        (
            "workspace_id",
            "Invocation workspace ownership",
        ),
        (
            "ticket_id",
            "Invocation ticket ownership",
        ),
        (
            "agent_run_id",
            "Invocation AgentRun ownership",
        ),
        (
            "agent_run_attempt_id",
            "Invocation AgentRunAttempt ownership",
        ),
    ],
)
def test_rejects_invocation_ownership_mismatch(
    field_name: str,
    expected_message: str,
) -> None:
    mismatched_id = UUID("14000000-0000-4000-8000-000000000019")
    base = _draft_invocation()
    if field_name == "workspace_id":
        draft = replace(
            base,
            workspace_id=mismatched_id,
        )
    elif field_name == "ticket_id":
        draft = replace(
            base,
            ticket_id=mismatched_id,
        )
    elif field_name == "agent_run_id":
        draft = replace(
            base,
            agent_run_id=mismatched_id,
        )
    else:
        draft = replace(
            base,
            agent_run_attempt_id=mismatched_id,
        )

    with pytest.raises(
        ValueError,
        match=expected_message,
    ):
        _command(
            invocations=(
                _decision_invocation(),
                draft,
            )
        )


@pytest.mark.parametrize(
    ("field_name", "expected_message"),
    [
        (
            "workspace_id",
            "Recommendation workspace ownership",
        ),
        (
            "ticket_id",
            "Recommendation ticket ownership",
        ),
        (
            "agent_run_id",
            "Recommendation AgentRun ownership",
        ),
    ],
)
def test_rejects_recommendation_ownership_mismatch(
    field_name: str,
    expected_message: str,
) -> None:
    mismatched_id = UUID("15000000-0000-4000-8000-000000000020")
    base = _recommendation()
    if field_name == "workspace_id":
        mismatched = replace(
            base,
            workspace_id=mismatched_id,
        )
    elif field_name == "ticket_id":
        mismatched = replace(
            base,
            ticket_id=mismatched_id,
        )
    else:
        mismatched = replace(
            base,
            agent_run_id=mismatched_id,
        )

    with pytest.raises(
        ValueError,
        match=expected_message,
    ):
        _command(
            recommendation=mismatched,
        )


def test_requires_accepted_invocation_in_command() -> None:
    recommendation = replace(
        _recommendation(),
        accepted_llm_invocation_id=UUID("16000000-0000-4000-8000-000000000021"),
    )

    with pytest.raises(
        ValueError,
        match="included in the persistence command",
    ):
        _command(
            recommendation=recommendation,
        )


def test_requires_accepted_invocation_to_be_final() -> None:
    recommendation = replace(
        _recommendation(),
        accepted_llm_invocation_id=(_DECISION_INVOCATION_ID),
    )

    with pytest.raises(
        ValueError,
        match="must be the final invocation",
    ):
        _command(
            recommendation=recommendation,
        )


def test_requires_successful_accepted_invocation() -> None:
    failed_draft = _draft_invocation(
        status=LLMInvocationStatus.TIMED_OUT,
        error_code=LLMErrorCode.TIMEOUT,
    )

    with pytest.raises(
        ValueError,
        match="must have succeeded",
    ):
        _command(
            invocations=(
                _decision_invocation(),
                failed_draft,
            )
        )


@pytest.mark.parametrize(
    "field_name",
    [
        "prompt_id",
        "prompt_version",
        "prompt_content_hash",
        "schema_version",
        "provider",
        "model",
    ],
)
def test_requires_accepted_invocation_provenance(
    field_name: str,
) -> None:
    base = _recommendation()
    if field_name == "prompt_id":
        recommendation = replace(
            base,
            prompt_id="different-prompt",
        )
    elif field_name == "prompt_version":
        recommendation = replace(
            base,
            prompt_version=2,
        )
    elif field_name == "prompt_content_hash":
        recommendation = replace(
            base,
            prompt_content_hash="c" * 64,
        )
    elif field_name == "schema_version":
        mismatched_draft = replace(
            _draft_invocation(),
            schema_version="different-schema-v1",
        )

        with pytest.raises(
            ValueError,
            match=f"share {field_name} provenance",
        ):
            _command(
                invocations=(
                    _decision_invocation(),
                    mismatched_draft,
                ),
            )
        return
    elif field_name == "provider":
        recommendation = replace(
            base,
            provider="different-provider",
        )
    else:
        recommendation = replace(
            base,
            model="different-model",
        )

    with pytest.raises(
        ValueError,
        match=f"share {field_name} provenance",
    ):
        _command(
            recommendation=recommendation,
        )


def test_rejects_citations_without_recommendation() -> None:
    with pytest.raises(
        ValueError,
        match="Citations require",
    ):
        PersistSupportRecommendationCommand(
            workspace_id=_WORKSPACE_ID,
            ticket_id=_TICKET_ID,
            agent_run_id=_AGENT_RUN_ID,
            agent_run_attempt_id=_ATTEMPT_ID,
            lease_token=_LEASE_TOKEN,
            persisted_at=_PERSISTED_AT,
            invocations=(_decision_invocation(),),
            recommendation=None,
            citations=_citations(),
        )


def test_requires_contiguous_citation_ordinals() -> None:
    first, second = _citations()

    with pytest.raises(
        ValueError,
        match="Citation ordinals must be contiguous",
    ):
        _command(
            citations=(
                first,
                replace(
                    second,
                    ordinal=3,
                ),
            )
        )


def test_requires_unique_citation_identifiers() -> None:
    first, second = _citations()

    with pytest.raises(
        ValueError,
        match="Citation identifiers must be unique",
    ):
        _command(
            citations=(
                first,
                replace(
                    second,
                    id=first.id,
                ),
            )
        )


def test_rejects_duplicate_cited_chunk() -> None:
    first, second = _citations()

    with pytest.raises(
        ValueError,
        match="same chunk more than once",
    ):
        _command(
            citations=(
                first,
                replace(
                    second,
                    chunk_id=first.chunk_id,
                ),
            )
        )


def test_rejects_citation_workspace_mismatch() -> None:
    first, second = _citations()

    with pytest.raises(
        ValueError,
        match="Citation workspace ownership",
    ):
        _command(
            citations=(
                replace(
                    first,
                    workspace_id=UUID("17000000-0000-4000-8000-000000000022"),
                ),
                second,
            )
        )


def test_rejects_citation_recommendation_mismatch() -> None:
    first, second = _citations()

    with pytest.raises(
        ValueError,
        match="Citation recommendation ownership",
    ):
        _command(
            citations=(
                replace(
                    first,
                    support_recommendation_id=UUID("18000000-0000-4000-8000-000000000023"),
                ),
                second,
            )
        )


@pytest.mark.parametrize(
    "timestamp_source",
    [
        "invocation",
        "recommendation",
        "citation",
    ],
)
def test_persistence_timestamp_cannot_precede_records(
    timestamp_source: str,
) -> None:
    future_timestamp = _PERSISTED_AT + timedelta(seconds=1)

    if timestamp_source == "invocation":
        invocations = (
            _decision_invocation(),
            replace(
                _draft_invocation(),
                created_at=future_timestamp,
            ),
        )
        recommendation = _recommendation()
        citations = _citations()
    elif timestamp_source == "recommendation":
        invocations = (
            _decision_invocation(),
            _draft_invocation(),
        )
        recommendation = replace(
            _recommendation(),
            created_at=future_timestamp,
        )
        citations = _citations()
    else:
        first, second = _citations()
        invocations = (
            _decision_invocation(),
            _draft_invocation(),
        )
        recommendation = _recommendation()
        citations = (
            replace(
                first,
                created_at=future_timestamp,
            ),
            second,
        )

    with pytest.raises(
        ValueError,
        match="persisted_at must not precede",
    ):
        _command(
            invocations=invocations,
            recommendation=recommendation,
            citations=citations,
        )


def test_requires_utc_persistence_timestamp() -> None:
    with pytest.raises(
        ValueError,
        match="persisted_at must be a UTC-aware timestamp",
    ):
        _command(persisted_at=_PERSISTED_AT.replace(tzinfo=None))


def test_command_is_immutable() -> None:
    command = _command()

    with pytest.raises(FrozenInstanceError):
        command.lease_token = UUID(  # type: ignore[misc]
            "19000000-0000-4000-8000-000000000024",
        )
