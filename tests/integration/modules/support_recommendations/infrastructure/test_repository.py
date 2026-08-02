"""Integration tests for fenced recommendation persistence."""

from dataclasses import replace
from datetime import UTC, datetime, timedelta
from uuid import UUID

import pytest
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from supportops.ai.gateway.errors import LLMErrorCode
from supportops.ai.gateway.results import LLMInvocationStatus
from supportops.ai.pricing.catalog import (
    PRICING_CATALOG_VERSION,
)
from supportops.ai.schemas.ticket_classification import (
    TICKET_CLASSIFICATION_SCHEMA_VERSION,
    TicketCategory,
    TicketIntent,
    TicketSentiment,
    TicketUrgency,
)
from supportops.infrastructure.postgresql.transaction import (
    SqlAlchemyTransactionManager,
)
from supportops.modules.agent_runs.domain.claiming import (
    AgentRunClaim,
    ClaimAgentRunCommand,
)
from supportops.modules.agent_runs.domain.models import (
    TICKET_CLASSIFICATION_WORKFLOW_VERSION,
    AgentRun,
)
from supportops.modules.agent_runs.infrastructure.repository import (
    SqlAlchemyAgentRunRepository,
)
from supportops.modules.knowledge_documents.domain.models import (
    Document,
    DocumentChunk,
    DocumentMediaType,
    DocumentVersion,
    KnowledgeIndexProfile,
)
from supportops.modules.knowledge_documents.infrastructure.repository import (
    SqlAlchemyDocumentChunkRepository,
    SqlAlchemyDocumentRepository,
    SqlAlchemyDocumentVersionRepository,
)
from supportops.modules.support_recommendations.application.persistence import (
    PersistSupportRecommendationCommand,
    SupportRecommendationPersistenceResult,
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
from supportops.modules.support_recommendations.infrastructure.repository import (
    SqlAlchemySupportRecommendationExecutionRepository,
)
from supportops.modules.ticket_classifications.application.persistence import (
    PersistClassificationExecutionCommand,
)
from supportops.modules.ticket_classifications.domain.models import (
    LLMInvocation,
    TicketClassification,
)
from supportops.modules.ticket_classifications.infrastructure.models import (
    LLMInvocationRecord,
)
from supportops.modules.ticket_classifications.infrastructure.repository import (
    SqlAlchemyClassificationPersistenceRepository,
)
from supportops.modules.tickets.domain.models import Ticket
from supportops.modules.tickets.infrastructure.repository import (
    SqlAlchemyTicketRepository,
)
from supportops.modules.workspaces.domain.models import Workspace
from supportops.modules.workspaces.infrastructure.repository import (
    SqlAlchemyWorkspaceRepository,
)

pytestmark = pytest.mark.integration

_WORKSPACE_ID = UUID("21000000-0000-4000-8000-000000000001")
_TICKET_ID = UUID("22000000-0000-4000-8000-000000000002")
_AGENT_RUN_ID = UUID("23000000-0000-4000-8000-000000000003")
_LEASE_TOKEN = UUID("24000000-0000-4000-8000-000000000004")
_EXECUTION_REQUEST_ID = UUID("25000000-0000-4000-8000-000000000005")

_CLASSIFICATION_INVOCATION_ID = UUID("26000000-0000-4000-8000-000000000006")
_CLASSIFICATION_ID = UUID("27000000-0000-4000-8000-000000000007")
_DECISION_INVOCATION_ID = UUID("28000000-0000-4000-8000-000000000008")
_DRAFT_INVOCATION_ID = UUID("29000000-0000-4000-8000-000000000009")
_RECOMMENDATION_ID = UUID("2a000000-0000-4000-8000-000000000010")
_CITATION_ID = UUID("2b000000-0000-4000-8000-000000000011")
_FAILURE_INVOCATION_ID = UUID("2c000000-0000-4000-8000-000000000012")

_DOCUMENT_ID = UUID("2d000000-0000-4000-8000-000000000013")
_DOCUMENT_VERSION_ID = UUID("2e000000-0000-4000-8000-000000000014")
_RETRIEVAL_QUERY_ID = UUID("2f000000-0000-4000-8000-000000000015")

_BASE_TIMESTAMP = datetime.now(UTC).replace(microsecond=0)
_CLAIMED_AT = _BASE_TIMESTAMP + timedelta(seconds=1)
_CLASSIFICATION_AT = _CLAIMED_AT + timedelta(seconds=1)
_DECISION_AT = _CLAIMED_AT + timedelta(seconds=2)
_DRAFT_AT = _CLAIMED_AT + timedelta(seconds=3)
_RECOMMENDATION_AT = _CLAIMED_AT + timedelta(seconds=4)
_PERSISTED_AT = _CLAIMED_AT + timedelta(seconds=5)
_LEASE_EXPIRES_AT = _CLAIMED_AT + timedelta(minutes=5)

_CLASSIFICATION_PROMPT_ID = "ticket-classification"
_CLASSIFICATION_PROMPT_HASH = "a" * 64
_DECISION_PROMPT_ID = "support-action-decision"
_DECISION_PROMPT_HASH = "b" * 64
_DRAFT_PROMPT_ID = "support-recommendation-draft"
_DRAFT_PROMPT_HASH = "c" * 64

_PROVIDER = "mock"
_MODEL = "mock-support-model-v1"


async def _create_running_claim(
    session: AsyncSession,
) -> AgentRunClaim:
    workspace = Workspace(
        id=_WORKSPACE_ID,
        name="Recommendation Repository",
        slug="recommendation-repository",
        created_at=_BASE_TIMESTAMP,
        updated_at=_BASE_TIMESTAMP,
    )
    ticket = Ticket.create(
        ticket_id=_TICKET_ID,
        workspace_id=_WORKSPACE_ID,
        subject="Unable to reset account access",
        description=(
            "The customer cannot complete the documented account-access recovery procedure."
        ),
        external_reference=None,
        ingestion_request_id=UUID("31000000-0000-4000-8000-000000000016"),
        correlation_id=UUID("32000000-0000-4000-8000-000000000017"),
        now=_BASE_TIMESTAMP,
    )
    run = AgentRun.create_initial(
        agent_run_id=_AGENT_RUN_ID,
        workspace_id=_WORKSPACE_ID,
        ticket_id=_TICKET_ID,
        ingestion_request_id=(ticket.ingestion_request_id),
        correlation_id=ticket.correlation_id,
        workflow_version=(TICKET_CLASSIFICATION_WORKFLOW_VERSION),
        max_attempts=3,
        now=_BASE_TIMESTAMP,
    )

    transaction_manager = SqlAlchemyTransactionManager(session)
    run_repository = SqlAlchemyAgentRunRepository(session)

    async with transaction_manager.transaction():
        await SqlAlchemyWorkspaceRepository(session).add(workspace)
        await SqlAlchemyTicketRepository(session).add(ticket)
        await run_repository.add(run)

    async with transaction_manager.transaction():
        claim = await run_repository.claim_next_available(
            ClaimAgentRunCommand(
                worker_id="recommendation-worker",
                lease_token=_LEASE_TOKEN,
                execution_request_id=(_EXECUTION_REQUEST_ID),
                claimed_at=_CLAIMED_AT,
                lease_expires_at=_LEASE_EXPIRES_AT,
            )
        )

    assert claim is not None

    return claim


def _successful_invocation(
    *,
    claim: AgentRunClaim,
    invocation_id: UUID,
    sequence: int,
    prompt_id: str,
    prompt_content_hash: str,
    schema_version: str,
    created_at: datetime,
) -> LLMInvocation:
    return LLMInvocation.create(
        invocation_id=invocation_id,
        workspace_id=_WORKSPACE_ID,
        ticket_id=_TICKET_ID,
        agent_run_id=_AGENT_RUN_ID,
        agent_run_attempt_id=claim.attempt.id,
        invocation_sequence=sequence,
        status=LLMInvocationStatus.SUCCEEDED,
        provider=_PROVIDER,
        model=_MODEL,
        provider_request_id=(f"mock-request-{sequence}"),
        prompt_id=prompt_id,
        prompt_version=1,
        prompt_content_hash=prompt_content_hash,
        schema_version=schema_version,
        input_tokens=20,
        cached_input_tokens=0,
        output_tokens=10,
        reasoning_tokens=0,
        total_tokens=30,
        pricing_catalog_version=(PRICING_CATALOG_VERSION),
        pricing_found=False,
        estimated_input_cost_usd=None,
        estimated_cached_input_cost_usd=None,
        estimated_output_cost_usd=None,
        estimated_total_cost_usd=None,
        latency_ms=25,
        error_code=None,
        now=created_at,
    )


def _classification_invocation(
    claim: AgentRunClaim,
) -> LLMInvocation:
    return _successful_invocation(
        claim=claim,
        invocation_id=_CLASSIFICATION_INVOCATION_ID,
        sequence=1,
        prompt_id=_CLASSIFICATION_PROMPT_ID,
        prompt_content_hash=(_CLASSIFICATION_PROMPT_HASH),
        schema_version=(TICKET_CLASSIFICATION_SCHEMA_VERSION),
        created_at=_CLASSIFICATION_AT,
    )


async def _persist_classification(
    session: AsyncSession,
    claim: AgentRunClaim,
) -> LLMInvocation:
    invocation = _classification_invocation(claim)
    classification = TicketClassification.create(
        classification_id=_CLASSIFICATION_ID,
        workspace_id=_WORKSPACE_ID,
        ticket_id=_TICKET_ID,
        agent_run_id=_AGENT_RUN_ID,
        category=TicketCategory.ACCOUNT_ACCESS,
        intent=TicketIntent.REQUEST_ACCESS,
        urgency=TicketUrgency.NORMAL,
        sentiment=TicketSentiment.NEUTRAL,
        requires_human_review=False,
        summary=("The customer needs documented account-access recovery guidance."),
        schema_version=(TICKET_CLASSIFICATION_SCHEMA_VERSION),
        prompt_id=_CLASSIFICATION_PROMPT_ID,
        prompt_version=1,
        prompt_content_hash=(_CLASSIFICATION_PROMPT_HASH),
        provider=_PROVIDER,
        model=_MODEL,
        accepted_llm_invocation_id=invocation.id,
        now=_CLASSIFICATION_AT,
    )
    command = PersistClassificationExecutionCommand(
        workspace_id=_WORKSPACE_ID,
        ticket_id=_TICKET_ID,
        agent_run_id=_AGENT_RUN_ID,
        agent_run_attempt_id=claim.attempt.id,
        lease_token=_LEASE_TOKEN,
        persisted_at=_CLASSIFICATION_AT,
        invocations=(invocation,),
        classification=classification,
    )

    async with SqlAlchemyTransactionManager(session).transaction():
        await SqlAlchemyClassificationPersistenceRepository(session).persist_fenced(command)

    return invocation


async def _persist_knowledge_chunk(
    session: AsyncSession,
) -> DocumentChunk:
    document = Document.create(
        document_id=_DOCUMENT_ID,
        workspace_id=_WORKSPACE_ID,
        title="Account access recovery",
        external_reference="runbook-account-access",
        now=_BASE_TIMESTAMP,
    )
    pending_version = DocumentVersion.create_pending(
        document_version_id=_DOCUMENT_VERSION_ID,
        workspace_id=_WORKSPACE_ID,
        document_id=document.id,
        version_number=1,
        media_type=DocumentMediaType.TEXT_MARKDOWN,
        content=(
            "# Account access recovery\n\n"
            "Verify the customer identity and start the "
            "documented password-reset procedure."
        ),
        now=_BASE_TIMESTAMP,
    )
    profile = KnowledgeIndexProfile(
        chunking_strategy="heading-aware",
        chunking_version="v1",
        tokenizer_encoding="cl100k_base",
        embedding_provider="mock",
        embedding_model="mock-embedding-v1",
        embedding_dimensions=8,
        knowledge_collection="supportops-knowledge",
        knowledge_vector_name="content",
    )
    indexed_version = pending_version.bind_index_profile(
        profile,
        now=_BASE_TIMESTAMP,
    )
    chunk = DocumentChunk.create(
        document_version=indexed_version,
        ordinal=0,
        section_path=("Account access recovery",),
        content=("Verify the customer identity and start the documented password-reset procedure."),
        token_count=14,
        now=_BASE_TIMESTAMP,
    )
    ready_version = indexed_version.mark_ready(
        chunk_count=1,
        embedding_input_tokens=14,
        embedding_estimated_cost_usd=None,
        embedding_pricing_catalog_version=(PRICING_CATALOG_VERSION),
        indexed_at=_BASE_TIMESTAMP,
    )

    async with SqlAlchemyTransactionManager(session).transaction():
        await SqlAlchemyDocumentRepository(session).add(document)
        await SqlAlchemyDocumentVersionRepository(session).add(ready_version)
        await SqlAlchemyDocumentChunkRepository(session).add_many((chunk,))

    return chunk


def _recommendation_command(
    *,
    claim: AgentRunClaim,
    classification_invocation: LLMInvocation,
    chunk: DocumentChunk,
    persisted_at: datetime = _PERSISTED_AT,
) -> PersistSupportRecommendationCommand:
    decision_invocation = _successful_invocation(
        claim=claim,
        invocation_id=_DECISION_INVOCATION_ID,
        sequence=2,
        prompt_id=_DECISION_PROMPT_ID,
        prompt_content_hash=_DECISION_PROMPT_HASH,
        schema_version="support-action-decision-v1",
        created_at=_DECISION_AT,
    )
    draft_invocation = _successful_invocation(
        claim=claim,
        invocation_id=_DRAFT_INVOCATION_ID,
        sequence=3,
        prompt_id=_DRAFT_PROMPT_ID,
        prompt_content_hash=_DRAFT_PROMPT_HASH,
        schema_version="support-recommendation-v1",
        created_at=_DRAFT_AT,
    )
    recommendation = SupportRecommendation.create(
        recommendation_id=_RECOMMENDATION_ID,
        workspace_id=_WORKSPACE_ID,
        ticket_id=_TICKET_ID,
        agent_run_id=_AGENT_RUN_ID,
        classification_id=_CLASSIFICATION_ID,
        accepted_llm_invocation_id=(draft_invocation.id),
        recommended_action=(SupportRecommendationAction.RESPOND),
        response_text=(
            "Follow the documented account-access recovery "
            "procedure after verifying the customer identity."
        ),
        requires_human_review=False,
        decision_summary=("The active runbook contains relevant recovery instructions."),
        prompt_id=_DRAFT_PROMPT_ID,
        prompt_version=1,
        prompt_content_hash=_DRAFT_PROMPT_HASH,
        provider=_PROVIDER,
        model=_MODEL,
        now=_RECOMMENDATION_AT,
    )
    citation = SupportRecommendationCitation.create(
        citation_id=_CITATION_ID,
        workspace_id=_WORKSPACE_ID,
        support_recommendation_id=(recommendation.id),
        ordinal=1,
        document_id=chunk.document_id,
        document_version_id=(chunk.document_version_id),
        chunk_id=chunk.id,
        retrieval_query_id=_RETRIEVAL_QUERY_ID,
        retrieval_rank=0,
        retrieval_score=0.91,
        now=_RECOMMENDATION_AT,
    )

    return PersistSupportRecommendationCommand(
        workspace_id=_WORKSPACE_ID,
        ticket_id=_TICKET_ID,
        agent_run_id=_AGENT_RUN_ID,
        agent_run_attempt_id=claim.attempt.id,
        lease_token=_LEASE_TOKEN,
        persisted_at=persisted_at,
        invocations=(
            classification_invocation,
            decision_invocation,
            draft_invocation,
        ),
        recommendation=recommendation,
        citations=(citation,),
    )


async def _counts(
    session: AsyncSession,
) -> tuple[int, int, int]:
    invocation_count = await session.scalar(select(func.count()).select_from(LLMInvocationRecord))
    recommendation_count = await session.scalar(
        select(func.count()).select_from(SupportRecommendationRecord)
    )
    citation_count = await session.scalar(
        select(func.count()).select_from(SupportRecommendationCitationRecord)
    )

    return (
        int(invocation_count or 0),
        int(recommendation_count or 0),
        int(citation_count or 0),
    )


async def test_persists_recommendation_and_citations_atomically(
    postgresql_session: AsyncSession,
    clean_business_tables: None,
) -> None:
    claim = await _create_running_claim(postgresql_session)
    classification_invocation = await _persist_classification(
        postgresql_session,
        claim,
    )
    chunk = await _persist_knowledge_chunk(postgresql_session)
    command = _recommendation_command(
        claim=claim,
        classification_invocation=(classification_invocation),
        chunk=chunk,
    )
    repository = SqlAlchemySupportRecommendationExecutionRepository(postgresql_session)

    async with SqlAlchemyTransactionManager(postgresql_session).transaction():
        result = await repository.persist_fenced(command)

    assert result is (SupportRecommendationPersistenceResult.APPLIED)
    assert await _counts(postgresql_session) == (
        3,
        1,
        1,
    )

    recommendation_result = await postgresql_session.execute(select(SupportRecommendationRecord))
    recommendation_record = recommendation_result.scalar_one()

    citation_result = await postgresql_session.execute(select(SupportRecommendationCitationRecord))
    citation_record = citation_result.scalar_one()

    assert recommendation_record.to_domain() == command.recommendation
    assert citation_record.to_domain() == (command.citations[0])


async def test_identical_replay_returns_already_recommended(
    postgresql_session: AsyncSession,
    clean_business_tables: None,
) -> None:
    claim = await _create_running_claim(postgresql_session)
    classification_invocation = await _persist_classification(
        postgresql_session,
        claim,
    )
    chunk = await _persist_knowledge_chunk(postgresql_session)
    command = _recommendation_command(
        claim=claim,
        classification_invocation=(classification_invocation),
        chunk=chunk,
    )
    repository = SqlAlchemySupportRecommendationExecutionRepository(postgresql_session)
    transaction_manager = SqlAlchemyTransactionManager(postgresql_session)

    async with transaction_manager.transaction():
        first_result = await repository.persist_fenced(command)

    async with transaction_manager.transaction():
        second_result = await repository.persist_fenced(command)

    assert first_result is (SupportRecommendationPersistenceResult.APPLIED)
    assert second_result is (SupportRecommendationPersistenceResult.ALREADY_RECOMMENDED)
    assert await _counts(postgresql_session) == (
        3,
        1,
        1,
    )


async def test_invocation_only_failure_is_idempotent(
    postgresql_session: AsyncSession,
    clean_business_tables: None,
) -> None:
    claim = await _create_running_claim(postgresql_session)
    failure_invocation = LLMInvocation.create(
        invocation_id=_FAILURE_INVOCATION_ID,
        workspace_id=_WORKSPACE_ID,
        ticket_id=_TICKET_ID,
        agent_run_id=_AGENT_RUN_ID,
        agent_run_attempt_id=claim.attempt.id,
        invocation_sequence=1,
        status=LLMInvocationStatus.TIMED_OUT,
        provider=_PROVIDER,
        model=_MODEL,
        provider_request_id="mock-request-1",
        prompt_id=_DECISION_PROMPT_ID,
        prompt_version=1,
        prompt_content_hash=_DECISION_PROMPT_HASH,
        schema_version="support-action-decision-v1",
        input_tokens=None,
        cached_input_tokens=None,
        output_tokens=None,
        reasoning_tokens=None,
        total_tokens=None,
        pricing_catalog_version=(PRICING_CATALOG_VERSION),
        pricing_found=False,
        estimated_input_cost_usd=None,
        estimated_cached_input_cost_usd=None,
        estimated_output_cost_usd=None,
        estimated_total_cost_usd=None,
        latency_ms=20_000,
        error_code=LLMErrorCode.TIMEOUT,
        now=_DECISION_AT,
    )
    command = PersistSupportRecommendationCommand(
        workspace_id=_WORKSPACE_ID,
        ticket_id=_TICKET_ID,
        agent_run_id=_AGENT_RUN_ID,
        agent_run_attempt_id=claim.attempt.id,
        lease_token=_LEASE_TOKEN,
        persisted_at=_PERSISTED_AT,
        invocations=(failure_invocation,),
        recommendation=None,
        citations=(),
    )
    repository = SqlAlchemySupportRecommendationExecutionRepository(postgresql_session)
    transaction_manager = SqlAlchemyTransactionManager(postgresql_session)

    async with transaction_manager.transaction():
        first_result = await repository.persist_fenced(command)

    async with transaction_manager.transaction():
        second_result = await repository.persist_fenced(command)

    assert first_result is (SupportRecommendationPersistenceResult.APPLIED)
    assert second_result is (SupportRecommendationPersistenceResult.ALREADY_RECORDED)
    assert await _counts(postgresql_session) == (
        1,
        0,
        0,
    )


async def test_conflicting_invocation_replay_fails_closed(
    postgresql_session: AsyncSession,
    clean_business_tables: None,
) -> None:
    claim = await _create_running_claim(postgresql_session)
    persisted_invocation = await _persist_classification(
        postgresql_session,
        claim,
    )
    conflicting_invocation = replace(
        persisted_invocation,
        provider_request_id="different-request",
    )
    command = PersistSupportRecommendationCommand(
        workspace_id=_WORKSPACE_ID,
        ticket_id=_TICKET_ID,
        agent_run_id=_AGENT_RUN_ID,
        agent_run_attempt_id=claim.attempt.id,
        lease_token=_LEASE_TOKEN,
        persisted_at=_PERSISTED_AT,
        invocations=(conflicting_invocation,),
        recommendation=None,
        citations=(),
    )

    with pytest.raises(
        RuntimeError,
        match=("invocation identity is already persisted with different invocation data"),
    ):
        async with SqlAlchemyTransactionManager(postgresql_session).transaction():
            await SqlAlchemySupportRecommendationExecutionRepository(
                postgresql_session
            ).persist_fenced(command)

    assert await _counts(postgresql_session) == (
        1,
        0,
        0,
    )


async def test_expired_lease_rejects_recommendation_writes(
    postgresql_session: AsyncSession,
    clean_business_tables: None,
) -> None:
    claim = await _create_running_claim(postgresql_session)
    classification_invocation = await _persist_classification(
        postgresql_session,
        claim,
    )
    chunk = await _persist_knowledge_chunk(postgresql_session)
    command = _recommendation_command(
        claim=claim,
        classification_invocation=(classification_invocation),
        chunk=chunk,
        persisted_at=_LEASE_EXPIRES_AT,
    )

    async with SqlAlchemyTransactionManager(postgresql_session).transaction():
        result = await SqlAlchemySupportRecommendationExecutionRepository(
            postgresql_session
        ).persist_fenced(command)

    assert result is (SupportRecommendationPersistenceResult.LEASE_LOST)
    assert await _counts(postgresql_session) == (
        1,
        0,
        0,
    )
