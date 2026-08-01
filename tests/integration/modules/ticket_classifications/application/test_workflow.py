"""PostgreSQL integration tests for durable classification workflows."""

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from uuid import UUID

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from supportops.ai.gateway.contracts import LLMTokenUsage
from supportops.ai.gateway.service import LLMGateway
from supportops.ai.providers.mock import (
    MOCK_TICKET_CLASSIFIER_MODEL,
    MockLLMOutcome,
    MockLLMProvider,
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
from supportops.modules.agent_runs.application.execution import (
    AgentRunExecutionContext,
)
from supportops.modules.agent_runs.application.processor import (
    ProcessClaimedAgentRun,
)
from supportops.modules.agent_runs.application.retry_policy import (
    AgentRunRetryPolicy,
)
from supportops.modules.agent_runs.domain.claiming import (
    AgentRunClaim,
    ClaimAgentRunCommand,
)
from supportops.modules.agent_runs.domain.models import (
    TICKET_CLASSIFICATION_WORKFLOW_VERSION,
    AgentRun,
    AgentRunAttemptOutcome,
    AgentRunStatus,
)
from supportops.modules.agent_runs.domain.transitions import (
    AgentRunTransitionResult,
)
from supportops.modules.agent_runs.infrastructure.models import (
    AgentRunAttemptRecord,
    AgentRunRecord,
)
from supportops.modules.agent_runs.infrastructure.repository import (
    SqlAlchemyAgentRunRepository,
)
from supportops.modules.ticket_classifications.infrastructure.models import (
    LLMInvocationRecord,
    TicketClassificationRecord,
)
from supportops.modules.tickets.domain.models import Ticket
from supportops.modules.tickets.infrastructure.repository import (
    SqlAlchemyTicketRepository,
)
from supportops.modules.workspaces.domain.models import Workspace
from supportops.modules.workspaces.infrastructure.repository import (
    SqlAlchemyWorkspaceRepository,
)
from supportops.worker.composition import (
    create_session_scoped_executor_registry,
)

pytestmark = pytest.mark.integration

_BASE_TIMESTAMP = datetime.now(UTC).replace(microsecond=0)
_FIRST_CLAIMED_AT = _BASE_TIMESTAMP + timedelta(seconds=1)
_FIRST_FINISHED_AT = _BASE_TIMESTAMP + timedelta(seconds=2)
_SECOND_CLAIMED_AT = _BASE_TIMESTAMP + timedelta(seconds=12)
_SECOND_FINISHED_AT = _BASE_TIMESTAMP + timedelta(seconds=13)

_WORKSPACE_ID = UUID(
    "10000000-0000-4000-8000-000000000001",
)
_TICKET_ID = UUID(
    "20000000-0000-4000-8000-000000000002",
)
_AGENT_RUN_ID = UUID(
    "30000000-0000-4000-8000-000000000003",
)
_FIRST_LEASE_TOKEN = UUID(
    "40000000-0000-4000-8000-000000000004",
)
_FIRST_EXECUTION_REQUEST_ID = UUID(
    "50000000-0000-4000-8000-000000000005",
)
_SECOND_LEASE_TOKEN = UUID(
    "60000000-0000-4000-8000-000000000006",
)
_SECOND_EXECUTION_REQUEST_ID = UUID(
    "70000000-0000-4000-8000-000000000007",
)

_ZERO_COST = Decimal("0.000000000000")


def _successful_outcome() -> MockLLMOutcome:
    """Return one deterministic valid classification response."""

    return MockLLMOutcome.success(
        {
            "category": TicketCategory.BILLING.value,
            "intent": TicketIntent.ASK_QUESTION.value,
            "urgency": TicketUrgency.NORMAL.value,
            "sentiment": TicketSentiment.NEUTRAL.value,
            "requires_human_review": False,
            "summary": ("The customer is asking about a duplicated invoice charge."),
            "schema_version": (TICKET_CLASSIFICATION_SCHEMA_VERSION),
        },
        usage=LLMTokenUsage(
            input_tokens=120,
            cached_input_tokens=0,
            output_tokens=24,
            reasoning_tokens=None,
            total_tokens=144,
        ),
    )


async def _persist_workspace_ticket_and_run(
    session: AsyncSession,
) -> Ticket:
    """Persist one queued classification AgentRun and its owners."""

    workspace = Workspace(
        id=_WORKSPACE_ID,
        name="Classification Support",
        slug="classification-support",
        created_at=_BASE_TIMESTAMP,
        updated_at=_BASE_TIMESTAMP,
    )
    ticket = Ticket.create(
        ticket_id=_TICKET_ID,
        workspace_id=_WORKSPACE_ID,
        subject="Duplicated invoice charge",
        description=("The latest invoice contains the same charge twice."),
        external_reference="SUP-CLASSIFICATION-1",
        ingestion_request_id=UUID(
            "81000000-0000-4000-8000-000000000008",
        ),
        correlation_id=UUID(
            "82000000-0000-4000-8000-000000000008",
        ),
        now=_BASE_TIMESTAMP,
    )
    run = AgentRun.create_initial(
        agent_run_id=_AGENT_RUN_ID,
        workspace_id=_WORKSPACE_ID,
        ticket_id=_TICKET_ID,
        ingestion_request_id=ticket.ingestion_request_id,
        correlation_id=ticket.correlation_id,
        workflow_version=(TICKET_CLASSIFICATION_WORKFLOW_VERSION),
        max_attempts=3,
        now=_BASE_TIMESTAMP,
    )

    async with SqlAlchemyTransactionManager(
        session,
    ).transaction():
        await SqlAlchemyWorkspaceRepository(
            session,
        ).add(workspace)
        await SqlAlchemyTicketRepository(
            session,
        ).add(ticket)
        await SqlAlchemyAgentRunRepository(
            session,
        ).add(run)

    return ticket


async def _claim_run(
    session: AsyncSession,
    *,
    worker_id: str,
    lease_token: UUID,
    execution_request_id: UUID,
    claimed_at: datetime,
) -> AgentRunClaim:
    """Claim the queued or retry-scheduled AgentRun."""

    repository = SqlAlchemyAgentRunRepository(session)

    async with SqlAlchemyTransactionManager(
        session,
    ).transaction():
        claim = await repository.claim_next_available(
            ClaimAgentRunCommand(
                worker_id=worker_id,
                lease_token=lease_token,
                execution_request_id=execution_request_id,
                claimed_at=claimed_at,
                lease_expires_at=(claimed_at + timedelta(minutes=5)),
            ),
        )

    assert claim is not None
    return claim


async def _process_claim(
    session: AsyncSession,
    *,
    claim: AgentRunClaim,
    gateway: LLMGateway,
    model: str,
    finished_at: datetime,
) -> AgentRunTransitionResult:
    """Process one claim using real repositories and executor registry."""

    transaction_manager = SqlAlchemyTransactionManager(
        session,
    )
    registry = create_session_scoped_executor_registry(
        session=session,
        transaction_manager=transaction_manager,
        gateway=gateway,
        model=model,
        request_timeout_seconds=12,
    )
    processor = ProcessClaimedAgentRun(
        ticket_repository=SqlAlchemyTicketRepository(
            session,
        ),
        agent_run_repository=SqlAlchemyAgentRunRepository(
            session,
        ),
        transaction_manager=transaction_manager,
        executor=registry,
        retry_policy=AgentRunRetryPolicy(
            base_delay_seconds=2.0,
            maximum_delay_seconds=60.0,
        ),
        execution_timeout_seconds=30.0,
        utc_now=lambda: finished_at,
    )

    return await processor.execute(claim)


async def _load_attempts(
    session: AsyncSession,
) -> list[AgentRunAttemptRecord]:
    """Load AgentRun attempts in attempt-number order."""

    result = await session.execute(
        select(AgentRunAttemptRecord)
        .where(
            AgentRunAttemptRecord.agent_run_id == _AGENT_RUN_ID,
        )
        .order_by(
            AgentRunAttemptRecord.attempt_number.asc(),
        ),
    )

    return list(result.scalars())


async def _load_invocations(
    session: AsyncSession,
) -> list[LLMInvocationRecord]:
    """Load all durable invocations for the AgentRun."""

    result = await session.execute(
        select(LLMInvocationRecord).where(
            LLMInvocationRecord.agent_run_id == _AGENT_RUN_ID,
        ),
    )

    return list(result.scalars())


async def _load_classification(
    session: AsyncSession,
) -> TicketClassificationRecord:
    """Load the accepted classification for the AgentRun."""

    result = await session.execute(
        select(TicketClassificationRecord).where(
            TicketClassificationRecord.agent_run_id == _AGENT_RUN_ID,
        ),
    )

    return result.scalar_one()


async def test_mock_workflow_persists_classification_and_completes_run(
    postgresql_session_factory: async_sessionmaker[AsyncSession],
    clean_business_tables: None,
) -> None:
    provider = MockLLMProvider.with_strict_outcomes(
        (_successful_outcome(),),
    )
    gateway = LLMGateway(
        provider=provider,
        max_repair_attempts=1,
    )

    try:
        async with postgresql_session_factory() as setup_session:
            await _persist_workspace_ticket_and_run(
                setup_session,
            )

        async with postgresql_session_factory() as claim_session:
            claim = await _claim_run(
                claim_session,
                worker_id="classification-worker-a",
                lease_token=_FIRST_LEASE_TOKEN,
                execution_request_id=(_FIRST_EXECUTION_REQUEST_ID),
                claimed_at=_FIRST_CLAIMED_AT,
            )

        async with postgresql_session_factory() as processing_session:
            result = await _process_claim(
                processing_session,
                claim=claim,
                gateway=gateway,
                model=MOCK_TICKET_CLASSIFIER_MODEL,
                finished_at=_FIRST_FINISHED_AT,
            )

        assert result is AgentRunTransitionResult.APPLIED
        assert provider.invocation_count == 1

        async with postgresql_session_factory() as verification_session:
            run = await verification_session.get(
                AgentRunRecord,
                _AGENT_RUN_ID,
            )
            attempts = await _load_attempts(
                verification_session,
            )
            invocations = await _load_invocations(
                verification_session,
            )
            classification = await _load_classification(
                verification_session,
            )

        assert run is not None
        assert run.status == AgentRunStatus.SUCCEEDED.value
        assert run.completed_at == _FIRST_FINISHED_AT
        assert run.attempt_count == 1
        assert run.last_error_code is None
        assert run.last_error_summary is None
        assert run.lease_owner is None
        assert run.lease_token is None
        assert run.lease_expires_at is None

        assert len(attempts) == 1
        assert attempts[0].outcome == (AgentRunAttemptOutcome.SUCCEEDED.value)
        assert attempts[0].error_code is None
        assert attempts[0].finished_at == (_FIRST_FINISHED_AT)

        assert len(invocations) == 1
        invocation = invocations[0]

        assert invocation.agent_run_attempt_id == (attempts[0].id)
        assert invocation.invocation_sequence == 1
        assert invocation.status == "succeeded"
        assert invocation.provider == "mock"
        assert invocation.model == (MOCK_TICKET_CLASSIFIER_MODEL)
        assert invocation.provider_request_id == ("mock-request-1")
        assert invocation.input_tokens == 120
        assert invocation.cached_input_tokens == 0
        assert invocation.output_tokens == 24
        assert invocation.reasoning_tokens is None
        assert invocation.total_tokens == 144
        assert invocation.pricing_found is True
        assert invocation.estimated_input_cost_usd == (_ZERO_COST)
        assert invocation.estimated_cached_input_cost_usd == _ZERO_COST
        assert invocation.estimated_output_cost_usd == (_ZERO_COST)
        assert invocation.estimated_total_cost_usd == (_ZERO_COST)
        assert invocation.error_code is None

        assert classification.workspace_id == (_WORKSPACE_ID)
        assert classification.ticket_id == _TICKET_ID
        assert classification.agent_run_id == (_AGENT_RUN_ID)
        assert classification.accepted_llm_invocation_id == (invocation.id)
        assert classification.category == "billing"
        assert classification.intent == "ask_question"
        assert classification.urgency == "normal"
        assert classification.sentiment == "neutral"
        assert classification.requires_human_review is False
        assert classification.summary == (
            "The customer is asking about a duplicated invoice charge."
        )
        assert classification.schema_version == (TICKET_CLASSIFICATION_SCHEMA_VERSION)
        assert classification.prompt_id == ("ticket-classification")
        assert classification.prompt_version == 1
        assert classification.provider == "mock"
        assert classification.model == (MOCK_TICKET_CLASSIFIER_MODEL)
        assert classification.updated_at == (classification.created_at)
    finally:
        await provider.close()


async def test_persisted_classification_prevents_repeated_provider_call(
    postgresql_session_factory: async_sessionmaker[AsyncSession],
    clean_business_tables: None,
) -> None:
    provider = MockLLMProvider.with_strict_outcomes(
        (_successful_outcome(),),
    )
    gateway = LLMGateway(
        provider=provider,
        max_repair_attempts=1,
    )

    try:
        async with postgresql_session_factory() as setup_session:
            ticket = await _persist_workspace_ticket_and_run(
                setup_session,
            )

        async with postgresql_session_factory() as claim_session:
            claim = await _claim_run(
                claim_session,
                worker_id="classification-worker-a",
                lease_token=_FIRST_LEASE_TOKEN,
                execution_request_id=(_FIRST_EXECUTION_REQUEST_ID),
                claimed_at=_FIRST_CLAIMED_AT,
            )

        async with postgresql_session_factory() as classification_session:
            transaction_manager = SqlAlchemyTransactionManager(
                classification_session,
            )
            registry = create_session_scoped_executor_registry(
                session=classification_session,
                transaction_manager=(transaction_manager),
                gateway=gateway,
                model=MOCK_TICKET_CLASSIFIER_MODEL,
                request_timeout_seconds=12,
            )
            executor = registry.resolve(
                workflow_name=(claim.agent_run.workflow_name),
                workflow_version=(TICKET_CLASSIFICATION_WORKFLOW_VERSION),
            )

            await executor.execute(
                AgentRunExecutionContext(
                    agent_run=claim.agent_run,
                    attempt=claim.attempt,
                    ticket=ticket,
                ),
            )

        assert provider.invocation_count == 1

        # Simulates recovery after classification commit but before
        # the AgentRun success transition was committed.
        async with postgresql_session_factory() as recovery_session:
            result = await _process_claim(
                recovery_session,
                claim=claim,
                gateway=gateway,
                model=MOCK_TICKET_CLASSIFIER_MODEL,
                finished_at=_FIRST_FINISHED_AT,
            )

        assert result is AgentRunTransitionResult.APPLIED

        # The provider uses a strict one-item queue. A repeated call
        # would raise MockLLMOutcomeQueueExhaustedError.
        assert provider.invocation_count == 1

        async with postgresql_session_factory() as verification_session:
            run = await verification_session.get(
                AgentRunRecord,
                _AGENT_RUN_ID,
            )
            invocations = await _load_invocations(
                verification_session,
            )
            classification = await _load_classification(
                verification_session,
            )

        assert run is not None
        assert run.status == AgentRunStatus.SUCCEEDED.value
        assert len(invocations) == 1
        assert classification.accepted_llm_invocation_id == (invocations[0].id)
    finally:
        await provider.close()


async def test_retry_creates_new_attempt_invocation_then_succeeds(
    postgresql_session_factory: async_sessionmaker[AsyncSession],
    clean_business_tables: None,
) -> None:
    provider = MockLLMProvider.with_strict_outcomes(
        (
            MockLLMOutcome.timeout(),
            _successful_outcome(),
        ),
    )
    gateway = LLMGateway(
        provider=provider,
        max_repair_attempts=1,
    )

    try:
        async with postgresql_session_factory() as setup_session:
            await _persist_workspace_ticket_and_run(
                setup_session,
            )

        async with postgresql_session_factory() as claim_session:
            first_claim = await _claim_run(
                claim_session,
                worker_id="classification-worker-a",
                lease_token=_FIRST_LEASE_TOKEN,
                execution_request_id=(_FIRST_EXECUTION_REQUEST_ID),
                claimed_at=_FIRST_CLAIMED_AT,
            )

        async with postgresql_session_factory() as first_processing_session:
            first_result = await _process_claim(
                first_processing_session,
                claim=first_claim,
                gateway=gateway,
                model=MOCK_TICKET_CLASSIFIER_MODEL,
                finished_at=_FIRST_FINISHED_AT,
            )

        assert first_result is AgentRunTransitionResult.APPLIED
        assert provider.invocation_count == 1

        async with postgresql_session_factory() as retry_claim_session:
            second_claim = await _claim_run(
                retry_claim_session,
                worker_id="classification-worker-b",
                lease_token=_SECOND_LEASE_TOKEN,
                execution_request_id=(_SECOND_EXECUTION_REQUEST_ID),
                claimed_at=_SECOND_CLAIMED_AT,
            )

        assert second_claim.agent_run.attempt_count == 2
        assert second_claim.attempt.attempt_number == 2

        async with postgresql_session_factory() as second_processing_session:
            second_result = await _process_claim(
                second_processing_session,
                claim=second_claim,
                gateway=gateway,
                model=MOCK_TICKET_CLASSIFIER_MODEL,
                finished_at=_SECOND_FINISHED_AT,
            )

        assert second_result is AgentRunTransitionResult.APPLIED
        assert provider.invocation_count == 2

        async with postgresql_session_factory() as verification_session:
            run = await verification_session.get(
                AgentRunRecord,
                _AGENT_RUN_ID,
            )
            attempts = await _load_attempts(
                verification_session,
            )
            invocations = await _load_invocations(
                verification_session,
            )
            classification = await _load_classification(
                verification_session,
            )

        assert run is not None
        assert run.status == AgentRunStatus.SUCCEEDED.value
        assert run.attempt_count == 2
        assert run.last_error_code is None
        assert run.last_error_summary is None

        assert len(attempts) == 2
        first_attempt, second_attempt = attempts

        assert first_attempt.outcome == (AgentRunAttemptOutcome.RETRYABLE_FAILURE.value)
        assert first_attempt.error_code == "llm_timeout"
        assert first_attempt.finished_at == (_FIRST_FINISHED_AT)

        assert second_attempt.outcome == (AgentRunAttemptOutcome.SUCCEEDED.value)
        assert second_attempt.error_code is None
        assert second_attempt.finished_at == (_SECOND_FINISHED_AT)

        assert len(invocations) == 2

        invocation_by_attempt = {
            invocation.agent_run_attempt_id: invocation for invocation in invocations
        }
        first_invocation = invocation_by_attempt[first_attempt.id]
        second_invocation = invocation_by_attempt[second_attempt.id]

        # Invocation sequences restart within each attempt.
        assert first_invocation.invocation_sequence == 1
        assert second_invocation.invocation_sequence == 1

        assert first_invocation.status == "timed_out"
        assert first_invocation.error_code == "llm_timeout"
        assert first_invocation.provider_request_id == ("mock-request-1")
        assert first_invocation.input_tokens is None
        assert first_invocation.estimated_total_cost_usd is None

        assert second_invocation.status == "succeeded"
        assert second_invocation.error_code is None
        assert second_invocation.provider_request_id == ("mock-request-2")
        assert second_invocation.pricing_found is True
        assert second_invocation.estimated_total_cost_usd == (_ZERO_COST)

        assert classification.accepted_llm_invocation_id == (second_invocation.id)
        assert classification.category == "billing"
        assert classification.provider == "mock"
    finally:
        await provider.close()
