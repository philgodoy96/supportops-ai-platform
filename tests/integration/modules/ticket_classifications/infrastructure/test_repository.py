"""PostgreSQL integration tests for fenced classification persistence."""

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
from supportops.modules.ticket_classifications.application.persistence import (
    ClassificationPersistenceResult,
    PersistClassificationExecutionCommand,
)
from supportops.modules.ticket_classifications.domain.models import (
    LLMInvocation,
)
from supportops.modules.ticket_classifications.infrastructure.models import (
    LLMInvocationRecord,
    TicketClassificationRecord,
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

_BASE_TIMESTAMP = datetime.now(UTC).replace(microsecond=0)
_CLAIMED_AT = _BASE_TIMESTAMP + timedelta(seconds=1)

_WORKSPACE_ID = UUID(
    "91000000-0000-4000-8000-000000000001",
)
_TICKET_ID = UUID(
    "92000000-0000-4000-8000-000000000002",
)
_AGENT_RUN_ID = UUID(
    "93000000-0000-4000-8000-000000000003",
)
_LEASE_TOKEN = UUID(
    "94000000-0000-4000-8000-000000000004",
)
_EXECUTION_REQUEST_ID = UUID(
    "95000000-0000-4000-8000-000000000005",
)
_INVOCATION_ID = UUID(
    "96000000-0000-4000-8000-000000000006",
)


async def _persist_and_claim(
    session: AsyncSession,
) -> AgentRunClaim:
    """Persist and claim one classification AgentRun."""

    workspace = Workspace(
        id=_WORKSPACE_ID,
        name="Classification Repository",
        slug="classification-repository",
        created_at=_BASE_TIMESTAMP,
        updated_at=_BASE_TIMESTAMP,
    )
    ticket = Ticket.create(
        ticket_id=_TICKET_ID,
        workspace_id=_WORKSPACE_ID,
        subject="Provider timeout",
        description=("The classification provider timeout is being tested."),
        external_reference=None,
        ingestion_request_id=UUID(
            "97000000-0000-4000-8000-000000000007",
        ),
        correlation_id=UUID(
            "98000000-0000-4000-8000-000000000008",
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
        max_retryable_failures=3,
        now=_BASE_TIMESTAMP,
    )
    transaction_manager = SqlAlchemyTransactionManager(
        session,
    )
    repository = SqlAlchemyAgentRunRepository(session)

    async with transaction_manager.transaction():
        await SqlAlchemyWorkspaceRepository(
            session,
        ).add(workspace)
        await SqlAlchemyTicketRepository(
            session,
        ).add(ticket)
        await repository.add(run)

    async with transaction_manager.transaction():
        claim = await repository.claim_next_available(
            ClaimAgentRunCommand(
                worker_id="repository-worker",
                lease_token=_LEASE_TOKEN,
                execution_request_id=(_EXECUTION_REQUEST_ID),
                claimed_at=_CLAIMED_AT,
                lease_expires_at=(_CLAIMED_AT + timedelta(minutes=5)),
            ),
        )

    assert claim is not None
    return claim


def _timeout_invocation(
    *,
    claim: AgentRunClaim,
    created_at: datetime,
) -> LLMInvocation:
    """Create one durable timeout trace."""

    return LLMInvocation.create(
        invocation_id=_INVOCATION_ID,
        workspace_id=_WORKSPACE_ID,
        ticket_id=_TICKET_ID,
        agent_run_id=_AGENT_RUN_ID,
        agent_run_attempt_id=claim.attempt.id,
        invocation_sequence=1,
        status=LLMInvocationStatus.TIMED_OUT,
        provider="mock",
        model="mock-ticket-classifier-v1",
        provider_request_id="mock-request-1",
        prompt_id="ticket-classification",
        prompt_version=1,
        prompt_content_hash="a" * 64,
        schema_version=(TICKET_CLASSIFICATION_SCHEMA_VERSION),
        input_tokens=None,
        cached_input_tokens=None,
        output_tokens=None,
        reasoning_tokens=None,
        total_tokens=None,
        pricing_catalog_version=(PRICING_CATALOG_VERSION),
        pricing_found=True,
        estimated_input_cost_usd=None,
        estimated_cached_input_cost_usd=None,
        estimated_output_cost_usd=None,
        estimated_total_cost_usd=None,
        latency_ms=12_000,
        error_code=LLMErrorCode.TIMEOUT,
        now=created_at,
    )


async def _count_invocations(
    session: AsyncSession,
) -> int:
    """Count persisted logical invocations."""

    result = await session.execute(
        select(func.count()).select_from(
            LLMInvocationRecord,
        ),
    )

    return result.scalar_one()


async def _count_classifications(
    session: AsyncSession,
) -> int:
    """Count persisted accepted classifications."""

    result = await session.execute(
        select(func.count()).select_from(
            TicketClassificationRecord,
        ),
    )

    return result.scalar_one()


async def test_repeated_failure_persistence_is_idempotent(
    postgresql_session: AsyncSession,
    clean_business_tables: None,
) -> None:
    claim = await _persist_and_claim(
        postgresql_session,
    )
    persisted_at = _CLAIMED_AT + timedelta(seconds=1)
    invocation = _timeout_invocation(
        claim=claim,
        created_at=persisted_at,
    )
    command = PersistClassificationExecutionCommand(
        workspace_id=_WORKSPACE_ID,
        ticket_id=_TICKET_ID,
        agent_run_id=_AGENT_RUN_ID,
        agent_run_attempt_id=claim.attempt.id,
        lease_token=_LEASE_TOKEN,
        persisted_at=persisted_at,
        invocations=(invocation,),
        classification=None,
    )
    repository = SqlAlchemyClassificationPersistenceRepository(
        postgresql_session,
    )
    transaction_manager = SqlAlchemyTransactionManager(
        postgresql_session,
    )

    async with transaction_manager.transaction():
        first_result = await repository.persist_fenced(
            command,
        )

    async with transaction_manager.transaction():
        repeated_result = await repository.persist_fenced(
            command,
        )

    assert first_result is (ClassificationPersistenceResult.APPLIED)
    assert repeated_result is (ClassificationPersistenceResult.ALREADY_RECORDED)
    assert await _count_invocations(postgresql_session) == 1
    assert await _count_classifications(postgresql_session) == 0

    result = await postgresql_session.execute(
        select(LLMInvocationRecord),
    )
    record = result.scalar_one()

    assert record.id == _INVOCATION_ID
    assert record.agent_run_attempt_id == claim.attempt.id
    assert record.invocation_sequence == 1
    assert record.status == "timed_out"
    assert record.error_code == "llm_timeout"


async def test_expired_lease_rejects_invocation_persistence(
    postgresql_session: AsyncSession,
    clean_business_tables: None,
) -> None:
    claim = await _persist_and_claim(
        postgresql_session,
    )
    lease_expires_at = claim.agent_run.lease_expires_at

    assert lease_expires_at is not None

    invocation = _timeout_invocation(
        claim=claim,
        created_at=lease_expires_at,
    )
    command = PersistClassificationExecutionCommand(
        workspace_id=_WORKSPACE_ID,
        ticket_id=_TICKET_ID,
        agent_run_id=_AGENT_RUN_ID,
        agent_run_attempt_id=claim.attempt.id,
        lease_token=_LEASE_TOKEN,
        persisted_at=lease_expires_at,
        invocations=(invocation,),
        classification=None,
    )
    repository = SqlAlchemyClassificationPersistenceRepository(
        postgresql_session,
    )

    async with SqlAlchemyTransactionManager(
        postgresql_session,
    ).transaction():
        result = await repository.persist_fenced(
            command,
        )

    assert result is (ClassificationPersistenceResult.LEASE_LOST)
    assert await _count_invocations(postgresql_session) == 0
    assert await _count_classifications(postgresql_session) == 0
