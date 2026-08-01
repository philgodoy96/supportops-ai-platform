"""Integration tests for processing claimed AgentRuns."""

from dataclasses import replace
from datetime import UTC, datetime, timedelta
from uuid import UUID

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from supportops.infrastructure.postgresql.transaction import (
    SqlAlchemyTransactionManager,
)
from supportops.modules.agent_runs.application.deterministic_executor import (
    DeterministicTicketProcessingExecutor,
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
    DETERMINISTIC_BASELINE_WORKFLOW_VERSION,
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
from supportops.modules.tickets.domain.models import Ticket
from supportops.modules.tickets.infrastructure.repository import (
    SqlAlchemyTicketRepository,
)
from supportops.modules.workspaces.domain.models import Workspace
from supportops.modules.workspaces.infrastructure.repository import (
    SqlAlchemyWorkspaceRepository,
)

pytestmark = pytest.mark.integration

_WORKSPACE_ID = UUID(
    "032c8c87-57cc-4d14-bfbd-04968b4e8cd4",
)
_TICKET_ID = UUID(
    "38bb60fe-d2ea-4615-b499-91aa45069019",
)
_RUN_ID = UUID(
    "69184ef1-4d71-452e-8070-0b784c29368e",
)
_LEASE_TOKEN = UUID(
    "dd0ae456-3467-41db-93d1-a908f40e8365",
)
_EXECUTION_REQUEST_ID = UUID(
    "d1fa068f-2278-47a8-b3c9-39ccf91f0a5e",
)
_BASE_TIMESTAMP = datetime(
    2026,
    7,
    31,
    12,
    0,
    tzinfo=UTC,
)
_CLAIMED_AT = _BASE_TIMESTAMP + timedelta(minutes=5)
_FINISHED_AT = _CLAIMED_AT + timedelta(seconds=1)


async def persist_workspace_ticket_and_run(
    session: AsyncSession,
    *,
    workflow_version: str | None = None,
) -> AgentRun:
    """Persist one workspace, ticket, and initial AgentRun."""

    workspace = Workspace(
        id=_WORKSPACE_ID,
        name="Platform Support",
        slug="platform-support",
        created_at=_BASE_TIMESTAMP,
        updated_at=_BASE_TIMESTAMP,
    )
    ticket = Ticket.create(
        ticket_id=_TICKET_ID,
        workspace_id=_WORKSPACE_ID,
        subject="Unable to access billing",
        description="The billing page returns an access error.",
        external_reference=None,
        ingestion_request_id=UUID(
            "725eec8a-c504-4071-ac96-c78cc907f26c",
        ),
        correlation_id=UUID(
            "1038c98e-62fd-45df-9839-138f7105cb78",
        ),
        now=_BASE_TIMESTAMP,
    )
    run = AgentRun.create_initial(
        agent_run_id=_RUN_ID,
        workspace_id=_WORKSPACE_ID,
        ticket_id=_TICKET_ID,
        ingestion_request_id=ticket.ingestion_request_id,
        correlation_id=ticket.correlation_id,
        workflow_version=DETERMINISTIC_BASELINE_WORKFLOW_VERSION,
        max_attempts=3,
        now=_BASE_TIMESTAMP,
    )

    if workflow_version is not None:
        run = replace(
            run,
            workflow_version=workflow_version,
        )

    async with SqlAlchemyTransactionManager(session).transaction():
        await SqlAlchemyWorkspaceRepository(session).add(workspace)
        await SqlAlchemyTicketRepository(session).add(ticket)
        await SqlAlchemyAgentRunRepository(session).add(run)

    return run


async def claim_and_commit(
    session: AsyncSession,
) -> AgentRunClaim:
    """Claim and commit the persisted AgentRun."""

    repository = SqlAlchemyAgentRunRepository(session)

    async with SqlAlchemyTransactionManager(session).transaction():
        claim = await repository.claim_next_available(
            ClaimAgentRunCommand(
                worker_id="worker-a",
                lease_token=_LEASE_TOKEN,
                execution_request_id=_EXECUTION_REQUEST_ID,
                claimed_at=_CLAIMED_AT,
                lease_expires_at=(_CLAIMED_AT + timedelta(seconds=45)),
            ),
        )

    assert claim is not None
    return claim


async def process_claim(
    session: AsyncSession,
    *,
    claim: AgentRunClaim,
) -> AgentRunTransitionResult:
    """Process one claim using real PostgreSQL repositories."""

    transaction_manager = SqlAlchemyTransactionManager(session)
    processor = ProcessClaimedAgentRun(
        ticket_repository=SqlAlchemyTicketRepository(session),
        agent_run_repository=SqlAlchemyAgentRunRepository(session),
        transaction_manager=transaction_manager,
        executor=DeterministicTicketProcessingExecutor(),
        retry_policy=AgentRunRetryPolicy(
            base_delay_seconds=2.0,
            maximum_delay_seconds=60.0,
        ),
        execution_timeout_seconds=30.0,
        utc_now=lambda: _FINISHED_AT,
    )

    return await processor.execute(claim)


async def load_run(
    session: AsyncSession,
) -> AgentRunRecord:
    """Load the persisted AgentRun."""

    run = await session.get(
        AgentRunRecord,
        _RUN_ID,
    )

    assert run is not None
    return run


async def load_attempts(
    session: AsyncSession,
) -> list[AgentRunAttemptRecord]:
    """Load persisted attempts in attempt-number order."""

    result = await session.execute(
        select(AgentRunAttemptRecord)
        .where(
            AgentRunAttemptRecord.agent_run_id == _RUN_ID,
        )
        .order_by(
            AgentRunAttemptRecord.attempt_number.asc(),
        ),
    )
    return list(result.scalars())


async def test_deterministic_processor_completes_claimed_run(
    postgresql_session_factory: async_sessionmaker[AsyncSession],
    clean_business_tables: None,
) -> None:
    async with postgresql_session_factory() as setup_session:
        await persist_workspace_ticket_and_run(setup_session)

    async with postgresql_session_factory() as claim_session:
        claim = await claim_and_commit(claim_session)

    assert claim.agent_run.status is AgentRunStatus.RUNNING
    assert claim.agent_run.attempt_count == 1
    assert claim.agent_run.lease_token == _LEASE_TOKEN
    assert claim.attempt.finished_at is None
    assert claim.attempt.outcome is None

    async with postgresql_session_factory() as processing_session:
        result = await process_claim(
            processing_session,
            claim=claim,
        )

    assert result is AgentRunTransitionResult.APPLIED

    async with postgresql_session_factory() as verification_session:
        run = await load_run(verification_session)
        attempts = await load_attempts(verification_session)

    assert run.status == AgentRunStatus.SUCCEEDED.value
    assert run.completed_at == _FINISHED_AT
    assert run.attempt_count == 1
    assert run.first_started_at == _CLAIMED_AT

    assert run.lease_owner is None
    assert run.lease_token is None
    assert run.lease_expires_at is None

    assert run.last_error_code is None
    assert run.last_error_summary is None

    assert len(attempts) == 1
    assert attempts[0].attempt_number == 1
    assert attempts[0].worker_id == "worker-a"
    assert attempts[0].lease_token == _LEASE_TOKEN
    assert attempts[0].execution_request_id == _EXECUTION_REQUEST_ID
    assert attempts[0].finished_at == _FINISHED_AT
    assert attempts[0].outcome == AgentRunAttemptOutcome.SUCCEEDED.value
    assert attempts[0].error_code is None
    assert attempts[0].error_summary is None


async def test_unsupported_workflow_version_fails_terminally(
    postgresql_session_factory: async_sessionmaker[AsyncSession],
    clean_business_tables: None,
) -> None:
    async with postgresql_session_factory() as setup_session:
        await persist_workspace_ticket_and_run(
            setup_session,
            workflow_version="unsupported-version",
        )

    async with postgresql_session_factory() as claim_session:
        claim = await claim_and_commit(claim_session)

    assert claim.agent_run.workflow_version == "unsupported-version"
    assert claim.agent_run.attempt_count == 1

    async with postgresql_session_factory() as processing_session:
        result = await process_claim(
            processing_session,
            claim=claim,
        )

    assert result is AgentRunTransitionResult.APPLIED

    async with postgresql_session_factory() as verification_session:
        run = await load_run(verification_session)
        attempts = await load_attempts(verification_session)

    assert run.status == AgentRunStatus.FAILED.value
    assert run.completed_at == _FINISHED_AT
    assert run.attempt_count == 1
    assert run.available_at == _BASE_TIMESTAMP

    assert run.lease_owner is None
    assert run.lease_token is None
    assert run.lease_expires_at is None

    assert run.last_error_code == "unsupported_workflow_version"
    assert run.last_error_summary == (
        "The AgentRun workflow version is not supported by the configured executor."
    )

    assert len(attempts) == 1
    assert attempts[0].attempt_number == 1
    assert attempts[0].finished_at == _FINISHED_AT
    assert attempts[0].outcome == AgentRunAttemptOutcome.TERMINAL_FAILURE.value
    assert attempts[0].error_code == "unsupported_workflow_version"
    assert attempts[0].error_summary == (
        "The AgentRun workflow version is not supported by the configured executor."
    )


async def test_processed_run_cannot_be_completed_again(
    postgresql_session_factory: async_sessionmaker[AsyncSession],
    clean_business_tables: None,
) -> None:
    async with postgresql_session_factory() as setup_session:
        await persist_workspace_ticket_and_run(setup_session)

    async with postgresql_session_factory() as claim_session:
        claim = await claim_and_commit(claim_session)

    async with postgresql_session_factory() as first_processing_session:
        first_result = await process_claim(
            first_processing_session,
            claim=claim,
        )

    async with postgresql_session_factory() as repeated_session:
        repeated_result = await process_claim(
            repeated_session,
            claim=claim,
        )

    assert first_result is AgentRunTransitionResult.APPLIED
    assert repeated_result is AgentRunTransitionResult.LEASE_LOST

    async with postgresql_session_factory() as verification_session:
        run = await load_run(verification_session)
        attempts = await load_attempts(verification_session)

    assert run.status == AgentRunStatus.SUCCEEDED.value
    assert run.completed_at == _FINISHED_AT
    assert len(attempts) == 1
    assert attempts[0].outcome == AgentRunAttemptOutcome.SUCCEEDED.value
