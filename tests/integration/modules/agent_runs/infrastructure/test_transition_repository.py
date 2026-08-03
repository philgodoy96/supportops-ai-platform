"""Integration tests for fenced PostgreSQL AgentRun transitions."""

import asyncio
from datetime import UTC, datetime, timedelta
from uuid import UUID

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from supportops.infrastructure.postgresql.transaction import (
    SqlAlchemyTransactionManager,
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
    AgentRunApprovalRequeueResult,
    AgentRunFailureDisposition,
    AgentRunTransitionResult,
    CompleteAgentRunCommand,
    FailAgentRunCommand,
    RequeueWaitingAgentRunCommand,
    WaitForApprovalAgentRunCommand,
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
_BASE_TIMESTAMP = datetime(
    2026,
    7,
    31,
    12,
    0,
    tzinfo=UTC,
)
_FIRST_CLAIMED_AT = _BASE_TIMESTAMP + timedelta(minutes=5)
_FIRST_LEASE_TOKEN = UUID(
    "dd0ae456-3467-41db-93d1-a908f40e8365",
)
_FIRST_EXECUTION_REQUEST_ID = UUID(
    "d1fa068f-2278-47a8-b3c9-39ccf91f0a5e",
)


async def persist_workspace_ticket_and_run(
    session: AsyncSession,
    *,
    max_retryable_failures: int = 3,
) -> AgentRun:
    """Persist one queued AgentRun with its workspace and ticket."""

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
        description="The dashboard returns an access error.",
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
        max_retryable_failures=max_retryable_failures,
        now=_BASE_TIMESTAMP,
    )

    async with SqlAlchemyTransactionManager(session).transaction():
        await SqlAlchemyWorkspaceRepository(session).add(workspace)
        await SqlAlchemyTicketRepository(session).add(ticket)
        await SqlAlchemyAgentRunRepository(session).add(run)

    return run


def create_claim_command(
    *,
    worker_id: str,
    lease_token: UUID,
    execution_request_id: UUID,
    claimed_at: datetime,
    lease_seconds: int = 45,
) -> ClaimAgentRunCommand:
    """Create deterministic ownership values for one claim."""

    return ClaimAgentRunCommand(
        worker_id=worker_id,
        lease_token=lease_token,
        execution_request_id=execution_request_id,
        claimed_at=claimed_at,
        lease_expires_at=claimed_at + timedelta(seconds=lease_seconds),
    )


async def claim_and_commit(
    session: AsyncSession,
    *,
    command: ClaimAgentRunCommand,
) -> AgentRunClaim:
    """Claim and commit one AgentRun."""

    repository = SqlAlchemyAgentRunRepository(session)

    async with SqlAlchemyTransactionManager(session).transaction():
        claim = await repository.claim_next_available(command)

    assert claim is not None
    return claim


async def mark_succeeded_and_commit(
    session: AsyncSession,
    *,
    command: CompleteAgentRunCommand,
) -> AgentRunTransitionResult:
    """Persist and commit a successful transition."""

    repository = SqlAlchemyAgentRunRepository(session)

    async with SqlAlchemyTransactionManager(session).transaction():
        return await repository.mark_succeeded(command)


async def mark_waiting_and_commit(
    session: AsyncSession,
    *,
    command: WaitForApprovalAgentRunCommand,
) -> AgentRunTransitionResult:
    """Persist and commit a waiting-for-approval transition."""

    repository = SqlAlchemyAgentRunRepository(session)

    async with SqlAlchemyTransactionManager(session).transaction():
        return await repository.mark_waiting_for_approval(command)


async def requeue_waiting_and_commit(
    session: AsyncSession,
    *,
    command: RequeueWaitingAgentRunCommand,
) -> AgentRunApprovalRequeueResult:
    """Persist and commit a waiting-for-approval requeue."""

    repository = SqlAlchemyAgentRunRepository(session)

    async with SqlAlchemyTransactionManager(session).transaction():
        return await repository.requeue_waiting_for_approval(command)


async def record_failure_and_commit(
    session: AsyncSession,
    *,
    command: FailAgentRunCommand,
) -> AgentRunTransitionResult:
    """Persist and commit a failure transition."""

    repository = SqlAlchemyAgentRunRepository(session)

    async with SqlAlchemyTransactionManager(session).transaction():
        return await repository.record_failure(command)


async def load_run(
    session: AsyncSession,
) -> AgentRunRecord:
    """Load the persisted AgentRun."""

    record = await session.get(
        AgentRunRecord,
        _RUN_ID,
    )

    assert record is not None
    return record


async def load_attempts(
    session: AsyncSession,
) -> list[AgentRunAttemptRecord]:
    """Load attempt history in ascending attempt order."""

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


async def test_success_closes_run_and_attempt_atomically(
    postgresql_session_factory: async_sessionmaker[AsyncSession],
    clean_business_tables: None,
) -> None:
    finished_at = _FIRST_CLAIMED_AT + timedelta(seconds=10)

    async with postgresql_session_factory() as setup_session:
        await persist_workspace_ticket_and_run(setup_session)

    async with postgresql_session_factory() as claim_session:
        await claim_and_commit(
            claim_session,
            command=create_claim_command(
                worker_id="worker-a",
                lease_token=_FIRST_LEASE_TOKEN,
                execution_request_id=_FIRST_EXECUTION_REQUEST_ID,
                claimed_at=_FIRST_CLAIMED_AT,
            ),
        )

    async with postgresql_session_factory() as completion_session:
        result = await mark_succeeded_and_commit(
            completion_session,
            command=CompleteAgentRunCommand(
                agent_run_id=_RUN_ID,
                lease_token=_FIRST_LEASE_TOKEN,
                finished_at=finished_at,
            ),
        )

    assert result is AgentRunTransitionResult.APPLIED

    async with postgresql_session_factory() as verification_session:
        run = await load_run(verification_session)
        attempts = await load_attempts(verification_session)

    assert run.status == AgentRunStatus.SUCCEEDED.value
    assert run.completed_at == finished_at
    assert run.lease_owner is None
    assert run.lease_token is None
    assert run.lease_expires_at is None
    assert run.last_error_code is None
    assert run.last_error_summary is None
    assert run.attempt_count == 1
    assert run.first_started_at == _FIRST_CLAIMED_AT

    assert len(attempts) == 1
    assert attempts[0].finished_at == finished_at
    assert attempts[0].outcome == AgentRunAttemptOutcome.SUCCEEDED.value
    assert attempts[0].error_code is None
    assert attempts[0].error_summary is None


async def test_repeated_completion_is_fenced_after_success(
    postgresql_session_factory: async_sessionmaker[AsyncSession],
    clean_business_tables: None,
) -> None:
    finished_at = _FIRST_CLAIMED_AT + timedelta(seconds=10)

    async with postgresql_session_factory() as setup_session:
        await persist_workspace_ticket_and_run(setup_session)

    async with postgresql_session_factory() as claim_session:
        await claim_and_commit(
            claim_session,
            command=create_claim_command(
                worker_id="worker-a",
                lease_token=_FIRST_LEASE_TOKEN,
                execution_request_id=_FIRST_EXECUTION_REQUEST_ID,
                claimed_at=_FIRST_CLAIMED_AT,
            ),
        )

    command = CompleteAgentRunCommand(
        agent_run_id=_RUN_ID,
        lease_token=_FIRST_LEASE_TOKEN,
        finished_at=finished_at,
    )

    async with postgresql_session_factory() as first_session:
        first_result = await mark_succeeded_and_commit(
            first_session,
            command=command,
        )

    async with postgresql_session_factory() as second_session:
        second_result = await mark_succeeded_and_commit(
            second_session,
            command=command,
        )

    assert first_result is AgentRunTransitionResult.APPLIED
    assert second_result is AgentRunTransitionResult.LEASE_LOST

    async with postgresql_session_factory() as verification_session:
        run = await load_run(verification_session)
        attempts = await load_attempts(verification_session)

    assert run.status == AgentRunStatus.SUCCEEDED.value
    assert run.completed_at == finished_at
    assert len(attempts) == 1
    assert attempts[0].outcome == AgentRunAttemptOutcome.SUCCEEDED.value


async def test_retryable_failure_schedules_future_retry(
    postgresql_session_factory: async_sessionmaker[AsyncSession],
    clean_business_tables: None,
) -> None:
    finished_at = _FIRST_CLAIMED_AT + timedelta(seconds=10)
    retry_available_at = finished_at + timedelta(seconds=2)

    async with postgresql_session_factory() as setup_session:
        await persist_workspace_ticket_and_run(setup_session)

    async with postgresql_session_factory() as claim_session:
        await claim_and_commit(
            claim_session,
            command=create_claim_command(
                worker_id="worker-a",
                lease_token=_FIRST_LEASE_TOKEN,
                execution_request_id=_FIRST_EXECUTION_REQUEST_ID,
                claimed_at=_FIRST_CLAIMED_AT,
            ),
        )

    async with postgresql_session_factory() as failure_session:
        result = await record_failure_and_commit(
            failure_session,
            command=FailAgentRunCommand(
                agent_run_id=_RUN_ID,
                lease_token=_FIRST_LEASE_TOKEN,
                finished_at=finished_at,
                outcome=AgentRunAttemptOutcome.RETRYABLE_FAILURE,
                disposition=AgentRunFailureDisposition.RETRY_SCHEDULED,
                error_code="retryable_executor_failure",
                error_summary=("The configured executor reported a retryable failure."),
                retry_available_at=retry_available_at,
            ),
        )

    assert result is AgentRunTransitionResult.APPLIED

    async with postgresql_session_factory() as verification_session:
        run = await load_run(verification_session)
        attempts = await load_attempts(verification_session)

    assert run.status == AgentRunStatus.RETRY_SCHEDULED.value
    assert run.available_at == retry_available_at
    assert run.completed_at is None
    assert run.lease_owner is None
    assert run.lease_token is None
    assert run.lease_expires_at is None
    assert run.last_error_code == "retryable_executor_failure"
    assert run.last_error_summary == ("The configured executor reported a retryable failure.")

    assert len(attempts) == 1
    assert attempts[0].outcome == AgentRunAttemptOutcome.RETRYABLE_FAILURE.value
    assert attempts[0].finished_at == finished_at
    assert attempts[0].error_code == "retryable_executor_failure"


async def test_timeout_schedules_retry_with_timed_out_outcome(
    postgresql_session_factory: async_sessionmaker[AsyncSession],
    clean_business_tables: None,
) -> None:
    finished_at = _FIRST_CLAIMED_AT + timedelta(seconds=30)
    retry_available_at = finished_at + timedelta(seconds=2)

    async with postgresql_session_factory() as setup_session:
        await persist_workspace_ticket_and_run(setup_session)

    async with postgresql_session_factory() as claim_session:
        await claim_and_commit(
            claim_session,
            command=create_claim_command(
                worker_id="worker-a",
                lease_token=_FIRST_LEASE_TOKEN,
                execution_request_id=_FIRST_EXECUTION_REQUEST_ID,
                claimed_at=_FIRST_CLAIMED_AT,
            ),
        )

    async with postgresql_session_factory() as failure_session:
        result = await record_failure_and_commit(
            failure_session,
            command=FailAgentRunCommand(
                agent_run_id=_RUN_ID,
                lease_token=_FIRST_LEASE_TOKEN,
                finished_at=finished_at,
                outcome=AgentRunAttemptOutcome.TIMED_OUT,
                disposition=AgentRunFailureDisposition.RETRY_SCHEDULED,
                error_code="executor_timeout",
                error_summary=("The configured executor exceeded its execution timeout."),
                retry_available_at=retry_available_at,
            ),
        )

    assert result is AgentRunTransitionResult.APPLIED

    async with postgresql_session_factory() as verification_session:
        run = await load_run(verification_session)
        attempts = await load_attempts(verification_session)

    assert run.status == AgentRunStatus.RETRY_SCHEDULED.value
    assert run.last_error_code == "executor_timeout"
    assert len(attempts) == 1
    assert attempts[0].outcome == AgentRunAttemptOutcome.TIMED_OUT.value
    assert attempts[0].error_code == "executor_timeout"


async def test_terminal_failure_marks_run_failed(
    postgresql_session_factory: async_sessionmaker[AsyncSession],
    clean_business_tables: None,
) -> None:
    finished_at = _FIRST_CLAIMED_AT + timedelta(seconds=10)

    async with postgresql_session_factory() as setup_session:
        await persist_workspace_ticket_and_run(setup_session)

    async with postgresql_session_factory() as claim_session:
        await claim_and_commit(
            claim_session,
            command=create_claim_command(
                worker_id="worker-a",
                lease_token=_FIRST_LEASE_TOKEN,
                execution_request_id=_FIRST_EXECUTION_REQUEST_ID,
                claimed_at=_FIRST_CLAIMED_AT,
            ),
        )

    async with postgresql_session_factory() as failure_session:
        result = await record_failure_and_commit(
            failure_session,
            command=FailAgentRunCommand(
                agent_run_id=_RUN_ID,
                lease_token=_FIRST_LEASE_TOKEN,
                finished_at=finished_at,
                outcome=AgentRunAttemptOutcome.TERMINAL_FAILURE,
                disposition=AgentRunFailureDisposition.FAILED,
                error_code="terminal_executor_failure",
                error_summary=("The configured executor reported a terminal failure."),
                retry_available_at=None,
            ),
        )

    assert result is AgentRunTransitionResult.APPLIED

    async with postgresql_session_factory() as verification_session:
        run = await load_run(verification_session)
        attempts = await load_attempts(verification_session)

    assert run.status == AgentRunStatus.FAILED.value
    assert run.completed_at == finished_at
    assert run.lease_owner is None
    assert run.lease_token is None
    assert run.lease_expires_at is None
    assert run.last_error_code == "terminal_executor_failure"

    assert len(attempts) == 1
    assert attempts[0].outcome == AgentRunAttemptOutcome.TERMINAL_FAILURE.value


async def test_exhausted_retryable_failure_marks_run_failed(
    postgresql_session_factory: async_sessionmaker[AsyncSession],
    clean_business_tables: None,
) -> None:
    finished_at = _FIRST_CLAIMED_AT + timedelta(seconds=10)

    async with postgresql_session_factory() as setup_session:
        await persist_workspace_ticket_and_run(
            setup_session,
            max_retryable_failures=1,
        )

    async with postgresql_session_factory() as claim_session:
        claim = await claim_and_commit(
            claim_session,
            command=create_claim_command(
                worker_id="worker-a",
                lease_token=_FIRST_LEASE_TOKEN,
                execution_request_id=_FIRST_EXECUTION_REQUEST_ID,
                claimed_at=_FIRST_CLAIMED_AT,
            ),
        )

    assert claim.agent_run.attempt_count == 1
    assert claim.agent_run.max_retryable_failures == 1

    async with postgresql_session_factory() as failure_session:
        result = await record_failure_and_commit(
            failure_session,
            command=FailAgentRunCommand(
                agent_run_id=_RUN_ID,
                lease_token=_FIRST_LEASE_TOKEN,
                finished_at=finished_at,
                outcome=AgentRunAttemptOutcome.RETRYABLE_FAILURE,
                disposition=AgentRunFailureDisposition.FAILED,
                error_code="unexpected_executor_failure",
                error_summary=("The executor failed unexpectedly and exhausted retries."),
                retry_available_at=None,
            ),
        )

    assert result is AgentRunTransitionResult.APPLIED

    async with postgresql_session_factory() as verification_session:
        run = await load_run(verification_session)
        attempts = await load_attempts(verification_session)

    assert run.status == AgentRunStatus.FAILED.value
    assert run.completed_at == finished_at
    assert run.attempt_count == 1
    assert attempts[0].outcome == AgentRunAttemptOutcome.RETRYABLE_FAILURE.value


async def test_stale_token_cannot_complete_reclaimed_run(
    postgresql_session_factory: async_sessionmaker[AsyncSession],
    clean_business_tables: None,
) -> None:
    first_finished_at = _FIRST_CLAIMED_AT + timedelta(seconds=10)
    retry_available_at = first_finished_at + timedelta(seconds=2)
    second_claimed_at = retry_available_at
    second_lease_token = UUID(
        "35e64ab6-d1b2-4386-a51e-48f11fa6d058",
    )
    second_execution_request_id = UUID(
        "43499fc4-3638-4097-aaf2-3c5300cf6cd6",
    )

    async with postgresql_session_factory() as setup_session:
        await persist_workspace_ticket_and_run(setup_session)

    async with postgresql_session_factory() as first_claim_session:
        await claim_and_commit(
            first_claim_session,
            command=create_claim_command(
                worker_id="worker-a",
                lease_token=_FIRST_LEASE_TOKEN,
                execution_request_id=_FIRST_EXECUTION_REQUEST_ID,
                claimed_at=_FIRST_CLAIMED_AT,
            ),
        )

    async with postgresql_session_factory() as retry_session:
        retry_result = await record_failure_and_commit(
            retry_session,
            command=FailAgentRunCommand(
                agent_run_id=_RUN_ID,
                lease_token=_FIRST_LEASE_TOKEN,
                finished_at=first_finished_at,
                outcome=AgentRunAttemptOutcome.RETRYABLE_FAILURE,
                disposition=AgentRunFailureDisposition.RETRY_SCHEDULED,
                error_code="retryable_executor_failure",
                error_summary=("The configured executor reported a retryable failure."),
                retry_available_at=retry_available_at,
            ),
        )

    assert retry_result is AgentRunTransitionResult.APPLIED

    async with postgresql_session_factory() as second_claim_session:
        second_claim = await claim_and_commit(
            second_claim_session,
            command=create_claim_command(
                worker_id="worker-b",
                lease_token=second_lease_token,
                execution_request_id=second_execution_request_id,
                claimed_at=second_claimed_at,
            ),
        )

    assert second_claim.agent_run.attempt_count == 2
    assert second_claim.agent_run.lease_token == second_lease_token

    async with postgresql_session_factory() as stale_session:
        stale_result = await mark_succeeded_and_commit(
            stale_session,
            command=CompleteAgentRunCommand(
                agent_run_id=_RUN_ID,
                lease_token=_FIRST_LEASE_TOKEN,
                finished_at=second_claimed_at + timedelta(seconds=1),
            ),
        )

    assert stale_result is AgentRunTransitionResult.LEASE_LOST

    async with postgresql_session_factory() as verification_session:
        run = await load_run(verification_session)
        attempts = await load_attempts(verification_session)

    assert run.status == AgentRunStatus.RUNNING.value
    assert run.attempt_count == 2
    assert run.lease_owner == "worker-b"
    assert run.lease_token == second_lease_token

    assert len(attempts) == 2
    assert attempts[0].outcome == AgentRunAttemptOutcome.RETRYABLE_FAILURE.value
    assert attempts[1].finished_at is None
    assert attempts[1].outcome is None
    assert attempts[1].lease_token == second_lease_token


async def test_expired_lease_cannot_complete_run(
    postgresql_session_factory: async_sessionmaker[AsyncSession],
    clean_business_tables: None,
) -> None:
    lease_seconds = 5
    finished_at = _FIRST_CLAIMED_AT + timedelta(
        seconds=lease_seconds + 1,
    )

    async with postgresql_session_factory() as setup_session:
        await persist_workspace_ticket_and_run(setup_session)

    async with postgresql_session_factory() as claim_session:
        await claim_and_commit(
            claim_session,
            command=create_claim_command(
                worker_id="worker-a",
                lease_token=_FIRST_LEASE_TOKEN,
                execution_request_id=_FIRST_EXECUTION_REQUEST_ID,
                claimed_at=_FIRST_CLAIMED_AT,
                lease_seconds=lease_seconds,
            ),
        )

    async with postgresql_session_factory() as completion_session:
        result = await mark_succeeded_and_commit(
            completion_session,
            command=CompleteAgentRunCommand(
                agent_run_id=_RUN_ID,
                lease_token=_FIRST_LEASE_TOKEN,
                finished_at=finished_at,
            ),
        )

    assert result is AgentRunTransitionResult.LEASE_LOST

    async with postgresql_session_factory() as verification_session:
        run = await load_run(verification_session)
        attempts = await load_attempts(verification_session)

    assert run.status == AgentRunStatus.RUNNING.value
    assert run.lease_token == _FIRST_LEASE_TOKEN
    assert run.completed_at is None
    assert len(attempts) == 1
    assert attempts[0].finished_at is None
    assert attempts[0].outcome is None


async def test_waiting_for_approval_closes_attempt_and_clears_lease(
    postgresql_session_factory: async_sessionmaker[AsyncSession],
    clean_business_tables: None,
) -> None:
    finished_at = _FIRST_CLAIMED_AT + timedelta(seconds=10)

    async with postgresql_session_factory() as setup_session:
        await persist_workspace_ticket_and_run(setup_session)

    async with postgresql_session_factory() as claim_session:
        await claim_and_commit(
            claim_session,
            command=create_claim_command(
                worker_id="worker-a",
                lease_token=_FIRST_LEASE_TOKEN,
                execution_request_id=_FIRST_EXECUTION_REQUEST_ID,
                claimed_at=_FIRST_CLAIMED_AT,
            ),
        )

    async with postgresql_session_factory() as waiting_session:
        result = await mark_waiting_and_commit(
            waiting_session,
            command=WaitForApprovalAgentRunCommand(
                agent_run_id=_RUN_ID,
                lease_token=_FIRST_LEASE_TOKEN,
                finished_at=finished_at,
            ),
        )

    assert result is AgentRunTransitionResult.APPLIED

    async with postgresql_session_factory() as verification_session:
        run = await load_run(verification_session)
        attempts = await load_attempts(verification_session)

    assert run.status == AgentRunStatus.WAITING_FOR_APPROVAL.value
    assert run.available_at is None
    assert run.completed_at is None
    assert run.lease_owner is None
    assert run.lease_token is None
    assert run.lease_expires_at is None
    assert run.last_error_code is None
    assert run.last_error_summary is None
    assert run.attempt_count == 1
    assert run.retryable_failure_count == 0
    assert len(attempts) == 1
    assert attempts[0].finished_at == finished_at
    assert attempts[0].outcome == (AgentRunAttemptOutcome.AWAITING_APPROVAL.value)
    assert attempts[0].error_code is None
    assert attempts[0].error_summary is None


async def test_stale_token_cannot_mark_waiting_for_approval(
    postgresql_session_factory: async_sessionmaker[AsyncSession],
    clean_business_tables: None,
) -> None:
    finished_at = _FIRST_CLAIMED_AT + timedelta(seconds=10)
    stale_token = UUID("aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee")

    async with postgresql_session_factory() as setup_session:
        await persist_workspace_ticket_and_run(setup_session)

    async with postgresql_session_factory() as claim_session:
        await claim_and_commit(
            claim_session,
            command=create_claim_command(
                worker_id="worker-a",
                lease_token=_FIRST_LEASE_TOKEN,
                execution_request_id=_FIRST_EXECUTION_REQUEST_ID,
                claimed_at=_FIRST_CLAIMED_AT,
            ),
        )

    async with postgresql_session_factory() as waiting_session:
        result = await mark_waiting_and_commit(
            waiting_session,
            command=WaitForApprovalAgentRunCommand(
                agent_run_id=_RUN_ID,
                lease_token=stale_token,
                finished_at=finished_at,
            ),
        )

    assert result is AgentRunTransitionResult.LEASE_LOST

    async with postgresql_session_factory() as verification_session:
        run = await load_run(verification_session)
        attempts = await load_attempts(verification_session)

    assert run.status == AgentRunStatus.RUNNING.value
    assert run.lease_token == _FIRST_LEASE_TOKEN
    assert run.available_at is not None
    assert run.completed_at is None
    assert len(attempts) == 1
    assert attempts[0].finished_at is None
    assert attempts[0].outcome is None


async def test_expired_lease_cannot_mark_waiting_for_approval(
    postgresql_session_factory: async_sessionmaker[AsyncSession],
    clean_business_tables: None,
) -> None:
    lease_seconds = 5
    finished_at = _FIRST_CLAIMED_AT + timedelta(
        seconds=lease_seconds + 1,
    )

    async with postgresql_session_factory() as setup_session:
        await persist_workspace_ticket_and_run(setup_session)

    async with postgresql_session_factory() as claim_session:
        await claim_and_commit(
            claim_session,
            command=create_claim_command(
                worker_id="worker-a",
                lease_token=_FIRST_LEASE_TOKEN,
                execution_request_id=_FIRST_EXECUTION_REQUEST_ID,
                claimed_at=_FIRST_CLAIMED_AT,
                lease_seconds=lease_seconds,
            ),
        )

    async with postgresql_session_factory() as waiting_session:
        result = await mark_waiting_and_commit(
            waiting_session,
            command=WaitForApprovalAgentRunCommand(
                agent_run_id=_RUN_ID,
                lease_token=_FIRST_LEASE_TOKEN,
                finished_at=finished_at,
            ),
        )

    assert result is AgentRunTransitionResult.LEASE_LOST

    async with postgresql_session_factory() as verification_session:
        run = await load_run(verification_session)
        attempts = await load_attempts(verification_session)

    assert run.status == AgentRunStatus.RUNNING.value
    assert run.lease_token == _FIRST_LEASE_TOKEN
    assert len(attempts) == 1
    assert attempts[0].finished_at is None
    assert attempts[0].outcome is None


async def test_requeue_waiting_for_approval_sets_queued_without_new_attempt(
    postgresql_session_factory: async_sessionmaker[AsyncSession],
    clean_business_tables: None,
) -> None:
    finished_at = _FIRST_CLAIMED_AT + timedelta(seconds=10)
    requeued_at = finished_at + timedelta(minutes=1)

    async with postgresql_session_factory() as setup_session:
        await persist_workspace_ticket_and_run(setup_session)

    async with postgresql_session_factory() as claim_session:
        await claim_and_commit(
            claim_session,
            command=create_claim_command(
                worker_id="worker-a",
                lease_token=_FIRST_LEASE_TOKEN,
                execution_request_id=_FIRST_EXECUTION_REQUEST_ID,
                claimed_at=_FIRST_CLAIMED_AT,
            ),
        )

    async with postgresql_session_factory() as waiting_session:
        waiting_result = await mark_waiting_and_commit(
            waiting_session,
            command=WaitForApprovalAgentRunCommand(
                agent_run_id=_RUN_ID,
                lease_token=_FIRST_LEASE_TOKEN,
                finished_at=finished_at,
            ),
        )

    assert waiting_result is AgentRunTransitionResult.APPLIED

    async with postgresql_session_factory() as requeue_session:
        result = await requeue_waiting_and_commit(
            requeue_session,
            command=RequeueWaitingAgentRunCommand(
                workspace_id=_WORKSPACE_ID,
                ticket_id=_TICKET_ID,
                agent_run_id=_RUN_ID,
                requeued_at=requeued_at,
            ),
        )

    assert result is AgentRunApprovalRequeueResult.APPLIED

    async with postgresql_session_factory() as verification_session:
        run = await load_run(verification_session)
        attempts = await load_attempts(verification_session)

    assert run.status == AgentRunStatus.QUEUED.value
    assert run.available_at == requeued_at
    assert run.updated_at == requeued_at
    assert run.completed_at is None
    assert run.last_error_code is None
    assert run.last_error_summary is None
    assert run.attempt_count == 1
    assert run.retryable_failure_count == 0
    assert len(attempts) == 1
    assert attempts[0].outcome == (AgentRunAttemptOutcome.AWAITING_APPROVAL.value)


async def test_requeue_waiting_for_approval_rejects_invalid_status(
    postgresql_session_factory: async_sessionmaker[AsyncSession],
    clean_business_tables: None,
) -> None:
    async with postgresql_session_factory() as setup_session:
        await persist_workspace_ticket_and_run(setup_session)

    async with postgresql_session_factory() as requeue_session:
        result = await requeue_waiting_and_commit(
            requeue_session,
            command=RequeueWaitingAgentRunCommand(
                workspace_id=_WORKSPACE_ID,
                ticket_id=_TICKET_ID,
                agent_run_id=_RUN_ID,
                requeued_at=_FIRST_CLAIMED_AT,
            ),
        )

    assert result is AgentRunApprovalRequeueResult.STATE_CONFLICT


async def test_requeue_waiting_for_approval_rejects_cross_scope(
    postgresql_session_factory: async_sessionmaker[AsyncSession],
    clean_business_tables: None,
) -> None:
    finished_at = _FIRST_CLAIMED_AT + timedelta(seconds=10)

    async with postgresql_session_factory() as setup_session:
        await persist_workspace_ticket_and_run(setup_session)

    async with postgresql_session_factory() as claim_session:
        await claim_and_commit(
            claim_session,
            command=create_claim_command(
                worker_id="worker-a",
                lease_token=_FIRST_LEASE_TOKEN,
                execution_request_id=_FIRST_EXECUTION_REQUEST_ID,
                claimed_at=_FIRST_CLAIMED_AT,
            ),
        )

    async with postgresql_session_factory() as waiting_session:
        await mark_waiting_and_commit(
            waiting_session,
            command=WaitForApprovalAgentRunCommand(
                agent_run_id=_RUN_ID,
                lease_token=_FIRST_LEASE_TOKEN,
                finished_at=finished_at,
            ),
        )

    async with postgresql_session_factory() as requeue_session:
        result = await requeue_waiting_and_commit(
            requeue_session,
            command=RequeueWaitingAgentRunCommand(
                workspace_id=UUID("aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"),
                ticket_id=_TICKET_ID,
                agent_run_id=_RUN_ID,
                requeued_at=finished_at + timedelta(minutes=1),
            ),
        )

    assert result is AgentRunApprovalRequeueResult.STATE_CONFLICT

    async with postgresql_session_factory() as requeue_session:
        wrong_ticket = await requeue_waiting_and_commit(
            requeue_session,
            command=RequeueWaitingAgentRunCommand(
                workspace_id=_WORKSPACE_ID,
                ticket_id=UUID("bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb"),
                agent_run_id=_RUN_ID,
                requeued_at=finished_at + timedelta(minutes=1),
            ),
        )

    assert wrong_ticket is AgentRunApprovalRequeueResult.STATE_CONFLICT


async def test_requeue_waiting_for_approval_preserves_lease_and_error_fields(
    postgresql_session_factory: async_sessionmaker[AsyncSession],
    clean_business_tables: None,
) -> None:
    finished_at = _FIRST_CLAIMED_AT + timedelta(seconds=10)
    requeued_at = finished_at + timedelta(minutes=1)

    async with postgresql_session_factory() as setup_session:
        await persist_workspace_ticket_and_run(setup_session)

    async with postgresql_session_factory() as claim_session:
        await claim_and_commit(
            claim_session,
            command=create_claim_command(
                worker_id="worker-a",
                lease_token=_FIRST_LEASE_TOKEN,
                execution_request_id=_FIRST_EXECUTION_REQUEST_ID,
                claimed_at=_FIRST_CLAIMED_AT,
            ),
        )

    async with postgresql_session_factory() as waiting_session:
        await mark_waiting_and_commit(
            waiting_session,
            command=WaitForApprovalAgentRunCommand(
                agent_run_id=_RUN_ID,
                lease_token=_FIRST_LEASE_TOKEN,
                finished_at=finished_at,
            ),
        )

    async with postgresql_session_factory() as requeue_session:
        result = await requeue_waiting_and_commit(
            requeue_session,
            command=RequeueWaitingAgentRunCommand(
                workspace_id=_WORKSPACE_ID,
                ticket_id=_TICKET_ID,
                agent_run_id=_RUN_ID,
                requeued_at=requeued_at,
            ),
        )

    assert result is AgentRunApprovalRequeueResult.APPLIED

    async with postgresql_session_factory() as verification_session:
        run = await load_run(verification_session)
        attempts = await load_attempts(verification_session)

    assert run.status == AgentRunStatus.QUEUED.value
    assert run.available_at == requeued_at
    assert run.lease_owner is None
    assert run.lease_token is None
    assert run.lease_expires_at is None
    assert run.completed_at is None
    assert run.last_error_code is None
    assert run.last_error_summary is None
    assert run.attempt_count == 1
    assert run.retryable_failure_count == 0
    assert len(attempts) == 1


async def test_repeated_requeue_waiting_for_approval_is_conflict(
    postgresql_session_factory: async_sessionmaker[AsyncSession],
    clean_business_tables: None,
) -> None:
    finished_at = _FIRST_CLAIMED_AT + timedelta(seconds=10)
    first_requeued_at = finished_at + timedelta(minutes=1)
    second_requeued_at = first_requeued_at + timedelta(minutes=1)

    async with postgresql_session_factory() as setup_session:
        await persist_workspace_ticket_and_run(setup_session)

    async with postgresql_session_factory() as claim_session:
        await claim_and_commit(
            claim_session,
            command=create_claim_command(
                worker_id="worker-a",
                lease_token=_FIRST_LEASE_TOKEN,
                execution_request_id=_FIRST_EXECUTION_REQUEST_ID,
                claimed_at=_FIRST_CLAIMED_AT,
            ),
        )

    async with postgresql_session_factory() as waiting_session:
        await mark_waiting_and_commit(
            waiting_session,
            command=WaitForApprovalAgentRunCommand(
                agent_run_id=_RUN_ID,
                lease_token=_FIRST_LEASE_TOKEN,
                finished_at=finished_at,
            ),
        )

    async with postgresql_session_factory() as requeue_session:
        first = await requeue_waiting_and_commit(
            requeue_session,
            command=RequeueWaitingAgentRunCommand(
                workspace_id=_WORKSPACE_ID,
                ticket_id=_TICKET_ID,
                agent_run_id=_RUN_ID,
                requeued_at=first_requeued_at,
            ),
        )

    async with postgresql_session_factory() as requeue_session:
        second = await requeue_waiting_and_commit(
            requeue_session,
            command=RequeueWaitingAgentRunCommand(
                workspace_id=_WORKSPACE_ID,
                ticket_id=_TICKET_ID,
                agent_run_id=_RUN_ID,
                requeued_at=second_requeued_at,
            ),
        )

    assert first is AgentRunApprovalRequeueResult.APPLIED
    assert second is AgentRunApprovalRequeueResult.STATE_CONFLICT

    async with postgresql_session_factory() as verification_session:
        run = await load_run(verification_session)

    assert run.available_at == first_requeued_at
    assert run.updated_at == first_requeued_at
    assert run.status == AgentRunStatus.QUEUED.value


async def test_requeue_waiting_for_approval_rejects_running_and_terminal(
    postgresql_session_factory: async_sessionmaker[AsyncSession],
    clean_business_tables: None,
) -> None:
    async with postgresql_session_factory() as setup_session:
        await persist_workspace_ticket_and_run(setup_session)

    async with postgresql_session_factory() as claim_session:
        await claim_and_commit(
            claim_session,
            command=create_claim_command(
                worker_id="worker-a",
                lease_token=_FIRST_LEASE_TOKEN,
                execution_request_id=_FIRST_EXECUTION_REQUEST_ID,
                claimed_at=_FIRST_CLAIMED_AT,
            ),
        )

    async with postgresql_session_factory() as requeue_session:
        running_conflict = await requeue_waiting_and_commit(
            requeue_session,
            command=RequeueWaitingAgentRunCommand(
                workspace_id=_WORKSPACE_ID,
                ticket_id=_TICKET_ID,
                agent_run_id=_RUN_ID,
                requeued_at=_FIRST_CLAIMED_AT + timedelta(minutes=1),
            ),
        )

    assert running_conflict is AgentRunApprovalRequeueResult.STATE_CONFLICT

    async with postgresql_session_factory() as complete_session:
        await mark_succeeded_and_commit(
            complete_session,
            command=CompleteAgentRunCommand(
                agent_run_id=_RUN_ID,
                lease_token=_FIRST_LEASE_TOKEN,
                finished_at=_FIRST_CLAIMED_AT + timedelta(seconds=10),
            ),
        )

    async with postgresql_session_factory() as requeue_session:
        succeeded_conflict = await requeue_waiting_and_commit(
            requeue_session,
            command=RequeueWaitingAgentRunCommand(
                workspace_id=_WORKSPACE_ID,
                ticket_id=_TICKET_ID,
                agent_run_id=_RUN_ID,
                requeued_at=_FIRST_CLAIMED_AT + timedelta(minutes=2),
            ),
        )

    assert succeeded_conflict is AgentRunApprovalRequeueResult.STATE_CONFLICT


async def test_concurrent_requeue_waiting_for_approval_has_one_winner(
    postgresql_session_factory: async_sessionmaker[AsyncSession],
    clean_business_tables: None,
) -> None:
    finished_at = _FIRST_CLAIMED_AT + timedelta(seconds=10)
    requeued_at = finished_at + timedelta(minutes=1)

    async with postgresql_session_factory() as setup_session:
        await persist_workspace_ticket_and_run(setup_session)

    async with postgresql_session_factory() as claim_session:
        await claim_and_commit(
            claim_session,
            command=create_claim_command(
                worker_id="worker-a",
                lease_token=_FIRST_LEASE_TOKEN,
                execution_request_id=_FIRST_EXECUTION_REQUEST_ID,
                claimed_at=_FIRST_CLAIMED_AT,
            ),
        )

    async with postgresql_session_factory() as waiting_session:
        await mark_waiting_and_commit(
            waiting_session,
            command=WaitForApprovalAgentRunCommand(
                agent_run_id=_RUN_ID,
                lease_token=_FIRST_LEASE_TOKEN,
                finished_at=finished_at,
            ),
        )

    ready = 0
    ready_lock = asyncio.Lock()
    release = asyncio.Event()

    async def attempt_requeue() -> AgentRunApprovalRequeueResult:
        nonlocal ready

        async with postgresql_session_factory() as session:
            async with ready_lock:
                ready += 1
                if ready == 2:
                    release.set()
            await release.wait()
            return await requeue_waiting_and_commit(
                session,
                command=RequeueWaitingAgentRunCommand(
                    workspace_id=_WORKSPACE_ID,
                    ticket_id=_TICKET_ID,
                    agent_run_id=_RUN_ID,
                    requeued_at=requeued_at,
                ),
            )

    results = await asyncio.gather(attempt_requeue(), attempt_requeue())

    assert set(results) == {
        AgentRunApprovalRequeueResult.APPLIED,
        AgentRunApprovalRequeueResult.STATE_CONFLICT,
    }

    async with postgresql_session_factory() as verification_session:
        run = await load_run(verification_session)
        attempts = await load_attempts(verification_session)

    assert run.status == AgentRunStatus.QUEUED.value
    assert run.available_at == requeued_at
    assert len(attempts) == 1
