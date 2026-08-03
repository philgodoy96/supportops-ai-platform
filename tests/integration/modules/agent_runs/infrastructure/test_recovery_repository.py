"""Integration tests for PostgreSQL AgentRun lease recovery."""

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
from supportops.modules.agent_runs.domain.recovery import (
    ExpiredAgentRunDisposition,
    RecoverExpiredAgentRunCommand,
    RecoverExpiredAgentRunResult,
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
_BASE_TIMESTAMP = datetime(
    2026,
    7,
    31,
    12,
    0,
    tzinfo=UTC,
)


async def persist_workspace(
    session: AsyncSession,
) -> None:
    """Persist the shared workspace."""

    workspace = Workspace(
        id=_WORKSPACE_ID,
        name="Platform Support",
        slug="platform-support",
        created_at=_BASE_TIMESTAMP,
        updated_at=_BASE_TIMESTAMP,
    )

    async with SqlAlchemyTransactionManager(session).transaction():
        await SqlAlchemyWorkspaceRepository(session).add(workspace)


async def persist_ticket_and_run(
    session: AsyncSession,
    *,
    ticket_id: UUID,
    run_id: UUID,
    max_retryable_failures: int = 3,
    created_at: datetime = _BASE_TIMESTAMP,
) -> AgentRun:
    """Persist one queued AgentRun and its ticket."""

    ticket = Ticket.create(
        ticket_id=ticket_id,
        workspace_id=_WORKSPACE_ID,
        subject=f"Ticket {ticket_id}",
        description="Lease recovery integration test.",
        external_reference=None,
        ingestion_request_id=UUID(
            "725eec8a-c504-4071-ac96-c78cc907f26c",
        ),
        correlation_id=UUID(
            "1038c98e-62fd-45df-9839-138f7105cb78",
        ),
        now=created_at,
    )
    run = AgentRun.create_initial(
        agent_run_id=run_id,
        workspace_id=_WORKSPACE_ID,
        ticket_id=ticket_id,
        ingestion_request_id=ticket.ingestion_request_id,
        correlation_id=ticket.correlation_id,
        workflow_version=DETERMINISTIC_BASELINE_WORKFLOW_VERSION,
        max_retryable_failures=max_retryable_failures,
        now=created_at,
    )

    async with SqlAlchemyTransactionManager(session).transaction():
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
    """Create deterministic claim ownership values."""

    return ClaimAgentRunCommand(
        worker_id=worker_id,
        lease_token=lease_token,
        execution_request_id=execution_request_id,
        claimed_at=claimed_at,
        lease_expires_at=claimed_at + timedelta(seconds=lease_seconds),
    )


def create_recovery_command(
    *,
    recovered_at: datetime,
    retry_base_delay_seconds: float = 2.0,
    retry_maximum_delay_seconds: float = 60.0,
) -> RecoverExpiredAgentRunCommand:
    """Create deterministic recovery values."""

    return RecoverExpiredAgentRunCommand(
        recovered_at=recovered_at,
        retry_base_delay_seconds=retry_base_delay_seconds,
        retry_maximum_delay_seconds=retry_maximum_delay_seconds,
        error_code="worker_lease_expired",
        error_summary=("The worker lease expired before execution completed."),
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


async def recover_and_commit(
    session: AsyncSession,
    *,
    command: RecoverExpiredAgentRunCommand,
) -> RecoverExpiredAgentRunResult | None:
    """Recover and commit one expired AgentRun."""

    repository = SqlAlchemyAgentRunRepository(session)

    async with SqlAlchemyTransactionManager(session).transaction():
        return await repository.recover_next_expired(command)


async def load_run(
    session: AsyncSession,
    *,
    run_id: UUID,
) -> AgentRunRecord:
    """Load one persisted AgentRun."""

    record = await session.get(
        AgentRunRecord,
        run_id,
    )

    assert record is not None
    return record


async def load_attempts(
    session: AsyncSession,
    *,
    run_id: UUID,
) -> list[AgentRunAttemptRecord]:
    """Load attempt history in ascending order."""

    result = await session.execute(
        select(AgentRunAttemptRecord)
        .where(
            AgentRunAttemptRecord.agent_run_id == run_id,
        )
        .order_by(
            AgentRunAttemptRecord.attempt_number.asc(),
        ),
    )
    return list(result.scalars())


async def test_recovery_closes_expired_attempt_and_schedules_retry(
    postgresql_session_factory: async_sessionmaker[AsyncSession],
    clean_business_tables: None,
) -> None:
    run_id = UUID(
        "10000000-0000-4000-8000-000000000001",
    )
    claimed_at = _BASE_TIMESTAMP + timedelta(minutes=5)
    recovered_at = claimed_at + timedelta(seconds=46)
    lease_token = UUID(
        "11000000-0000-4000-8000-000000000001",
    )

    async with postgresql_session_factory() as setup_session:
        await persist_workspace(setup_session)
        await persist_ticket_and_run(
            setup_session,
            ticket_id=UUID(
                "12000000-0000-4000-8000-000000000001",
            ),
            run_id=run_id,
        )

    async with postgresql_session_factory() as claim_session:
        await claim_and_commit(
            claim_session,
            command=create_claim_command(
                worker_id="worker-a",
                lease_token=lease_token,
                execution_request_id=UUID(
                    "13000000-0000-4000-8000-000000000001",
                ),
                claimed_at=claimed_at,
            ),
        )

    async with postgresql_session_factory() as recovery_session:
        result = await recover_and_commit(
            recovery_session,
            command=create_recovery_command(
                recovered_at=recovered_at,
            ),
        )

    assert result is not None
    assert result.disposition is ExpiredAgentRunDisposition.RETRY_SCHEDULED
    assert result.expired_lease_token == lease_token

    async with postgresql_session_factory() as verification_session:
        run = await load_run(
            verification_session,
            run_id=run_id,
        )
        attempts = await load_attempts(
            verification_session,
            run_id=run_id,
        )

    assert run.status == AgentRunStatus.RETRY_SCHEDULED.value
    assert run.available_at == recovered_at + timedelta(seconds=2)
    assert run.completed_at is None
    assert run.attempt_count == 1
    assert run.lease_owner is None
    assert run.lease_token is None
    assert run.lease_expires_at is None
    assert run.last_error_code == "worker_lease_expired"
    assert run.last_error_summary == ("The worker lease expired before execution completed.")

    assert len(attempts) == 1
    assert attempts[0].attempt_number == 1
    assert attempts[0].finished_at == recovered_at
    assert attempts[0].outcome == AgentRunAttemptOutcome.LEASE_EXPIRED.value
    assert attempts[0].error_code == "worker_lease_expired"


async def test_valid_lease_is_not_recovered(
    postgresql_session_factory: async_sessionmaker[AsyncSession],
    clean_business_tables: None,
) -> None:
    run_id = UUID(
        "20000000-0000-4000-8000-000000000002",
    )
    claimed_at = _BASE_TIMESTAMP + timedelta(minutes=5)

    async with postgresql_session_factory() as setup_session:
        await persist_workspace(setup_session)
        await persist_ticket_and_run(
            setup_session,
            ticket_id=UUID(
                "21000000-0000-4000-8000-000000000002",
            ),
            run_id=run_id,
        )

    async with postgresql_session_factory() as claim_session:
        await claim_and_commit(
            claim_session,
            command=create_claim_command(
                worker_id="worker-a",
                lease_token=UUID(
                    "22000000-0000-4000-8000-000000000002",
                ),
                execution_request_id=UUID(
                    "23000000-0000-4000-8000-000000000002",
                ),
                claimed_at=claimed_at,
                lease_seconds=45,
            ),
        )

    async with postgresql_session_factory() as recovery_session:
        result = await recover_and_commit(
            recovery_session,
            command=create_recovery_command(
                recovered_at=claimed_at + timedelta(seconds=44),
            ),
        )

    assert result is None

    async with postgresql_session_factory() as verification_session:
        run = await load_run(
            verification_session,
            run_id=run_id,
        )
        attempts = await load_attempts(
            verification_session,
            run_id=run_id,
        )

    assert run.status == AgentRunStatus.RUNNING.value
    assert run.lease_owner == "worker-a"
    assert run.lease_token is not None
    assert run.attempt_count == 1
    assert len(attempts) == 1
    assert attempts[0].finished_at is None
    assert attempts[0].outcome is None


async def test_recovered_run_can_be_claimed_again(
    postgresql_session_factory: async_sessionmaker[AsyncSession],
    clean_business_tables: None,
) -> None:
    run_id = UUID(
        "30000000-0000-4000-8000-000000000003",
    )
    first_claimed_at = _BASE_TIMESTAMP + timedelta(minutes=5)
    recovered_at = first_claimed_at + timedelta(seconds=46)
    retry_available_at = recovered_at + timedelta(seconds=2)
    first_lease_token = UUID(
        "31000000-0000-4000-8000-000000000003",
    )
    second_lease_token = UUID(
        "32000000-0000-4000-8000-000000000003",
    )

    async with postgresql_session_factory() as setup_session:
        await persist_workspace(setup_session)
        await persist_ticket_and_run(
            setup_session,
            ticket_id=UUID(
                "33000000-0000-4000-8000-000000000003",
            ),
            run_id=run_id,
        )

    async with postgresql_session_factory() as first_claim_session:
        await claim_and_commit(
            first_claim_session,
            command=create_claim_command(
                worker_id="worker-a",
                lease_token=first_lease_token,
                execution_request_id=UUID(
                    "34000000-0000-4000-8000-000000000003",
                ),
                claimed_at=first_claimed_at,
            ),
        )

    async with postgresql_session_factory() as recovery_session:
        recovery_result = await recover_and_commit(
            recovery_session,
            command=create_recovery_command(
                recovered_at=recovered_at,
            ),
        )

    assert recovery_result is not None

    async with postgresql_session_factory() as second_claim_session:
        second_claim = await claim_and_commit(
            second_claim_session,
            command=create_claim_command(
                worker_id="worker-b",
                lease_token=second_lease_token,
                execution_request_id=UUID(
                    "35000000-0000-4000-8000-000000000003",
                ),
                claimed_at=retry_available_at,
            ),
        )

    assert second_claim.agent_run.id == run_id
    assert second_claim.agent_run.status is AgentRunStatus.RUNNING
    assert second_claim.agent_run.attempt_count == 2
    assert second_claim.agent_run.lease_owner == "worker-b"
    assert second_claim.agent_run.lease_token == second_lease_token
    assert second_claim.attempt.attempt_number == 2
    assert second_claim.attempt.lease_token == second_lease_token

    async with postgresql_session_factory() as verification_session:
        attempts = await load_attempts(
            verification_session,
            run_id=run_id,
        )

    assert len(attempts) == 2
    assert attempts[0].outcome == AgentRunAttemptOutcome.LEASE_EXPIRED.value
    assert attempts[0].lease_token == first_lease_token
    assert attempts[1].outcome is None
    assert attempts[1].finished_at is None
    assert attempts[1].lease_token == second_lease_token


async def test_exhausted_expired_run_is_marked_failed(
    postgresql_session_factory: async_sessionmaker[AsyncSession],
    clean_business_tables: None,
) -> None:
    run_id = UUID(
        "40000000-0000-4000-8000-000000000004",
    )
    claimed_at = _BASE_TIMESTAMP + timedelta(minutes=5)
    recovered_at = claimed_at + timedelta(seconds=46)

    async with postgresql_session_factory() as setup_session:
        await persist_workspace(setup_session)
        await persist_ticket_and_run(
            setup_session,
            ticket_id=UUID(
                "41000000-0000-4000-8000-000000000004",
            ),
            run_id=run_id,
            max_retryable_failures=1,
        )

    async with postgresql_session_factory() as claim_session:
        claim = await claim_and_commit(
            claim_session,
            command=create_claim_command(
                worker_id="worker-a",
                lease_token=UUID(
                    "42000000-0000-4000-8000-000000000004",
                ),
                execution_request_id=UUID(
                    "43000000-0000-4000-8000-000000000004",
                ),
                claimed_at=claimed_at,
            ),
        )

    assert claim.agent_run.attempt_count == 1
    assert claim.agent_run.max_retryable_failures == 1

    async with postgresql_session_factory() as recovery_session:
        result = await recover_and_commit(
            recovery_session,
            command=create_recovery_command(
                recovered_at=recovered_at,
            ),
        )

    assert result is not None
    assert result.disposition is ExpiredAgentRunDisposition.FAILED

    async with postgresql_session_factory() as verification_session:
        run = await load_run(
            verification_session,
            run_id=run_id,
        )
        attempts = await load_attempts(
            verification_session,
            run_id=run_id,
        )

    assert run.status == AgentRunStatus.FAILED.value
    assert run.completed_at == recovered_at
    assert run.attempt_count == 1
    assert run.lease_owner is None
    assert run.lease_token is None
    assert run.lease_expires_at is None

    assert len(attempts) == 1
    assert attempts[0].outcome == AgentRunAttemptOutcome.LEASE_EXPIRED.value


async def test_same_expired_lease_is_not_recovered_twice(
    postgresql_session_factory: async_sessionmaker[AsyncSession],
    clean_business_tables: None,
) -> None:
    run_id = UUID(
        "50000000-0000-4000-8000-000000000005",
    )
    claimed_at = _BASE_TIMESTAMP + timedelta(minutes=5)
    recovered_at = claimed_at + timedelta(seconds=46)

    async with postgresql_session_factory() as setup_session:
        await persist_workspace(setup_session)
        await persist_ticket_and_run(
            setup_session,
            ticket_id=UUID(
                "51000000-0000-4000-8000-000000000005",
            ),
            run_id=run_id,
        )

    async with postgresql_session_factory() as claim_session:
        await claim_and_commit(
            claim_session,
            command=create_claim_command(
                worker_id="worker-a",
                lease_token=UUID(
                    "52000000-0000-4000-8000-000000000005",
                ),
                execution_request_id=UUID(
                    "53000000-0000-4000-8000-000000000005",
                ),
                claimed_at=claimed_at,
            ),
        )

    command = create_recovery_command(
        recovered_at=recovered_at,
    )

    async with postgresql_session_factory() as first_session:
        first_result = await recover_and_commit(
            first_session,
            command=command,
        )

    async with postgresql_session_factory() as second_session:
        second_result = await recover_and_commit(
            second_session,
            command=command,
        )

    assert first_result is not None
    assert second_result is None

    async with postgresql_session_factory() as verification_session:
        attempts = await load_attempts(
            verification_session,
            run_id=run_id,
        )

    assert len(attempts) == 1
    assert attempts[0].outcome == AgentRunAttemptOutcome.LEASE_EXPIRED.value


async def test_skip_locked_allows_recovery_workers_to_recover_distinct_runs(
    postgresql_session_factory: async_sessionmaker[AsyncSession],
    clean_business_tables: None,
) -> None:
    first_run_id = UUID(
        "60000000-0000-4000-8000-000000000006",
    )
    second_run_id = UUID(
        "61000000-0000-4000-8000-000000000006",
    )
    claimed_at = _BASE_TIMESTAMP + timedelta(minutes=5)
    recovered_at = claimed_at + timedelta(seconds=46)

    async with postgresql_session_factory() as setup_session:
        await persist_workspace(setup_session)
        await persist_ticket_and_run(
            setup_session,
            ticket_id=UUID(
                "62000000-0000-4000-8000-000000000006",
            ),
            run_id=first_run_id,
        )
        await persist_ticket_and_run(
            setup_session,
            ticket_id=UUID(
                "63000000-0000-4000-8000-000000000006",
            ),
            run_id=second_run_id,
        )

    async with postgresql_session_factory() as first_claim_session:
        await claim_and_commit(
            first_claim_session,
            command=create_claim_command(
                worker_id="worker-a",
                lease_token=UUID(
                    "64000000-0000-4000-8000-000000000006",
                ),
                execution_request_id=UUID(
                    "65000000-0000-4000-8000-000000000006",
                ),
                claimed_at=claimed_at,
            ),
        )

    async with postgresql_session_factory() as second_claim_session:
        await claim_and_commit(
            second_claim_session,
            command=create_claim_command(
                worker_id="worker-b",
                lease_token=UUID(
                    "66000000-0000-4000-8000-000000000006",
                ),
                execution_request_id=UUID(
                    "67000000-0000-4000-8000-000000000006",
                ),
                claimed_at=claimed_at,
            ),
        )

    command = create_recovery_command(
        recovered_at=recovered_at,
    )

    async with (
        postgresql_session_factory() as first_recovery_session,
        postgresql_session_factory() as second_recovery_session,
    ):
        first_repository = SqlAlchemyAgentRunRepository(
            first_recovery_session,
        )
        second_repository = SqlAlchemyAgentRunRepository(
            second_recovery_session,
        )

        async with first_recovery_session.begin():
            first_result = await first_repository.recover_next_expired(
                command,
            )

            async with second_recovery_session.begin():
                second_result = await second_repository.recover_next_expired(
                    command,
                )

    assert first_result is not None
    assert second_result is not None
    assert first_result.agent_run.id != second_result.agent_run.id
    assert {
        first_result.agent_run.id,
        second_result.agent_run.id,
    } == {
        first_run_id,
        second_run_id,
    }

    async with postgresql_session_factory() as verification_session:
        first_run = await load_run(
            verification_session,
            run_id=first_run_id,
        )
        second_run = await load_run(
            verification_session,
            run_id=second_run_id,
        )

    assert first_run.status == AgentRunStatus.RETRY_SCHEDULED.value
    assert second_run.status == AgentRunStatus.RETRY_SCHEDULED.value
    assert first_run.attempt_count == 1
    assert second_run.attempt_count == 1


async def test_recovery_excludes_waiting_for_approval_runs(
    postgresql_session_factory: async_sessionmaker[AsyncSession],
    clean_business_tables: None,
) -> None:
    recovered_at = _BASE_TIMESTAMP + timedelta(minutes=10)
    run_id = UUID("b2000000-0000-4000-8000-000000000002")
    ticket_id = UUID("b1000000-0000-4000-8000-000000000001")

    async with postgresql_session_factory() as setup_session:
        await persist_workspace(setup_session)
        await persist_ticket_and_run(
            setup_session,
            ticket_id=ticket_id,
            run_id=run_id,
        )
        async with SqlAlchemyTransactionManager(setup_session).transaction():
            record = await setup_session.get(AgentRunRecord, run_id)
            assert record is not None
            record.status = AgentRunStatus.WAITING_FOR_APPROVAL.value
            record.available_at = None
            record.attempt_count = 1
            record.first_started_at = _BASE_TIMESTAMP + timedelta(minutes=1)
            record.lease_owner = None
            record.lease_token = None
            record.lease_expires_at = None
            record.updated_at = _BASE_TIMESTAMP + timedelta(minutes=2)
            await setup_session.flush()

    async with postgresql_session_factory() as recovery_session:
        repository = SqlAlchemyAgentRunRepository(recovery_session)
        async with SqlAlchemyTransactionManager(recovery_session).transaction():
            result = await repository.recover_next_expired(
                create_recovery_command(recovered_at=recovered_at),
            )

    assert result is None
