"""Integration tests for PostgreSQL AgentRun claiming."""

from collections.abc import Callable
from dataclasses import replace
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


def create_claim_command(
    *,
    worker_id: str,
    lease_token: UUID,
    execution_request_id: UUID,
    claimed_at: datetime,
) -> ClaimAgentRunCommand:
    """Create deterministic ownership values for one claim."""

    return ClaimAgentRunCommand(
        worker_id=worker_id,
        lease_token=lease_token,
        execution_request_id=execution_request_id,
        claimed_at=claimed_at,
        lease_expires_at=claimed_at + timedelta(seconds=45),
    )


async def persist_workspace(
    session: AsyncSession,
) -> Workspace:
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

    return workspace


async def persist_ticket_and_run(
    session: AsyncSession,
    *,
    ticket_id: UUID,
    run_id: UUID,
    created_at: datetime,
    available_at: datetime | None = None,
    run_transform: Callable[[AgentRun], AgentRun] | None = None,
) -> tuple[Ticket, AgentRun]:
    """Persist one ticket and its associated AgentRun."""

    ticket = Ticket.create(
        ticket_id=ticket_id,
        workspace_id=_WORKSPACE_ID,
        subject=f"Ticket {ticket_id}",
        description="Deterministic claim integration test.",
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
        ticket_id=ticket.id,
        ingestion_request_id=ticket.ingestion_request_id,
        correlation_id=ticket.correlation_id,
        workflow_version=DETERMINISTIC_BASELINE_WORKFLOW_VERSION,
        max_retryable_failures=3,
        now=created_at,
    )

    if available_at is not None:
        run = replace(
            run,
            available_at=available_at,
        )

    if run_transform is not None:
        run = run_transform(run)

    async with SqlAlchemyTransactionManager(session).transaction():
        await SqlAlchemyTicketRepository(session).add(ticket)
        await SqlAlchemyAgentRunRepository(session).add(run)

    return ticket, run


async def load_attempts(
    session: AsyncSession,
    *,
    agent_run_id: UUID,
) -> list[AgentRunAttemptRecord]:
    """Load persisted attempts in attempt-number order."""

    result = await session.execute(
        select(AgentRunAttemptRecord)
        .where(
            AgentRunAttemptRecord.agent_run_id == agent_run_id,
        )
        .order_by(
            AgentRunAttemptRecord.attempt_number.asc(),
        ),
    )
    return list(result.scalars())


async def claim_and_commit(
    session: AsyncSession,
    *,
    command: ClaimAgentRunCommand,
) -> AgentRunClaim | None:
    """Claim one run and commit the ownership transaction."""

    repository = SqlAlchemyAgentRunRepository(session)

    async with SqlAlchemyTransactionManager(session).transaction():
        return await repository.claim_next_available(command)


async def test_claim_persists_running_run_and_active_attempt(
    postgresql_session_factory: async_sessionmaker[AsyncSession],
    clean_business_tables: None,
) -> None:
    claimed_at = _BASE_TIMESTAMP + timedelta(minutes=5)
    run_id = UUID(
        "69184ef1-4d71-452e-8070-0b784c29368e",
    )
    lease_token = UUID(
        "dd0ae456-3467-41db-93d1-a908f40e8365",
    )
    execution_request_id = UUID(
        "d1fa068f-2278-47a8-b3c9-39ccf91f0a5e",
    )

    async with postgresql_session_factory() as setup_session:
        await persist_workspace(setup_session)
        _, run = await persist_ticket_and_run(
            setup_session,
            ticket_id=UUID(
                "38bb60fe-d2ea-4615-b499-91aa45069019",
            ),
            run_id=run_id,
            created_at=_BASE_TIMESTAMP,
        )

    async with postgresql_session_factory() as claim_session:
        claim = await claim_and_commit(
            claim_session,
            command=create_claim_command(
                worker_id="worker-a",
                lease_token=lease_token,
                execution_request_id=execution_request_id,
                claimed_at=claimed_at,
            ),
        )

    assert claim is not None
    assert claim.agent_run.id == run.id
    assert claim.agent_run.status is AgentRunStatus.RUNNING
    assert claim.agent_run.attempt_count == 1
    assert claim.agent_run.lease_owner == "worker-a"
    assert claim.agent_run.lease_token == lease_token
    assert claim.agent_run.first_started_at == claimed_at
    assert claim.attempt.attempt_number == 1
    assert claim.attempt.execution_request_id == execution_request_id

    async with postgresql_session_factory() as verification_session:
        persisted_run = await verification_session.get(
            AgentRunRecord,
            run_id,
        )
        attempts = await load_attempts(
            verification_session,
            agent_run_id=run_id,
        )

    assert persisted_run is not None
    assert persisted_run.status == AgentRunStatus.RUNNING.value
    assert persisted_run.attempt_count == 1
    assert persisted_run.lease_owner == "worker-a"
    assert persisted_run.lease_token == lease_token
    assert len(attempts) == 1
    assert attempts[0].attempt_number == 1
    assert attempts[0].worker_id == "worker-a"
    assert attempts[0].lease_token == lease_token
    assert attempts[0].execution_request_id == execution_request_id
    assert attempts[0].finished_at is None
    assert attempts[0].outcome is None


async def test_claim_uses_deterministic_available_order(
    postgresql_session_factory: async_sessionmaker[AsyncSession],
    clean_business_tables: None,
) -> None:
    claimed_at = _BASE_TIMESTAMP + timedelta(minutes=10)

    earliest_run_id = UUID(
        "10000000-0000-4000-8000-000000000001",
    )
    later_run_id = UUID(
        "20000000-0000-4000-8000-000000000002",
    )

    async with postgresql_session_factory() as setup_session:
        await persist_workspace(setup_session)

        await persist_ticket_and_run(
            setup_session,
            ticket_id=UUID(
                "11000000-0000-4000-8000-000000000001",
            ),
            run_id=later_run_id,
            created_at=_BASE_TIMESTAMP,
            available_at=_BASE_TIMESTAMP + timedelta(minutes=2),
        )
        await persist_ticket_and_run(
            setup_session,
            ticket_id=UUID(
                "12000000-0000-4000-8000-000000000002",
            ),
            run_id=earliest_run_id,
            created_at=_BASE_TIMESTAMP,
            available_at=_BASE_TIMESTAMP + timedelta(minutes=1),
        )

    async with postgresql_session_factory() as claim_session:
        claim = await claim_and_commit(
            claim_session,
            command=create_claim_command(
                worker_id="worker-a",
                lease_token=UUID(
                    "30000000-0000-4000-8000-000000000003",
                ),
                execution_request_id=UUID(
                    "40000000-0000-4000-8000-000000000004",
                ),
                claimed_at=claimed_at,
            ),
        )

    assert claim is not None
    assert claim.agent_run.id == earliest_run_id


async def test_claim_excludes_future_available_run(
    postgresql_session_factory: async_sessionmaker[AsyncSession],
    clean_business_tables: None,
) -> None:
    claimed_at = _BASE_TIMESTAMP + timedelta(minutes=1)

    async with postgresql_session_factory() as setup_session:
        await persist_workspace(setup_session)
        await persist_ticket_and_run(
            setup_session,
            ticket_id=UUID(
                "51000000-0000-4000-8000-000000000001",
            ),
            run_id=UUID(
                "52000000-0000-4000-8000-000000000002",
            ),
            created_at=_BASE_TIMESTAMP,
            available_at=_BASE_TIMESTAMP + timedelta(minutes=2),
        )

    async with postgresql_session_factory() as claim_session:
        claim = await claim_and_commit(
            claim_session,
            command=create_claim_command(
                worker_id="worker-a",
                lease_token=UUID(
                    "53000000-0000-4000-8000-000000000003",
                ),
                execution_request_id=UUID(
                    "54000000-0000-4000-8000-000000000004",
                ),
                claimed_at=claimed_at,
            ),
        )

    assert claim is None


async def test_claim_excludes_terminal_and_exhausted_runs(
    postgresql_session_factory: async_sessionmaker[AsyncSession],
    clean_business_tables: None,
) -> None:
    claimed_at = _BASE_TIMESTAMP + timedelta(minutes=10)

    def make_succeeded(run: AgentRun) -> AgentRun:
        return replace(
            run,
            status=AgentRunStatus.SUCCEEDED,
            attempt_count=1,
            first_started_at=_BASE_TIMESTAMP + timedelta(minutes=1),
            completed_at=_BASE_TIMESTAMP + timedelta(minutes=2),
            updated_at=_BASE_TIMESTAMP + timedelta(minutes=2),
        )

    def make_exhausted(run: AgentRun) -> AgentRun:
        return replace(
            run,
            status=AgentRunStatus.RETRY_SCHEDULED,
            attempt_count=3,
            retryable_failure_count=3,
            first_started_at=_BASE_TIMESTAMP + timedelta(minutes=1),
            last_error_code="retryable_executor_failure",
            last_error_summary=("The configured executor reported a retryable failure."),
            updated_at=_BASE_TIMESTAMP + timedelta(minutes=3),
        )

    async with postgresql_session_factory() as setup_session:
        await persist_workspace(setup_session)
        await persist_ticket_and_run(
            setup_session,
            ticket_id=UUID(
                "61000000-0000-4000-8000-000000000001",
            ),
            run_id=UUID(
                "62000000-0000-4000-8000-000000000002",
            ),
            created_at=_BASE_TIMESTAMP,
            run_transform=make_succeeded,
        )
        await persist_ticket_and_run(
            setup_session,
            ticket_id=UUID(
                "63000000-0000-4000-8000-000000000003",
            ),
            run_id=UUID(
                "64000000-0000-4000-8000-000000000004",
            ),
            created_at=_BASE_TIMESTAMP,
            run_transform=make_exhausted,
        )

    async with postgresql_session_factory() as claim_session:
        claim = await claim_and_commit(
            claim_session,
            command=create_claim_command(
                worker_id="worker-a",
                lease_token=UUID(
                    "65000000-0000-4000-8000-000000000005",
                ),
                execution_request_id=UUID(
                    "66000000-0000-4000-8000-000000000006",
                ),
                claimed_at=claimed_at,
            ),
        )

    assert claim is None


async def test_committed_run_cannot_be_claimed_twice(
    postgresql_session_factory: async_sessionmaker[AsyncSession],
    clean_business_tables: None,
) -> None:
    claimed_at = _BASE_TIMESTAMP + timedelta(minutes=5)

    async with postgresql_session_factory() as setup_session:
        await persist_workspace(setup_session)
        await persist_ticket_and_run(
            setup_session,
            ticket_id=UUID(
                "71000000-0000-4000-8000-000000000001",
            ),
            run_id=UUID(
                "72000000-0000-4000-8000-000000000002",
            ),
            created_at=_BASE_TIMESTAMP,
        )

    async with postgresql_session_factory() as first_session:
        first_claim = await claim_and_commit(
            first_session,
            command=create_claim_command(
                worker_id="worker-a",
                lease_token=UUID(
                    "73000000-0000-4000-8000-000000000003",
                ),
                execution_request_id=UUID(
                    "74000000-0000-4000-8000-000000000004",
                ),
                claimed_at=claimed_at,
            ),
        )

    async with postgresql_session_factory() as second_session:
        second_claim = await claim_and_commit(
            second_session,
            command=create_claim_command(
                worker_id="worker-b",
                lease_token=UUID(
                    "75000000-0000-4000-8000-000000000005",
                ),
                execution_request_id=UUID(
                    "76000000-0000-4000-8000-000000000006",
                ),
                claimed_at=claimed_at,
            ),
        )

    assert first_claim is not None
    assert second_claim is None


async def test_skip_locked_allows_workers_to_claim_different_runs(
    postgresql_session_factory: async_sessionmaker[AsyncSession],
    clean_business_tables: None,
) -> None:
    claimed_at = _BASE_TIMESTAMP + timedelta(minutes=5)

    first_run_id = UUID(
        "81000000-0000-4000-8000-000000000001",
    )
    second_run_id = UUID(
        "82000000-0000-4000-8000-000000000002",
    )

    async with postgresql_session_factory() as setup_session:
        await persist_workspace(setup_session)
        await persist_ticket_and_run(
            setup_session,
            ticket_id=UUID(
                "83000000-0000-4000-8000-000000000003",
            ),
            run_id=first_run_id,
            created_at=_BASE_TIMESTAMP,
        )
        await persist_ticket_and_run(
            setup_session,
            ticket_id=UUID(
                "84000000-0000-4000-8000-000000000004",
            ),
            run_id=second_run_id,
            created_at=_BASE_TIMESTAMP,
        )

    async with (
        postgresql_session_factory() as first_session,
        postgresql_session_factory() as second_session,
    ):
        first_repository = SqlAlchemyAgentRunRepository(first_session)
        second_repository = SqlAlchemyAgentRunRepository(second_session)

        async with first_session.begin():
            first_claim = await first_repository.claim_next_available(
                create_claim_command(
                    worker_id="worker-a",
                    lease_token=UUID(
                        "85000000-0000-4000-8000-000000000005",
                    ),
                    execution_request_id=UUID(
                        "86000000-0000-4000-8000-000000000006",
                    ),
                    claimed_at=claimed_at,
                ),
            )

            async with second_session.begin():
                second_claim = await second_repository.claim_next_available(
                    create_claim_command(
                        worker_id="worker-b",
                        lease_token=UUID(
                            "87000000-0000-4000-8000-000000000007",
                        ),
                        execution_request_id=UUID(
                            "88000000-0000-4000-8000-000000000008",
                        ),
                        claimed_at=claimed_at,
                    ),
                )

    assert first_claim is not None
    assert second_claim is not None
    assert first_claim.agent_run.id != second_claim.agent_run.id
    assert {
        first_claim.agent_run.id,
        second_claim.agent_run.id,
    } == {
        first_run_id,
        second_run_id,
    }


async def test_retry_claim_increments_attempt_and_changes_lease_token(
    postgresql_session_factory: async_sessionmaker[AsyncSession],
    clean_business_tables: None,
) -> None:
    previous_lease_token = UUID(
        "91000000-0000-4000-8000-000000000001",
    )
    new_lease_token = UUID(
        "92000000-0000-4000-8000-000000000002",
    )
    first_started_at = _BASE_TIMESTAMP + timedelta(minutes=1)
    claimed_at = _BASE_TIMESTAMP + timedelta(minutes=5)

    def make_retry_scheduled(run: AgentRun) -> AgentRun:
        return replace(
            run,
            status=AgentRunStatus.RETRY_SCHEDULED,
            attempt_count=1,
            first_started_at=first_started_at,
            last_error_code="retryable_executor_failure",
            last_error_summary=("The configured executor reported a retryable failure."),
            updated_at=_BASE_TIMESTAMP + timedelta(minutes=2),
        )

    async with postgresql_session_factory() as setup_session:
        await persist_workspace(setup_session)
        _, run = await persist_ticket_and_run(
            setup_session,
            ticket_id=UUID(
                "93000000-0000-4000-8000-000000000003",
            ),
            run_id=UUID(
                "94000000-0000-4000-8000-000000000004",
            ),
            created_at=_BASE_TIMESTAMP,
            run_transform=make_retry_scheduled,
        )

        previous_attempt = AgentRunAttemptRecord(
            id=UUID(
                "95000000-0000-4000-8000-000000000005",
            ),
            agent_run_id=run.id,
            attempt_number=1,
            worker_id="worker-previous",
            lease_token=previous_lease_token,
            execution_request_id=UUID(
                "96000000-0000-4000-8000-000000000006",
            ),
            started_at=first_started_at,
            finished_at=_BASE_TIMESTAMP + timedelta(minutes=2),
            outcome=AgentRunAttemptOutcome.RETRYABLE_FAILURE.value,
            error_code="retryable_executor_failure",
            error_summary=("The configured executor reported a retryable failure."),
        )
        setup_session.add(previous_attempt)
        await setup_session.commit()

    async with postgresql_session_factory() as claim_session:
        claim = await claim_and_commit(
            claim_session,
            command=create_claim_command(
                worker_id="worker-new",
                lease_token=new_lease_token,
                execution_request_id=UUID(
                    "97000000-0000-4000-8000-000000000007",
                ),
                claimed_at=claimed_at,
            ),
        )

    assert claim is not None
    assert claim.agent_run.attempt_count == 2
    assert claim.agent_run.first_started_at == first_started_at
    assert claim.agent_run.lease_token == new_lease_token
    assert claim.agent_run.lease_token != previous_lease_token
    assert claim.attempt.attempt_number == 2
    assert claim.attempt.lease_token == new_lease_token

    async with postgresql_session_factory() as verification_session:
        attempts = await load_attempts(
            verification_session,
            agent_run_id=run.id,
        )

    assert len(attempts) == 2
    assert attempts[0].attempt_number == 1
    assert attempts[1].attempt_number == 2
    assert attempts[0].lease_token == previous_lease_token
    assert attempts[1].lease_token == new_lease_token


async def test_claim_excludes_waiting_for_approval_runs(
    postgresql_session_factory: async_sessionmaker[AsyncSession],
    clean_business_tables: None,
) -> None:
    claimed_at = _BASE_TIMESTAMP + timedelta(minutes=10)

    def make_waiting(run: AgentRun) -> AgentRun:
        return replace(
            run,
            status=AgentRunStatus.WAITING_FOR_APPROVAL,
            available_at=None,
            attempt_count=1,
            first_started_at=_BASE_TIMESTAMP + timedelta(minutes=1),
            updated_at=_BASE_TIMESTAMP + timedelta(minutes=2),
        )

    async with postgresql_session_factory() as setup_session:
        await persist_workspace(setup_session)
        await persist_ticket_and_run(
            setup_session,
            ticket_id=UUID(
                "a1000000-0000-4000-8000-000000000001",
            ),
            run_id=UUID(
                "a2000000-0000-4000-8000-000000000002",
            ),
            created_at=_BASE_TIMESTAMP,
            run_transform=make_waiting,
        )

    async with postgresql_session_factory() as claim_session:
        claim = await claim_and_commit(
            claim_session,
            command=create_claim_command(
                worker_id="worker-a",
                lease_token=UUID(
                    "a3000000-0000-4000-8000-000000000003",
                ),
                execution_request_id=UUID(
                    "a4000000-0000-4000-8000-000000000004",
                ),
                claimed_at=claimed_at,
            ),
        )

    assert claim is None
