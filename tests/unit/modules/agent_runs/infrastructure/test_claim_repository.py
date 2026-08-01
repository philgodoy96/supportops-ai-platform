"""Unit tests for atomic AgentRun repository claiming."""

from dataclasses import replace
from datetime import UTC, datetime, timedelta
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock, create_autospec
from uuid import UUID

from sqlalchemy.dialects import postgresql
from sqlalchemy.ext.asyncio import AsyncSession

from supportops.modules.agent_runs.domain.claiming import (
    ClaimAgentRunCommand,
)
from supportops.modules.agent_runs.domain.models import (
    DETERMINISTIC_BASELINE_WORKFLOW_VERSION,
    AgentRun,
    AgentRunStatus,
)
from supportops.modules.agent_runs.infrastructure.models import (
    AgentRunAttemptRecord,
    AgentRunRecord,
)
from supportops.modules.agent_runs.infrastructure.repository import (
    SqlAlchemyAgentRunRepository,
)

_RUN_ID = UUID(
    "69184ef1-4d71-452e-8070-0b784c29368e",
)
_WORKSPACE_ID = UUID(
    "032c8c87-57cc-4d14-bfbd-04968b4e8cd4",
)
_TICKET_ID = UUID(
    "38bb60fe-d2ea-4615-b499-91aa45069019",
)
_LEASE_TOKEN = UUID(
    "dd0ae456-3467-41db-93d1-a908f40e8365",
)
_EXECUTION_REQUEST_ID = UUID(
    "d1fa068f-2278-47a8-b3c9-39ccf91f0a5e",
)
_CREATED_AT = datetime(
    2026,
    7,
    31,
    16,
    0,
    tzinfo=UTC,
)
_CLAIMED_AT = datetime(
    2026,
    7,
    31,
    17,
    0,
    tzinfo=UTC,
)


def create_initial_run() -> AgentRun:
    """Create a deterministic queued AgentRun."""

    return AgentRun.create_initial(
        agent_run_id=_RUN_ID,
        workspace_id=_WORKSPACE_ID,
        ticket_id=_TICKET_ID,
        ingestion_request_id=UUID(
            "725eec8a-c504-4071-ac96-c78cc907f26c",
        ),
        correlation_id=UUID(
            "1038c98e-62fd-45df-9839-138f7105cb78",
        ),
        workflow_version=DETERMINISTIC_BASELINE_WORKFLOW_VERSION,
        max_attempts=3,
        now=_CREATED_AT,
    )


def create_retry_scheduled_run() -> AgentRun:
    """Create a deterministic run waiting for its second attempt."""

    return replace(
        create_initial_run(),
        status=AgentRunStatus.RETRY_SCHEDULED,
        available_at=_CLAIMED_AT,
        attempt_count=1,
        first_started_at=_CREATED_AT + timedelta(minutes=10),
        last_error_code="retryable_executor_failure",
        last_error_summary=("The configured executor reported a retryable failure."),
        updated_at=_CREATED_AT + timedelta(minutes=10),
    )


def create_claim_command() -> ClaimAgentRunCommand:
    """Create deterministic ownership values for one claim."""

    return ClaimAgentRunCommand(
        worker_id="worker-a",
        lease_token=_LEASE_TOKEN,
        execution_request_id=_EXECUTION_REQUEST_ID,
        claimed_at=_CLAIMED_AT,
        lease_expires_at=_CLAIMED_AT + timedelta(seconds=45),
    )


def create_session() -> MagicMock:
    """Create an autospecced SQLAlchemy session."""

    session = create_autospec(
        AsyncSession,
        instance=True,
    )
    session.execute = AsyncMock()
    session.add = MagicMock()
    session.flush = AsyncMock()
    session.commit = AsyncMock()
    session.rollback = AsyncMock()
    return cast(MagicMock, session)


async def test_claim_next_available_returns_none_when_no_run_exists() -> None:
    session = create_session()
    result = MagicMock()
    result.scalar_one_or_none.return_value = None
    session.execute.return_value = result

    repository = SqlAlchemyAgentRunRepository(session)

    claim = await repository.claim_next_available(
        create_claim_command(),
    )

    assert claim is None
    session.add.assert_not_called()
    session.flush.assert_not_awaited()
    session.commit.assert_not_awaited()
    session.rollback.assert_not_awaited()


async def test_claim_next_available_creates_running_ownership() -> None:
    session = create_session()
    record = AgentRunRecord.from_domain(create_initial_run())
    result = MagicMock()
    result.scalar_one_or_none.return_value = record
    session.execute.return_value = result

    repository = SqlAlchemyAgentRunRepository(session)

    claim = await repository.claim_next_available(
        create_claim_command(),
    )

    assert claim is not None
    assert claim.agent_run.status is AgentRunStatus.RUNNING
    assert claim.agent_run.attempt_count == 1
    assert claim.agent_run.lease_owner == "worker-a"
    assert claim.agent_run.lease_token == _LEASE_TOKEN
    assert claim.agent_run.lease_expires_at == _CLAIMED_AT + timedelta(seconds=45)
    assert claim.agent_run.first_started_at == _CLAIMED_AT
    assert claim.agent_run.updated_at == _CLAIMED_AT
    assert claim.agent_run.completed_at is None


async def test_claim_next_available_creates_matching_attempt() -> None:
    session = create_session()
    record = AgentRunRecord.from_domain(create_initial_run())
    result = MagicMock()
    result.scalar_one_or_none.return_value = record
    session.execute.return_value = result

    repository = SqlAlchemyAgentRunRepository(session)

    claim = await repository.claim_next_available(
        create_claim_command(),
    )

    assert claim is not None
    assert claim.attempt.agent_run_id == _RUN_ID
    assert claim.attempt.attempt_number == 1
    assert claim.attempt.worker_id == "worker-a"
    assert claim.attempt.lease_token == _LEASE_TOKEN
    assert claim.attempt.execution_request_id == _EXECUTION_REQUEST_ID
    assert claim.attempt.started_at == _CLAIMED_AT
    assert claim.attempt.finished_at is None
    assert claim.attempt.outcome is None

    session.add.assert_called_once()
    persisted_attempt = session.add.call_args.args[0]

    assert isinstance(persisted_attempt, AgentRunAttemptRecord)
    assert persisted_attempt.to_domain() == claim.attempt
    session.flush.assert_awaited_once_with()


async def test_claim_next_available_preserves_first_started_at() -> None:
    session = create_session()
    retry_scheduled_run = create_retry_scheduled_run()
    record = AgentRunRecord.from_domain(retry_scheduled_run)
    result = MagicMock()
    result.scalar_one_or_none.return_value = record
    session.execute.return_value = result

    repository = SqlAlchemyAgentRunRepository(session)

    claim = await repository.claim_next_available(
        create_claim_command(),
    )

    assert claim is not None
    assert claim.agent_run.status is AgentRunStatus.RUNNING
    assert claim.agent_run.attempt_count == 2
    assert claim.agent_run.first_started_at == retry_scheduled_run.first_started_at
    assert claim.attempt.attempt_number == 2


async def test_claim_next_available_preserves_previous_error() -> None:
    session = create_session()
    retry_scheduled_run = create_retry_scheduled_run()
    record = AgentRunRecord.from_domain(retry_scheduled_run)
    result = MagicMock()
    result.scalar_one_or_none.return_value = record
    session.execute.return_value = result

    repository = SqlAlchemyAgentRunRepository(session)

    claim = await repository.claim_next_available(
        create_claim_command(),
    )

    assert claim is not None
    assert claim.agent_run.last_error_code == "retryable_executor_failure"
    assert (
        claim.agent_run.last_error_summary
        == "The configured executor reported a retryable failure."
    )


async def test_claim_next_available_does_not_own_transaction() -> None:
    session = create_session()
    record = AgentRunRecord.from_domain(create_initial_run())
    result = MagicMock()
    result.scalar_one_or_none.return_value = record
    session.execute.return_value = result

    repository = SqlAlchemyAgentRunRepository(session)

    await repository.claim_next_available(
        create_claim_command(),
    )

    session.flush.assert_awaited_once_with()
    session.commit.assert_not_awaited()
    session.rollback.assert_not_awaited()


async def test_claim_query_uses_postgresql_skip_locked_ordering() -> None:
    session = create_session()
    result = MagicMock()
    result.scalar_one_or_none.return_value = None
    session.execute.return_value = result

    repository = SqlAlchemyAgentRunRepository(session)

    await repository.claim_next_available(
        create_claim_command(),
    )

    statement = session.execute.await_args.args[0]
    compiled = str(
        statement.compile(
            dialect=cast(Any, postgresql.dialect)(),
            compile_kwargs={"literal_binds": True},
        ),
    )

    normalized = " ".join(compiled.split())

    assert "FOR UPDATE SKIP LOCKED" in normalized
    assert (
        "ORDER BY agent_runs.available_at ASC, "
        "agent_runs.created_at ASC, agent_runs.id ASC" in normalized
    )
    assert "agent_runs.available_at <=" in normalized
    assert "agent_runs.attempt_count < agent_runs.max_attempts" in normalized
    assert "LIMIT 1" in normalized
