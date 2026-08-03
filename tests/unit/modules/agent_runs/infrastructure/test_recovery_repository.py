"""Unit tests for expired AgentRun lease recovery."""

from datetime import UTC, datetime, timedelta
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock, create_autospec
from uuid import UUID

import pytest
from sqlalchemy.dialects import postgresql
from sqlalchemy.ext.asyncio import AsyncSession

from supportops.modules.agent_runs.domain.models import (
    DETERMINISTIC_BASELINE_WORKFLOW_VERSION,
    AgentRun,
    AgentRunAttempt,
    AgentRunAttemptOutcome,
    AgentRunStatus,
)
from supportops.modules.agent_runs.domain.recovery import (
    ExpiredAgentRunDisposition,
    RecoverExpiredAgentRunCommand,
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
_EXPIRED_LEASE_TOKEN = UUID(
    "dd0ae456-3467-41db-93d1-a908f40e8365",
)
_STARTED_AT = datetime(
    2026,
    7,
    31,
    17,
    0,
    tzinfo=UTC,
)
_RECOVERED_AT = datetime(
    2026,
    7,
    31,
    17,
    1,
    tzinfo=UTC,
)


def create_recovery_command() -> RecoverExpiredAgentRunCommand:
    """Create deterministic recovery values."""

    return RecoverExpiredAgentRunCommand(
        recovered_at=_RECOVERED_AT,
        retry_base_delay_seconds=2.0,
        retry_maximum_delay_seconds=60.0,
        error_code="worker_lease_expired",
        error_summary=("The worker lease expired before execution completed."),
    )


def create_expired_run_record(
    *,
    attempt_count: int = 1,
    retryable_failure_count: int = 0,
    max_retryable_failures: int = 3,
) -> AgentRunRecord:
    """Create an expired actively leased AgentRun record."""

    initial_run = AgentRun.create_initial(
        agent_run_id=_RUN_ID,
        workspace_id=UUID(
            "032c8c87-57cc-4d14-bfbd-04968b4e8cd4",
        ),
        ticket_id=UUID(
            "38bb60fe-d2ea-4615-b499-91aa45069019",
        ),
        ingestion_request_id=UUID(
            "725eec8a-c504-4071-ac96-c78cc907f26c",
        ),
        correlation_id=UUID(
            "1038c98e-62fd-45df-9839-138f7105cb78",
        ),
        workflow_version=DETERMINISTIC_BASELINE_WORKFLOW_VERSION,
        max_retryable_failures=max_retryable_failures,
        now=_STARTED_AT - timedelta(minutes=1),
    )

    running_run = AgentRun(
        id=initial_run.id,
        workspace_id=initial_run.workspace_id,
        ticket_id=initial_run.ticket_id,
        workflow_name=initial_run.workflow_name,
        workflow_version=initial_run.workflow_version,
        trigger_key=initial_run.trigger_key,
        status=AgentRunStatus.RUNNING,
        available_at=initial_run.available_at,
        attempt_count=attempt_count,
        retryable_failure_count=retryable_failure_count,
        max_retryable_failures=max_retryable_failures,
        lease_owner="worker-a",
        lease_token=_EXPIRED_LEASE_TOKEN,
        lease_expires_at=_RECOVERED_AT - timedelta(seconds=1),
        first_started_at=_STARTED_AT,
        completed_at=None,
        last_error_code=None,
        last_error_summary=None,
        ingestion_request_id=initial_run.ingestion_request_id,
        correlation_id=initial_run.correlation_id,
        created_at=initial_run.created_at,
        updated_at=_STARTED_AT,
    )
    return AgentRunRecord.from_domain(running_run)


def create_active_attempt_record(
    *,
    attempt_number: int = 1,
) -> AgentRunAttemptRecord:
    """Create the active attempt owned by the expired lease."""

    attempt = AgentRunAttempt.start(
        attempt_id=UUID(
            "2b39f5b7-b2a4-48d0-b079-fdad286d5315",
        ),
        agent_run_id=_RUN_ID,
        attempt_number=attempt_number,
        worker_id="worker-a",
        lease_token=_EXPIRED_LEASE_TOKEN,
        execution_request_id=UUID(
            "d1fa068f-2278-47a8-b3c9-39ccf91f0a5e",
        ),
        now=_STARTED_AT,
    )
    return AgentRunAttemptRecord.from_domain(attempt)


def create_session(
    *,
    run_record: AgentRunRecord | None,
    attempt_record: AgentRunAttemptRecord | None = None,
) -> MagicMock:
    """Create a session returning the requested recovery records."""

    session = create_autospec(
        AsyncSession,
        instance=True,
    )
    session.flush = AsyncMock()
    session.commit = AsyncMock()
    session.rollback = AsyncMock()

    run_result = MagicMock()
    run_result.scalar_one_or_none.return_value = run_record

    attempt_result = MagicMock()
    attempt_result.scalar_one_or_none.return_value = attempt_record

    session.execute = AsyncMock(
        side_effect=[
            run_result,
            attempt_result,
        ],
    )

    return cast(MagicMock, session)


async def test_recover_next_expired_returns_none_when_none_exists() -> None:
    session = create_session(run_record=None)
    repository = SqlAlchemyAgentRunRepository(session)

    result = await repository.recover_next_expired(
        create_recovery_command(),
    )

    assert result is None
    assert session.execute.await_count == 1
    session.flush.assert_not_awaited()
    session.commit.assert_not_awaited()
    session.rollback.assert_not_awaited()


async def test_recover_next_expired_schedules_retry() -> None:
    run_record = create_expired_run_record(
        attempt_count=1,
        max_retryable_failures=3,
    )
    attempt_record = create_active_attempt_record(
        attempt_number=1,
    )
    session = create_session(
        run_record=run_record,
        attempt_record=attempt_record,
    )
    repository = SqlAlchemyAgentRunRepository(session)

    result = await repository.recover_next_expired(
        create_recovery_command(),
    )

    assert result is not None
    assert result.disposition is ExpiredAgentRunDisposition.RETRY_SCHEDULED
    assert result.expired_lease_token == _EXPIRED_LEASE_TOKEN

    assert run_record.status == AgentRunStatus.RETRY_SCHEDULED.value
    assert run_record.available_at == _RECOVERED_AT + timedelta(seconds=2)
    assert run_record.completed_at is None
    assert run_record.attempt_count == 1
    assert run_record.retryable_failure_count == 1
    assert run_record.first_started_at == _STARTED_AT
    assert run_record.lease_owner is None
    assert run_record.lease_token is None
    assert run_record.lease_expires_at is None
    assert run_record.last_error_code == "worker_lease_expired"
    assert run_record.last_error_summary == ("The worker lease expired before execution completed.")
    assert run_record.updated_at == _RECOVERED_AT

    assert attempt_record.finished_at == _RECOVERED_AT
    assert attempt_record.outcome == AgentRunAttemptOutcome.LEASE_EXPIRED.value
    assert attempt_record.error_code == "worker_lease_expired"
    assert attempt_record.error_summary == ("The worker lease expired before execution completed.")

    session.flush.assert_awaited_once_with()
    session.commit.assert_not_awaited()
    session.rollback.assert_not_awaited()


async def test_recover_next_expired_marks_exhausted_run_failed() -> None:
    run_record = create_expired_run_record(
        attempt_count=5,
        retryable_failure_count=2,
        max_retryable_failures=3,
    )
    original_available_at = run_record.available_at
    attempt_record = create_active_attempt_record(
        attempt_number=5,
    )
    session = create_session(
        run_record=run_record,
        attempt_record=attempt_record,
    )
    repository = SqlAlchemyAgentRunRepository(session)

    result = await repository.recover_next_expired(
        create_recovery_command(),
    )

    assert result is not None
    assert result.disposition is ExpiredAgentRunDisposition.FAILED
    assert result.expired_lease_token == _EXPIRED_LEASE_TOKEN

    assert run_record.status == AgentRunStatus.FAILED.value
    assert run_record.completed_at == _RECOVERED_AT
    assert run_record.available_at == original_available_at
    assert run_record.attempt_count == 5
    assert run_record.retryable_failure_count == 3
    assert run_record.lease_owner is None
    assert run_record.lease_token is None
    assert run_record.lease_expires_at is None
    assert run_record.last_error_code == "worker_lease_expired"

    assert attempt_record.finished_at == _RECOVERED_AT
    assert attempt_record.outcome == AgentRunAttemptOutcome.LEASE_EXPIRED.value

    session.flush.assert_awaited_once_with()


async def test_recovery_does_not_create_or_increment_attempt() -> None:
    run_record = create_expired_run_record(
        attempt_count=2,
        max_retryable_failures=3,
    )
    attempt_record = create_active_attempt_record(
        attempt_number=2,
    )
    session = create_session(
        run_record=run_record,
        attempt_record=attempt_record,
    )
    session.add = MagicMock()
    repository = SqlAlchemyAgentRunRepository(session)

    result = await repository.recover_next_expired(
        create_recovery_command(),
    )

    assert result is not None
    assert result.agent_run.attempt_count == 2
    assert attempt_record.attempt_number == 2
    session.add.assert_not_called()


async def test_recovery_requires_expired_run_lease_token() -> None:
    run_record = create_expired_run_record()
    run_record.lease_token = None
    session = create_session(
        run_record=run_record,
    )
    repository = SqlAlchemyAgentRunRepository(session)

    with pytest.raises(
        RuntimeError,
        match=(
            r"Expired running AgentRun does not have "
            r"a lease token\."
        ),
    ):
        await repository.recover_next_expired(
            create_recovery_command(),
        )

    assert session.execute.await_count == 1
    session.flush.assert_not_awaited()


async def test_recovery_requires_active_attempt_for_expired_lease() -> None:
    session = create_session(
        run_record=create_expired_run_record(),
        attempt_record=None,
    )
    repository = SqlAlchemyAgentRunRepository(session)

    with pytest.raises(
        RuntimeError,
        match=(
            r"Active AgentRun attempt was not found "
            r"for the expired lease\."
        ),
    ):
        await repository.recover_next_expired(
            create_recovery_command(),
        )

    assert session.execute.await_count == 2
    session.flush.assert_not_awaited()


async def test_recovery_query_uses_skip_locked_and_expiry_ordering() -> None:
    session = create_session(run_record=None)
    repository = SqlAlchemyAgentRunRepository(session)

    await repository.recover_next_expired(
        create_recovery_command(),
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
        "ORDER BY agent_runs.lease_expires_at ASC, "
        "agent_runs.created_at ASC, agent_runs.id ASC" in normalized
    )
    assert "agent_runs.status = 'running'" in normalized
    assert "agent_runs.lease_expires_at IS NOT NULL" in normalized
    assert "agent_runs.lease_expires_at <=" in normalized
    assert "LIMIT 1" in normalized
    assert "waiting_for_approval" not in normalized
