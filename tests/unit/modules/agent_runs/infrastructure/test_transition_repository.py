"""Unit tests for fenced AgentRun repository transitions."""

from datetime import UTC, datetime, timedelta
from typing import cast
from unittest.mock import AsyncMock, MagicMock, create_autospec
from uuid import UUID

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from supportops.modules.agent_runs.domain.models import (
    DETERMINISTIC_BASELINE_WORKFLOW_VERSION,
    AgentRun,
    AgentRunAttempt,
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

_RUN_ID = UUID(
    "69184ef1-4d71-452e-8070-0b784c29368e",
)
_LEASE_TOKEN = UUID(
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
_FINISHED_AT = datetime(
    2026,
    7,
    31,
    17,
    0,
    10,
    tzinfo=UTC,
)


def create_running_run_record() -> AgentRunRecord:
    """Create a deterministic actively leased AgentRun record."""

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
        max_retryable_failures=3,
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
        attempt_count=1,
        retryable_failure_count=0,
        max_retryable_failures=initial_run.max_retryable_failures,
        lease_owner="worker-a",
        lease_token=_LEASE_TOKEN,
        lease_expires_at=_STARTED_AT + timedelta(seconds=45),
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


def create_waiting_run_record() -> AgentRunRecord:
    """Create a deterministic waiting-for-approval AgentRun record."""

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
        max_retryable_failures=3,
        now=_STARTED_AT - timedelta(minutes=1),
    )

    waiting_run = AgentRun(
        id=initial_run.id,
        workspace_id=initial_run.workspace_id,
        ticket_id=initial_run.ticket_id,
        workflow_name=initial_run.workflow_name,
        workflow_version=initial_run.workflow_version,
        trigger_key=initial_run.trigger_key,
        status=AgentRunStatus.WAITING_FOR_APPROVAL,
        available_at=None,
        attempt_count=1,
        retryable_failure_count=0,
        max_retryable_failures=initial_run.max_retryable_failures,
        lease_owner=None,
        lease_token=None,
        lease_expires_at=None,
        first_started_at=_STARTED_AT,
        completed_at=None,
        last_error_code=None,
        last_error_summary=None,
        ingestion_request_id=initial_run.ingestion_request_id,
        correlation_id=initial_run.correlation_id,
        created_at=initial_run.created_at,
        updated_at=_FINISHED_AT,
    )
    return AgentRunRecord.from_domain(waiting_run)


def create_active_attempt_record() -> AgentRunAttemptRecord:
    """Create a deterministic active attempt record."""

    attempt = AgentRunAttempt.start(
        attempt_id=UUID(
            "2b39f5b7-b2a4-48d0-b079-fdad286d5315",
        ),
        agent_run_id=_RUN_ID,
        attempt_number=1,
        worker_id="worker-a",
        lease_token=_LEASE_TOKEN,
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
    """Create a session returning the requested run and attempt records."""

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


async def test_mark_succeeded_closes_attempt_and_run() -> None:
    run_record = create_running_run_record()
    attempt_record = create_active_attempt_record()
    original_available_at = run_record.available_at
    session = create_session(
        run_record=run_record,
        attempt_record=attempt_record,
    )
    repository = SqlAlchemyAgentRunRepository(session)

    result = await repository.mark_succeeded(
        CompleteAgentRunCommand(
            agent_run_id=_RUN_ID,
            lease_token=_LEASE_TOKEN,
            finished_at=_FINISHED_AT,
        ),
    )

    assert result is AgentRunTransitionResult.APPLIED

    assert run_record.status == AgentRunStatus.SUCCEEDED.value
    assert run_record.completed_at == _FINISHED_AT
    assert run_record.available_at == original_available_at
    assert run_record.lease_owner is None
    assert run_record.lease_token is None
    assert run_record.lease_expires_at is None
    assert run_record.last_error_code is None
    assert run_record.last_error_summary is None
    assert run_record.updated_at == _FINISHED_AT
    assert run_record.attempt_count == 1
    assert run_record.first_started_at == _STARTED_AT

    assert attempt_record.finished_at == _FINISHED_AT
    assert attempt_record.outcome == AgentRunAttemptOutcome.SUCCEEDED.value
    assert attempt_record.error_code is None
    assert attempt_record.error_summary is None

    session.flush.assert_awaited_once_with()
    session.commit.assert_not_awaited()
    session.rollback.assert_not_awaited()


async def test_mark_succeeded_returns_lease_lost_for_stale_ownership() -> None:
    session = create_session(run_record=None)
    repository = SqlAlchemyAgentRunRepository(session)

    result = await repository.mark_succeeded(
        CompleteAgentRunCommand(
            agent_run_id=_RUN_ID,
            lease_token=_LEASE_TOKEN,
            finished_at=_FINISHED_AT,
        ),
    )

    assert result is AgentRunTransitionResult.LEASE_LOST
    assert session.execute.await_count == 1
    session.flush.assert_not_awaited()
    session.commit.assert_not_awaited()
    session.rollback.assert_not_awaited()


async def test_mark_succeeded_requires_active_attempt() -> None:
    session = create_session(
        run_record=create_running_run_record(),
        attempt_record=None,
    )
    repository = SqlAlchemyAgentRunRepository(session)

    with pytest.raises(
        RuntimeError,
        match=(
            r"Active AgentRun attempt was not found "
            r"for the current lease\."
        ),
    ):
        await repository.mark_succeeded(
            CompleteAgentRunCommand(
                agent_run_id=_RUN_ID,
                lease_token=_LEASE_TOKEN,
                finished_at=_FINISHED_AT,
            ),
        )

    session.flush.assert_not_awaited()


async def test_record_failure_schedules_retry() -> None:
    run_record = create_running_run_record()
    attempt_record = create_active_attempt_record()
    retry_available_at = _FINISHED_AT + timedelta(seconds=2)
    session = create_session(
        run_record=run_record,
        attempt_record=attempt_record,
    )
    repository = SqlAlchemyAgentRunRepository(session)

    result = await repository.record_failure(
        FailAgentRunCommand(
            agent_run_id=_RUN_ID,
            lease_token=_LEASE_TOKEN,
            finished_at=_FINISHED_AT,
            outcome=AgentRunAttemptOutcome.RETRYABLE_FAILURE,
            disposition=AgentRunFailureDisposition.RETRY_SCHEDULED,
            error_code="retryable_executor_failure",
            error_summary=("The configured executor reported a retryable failure."),
            retry_available_at=retry_available_at,
        ),
    )

    assert result is AgentRunTransitionResult.APPLIED

    assert run_record.status == AgentRunStatus.RETRY_SCHEDULED.value
    assert run_record.available_at == retry_available_at
    assert run_record.completed_at is None
    assert run_record.lease_owner is None
    assert run_record.lease_token is None
    assert run_record.lease_expires_at is None
    assert run_record.last_error_code == "retryable_executor_failure"
    assert run_record.last_error_summary == (
        "The configured executor reported a retryable failure."
    )
    assert run_record.updated_at == _FINISHED_AT

    assert attempt_record.finished_at == _FINISHED_AT
    assert attempt_record.outcome == AgentRunAttemptOutcome.RETRYABLE_FAILURE.value
    assert attempt_record.error_code == "retryable_executor_failure"
    assert attempt_record.error_summary == ("The configured executor reported a retryable failure.")

    session.flush.assert_awaited_once_with()


async def test_record_failure_persists_timeout_outcome() -> None:
    run_record = create_running_run_record()
    attempt_record = create_active_attempt_record()
    session = create_session(
        run_record=run_record,
        attempt_record=attempt_record,
    )
    repository = SqlAlchemyAgentRunRepository(session)

    result = await repository.record_failure(
        FailAgentRunCommand(
            agent_run_id=_RUN_ID,
            lease_token=_LEASE_TOKEN,
            finished_at=_FINISHED_AT,
            outcome=AgentRunAttemptOutcome.TIMED_OUT,
            disposition=AgentRunFailureDisposition.RETRY_SCHEDULED,
            error_code="executor_timeout",
            error_summary=("The configured executor exceeded its execution timeout."),
            retry_available_at=_FINISHED_AT + timedelta(seconds=2),
        ),
    )

    assert result is AgentRunTransitionResult.APPLIED
    assert run_record.status == AgentRunStatus.RETRY_SCHEDULED.value
    assert attempt_record.outcome == AgentRunAttemptOutcome.TIMED_OUT.value
    assert attempt_record.error_code == "executor_timeout"


async def test_record_failure_marks_terminal_run_failed() -> None:
    run_record = create_running_run_record()
    attempt_record = create_active_attempt_record()
    original_available_at = run_record.available_at
    session = create_session(
        run_record=run_record,
        attempt_record=attempt_record,
    )
    repository = SqlAlchemyAgentRunRepository(session)

    result = await repository.record_failure(
        FailAgentRunCommand(
            agent_run_id=_RUN_ID,
            lease_token=_LEASE_TOKEN,
            finished_at=_FINISHED_AT,
            outcome=AgentRunAttemptOutcome.TERMINAL_FAILURE,
            disposition=AgentRunFailureDisposition.FAILED,
            error_code="terminal_executor_failure",
            error_summary=("The configured executor reported a terminal failure."),
            retry_available_at=None,
        ),
    )

    assert result is AgentRunTransitionResult.APPLIED

    assert run_record.status == AgentRunStatus.FAILED.value
    assert run_record.completed_at == _FINISHED_AT
    assert run_record.available_at == original_available_at
    assert run_record.lease_owner is None
    assert run_record.lease_token is None
    assert run_record.lease_expires_at is None
    assert run_record.last_error_code == "terminal_executor_failure"
    assert run_record.last_error_summary == ("The configured executor reported a terminal failure.")

    assert attempt_record.finished_at == _FINISHED_AT
    assert attempt_record.outcome == AgentRunAttemptOutcome.TERMINAL_FAILURE.value


async def test_record_failure_can_fail_exhausted_retryable_run() -> None:
    run_record = create_running_run_record()
    run_record.retryable_failure_count = run_record.max_retryable_failures - 1
    attempt_record = create_active_attempt_record()
    session = create_session(
        run_record=run_record,
        attempt_record=attempt_record,
    )
    repository = SqlAlchemyAgentRunRepository(session)

    result = await repository.record_failure(
        FailAgentRunCommand(
            agent_run_id=_RUN_ID,
            lease_token=_LEASE_TOKEN,
            finished_at=_FINISHED_AT,
            outcome=AgentRunAttemptOutcome.RETRYABLE_FAILURE,
            disposition=AgentRunFailureDisposition.FAILED,
            error_code="unexpected_executor_failure",
            error_summary=("The executor failed unexpectedly and exhausted retries."),
            retry_available_at=None,
        ),
    )

    assert result is AgentRunTransitionResult.APPLIED
    assert run_record.status == AgentRunStatus.FAILED.value
    assert run_record.completed_at == _FINISHED_AT
    assert run_record.retryable_failure_count == run_record.max_retryable_failures
    assert attempt_record.outcome == AgentRunAttemptOutcome.RETRYABLE_FAILURE.value


async def test_record_failure_increments_retryable_failure_count() -> None:
    run_record = create_running_run_record()
    attempt_record = create_active_attempt_record()
    session = create_session(
        run_record=run_record,
        attempt_record=attempt_record,
    )
    repository = SqlAlchemyAgentRunRepository(session)

    result = await repository.record_failure(
        FailAgentRunCommand(
            agent_run_id=_RUN_ID,
            lease_token=_LEASE_TOKEN,
            finished_at=_FINISHED_AT,
            outcome=AgentRunAttemptOutcome.RETRYABLE_FAILURE,
            disposition=AgentRunFailureDisposition.RETRY_SCHEDULED,
            error_code="unexpected_executor_failure",
            error_summary=("The executor failed unexpectedly and may be retried."),
            retry_available_at=_FINISHED_AT + timedelta(seconds=2),
        ),
    )

    assert result is AgentRunTransitionResult.APPLIED
    assert run_record.retryable_failure_count == 1
    assert run_record.status == AgentRunStatus.RETRY_SCHEDULED.value


async def test_terminal_failure_does_not_increment_failure_count() -> None:
    run_record = create_running_run_record()
    attempt_record = create_active_attempt_record()
    session = create_session(
        run_record=run_record,
        attempt_record=attempt_record,
    )
    repository = SqlAlchemyAgentRunRepository(session)

    result = await repository.record_failure(
        FailAgentRunCommand(
            agent_run_id=_RUN_ID,
            lease_token=_LEASE_TOKEN,
            finished_at=_FINISHED_AT,
            outcome=AgentRunAttemptOutcome.TERMINAL_FAILURE,
            disposition=AgentRunFailureDisposition.FAILED,
            error_code="terminal_executor_failure",
            error_summary=("The configured executor reported a terminal failure."),
            retry_available_at=None,
        ),
    )

    assert result is AgentRunTransitionResult.APPLIED
    assert run_record.retryable_failure_count == 0
    assert run_record.status == AgentRunStatus.FAILED.value


async def test_record_failure_rejects_inconsistent_disposition() -> None:
    run_record = create_running_run_record()
    attempt_record = create_active_attempt_record()
    session = create_session(
        run_record=run_record,
        attempt_record=attempt_record,
    )
    repository = SqlAlchemyAgentRunRepository(session)

    with pytest.raises(
        RuntimeError,
        match=("disposition does not match the prospective retryable failure budget"),
    ):
        await repository.record_failure(
            FailAgentRunCommand(
                agent_run_id=_RUN_ID,
                lease_token=_LEASE_TOKEN,
                finished_at=_FINISHED_AT,
                outcome=AgentRunAttemptOutcome.RETRYABLE_FAILURE,
                disposition=AgentRunFailureDisposition.FAILED,
                error_code="unexpected_executor_failure",
                error_summary=("The executor failed unexpectedly and may be retried."),
                retry_available_at=None,
            ),
        )

    assert run_record.retryable_failure_count == 0
    session.flush.assert_not_awaited()
    session.commit.assert_not_awaited()


async def test_record_failure_returns_lease_lost_for_stale_ownership() -> None:
    session = create_session(run_record=None)
    repository = SqlAlchemyAgentRunRepository(session)

    result = await repository.record_failure(
        FailAgentRunCommand(
            agent_run_id=_RUN_ID,
            lease_token=_LEASE_TOKEN,
            finished_at=_FINISHED_AT,
            outcome=AgentRunAttemptOutcome.RETRYABLE_FAILURE,
            disposition=AgentRunFailureDisposition.RETRY_SCHEDULED,
            error_code="retryable_executor_failure",
            error_summary=("The configured executor reported a retryable failure."),
            retry_available_at=_FINISHED_AT + timedelta(seconds=2),
        ),
    )

    assert result is AgentRunTransitionResult.LEASE_LOST
    assert session.execute.await_count == 1
    session.flush.assert_not_awaited()


async def test_record_failure_requires_active_attempt() -> None:
    session = create_session(
        run_record=create_running_run_record(),
        attempt_record=None,
    )
    repository = SqlAlchemyAgentRunRepository(session)

    with pytest.raises(
        RuntimeError,
        match=(
            r"Active AgentRun attempt was not found "
            r"for the current lease\."
        ),
    ):
        await repository.record_failure(
            FailAgentRunCommand(
                agent_run_id=_RUN_ID,
                lease_token=_LEASE_TOKEN,
                finished_at=_FINISHED_AT,
                outcome=AgentRunAttemptOutcome.RETRYABLE_FAILURE,
                disposition=AgentRunFailureDisposition.RETRY_SCHEDULED,
                error_code="retryable_executor_failure",
                error_summary=("The configured executor reported a retryable failure."),
                retry_available_at=_FINISHED_AT + timedelta(seconds=2),
            ),
        )

    session.flush.assert_not_awaited()


async def test_mark_waiting_for_approval_closes_attempt_and_clears_lease() -> None:
    run_record = create_running_run_record()
    attempt_record = create_active_attempt_record()
    original_attempt_count = run_record.attempt_count
    original_retryable_failure_count = run_record.retryable_failure_count
    session = create_session(
        run_record=run_record,
        attempt_record=attempt_record,
    )
    repository = SqlAlchemyAgentRunRepository(session)

    result = await repository.mark_waiting_for_approval(
        WaitForApprovalAgentRunCommand(
            agent_run_id=_RUN_ID,
            lease_token=_LEASE_TOKEN,
            finished_at=_FINISHED_AT,
        ),
    )

    assert result is AgentRunTransitionResult.APPLIED
    assert run_record.status == AgentRunStatus.WAITING_FOR_APPROVAL.value
    assert run_record.available_at is None
    assert run_record.completed_at is None
    assert run_record.lease_owner is None
    assert run_record.lease_token is None
    assert run_record.lease_expires_at is None
    assert run_record.last_error_code is None
    assert run_record.last_error_summary is None
    assert run_record.attempt_count == original_attempt_count
    assert run_record.retryable_failure_count == (original_retryable_failure_count)
    assert run_record.updated_at == _FINISHED_AT
    assert attempt_record.finished_at == _FINISHED_AT
    assert attempt_record.outcome == (AgentRunAttemptOutcome.AWAITING_APPROVAL.value)
    assert attempt_record.error_code is None
    assert attempt_record.error_summary is None
    session.flush.assert_awaited_once_with()
    session.commit.assert_not_awaited()


async def test_mark_waiting_for_approval_returns_lease_lost_for_stale_token() -> None:
    session = create_session(run_record=None)
    repository = SqlAlchemyAgentRunRepository(session)

    result = await repository.mark_waiting_for_approval(
        WaitForApprovalAgentRunCommand(
            agent_run_id=_RUN_ID,
            lease_token=_LEASE_TOKEN,
            finished_at=_FINISHED_AT,
        ),
    )

    assert result is AgentRunTransitionResult.LEASE_LOST
    session.flush.assert_not_awaited()


async def test_mark_waiting_for_approval_requires_active_attempt() -> None:
    session = create_session(
        run_record=create_running_run_record(),
        attempt_record=None,
    )
    repository = SqlAlchemyAgentRunRepository(session)

    with pytest.raises(
        RuntimeError,
        match=(
            r"Active AgentRun attempt was not found "
            r"for the current lease\."
        ),
    ):
        await repository.mark_waiting_for_approval(
            WaitForApprovalAgentRunCommand(
                agent_run_id=_RUN_ID,
                lease_token=_LEASE_TOKEN,
                finished_at=_FINISHED_AT,
            ),
        )

    session.flush.assert_not_awaited()


async def test_requeue_waiting_for_approval_sets_queued_available_at() -> None:
    run_record = create_waiting_run_record()
    original_attempt_count = run_record.attempt_count
    original_retryable_failure_count = run_record.retryable_failure_count
    session = create_session(run_record=run_record)
    repository = SqlAlchemyAgentRunRepository(session)
    requeued_at = _FINISHED_AT + timedelta(minutes=1)

    result = await repository.requeue_waiting_for_approval(
        RequeueWaitingAgentRunCommand(
            workspace_id=run_record.workspace_id,
            ticket_id=run_record.ticket_id,
            agent_run_id=_RUN_ID,
            requeued_at=requeued_at,
        ),
    )

    assert result is AgentRunApprovalRequeueResult.APPLIED
    assert run_record.status == AgentRunStatus.QUEUED.value
    assert run_record.available_at == requeued_at
    assert run_record.updated_at == requeued_at
    assert run_record.completed_at is None
    assert run_record.last_error_code is None
    assert run_record.last_error_summary is None
    assert run_record.attempt_count == original_attempt_count
    assert run_record.retryable_failure_count == (original_retryable_failure_count)
    assert run_record.lease_owner is None
    assert run_record.lease_token is None
    assert run_record.lease_expires_at is None
    session.flush.assert_awaited_once_with()
    session.commit.assert_not_awaited()
    session.add.assert_not_called()


async def test_requeue_waiting_for_approval_returns_conflict_for_invalid_status() -> None:
    run_record = create_running_run_record()
    session = create_session(run_record=run_record)
    repository = SqlAlchemyAgentRunRepository(session)

    result = await repository.requeue_waiting_for_approval(
        RequeueWaitingAgentRunCommand(
            workspace_id=run_record.workspace_id,
            ticket_id=run_record.ticket_id,
            agent_run_id=_RUN_ID,
            requeued_at=_FINISHED_AT,
        ),
    )

    assert result is AgentRunApprovalRequeueResult.STATE_CONFLICT
    session.flush.assert_not_awaited()


async def test_requeue_waiting_for_approval_returns_conflict_for_cross_scope() -> None:
    session = create_session(run_record=None)
    repository = SqlAlchemyAgentRunRepository(session)

    result = await repository.requeue_waiting_for_approval(
        RequeueWaitingAgentRunCommand(
            workspace_id=UUID("aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"),
            ticket_id=UUID("bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb"),
            agent_run_id=_RUN_ID,
            requeued_at=_FINISHED_AT,
        ),
    )

    assert result is AgentRunApprovalRequeueResult.STATE_CONFLICT
    session.flush.assert_not_awaited()
