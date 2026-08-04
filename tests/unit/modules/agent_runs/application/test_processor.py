"""Unit tests for processing one claimed AgentRun."""

import asyncio
from collections.abc import AsyncIterator, Mapping
from contextlib import AbstractContextManager, asynccontextmanager
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime, timedelta
from types import TracebackType
from typing import Literal
from unittest.mock import AsyncMock
from uuid import UUID

import pytest

from supportops.modules.agent_runs.application.execution import (
    AgentRunExecutionContext,
    CompletedExecution,
    PausedForApproval,
    RetryableAgentRunExecutionError,
    TerminalAgentRunExecutionError,
)
from supportops.modules.agent_runs.application.processor import (
    ProcessClaimedAgentRun,
)
from supportops.modules.agent_runs.application.retry_policy import (
    AgentRunRetryPolicy,
)
from supportops.modules.agent_runs.domain.claiming import AgentRunClaim
from supportops.modules.agent_runs.domain.models import (
    DETERMINISTIC_BASELINE_WORKFLOW_VERSION,
    AgentRun,
    AgentRunAttempt,
    AgentRunAttemptOutcome,
    AgentRunStatus,
)
from supportops.modules.agent_runs.domain.transitions import (
    AgentRunFailureDisposition,
    AgentRunTransitionResult,
)
from supportops.modules.tickets.domain.models import Ticket
from supportops.observability.context import (
    current_observation_context,
    current_trace_context,
)
from supportops.observability.models import (
    EventObservation,
    ObservationAttributes,
    ObservationStatus,
    ObservationType,
    ObservationUpdate,
    TraceAttributes,
)

_NOW = datetime(
    2026,
    7,
    31,
    18,
    0,
    tzinfo=UTC,
)
_FINISHED_AT = _NOW + timedelta(seconds=1)
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


class RecordingTransactionManager:
    """Record transaction boundaries and expose active depth."""

    def __init__(self) -> None:
        self.entries = 0
        self.active_depth = 0

    @asynccontextmanager
    async def transaction(self) -> AsyncIterator[None]:
        self.entries += 1
        self.active_depth += 1

        try:
            yield
        finally:
            self.active_depth -= 1


class RecordingExecutor:
    """Record whether execution occurs inside a transaction."""

    def __init__(
        self,
        *,
        transaction_manager: RecordingTransactionManager,
        error: Exception | None = None,
    ) -> None:
        self._transaction_manager = transaction_manager
        self._error = error
        self.contexts: list[AgentRunExecutionContext] = []

    async def execute(
        self,
        context: AgentRunExecutionContext,
    ) -> CompletedExecution:
        assert self._transaction_manager.active_depth == 0
        self.contexts.append(context)

        if self._error is not None:
            raise self._error

        return CompletedExecution()


class BlockingExecutor:
    """Block long enough for the processor timeout to expire."""

    async def execute(
        self,
        context: AgentRunExecutionContext,
    ) -> CompletedExecution:
        del context
        await asyncio.sleep(10)
        return CompletedExecution()


class PausingExecutor:
    """Return a paused-for-approval execution result."""

    def __init__(
        self,
        *,
        approval_request_id: UUID,
        graph_thread_id: str,
    ) -> None:
        self._approval_request_id = approval_request_id
        self._graph_thread_id = graph_thread_id
        self.contexts: list[AgentRunExecutionContext] = []

    async def execute(
        self,
        context: AgentRunExecutionContext,
    ) -> PausedForApproval:
        self.contexts.append(context)
        return PausedForApproval(
            approval_request_id=self._approval_request_id,
            graph_thread_id=self._graph_thread_id,
        )


class UnknownResultExecutor:
    """Return an unsupported execution result object."""

    async def execute(
        self,
        context: AgentRunExecutionContext,
    ) -> object:
        del context
        return object()


def create_ticket() -> Ticket:
    return Ticket.create(
        ticket_id=_TICKET_ID,
        workspace_id=_WORKSPACE_ID,
        subject="Unable to access billing",
        description="The billing page returns an access error.",
        ingestion_request_id=UUID(
            "725eec8a-c504-4071-ac96-c78cc907f26c",
        ),
        correlation_id=UUID(
            "1038c98e-62fd-45df-9839-138f7105cb78",
        ),
        now=_NOW - timedelta(minutes=1),
    )


def create_claim(
    *,
    attempt_count: int = 1,
    retryable_failure_count: int = 0,
    max_retryable_failures: int = 3,
) -> AgentRunClaim:
    initial_run = AgentRun.create_initial(
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
        max_retryable_failures=max_retryable_failures,
        now=_NOW - timedelta(minutes=1),
    )
    running_run = replace(
        initial_run,
        status=AgentRunStatus.RUNNING,
        attempt_count=attempt_count,
        retryable_failure_count=retryable_failure_count,
        lease_owner="worker-a",
        lease_token=_LEASE_TOKEN,
        lease_expires_at=_NOW + timedelta(seconds=45),
        first_started_at=_NOW,
        updated_at=_NOW,
    )
    attempt = AgentRunAttempt.start(
        attempt_id=UUID(
            "2b39f5b7-b2a4-48d0-b079-fdad286d5315",
        ),
        agent_run_id=_RUN_ID,
        attempt_number=attempt_count,
        worker_id="worker-a",
        lease_token=_LEASE_TOKEN,
        execution_request_id=UUID(
            "d1fa068f-2278-47a8-b3c9-39ccf91f0a5e",
        ),
        now=_NOW,
    )

    return AgentRunClaim(
        agent_run=running_run,
        attempt=attempt,
    )


def create_processor(
    *,
    executor: object,
    ticket: Ticket | None = None,
    transition_result: AgentRunTransitionResult = (AgentRunTransitionResult.APPLIED),
    execution_timeout_seconds: float = 30.0,
) -> tuple[
    ProcessClaimedAgentRun,
    AsyncMock,
    AsyncMock,
    RecordingTransactionManager,
]:
    ticket_repository = AsyncMock()
    ticket_repository.get.return_value = create_ticket() if ticket is None else ticket

    agent_run_repository = AsyncMock()
    agent_run_repository.mark_succeeded.return_value = transition_result
    agent_run_repository.mark_waiting_for_approval.return_value = transition_result
    agent_run_repository.record_failure.return_value = transition_result

    transaction_manager = RecordingTransactionManager()

    processor = ProcessClaimedAgentRun(
        ticket_repository=ticket_repository,
        agent_run_repository=agent_run_repository,
        transaction_manager=transaction_manager,
        executor=executor,  # type: ignore[arg-type]
        retry_policy=AgentRunRetryPolicy(
            base_delay_seconds=2.0,
            maximum_delay_seconds=60.0,
        ),
        execution_timeout_seconds=execution_timeout_seconds,
        utc_now=lambda: _FINISHED_AT,
    )

    return (
        processor,
        ticket_repository,
        agent_run_repository,
        transaction_manager,
    )


async def test_success_executes_outside_transaction_and_completes() -> None:
    transaction_manager = RecordingTransactionManager()
    executor = RecordingExecutor(
        transaction_manager=transaction_manager,
    )
    ticket_repository = AsyncMock()
    ticket_repository.get.return_value = create_ticket()
    agent_run_repository = AsyncMock()
    agent_run_repository.mark_succeeded.return_value = AgentRunTransitionResult.APPLIED

    processor = ProcessClaimedAgentRun(
        ticket_repository=ticket_repository,
        agent_run_repository=agent_run_repository,
        transaction_manager=transaction_manager,
        executor=executor,
        retry_policy=AgentRunRetryPolicy(
            base_delay_seconds=2.0,
            maximum_delay_seconds=60.0,
        ),
        execution_timeout_seconds=30.0,
        utc_now=lambda: _FINISHED_AT,
    )

    claim = create_claim()
    result = await processor.execute(claim)

    assert result is AgentRunTransitionResult.APPLIED
    assert transaction_manager.entries == 2
    assert len(executor.contexts) == 1
    assert executor.contexts[0].agent_run == claim.agent_run
    assert executor.contexts[0].attempt == claim.attempt
    assert executor.contexts[0].ticket == create_ticket()

    command = agent_run_repository.mark_succeeded.await_args.args[0]
    assert command.agent_run_id == _RUN_ID
    assert command.lease_token == _LEASE_TOKEN
    assert command.finished_at == _FINISHED_AT

    agent_run_repository.record_failure.assert_not_awaited()


async def test_missing_ticket_is_terminal_without_executor_call() -> None:
    transaction_manager = RecordingTransactionManager()
    executor = RecordingExecutor(
        transaction_manager=transaction_manager,
    )
    ticket_repository = AsyncMock()
    ticket_repository.get.return_value = None
    agent_run_repository = AsyncMock()
    agent_run_repository.record_failure.return_value = AgentRunTransitionResult.APPLIED

    processor = ProcessClaimedAgentRun(
        ticket_repository=ticket_repository,
        agent_run_repository=agent_run_repository,
        transaction_manager=transaction_manager,
        executor=executor,
        retry_policy=AgentRunRetryPolicy(
            base_delay_seconds=2.0,
            maximum_delay_seconds=60.0,
        ),
        execution_timeout_seconds=30.0,
        utc_now=lambda: _FINISHED_AT,
    )

    result = await processor.execute(create_claim())

    assert result is AgentRunTransitionResult.APPLIED
    assert executor.contexts == []

    command = agent_run_repository.record_failure.await_args.args[0]
    assert command.outcome is AgentRunAttemptOutcome.TERMINAL_FAILURE
    assert command.disposition is AgentRunFailureDisposition.FAILED
    assert command.error_code == "ticket_not_found"
    assert command.retry_available_at is None


async def test_retryable_error_schedules_retry() -> None:
    transaction_manager = RecordingTransactionManager()
    executor = RecordingExecutor(
        transaction_manager=transaction_manager,
        error=RetryableAgentRunExecutionError(
            error_code="provider_unavailable",
            error_summary="The processing provider is temporarily unavailable.",
        ),
    )
    ticket_repository = AsyncMock()
    ticket_repository.get.return_value = create_ticket()
    agent_run_repository = AsyncMock()
    agent_run_repository.record_failure.return_value = AgentRunTransitionResult.APPLIED

    processor = ProcessClaimedAgentRun(
        ticket_repository=ticket_repository,
        agent_run_repository=agent_run_repository,
        transaction_manager=transaction_manager,
        executor=executor,
        retry_policy=AgentRunRetryPolicy(
            base_delay_seconds=2.0,
            maximum_delay_seconds=60.0,
        ),
        execution_timeout_seconds=30.0,
        utc_now=lambda: _FINISHED_AT,
    )

    await processor.execute(create_claim())

    command = agent_run_repository.record_failure.await_args.args[0]
    assert command.outcome is AgentRunAttemptOutcome.RETRYABLE_FAILURE
    assert command.disposition is AgentRunFailureDisposition.RETRY_SCHEDULED
    assert command.error_code == "provider_unavailable"
    assert command.retry_available_at == _FINISHED_AT + timedelta(seconds=2)


async def test_retryable_error_fails_when_budget_is_exhausted() -> None:
    transaction_manager = RecordingTransactionManager()
    executor = RecordingExecutor(
        transaction_manager=transaction_manager,
        error=RetryableAgentRunExecutionError(
            error_code="provider_unavailable",
            error_summary="The processing provider is temporarily unavailable.",
        ),
    )
    ticket_repository = AsyncMock()
    ticket_repository.get.return_value = create_ticket()
    agent_run_repository = AsyncMock()
    agent_run_repository.record_failure.return_value = AgentRunTransitionResult.APPLIED

    processor = ProcessClaimedAgentRun(
        ticket_repository=ticket_repository,
        agent_run_repository=agent_run_repository,
        transaction_manager=transaction_manager,
        executor=executor,
        retry_policy=AgentRunRetryPolicy(
            base_delay_seconds=2.0,
            maximum_delay_seconds=60.0,
        ),
        execution_timeout_seconds=30.0,
        utc_now=lambda: _FINISHED_AT,
    )

    await processor.execute(
        create_claim(
            attempt_count=5,
            retryable_failure_count=2,
            max_retryable_failures=3,
        ),
    )

    command = agent_run_repository.record_failure.await_args.args[0]
    assert command.outcome is AgentRunAttemptOutcome.RETRYABLE_FAILURE
    assert command.disposition is AgentRunFailureDisposition.FAILED
    assert command.retry_available_at is None


async def test_retryable_error_uses_failure_count_not_attempt_count() -> None:
    transaction_manager = RecordingTransactionManager()
    executor = RecordingExecutor(
        transaction_manager=transaction_manager,
        error=RetryableAgentRunExecutionError(
            error_code="provider_unavailable",
            error_summary="The processing provider is temporarily unavailable.",
        ),
    )
    ticket_repository = AsyncMock()
    ticket_repository.get.return_value = create_ticket()
    agent_run_repository = AsyncMock()
    agent_run_repository.record_failure.return_value = AgentRunTransitionResult.APPLIED

    processor = ProcessClaimedAgentRun(
        ticket_repository=ticket_repository,
        agent_run_repository=agent_run_repository,
        transaction_manager=transaction_manager,
        executor=executor,
        retry_policy=AgentRunRetryPolicy(
            base_delay_seconds=2.0,
            maximum_delay_seconds=60.0,
        ),
        execution_timeout_seconds=30.0,
        utc_now=lambda: _FINISHED_AT,
    )

    await processor.execute(
        create_claim(
            attempt_count=10,
            retryable_failure_count=1,
            max_retryable_failures=3,
        ),
    )

    command = agent_run_repository.record_failure.await_args.args[0]
    assert command.disposition is AgentRunFailureDisposition.RETRY_SCHEDULED
    assert command.retry_available_at == _FINISHED_AT + timedelta(seconds=4)


async def test_terminal_error_fails_without_retry() -> None:
    transaction_manager = RecordingTransactionManager()
    executor = RecordingExecutor(
        transaction_manager=transaction_manager,
        error=TerminalAgentRunExecutionError(
            error_code="unsupported_workflow",
            error_summary=("The AgentRun workflow is not supported by the executor."),
        ),
    )
    ticket_repository = AsyncMock()
    ticket_repository.get.return_value = create_ticket()
    agent_run_repository = AsyncMock()
    agent_run_repository.record_failure.return_value = AgentRunTransitionResult.APPLIED

    processor = ProcessClaimedAgentRun(
        ticket_repository=ticket_repository,
        agent_run_repository=agent_run_repository,
        transaction_manager=transaction_manager,
        executor=executor,
        retry_policy=AgentRunRetryPolicy(
            base_delay_seconds=2.0,
            maximum_delay_seconds=60.0,
        ),
        execution_timeout_seconds=30.0,
        utc_now=lambda: _FINISHED_AT,
    )

    await processor.execute(create_claim())

    command = agent_run_repository.record_failure.await_args.args[0]
    assert command.outcome is AgentRunAttemptOutcome.TERMINAL_FAILURE
    assert command.disposition is AgentRunFailureDisposition.FAILED
    assert command.retry_available_at is None


async def test_unexpected_error_uses_safe_retryable_failure() -> None:
    transaction_manager = RecordingTransactionManager()
    executor = RecordingExecutor(
        transaction_manager=transaction_manager,
        error=RuntimeError("database password leaked in raw exception"),
    )
    ticket_repository = AsyncMock()
    ticket_repository.get.return_value = create_ticket()
    agent_run_repository = AsyncMock()
    agent_run_repository.record_failure.return_value = AgentRunTransitionResult.APPLIED

    processor = ProcessClaimedAgentRun(
        ticket_repository=ticket_repository,
        agent_run_repository=agent_run_repository,
        transaction_manager=transaction_manager,
        executor=executor,
        retry_policy=AgentRunRetryPolicy(
            base_delay_seconds=2.0,
            maximum_delay_seconds=60.0,
        ),
        execution_timeout_seconds=30.0,
        utc_now=lambda: _FINISHED_AT,
    )

    await processor.execute(create_claim())

    command = agent_run_repository.record_failure.await_args.args[0]
    assert command.error_code == "unexpected_executor_failure"
    assert "password" not in command.error_summary
    assert command.outcome is AgentRunAttemptOutcome.RETRYABLE_FAILURE


async def test_timeout_is_recorded_as_timed_out() -> None:
    ticket_repository = AsyncMock()
    ticket_repository.get.return_value = create_ticket()
    agent_run_repository = AsyncMock()
    agent_run_repository.record_failure.return_value = AgentRunTransitionResult.APPLIED
    transaction_manager = RecordingTransactionManager()

    processor = ProcessClaimedAgentRun(
        ticket_repository=ticket_repository,
        agent_run_repository=agent_run_repository,
        transaction_manager=transaction_manager,
        executor=BlockingExecutor(),
        retry_policy=AgentRunRetryPolicy(
            base_delay_seconds=2.0,
            maximum_delay_seconds=60.0,
        ),
        execution_timeout_seconds=0.001,
        utc_now=lambda: _FINISHED_AT,
    )

    await processor.execute(create_claim())

    command = agent_run_repository.record_failure.await_args.args[0]
    assert command.outcome is AgentRunAttemptOutcome.TIMED_OUT
    assert command.error_code == "executor_timeout"
    assert command.disposition is AgentRunFailureDisposition.RETRY_SCHEDULED


async def test_lease_lost_result_is_propagated() -> None:
    transaction_manager = RecordingTransactionManager()
    executor = RecordingExecutor(
        transaction_manager=transaction_manager,
    )
    ticket_repository = AsyncMock()
    ticket_repository.get.return_value = create_ticket()
    agent_run_repository = AsyncMock()
    agent_run_repository.mark_succeeded.return_value = AgentRunTransitionResult.LEASE_LOST

    processor = ProcessClaimedAgentRun(
        ticket_repository=ticket_repository,
        agent_run_repository=agent_run_repository,
        transaction_manager=transaction_manager,
        executor=executor,
        retry_policy=AgentRunRetryPolicy(
            base_delay_seconds=2.0,
            maximum_delay_seconds=60.0,
        ),
        execution_timeout_seconds=30.0,
        utc_now=lambda: _FINISHED_AT,
    )

    result = await processor.execute(create_claim())

    assert result is AgentRunTransitionResult.LEASE_LOST


async def test_paused_for_approval_marks_waiting_without_success_or_failure() -> None:
    approval_request_id = UUID("aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee")
    graph_thread_id = "controlled-support:controlled-support-v1:thread"
    transaction_manager = RecordingTransactionManager()
    executor = PausingExecutor(
        approval_request_id=approval_request_id,
        graph_thread_id=graph_thread_id,
    )
    ticket_repository = AsyncMock()
    ticket_repository.get.return_value = create_ticket()
    agent_run_repository = AsyncMock()
    agent_run_repository.mark_waiting_for_approval.return_value = AgentRunTransitionResult.APPLIED

    processor = ProcessClaimedAgentRun(
        ticket_repository=ticket_repository,
        agent_run_repository=agent_run_repository,
        transaction_manager=transaction_manager,
        executor=executor,
        retry_policy=AgentRunRetryPolicy(
            base_delay_seconds=2.0,
            maximum_delay_seconds=60.0,
        ),
        execution_timeout_seconds=30.0,
        utc_now=lambda: _FINISHED_AT,
    )

    claim = create_claim(retryable_failure_count=1)
    result = await processor.execute(claim)

    assert result is AgentRunTransitionResult.APPLIED
    command = agent_run_repository.mark_waiting_for_approval.await_args.args[0]
    assert command.agent_run_id == _RUN_ID
    assert command.lease_token == _LEASE_TOKEN
    assert command.finished_at == _FINISHED_AT
    assert claim.agent_run.retryable_failure_count == 1
    agent_run_repository.mark_succeeded.assert_not_awaited()
    agent_run_repository.record_failure.assert_not_awaited()


async def test_paused_for_approval_preserves_max_retryable_failures() -> None:
    approval_request_id = UUID("aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee")
    transaction_manager = RecordingTransactionManager()
    executor = PausingExecutor(
        approval_request_id=approval_request_id,
        graph_thread_id="ticket-processing:human-approved-support-v1:graph-v1:x",
    )
    ticket_repository = AsyncMock()
    ticket_repository.get.return_value = create_ticket()
    agent_run_repository = AsyncMock()
    agent_run_repository.mark_waiting_for_approval.return_value = AgentRunTransitionResult.APPLIED

    processor = ProcessClaimedAgentRun(
        ticket_repository=ticket_repository,
        agent_run_repository=agent_run_repository,
        transaction_manager=transaction_manager,
        executor=executor,
        retry_policy=AgentRunRetryPolicy(
            base_delay_seconds=2.0,
            maximum_delay_seconds=60.0,
        ),
        execution_timeout_seconds=30.0,
        utc_now=lambda: _FINISHED_AT,
    )

    claim = create_claim(
        attempt_count=1,
        retryable_failure_count=0,
        max_retryable_failures=3,
    )
    result = await processor.execute(claim)

    assert result is AgentRunTransitionResult.APPLIED
    assert claim.agent_run.retryable_failure_count == 0
    assert claim.agent_run.max_retryable_failures == 3
    assert claim.agent_run.attempt_count == 1
    agent_run_repository.mark_waiting_for_approval.assert_awaited_once()
    agent_run_repository.mark_succeeded.assert_not_awaited()
    agent_run_repository.record_failure.assert_not_awaited()


async def test_successful_resume_preserves_attempt_and_retry_counters() -> None:
    transaction_manager = RecordingTransactionManager()
    executor = RecordingExecutor(
        transaction_manager=transaction_manager,
    )
    ticket_repository = AsyncMock()
    ticket_repository.get.return_value = create_ticket()
    agent_run_repository = AsyncMock()
    agent_run_repository.mark_succeeded.return_value = AgentRunTransitionResult.APPLIED

    processor = ProcessClaimedAgentRun(
        ticket_repository=ticket_repository,
        agent_run_repository=agent_run_repository,
        transaction_manager=transaction_manager,
        executor=executor,
        retry_policy=AgentRunRetryPolicy(
            base_delay_seconds=2.0,
            maximum_delay_seconds=60.0,
        ),
        execution_timeout_seconds=30.0,
        utc_now=lambda: _FINISHED_AT,
    )

    claim = create_claim(
        attempt_count=2,
        retryable_failure_count=0,
        max_retryable_failures=3,
    )
    result = await processor.execute(claim)

    assert result is AgentRunTransitionResult.APPLIED
    assert claim.agent_run.attempt_count == 2
    assert claim.agent_run.retryable_failure_count == 0
    assert claim.agent_run.max_retryable_failures == 3
    agent_run_repository.mark_succeeded.assert_awaited_once()
    agent_run_repository.record_failure.assert_not_awaited()
    agent_run_repository.mark_waiting_for_approval.assert_not_awaited()


async def test_successful_resume_completes_without_changing_retry_counter() -> None:
    transaction_manager = RecordingTransactionManager()
    executor = RecordingExecutor(
        transaction_manager=transaction_manager,
    )
    ticket_repository = AsyncMock()
    ticket_repository.get.return_value = create_ticket()
    agent_run_repository = AsyncMock()
    agent_run_repository.mark_succeeded.return_value = AgentRunTransitionResult.APPLIED

    processor = ProcessClaimedAgentRun(
        ticket_repository=ticket_repository,
        agent_run_repository=agent_run_repository,
        transaction_manager=transaction_manager,
        executor=executor,
        retry_policy=AgentRunRetryPolicy(
            base_delay_seconds=2.0,
            maximum_delay_seconds=60.0,
        ),
        execution_timeout_seconds=30.0,
        utc_now=lambda: _FINISHED_AT,
    )

    claim = create_claim(
        attempt_count=2,
        retryable_failure_count=1,
    )
    result = await processor.execute(claim)

    assert result is AgentRunTransitionResult.APPLIED
    assert claim.agent_run.retryable_failure_count == 1
    agent_run_repository.mark_succeeded.assert_awaited_once()
    agent_run_repository.mark_waiting_for_approval.assert_not_awaited()
    agent_run_repository.record_failure.assert_not_awaited()


async def test_paused_for_approval_lease_lost_is_propagated() -> None:
    transaction_manager = RecordingTransactionManager()
    executor = PausingExecutor(
        approval_request_id=UUID("aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee"),
        graph_thread_id="controlled-support:thread",
    )
    ticket_repository = AsyncMock()
    ticket_repository.get.return_value = create_ticket()
    agent_run_repository = AsyncMock()
    agent_run_repository.mark_waiting_for_approval.return_value = (
        AgentRunTransitionResult.LEASE_LOST
    )

    processor = ProcessClaimedAgentRun(
        ticket_repository=ticket_repository,
        agent_run_repository=agent_run_repository,
        transaction_manager=transaction_manager,
        executor=executor,
        retry_policy=AgentRunRetryPolicy(
            base_delay_seconds=2.0,
            maximum_delay_seconds=60.0,
        ),
        execution_timeout_seconds=30.0,
        utc_now=lambda: _FINISHED_AT,
    )

    result = await processor.execute(create_claim())

    assert result is AgentRunTransitionResult.LEASE_LOST
    agent_run_repository.mark_succeeded.assert_not_awaited()
    agent_run_repository.record_failure.assert_not_awaited()


async def test_unknown_execution_result_fails_loudly() -> None:
    transaction_manager = RecordingTransactionManager()
    ticket_repository = AsyncMock()
    ticket_repository.get.return_value = create_ticket()
    agent_run_repository = AsyncMock()

    processor = ProcessClaimedAgentRun(
        ticket_repository=ticket_repository,
        agent_run_repository=agent_run_repository,
        transaction_manager=transaction_manager,
        executor=UnknownResultExecutor(),  # type: ignore[arg-type]
        retry_policy=AgentRunRetryPolicy(
            base_delay_seconds=2.0,
            maximum_delay_seconds=60.0,
        ),
        execution_timeout_seconds=30.0,
        utc_now=lambda: _FINISHED_AT,
    )

    with pytest.raises(
        RuntimeError,
        match="unsupported execution result",
    ):
        await processor.execute(create_claim())

    agent_run_repository.mark_succeeded.assert_not_awaited()
    agent_run_repository.mark_waiting_for_approval.assert_not_awaited()
    agent_run_repository.record_failure.assert_not_awaited()


@pytest.mark.parametrize(
    "execution_timeout_seconds",
    [
        0.0,
        -1.0,
    ],
)
def test_processor_requires_positive_timeout(
    execution_timeout_seconds: float,
) -> None:
    with pytest.raises(
        ValueError,
        match="execution_timeout_seconds must be greater than zero",
    ):
        ProcessClaimedAgentRun(
            ticket_repository=AsyncMock(),
            agent_run_repository=AsyncMock(),
            transaction_manager=RecordingTransactionManager(),
            executor=AsyncMock(),
            retry_policy=AgentRunRetryPolicy(
                base_delay_seconds=2.0,
                maximum_delay_seconds=60.0,
            ),
            execution_timeout_seconds=execution_timeout_seconds,
        )


@dataclass
class RecordingObservationScope:
    attributes: ObservationAttributes
    updates: list[ObservationUpdate] = field(default_factory=list)
    events: list[EventObservation] = field(default_factory=list)
    closed: bool = False
    children: list["RecordingObservationScope"] = field(default_factory=list)

    @property
    def observation_id(self) -> str | None:
        return "observation-1"

    def update(self, update: ObservationUpdate) -> None:
        self.updates.append(update)

    def start_observation(
        self,
        attributes: ObservationAttributes,
    ) -> AbstractContextManager["RecordingObservationScope"]:
        child = RecordingObservationScope(attributes=attributes)
        self.children.append(child)
        return _RecordingObservationManager(child)

    def record_event(self, event: EventObservation) -> None:
        self.events.append(event)


@dataclass
class RecordingTraceScope:
    attributes: TraceAttributes
    updates: list[ObservationUpdate] = field(default_factory=list)
    events: list[EventObservation] = field(default_factory=list)
    observations: list[RecordingObservationScope] = field(default_factory=list)
    closed: bool = False

    @property
    def trace_seed(self) -> str:
        return self.attributes.trace_seed

    @property
    def trace_id(self) -> str | None:
        return None

    @property
    def session_id(self) -> str | None:
        return self.attributes.session_id

    def update(self, update: ObservationUpdate) -> None:
        self.updates.append(update)

    def start_observation(
        self,
        attributes: ObservationAttributes,
    ) -> AbstractContextManager[RecordingObservationScope]:
        observation = RecordingObservationScope(attributes=attributes)
        self.observations.append(observation)
        return _RecordingObservationManager(observation)

    def record_event(self, event: EventObservation) -> None:
        self.events.append(event)


class _RecordingObservationManager(AbstractContextManager[RecordingObservationScope]):
    def __init__(self, scope: RecordingObservationScope) -> None:
        self._scope = scope

    def __enter__(self) -> RecordingObservationScope:
        return self._scope

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> Literal[False]:
        del exc_type, exc, traceback
        self._scope.closed = True
        return False


class RecordingObservabilityClient:
    """Capture AgentRun telemetry without exporting to a provider."""

    def __init__(
        self,
        *,
        fail_start_trace: bool = False,
        fail_start_observation: bool = False,
        fail_update: bool = False,
        fail_exit: bool = False,
    ) -> None:
        self.provider = type("Provider", (), {"value": "recording"})()
        self.enabled = True
        self.traces: list[RecordingTraceScope] = []
        self.flush_calls = 0
        self.shutdown_calls = 0
        self.fail_start_trace = fail_start_trace
        self.fail_start_observation = fail_start_observation
        self.fail_update = fail_update
        self.fail_exit = fail_exit

    def start_trace(
        self,
        attributes: TraceAttributes,
    ) -> AbstractContextManager[RecordingTraceScope]:
        if self.fail_start_trace:
            raise RuntimeError("trace start failed")

        trace = RecordingTraceScope(attributes=attributes)
        self.traces.append(trace)

        client = self

        class Manager(AbstractContextManager[RecordingTraceScope]):
            def __enter__(self) -> RecordingTraceScope:
                return trace

            def __exit__(
                self,
                exc_type: type[BaseException] | None,
                exc: BaseException | None,
                traceback: TracebackType | None,
            ) -> Literal[False]:
                del exc_type, exc, traceback
                if client.fail_exit:
                    raise RuntimeError("trace exit failed")
                trace.closed = True
                return False

        return Manager()

    def start_observation(
        self,
        attributes: ObservationAttributes,
    ) -> AbstractContextManager[RecordingObservationScope]:
        del attributes
        raise AssertionError("processor must start observations under the AgentRun trace")

    def record_event(self, event: EventObservation) -> None:
        del event

    def record_trace_event(self, *, identity: object, event: EventObservation) -> None:
        del identity, event

    def flush(self) -> None:
        self.flush_calls += 1

    def shutdown(self) -> None:
        self.shutdown_calls += 1


class ContextTrackingObservabilityClient(RecordingObservabilityClient):
    """Recording client that also installs ContextVars like production adapters."""

    def start_trace(
        self,
        attributes: TraceAttributes,
    ) -> AbstractContextManager[RecordingTraceScope]:
        from supportops.observability.context import (
            ActiveObservationContext,
            ActiveTraceContext,
            observation_context_scope,
            trace_context_scope,
        )

        if self.fail_start_trace:
            raise RuntimeError("trace start failed")

        trace = RecordingTraceScope(attributes=attributes)
        self.traces.append(trace)
        client = self

        class TraceManager(AbstractContextManager[RecordingTraceScope]):
            def __init__(self) -> None:
                self._trace_context = trace_context_scope(
                    ActiveTraceContext(
                        trace_seed=attributes.trace_seed,
                        session_id=attributes.session_id,
                    )
                )
                self._observation_manager: (
                    AbstractContextManager[RecordingObservationScope] | None
                ) = None

            def __enter__(self) -> RecordingTraceScope:
                self._trace_context.__enter__()

                original_start = trace.start_observation

                def start_observation(
                    observation_attributes: ObservationAttributes,
                ) -> AbstractContextManager[RecordingObservationScope]:
                    if client.fail_start_observation:
                        raise RuntimeError("observation start failed")

                    observation = RecordingObservationScope(
                        attributes=observation_attributes,
                    )
                    trace.observations.append(observation)

                    class ObservationManager(
                        AbstractContextManager[RecordingObservationScope],
                    ):
                        def __init__(self) -> None:
                            self._observation_context = observation_context_scope(
                                ActiveObservationContext(
                                    name=observation_attributes.name,
                                    observation_id=observation.observation_id,
                                )
                            )

                        def __enter__(self) -> RecordingObservationScope:
                            self._observation_context.__enter__()

                            original_update = observation.update

                            def update(update: ObservationUpdate) -> None:
                                if client.fail_update:
                                    raise RuntimeError("observation update failed")
                                original_update(update)

                            observation.update = update  # type: ignore[method-assign]
                            return observation

                        def __exit__(
                            self,
                            exc_type: type[BaseException] | None,
                            exc: BaseException | None,
                            traceback: TracebackType | None,
                        ) -> Literal[False]:
                            self._observation_context.__exit__(
                                exc_type,
                                exc,
                                traceback,
                            )
                            if client.fail_exit:
                                raise RuntimeError("observation exit failed")
                            observation.closed = True
                            return False

                    return ObservationManager()

                trace.start_observation = start_observation  # type: ignore[assignment]

                original_update = trace.update

                def update(update: ObservationUpdate) -> None:
                    if client.fail_update:
                        raise RuntimeError("trace update failed")
                    original_update(update)

                trace.update = update  # type: ignore[method-assign]
                del original_start
                return trace

            def __exit__(
                self,
                exc_type: type[BaseException] | None,
                exc: BaseException | None,
                traceback: TracebackType | None,
            ) -> Literal[False]:
                self._trace_context.__exit__(exc_type, exc, traceback)
                if client.fail_exit:
                    raise RuntimeError("trace exit failed")
                trace.closed = True
                return False

        return TraceManager()


def _create_traced_processor(
    *,
    executor: object,
    observability_client: RecordingObservabilityClient,
    transition_result: AgentRunTransitionResult = AgentRunTransitionResult.APPLIED,
    ticket: Ticket | None = None,
    missing_ticket: bool = False,
    execution_timeout_seconds: float = 30.0,
) -> tuple[
    ProcessClaimedAgentRun,
    AsyncMock,
    RecordingTransactionManager,
]:
    ticket_repository = AsyncMock()
    if missing_ticket:
        ticket_repository.get.return_value = None
    elif ticket is None:
        ticket_repository.get.return_value = create_ticket()
    else:
        ticket_repository.get.return_value = ticket

    agent_run_repository = AsyncMock()
    agent_run_repository.mark_succeeded.return_value = transition_result
    agent_run_repository.mark_waiting_for_approval.return_value = transition_result
    agent_run_repository.record_failure.return_value = transition_result
    transaction_manager = RecordingTransactionManager()

    processor = ProcessClaimedAgentRun(
        ticket_repository=ticket_repository,
        agent_run_repository=agent_run_repository,
        transaction_manager=transaction_manager,
        executor=executor,  # type: ignore[arg-type]
        retry_policy=AgentRunRetryPolicy(
            base_delay_seconds=2.0,
            maximum_delay_seconds=60.0,
        ),
        execution_timeout_seconds=execution_timeout_seconds,
        utc_now=lambda: _FINISHED_AT,
        observability_client=observability_client,
    )
    return processor, agent_run_repository, transaction_manager


def _assert_safe_attempt_metadata(metadata: Mapping[str, object]) -> None:
    assert "lease_token" not in metadata
    assert "execution_grant" not in metadata
    assert "ticket_subject" not in metadata
    assert "ticket_description" not in metadata
    assert "Unable to access billing" not in str(metadata)
    assert "password" not in str(metadata).lower()


async def test_claimed_attempt_creates_one_agent_run_trace_and_worker_attempt() -> None:
    observability = ContextTrackingObservabilityClient()
    transaction_manager = RecordingTransactionManager()
    executor = RecordingExecutor(transaction_manager=transaction_manager)
    processor, _, _ = _create_traced_processor(
        executor=executor,
        observability_client=observability,
    )
    claim = create_claim()

    result = await processor.execute(claim)

    assert result is AgentRunTransitionResult.APPLIED
    assert len(observability.traces) == 1
    trace = observability.traces[0]
    assert trace.attributes.trace_seed == f"agent-run:{_RUN_ID}"
    assert trace.attributes.session_id == f"ticket:{_TICKET_ID}"
    assert trace.attributes.name == "agent-run"
    assert trace.attributes.tags == ("supportops", "agent-run")
    assert len(trace.observations) == 1

    attempt = trace.observations[0]
    assert attempt.attributes.name == "worker-attempt"
    assert attempt.attributes.observation_type is ObservationType.SPAN
    assert attempt.attributes.input_paths == frozenset()
    assert attempt.attributes.output_paths == frozenset()

    metadata = dict(attempt.attributes.metadata)
    assert metadata["agent_run_id"] == str(_RUN_ID)
    assert metadata["agent_run_attempt_id"] == str(claim.attempt.id)
    assert metadata["attempt_number"] == 1
    assert metadata["execution_request_id"] == str(claim.attempt.execution_request_id)
    assert metadata["workspace_id"] == str(_WORKSPACE_ID)
    assert metadata["ticket_id"] == str(_TICKET_ID)
    assert metadata["workflow_name"] == claim.agent_run.workflow_name
    assert metadata["workflow_version"] == claim.agent_run.workflow_version
    assert metadata["trigger_key"] == claim.agent_run.trigger_key
    assert metadata["correlation_id"] == str(claim.agent_run.correlation_id)
    assert metadata["worker_id"] == "worker-a"
    _assert_safe_attempt_metadata(metadata)
    _assert_safe_attempt_metadata(dict(trace.attributes.metadata))

    assert attempt.updates[-1].status is ObservationStatus.OK
    assert attempt.updates[-1].metadata["attempt_outcome"] == "succeeded"
    assert trace.updates[-1].status is ObservationStatus.OK
    assert trace.updates[-1].metadata["agent_run_status"] == "succeeded"
    assert attempt.closed is True
    assert trace.closed is True
    assert current_trace_context() is None
    assert current_observation_context() is None
    assert observability.flush_calls == 0


async def test_retryable_failure_updates_attempt_and_trace_as_error() -> None:
    observability = RecordingObservabilityClient()
    transaction_manager = RecordingTransactionManager()
    executor = RecordingExecutor(
        transaction_manager=transaction_manager,
        error=RetryableAgentRunExecutionError(
            error_code="provider_unavailable",
            error_summary="The processing provider is temporarily unavailable.",
        ),
    )
    processor, _, _ = _create_traced_processor(
        executor=executor,
        observability_client=observability,
    )

    await processor.execute(create_claim())

    trace = observability.traces[0]
    attempt = trace.observations[0]
    assert attempt.updates[-1].status is ObservationStatus.ERROR
    assert attempt.updates[-1].metadata["attempt_outcome"] == "retryable_failure"
    assert attempt.updates[-1].error_code == "provider_unavailable"
    assert trace.updates[-1].status is ObservationStatus.ERROR
    assert trace.updates[-1].metadata["agent_run_status"] == "retry_scheduled"
    assert trace.updates[-1].metadata["latest_attempt_outcome"] == "retryable_failure"


async def test_terminal_failure_updates_attempt_and_trace_as_error() -> None:
    observability = RecordingObservabilityClient()
    transaction_manager = RecordingTransactionManager()
    executor = RecordingExecutor(
        transaction_manager=transaction_manager,
        error=TerminalAgentRunExecutionError(
            error_code="unsupported_workflow",
            error_summary="The AgentRun workflow is not supported by the executor.",
        ),
    )
    processor, _, _ = _create_traced_processor(
        executor=executor,
        observability_client=observability,
    )

    await processor.execute(create_claim())

    trace = observability.traces[0]
    attempt = trace.observations[0]
    assert attempt.updates[-1].status is ObservationStatus.ERROR
    assert attempt.updates[-1].metadata["attempt_outcome"] == "terminal_failure"
    assert attempt.updates[-1].error_code == "unsupported_workflow"
    assert trace.updates[-1].status is ObservationStatus.ERROR
    assert trace.updates[-1].metadata["agent_run_status"] == "failed"


async def test_timeout_maps_to_timed_out_outcome() -> None:
    observability = RecordingObservabilityClient()
    processor, _, _ = _create_traced_processor(
        executor=BlockingExecutor(),
        observability_client=observability,
        execution_timeout_seconds=0.001,
    )

    await processor.execute(create_claim())

    attempt = observability.traces[0].observations[0]
    trace = observability.traces[0]
    assert attempt.updates[-1].metadata["attempt_outcome"] == "timed_out"
    assert attempt.updates[-1].error_code == "executor_timeout"
    assert attempt.updates[-1].status is ObservationStatus.ERROR
    assert trace.updates[-1].metadata["agent_run_status"] == "retry_scheduled"
    assert trace.updates[-1].metadata["latest_attempt_outcome"] == "timed_out"


async def test_pause_maps_to_ok_awaiting_approval() -> None:
    observability = RecordingObservabilityClient()
    executor = PausingExecutor(
        approval_request_id=UUID("aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee"),
        graph_thread_id="controlled-support:thread",
    )
    processor, _, _ = _create_traced_processor(
        executor=executor,
        observability_client=observability,
    )

    await processor.execute(create_claim())

    attempt = observability.traces[0].observations[0]
    trace = observability.traces[0]
    assert attempt.updates[-1].status is ObservationStatus.OK
    assert attempt.updates[-1].metadata["attempt_outcome"] == "awaiting_approval"
    assert trace.updates[-1].status is ObservationStatus.OK
    assert trace.updates[-1].metadata["agent_run_status"] == "waiting_for_approval"


async def test_lease_lost_finalization_maps_to_error_without_model_status() -> None:
    observability = RecordingObservabilityClient()
    transaction_manager = RecordingTransactionManager()
    executor = RecordingExecutor(transaction_manager=transaction_manager)
    processor, _, _ = _create_traced_processor(
        executor=executor,
        observability_client=observability,
        transition_result=AgentRunTransitionResult.LEASE_LOST,
    )

    result = await processor.execute(create_claim())

    assert result is AgentRunTransitionResult.LEASE_LOST
    attempt = observability.traces[0].observations[0]
    trace = observability.traces[0]
    assert attempt.updates[-1].status is ObservationStatus.ERROR
    assert attempt.updates[-1].metadata["attempt_outcome"] == "lease_lost"
    assert attempt.updates[-1].error_code == "lease_lost"
    assert trace.updates[-1].status is ObservationStatus.ERROR
    assert trace.updates[-1].metadata["latest_attempt_outcome"] == "lease_lost"
    assert "agent_run_status" not in trace.updates[-1].metadata
    assert "lease_lost" not in {status.value for status in AgentRunStatus}
    assert "lease_lost" not in {outcome.value for outcome in AgentRunAttemptOutcome}


async def test_retry_and_resume_reuse_same_agent_run_trace_seed() -> None:
    observability = RecordingObservabilityClient()
    transaction_manager = RecordingTransactionManager()
    processor, _, _ = _create_traced_processor(
        executor=RecordingExecutor(transaction_manager=transaction_manager),
        observability_client=observability,
    )

    await processor.execute(create_claim(attempt_count=1))
    await processor.execute(create_claim(attempt_count=2))

    assert len(observability.traces) == 2
    assert observability.traces[0].attributes.trace_seed == f"agent-run:{_RUN_ID}"
    assert observability.traces[1].attributes.trace_seed == f"agent-run:{_RUN_ID}"


async def test_different_agent_runs_use_different_trace_seeds() -> None:
    observability = RecordingObservabilityClient()
    transaction_manager = RecordingTransactionManager()
    processor, _, _ = _create_traced_processor(
        executor=RecordingExecutor(transaction_manager=transaction_manager),
        observability_client=observability,
    )
    other_run_id = UUID("aaaaaaaa-bbbb-4ccc-8ddd-111111111111")

    claim = create_claim()
    other_claim = create_claim()
    other_claim = AgentRunClaim(
        agent_run=replace(other_claim.agent_run, id=other_run_id),
        attempt=replace(other_claim.attempt, agent_run_id=other_run_id),
    )

    await processor.execute(claim)
    await processor.execute(other_claim)

    assert observability.traces[0].attributes.trace_seed == f"agent-run:{_RUN_ID}"
    assert observability.traces[1].attributes.trace_seed == f"agent-run:{other_run_id}"


async def test_telemetry_failures_preserve_business_result() -> None:
    observability = ContextTrackingObservabilityClient(
        fail_start_trace=True,
    )
    transaction_manager = RecordingTransactionManager()
    processor, agent_run_repository, _ = _create_traced_processor(
        executor=RecordingExecutor(transaction_manager=transaction_manager),
        observability_client=observability,
    )

    result = await processor.execute(create_claim())

    assert result is AgentRunTransitionResult.APPLIED
    agent_run_repository.mark_succeeded.assert_awaited_once()
    assert observability.traces == []


async def test_observation_update_failure_preserves_business_result() -> None:
    observability = ContextTrackingObservabilityClient(fail_update=True)
    transaction_manager = RecordingTransactionManager()
    processor, agent_run_repository, _ = _create_traced_processor(
        executor=RecordingExecutor(transaction_manager=transaction_manager),
        observability_client=observability,
    )

    result = await processor.execute(create_claim())

    assert result is AgentRunTransitionResult.APPLIED
    agent_run_repository.mark_succeeded.assert_awaited_once()
    assert observability.traces[0].closed is True


async def test_unknown_execution_result_preserves_exception_identity() -> None:
    observability = ContextTrackingObservabilityClient()
    processor, _, _ = _create_traced_processor(
        executor=UnknownResultExecutor(),
        observability_client=observability,
    )

    with pytest.raises(RuntimeError, match="unsupported execution result") as raised:
        await processor.execute(create_claim())

    assert type(raised.value) is RuntimeError
    assert current_trace_context() is None
    assert current_observation_context() is None
    assert observability.traces[0].closed is True
    assert observability.traces[0].observations[0].closed is True


async def test_trace_closes_after_authoritative_persistence() -> None:
    transaction_manager = RecordingTransactionManager()
    persistence_order: list[str] = []

    agent_run_repository = AsyncMock()

    async def mark_succeeded(command: object) -> AgentRunTransitionResult:
        del command
        persistence_order.append("persist")
        return AgentRunTransitionResult.APPLIED

    agent_run_repository.mark_succeeded.side_effect = mark_succeeded
    ticket_repository = AsyncMock()
    ticket_repository.get.return_value = create_ticket()

    class OrderedClient(RecordingObservabilityClient):
        def start_trace(
            self,
            attributes: TraceAttributes,
        ) -> AbstractContextManager[RecordingTraceScope]:
            manager = super().start_trace(attributes)
            trace = self.traces[-1]
            original_update = trace.update

            def update(update: ObservationUpdate) -> None:
                persistence_order.append("telemetry_update")
                original_update(update)

            trace.update = update  # type: ignore[method-assign]
            return manager

    client = OrderedClient()
    processor = ProcessClaimedAgentRun(
        ticket_repository=ticket_repository,
        agent_run_repository=agent_run_repository,
        transaction_manager=transaction_manager,
        executor=RecordingExecutor(transaction_manager=transaction_manager),
        retry_policy=AgentRunRetryPolicy(
            base_delay_seconds=2.0,
            maximum_delay_seconds=60.0,
        ),
        execution_timeout_seconds=30.0,
        utc_now=lambda: _FINISHED_AT,
        observability_client=client,
    )

    await processor.execute(create_claim())

    assert persistence_order[0] == "persist"
    assert "telemetry_update" in persistence_order
    assert persistence_order.index("persist") < persistence_order.index(
        "telemetry_update",
    )
    assert client.traces[0].closed is True
    assert client.traces[0].observations[0].closed is True
