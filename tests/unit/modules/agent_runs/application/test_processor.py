"""Unit tests for processing one claimed AgentRun."""

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock
from uuid import UUID

import pytest

from supportops.modules.agent_runs.application.execution import (
    AgentRunExecutionContext,
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
    ) -> None:
        assert self._transaction_manager.active_depth == 0
        self.contexts.append(context)

        if self._error is not None:
            raise self._error


class BlockingExecutor:
    """Block long enough for the processor timeout to expire."""

    async def execute(
        self,
        context: AgentRunExecutionContext,
    ) -> None:
        del context
        await asyncio.sleep(10)


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
    max_attempts: int = 3,
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
        max_attempts=max_attempts,
        now=_NOW - timedelta(minutes=1),
    )
    running_run = replace(
        initial_run,
        status=AgentRunStatus.RUNNING,
        attempt_count=attempt_count,
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
            attempt_count=3,
            max_attempts=3,
        ),
    )

    command = agent_run_repository.record_failure.await_args.args[0]
    assert command.outcome is AgentRunAttemptOutcome.RETRYABLE_FAILURE
    assert command.disposition is AgentRunFailureDisposition.FAILED
    assert command.retry_available_at is None


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
