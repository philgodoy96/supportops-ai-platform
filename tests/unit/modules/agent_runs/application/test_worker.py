"""Unit tests for one PostgreSQL worker cycle."""

from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock
from uuid import UUID

import pytest

from supportops.modules.agent_runs.application.retry_policy import (
    AgentRunRetryPolicy,
)
from supportops.modules.agent_runs.application.worker import (
    RunAgentWorkerCycle,
    WorkerCycleOutcome,
    WorkerCycleResult,
)
from supportops.modules.agent_runs.domain.claiming import AgentRunClaim
from supportops.modules.agent_runs.domain.models import (
    DETERMINISTIC_BASELINE_WORKFLOW_VERSION,
    AgentRun,
    AgentRunAttempt,
    AgentRunStatus,
)
from supportops.modules.agent_runs.domain.recovery import (
    ExpiredAgentRunDisposition,
    RecoverExpiredAgentRunResult,
)
from supportops.modules.agent_runs.domain.transitions import (
    AgentRunTransitionResult,
)
from supportops.modules.approvals.application.models import (
    ApprovalExpirationBatchResult,
    ExpirePendingApprovalRequestsCommand,
)

_NOW = datetime(
    2026,
    7,
    31,
    18,
    0,
    tzinfo=UTC,
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


class RecordingTransactionManager:
    """Record transaction boundaries used by one worker cycle."""

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


def create_claim() -> AgentRunClaim:
    """Create a deterministic claimed AgentRun."""

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
        now=_NOW - timedelta(minutes=1),
    )
    running_run = replace(
        initial_run,
        status=AgentRunStatus.RUNNING,
        attempt_count=1,
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
        attempt_number=1,
        worker_id="worker-a",
        lease_token=_LEASE_TOKEN,
        execution_request_id=_EXECUTION_REQUEST_ID,
        now=_NOW,
    )

    return AgentRunClaim(
        agent_run=running_run,
        attempt=attempt,
    )


def create_recovery_result() -> RecoverExpiredAgentRunResult:
    """Create a deterministic successful recovery result."""

    claim = create_claim()
    recovered_run = replace(
        claim.agent_run,
        status=AgentRunStatus.RETRY_SCHEDULED,
        available_at=_NOW + timedelta(seconds=2),
        lease_owner=None,
        lease_token=None,
        lease_expires_at=None,
        last_error_code="worker_lease_expired",
        last_error_summary=("The worker lease expired before execution completed."),
        updated_at=_NOW,
    )

    return RecoverExpiredAgentRunResult(
        agent_run=recovered_run,
        expired_lease_token=_LEASE_TOKEN,
        disposition=ExpiredAgentRunDisposition.RETRY_SCHEDULED,
    )


def uuid_values(*values: UUID) -> Iterator[UUID]:
    """Yield deterministic UUID values."""

    return iter(values)


def create_worker(
    *,
    recovery_result: RecoverExpiredAgentRunResult | None = None,
    claim: AgentRunClaim | None = None,
    transition_result: AgentRunTransitionResult = (AgentRunTransitionResult.APPLIED),
    approval_expiration_batch_size: int = 100,
) -> tuple[
    RunAgentWorkerCycle,
    AsyncMock,
    AsyncMock,
    AsyncMock,
    RecordingTransactionManager,
]:
    repository = AsyncMock()
    repository.recover_next_expired.return_value = recovery_result
    repository.claim_next_available.return_value = claim

    processor = AsyncMock()
    processor.execute.return_value = transition_result

    expire_pending_approvals = AsyncMock()
    expire_pending_approvals.execute.return_value = ApprovalExpirationBatchResult(
        approval_request_ids=(),
    )

    transaction_manager = RecordingTransactionManager()
    generated_values = uuid_values(
        _LEASE_TOKEN,
        _EXECUTION_REQUEST_ID,
    )

    worker = RunAgentWorkerCycle(
        worker_id="worker-a",
        agent_run_repository=repository,
        transaction_manager=transaction_manager,
        processor=processor,
        retry_policy=AgentRunRetryPolicy(
            base_delay_seconds=2.0,
            maximum_delay_seconds=60.0,
        ),
        lease_seconds=45.0,
        expire_pending_approvals=expire_pending_approvals,
        approval_expiration_batch_size=approval_expiration_batch_size,
        utc_now=lambda: _NOW,
        uuid_provider=lambda: next(generated_values),
    )

    return (
        worker,
        repository,
        processor,
        expire_pending_approvals,
        transaction_manager,
    )


async def test_idle_cycle_recovers_expires_then_attempts_claim() -> None:
    (
        worker,
        repository,
        processor,
        expire_pending_approvals,
        transaction_manager,
    ) = create_worker()

    result = await worker.execute()

    assert result == WorkerCycleResult(
        outcome=WorkerCycleOutcome.IDLE,
        recovered_expired_run=False,
        agent_run_id=None,
    )
    assert transaction_manager.entries == 2
    repository.recover_next_expired.assert_awaited_once()
    expire_pending_approvals.execute.assert_awaited_once()
    repository.claim_next_available.assert_awaited_once()
    assert (
        repository.recover_next_expired.await_args_list[0]
        and expire_pending_approvals.execute.await_args_list[0]
        and repository.claim_next_available.await_args_list[0]
    )
    assert repository.mock_calls[0][0] == "recover_next_expired"
    assert expire_pending_approvals.mock_calls[0][0] == "execute"
    assert repository.mock_calls[1][0] == "claim_next_available"
    processor.execute.assert_not_awaited()

    expiration_command = expire_pending_approvals.execute.await_args.args[0]
    assert expiration_command.now == _NOW
    assert expiration_command.batch_size == 100


async def test_cycle_reports_recovery_even_when_no_run_is_claimed() -> None:
    worker, _, processor, expire_pending_approvals, _ = create_worker(
        recovery_result=create_recovery_result(),
    )

    result = await worker.execute()

    assert result.outcome is WorkerCycleOutcome.IDLE
    assert result.recovered_expired_run is True
    assert result.agent_run_id is None
    expire_pending_approvals.execute.assert_awaited_once()
    processor.execute.assert_not_awaited()


async def test_cycle_claims_and_processes_one_run() -> None:
    claim = create_claim()
    (
        worker,
        repository,
        processor,
        expire_pending_approvals,
        transaction_manager,
    ) = create_worker(
        claim=claim,
    )

    result = await worker.execute()

    assert result == WorkerCycleResult(
        outcome=WorkerCycleOutcome.PROCESSED,
        recovered_expired_run=False,
        agent_run_id=_RUN_ID,
    )
    assert transaction_manager.entries == 2
    expire_pending_approvals.execute.assert_awaited_once()
    processor.execute.assert_awaited_once_with(claim)

    command = repository.claim_next_available.await_args.args[0]
    assert command.worker_id == "worker-a"
    assert command.lease_token == _LEASE_TOKEN
    assert command.execution_request_id == _EXECUTION_REQUEST_ID
    assert command.claimed_at == _NOW
    assert command.lease_expires_at == _NOW + timedelta(seconds=45)


async def test_cycle_reports_lease_loss_from_processor() -> None:
    worker, _, _, _, _ = create_worker(
        claim=create_claim(),
        transition_result=AgentRunTransitionResult.LEASE_LOST,
    )

    result = await worker.execute()

    assert result == WorkerCycleResult(
        outcome=WorkerCycleOutcome.LEASE_LOST,
        recovered_expired_run=False,
        agent_run_id=_RUN_ID,
    )


async def test_cycle_can_recover_and_process_in_same_iteration() -> None:
    (
        worker,
        _,
        processor,
        expire_pending_approvals,
        transaction_manager,
    ) = create_worker(
        recovery_result=create_recovery_result(),
        claim=create_claim(),
    )

    result = await worker.execute()

    assert result.outcome is WorkerCycleOutcome.PROCESSED
    assert result.recovered_expired_run is True
    assert result.agent_run_id == _RUN_ID
    assert transaction_manager.entries == 2
    expire_pending_approvals.execute.assert_awaited_once()
    processor.execute.assert_awaited_once()


async def test_recovery_uses_safe_error_and_policy_bounds() -> None:
    worker, repository, _, _, _ = create_worker()

    await worker.execute()

    command = repository.recover_next_expired.await_args.args[0]

    assert command.recovered_at == _NOW
    assert command.retry_base_delay_seconds == 2.0
    assert command.retry_maximum_delay_seconds == 60.0
    assert command.error_code == "worker_lease_expired"
    assert command.error_summary == ("The worker lease expired before execution completed.")


async def test_expiration_runs_before_claim_with_configured_batch_size() -> None:
    call_order: list[str] = []

    worker, repository, _, expire_pending_approvals, _ = create_worker(
        approval_expiration_batch_size=25,
    )

    async def track_recovery(command: object) -> None:
        del command
        call_order.append("recover")
        return None

    async def track_expiration(
        command: ExpirePendingApprovalRequestsCommand,
    ) -> ApprovalExpirationBatchResult:
        call_order.append("expire")
        assert command.batch_size == 25
        return ApprovalExpirationBatchResult(approval_request_ids=())

    async def track_claim(command: object) -> None:
        del command
        call_order.append("claim")
        return None

    repository.recover_next_expired.side_effect = track_recovery
    expire_pending_approvals.execute.side_effect = track_expiration
    repository.claim_next_available.side_effect = track_claim

    result = await worker.execute()

    assert result.outcome is WorkerCycleOutcome.IDLE
    assert call_order == ["recover", "expire", "claim"]


@pytest.mark.parametrize(
    "worker_id",
    [
        "",
        " worker-a",
        "worker-a ",
    ],
)
def test_worker_cycle_requires_valid_worker_id(
    worker_id: str,
) -> None:
    with pytest.raises(ValueError):
        RunAgentWorkerCycle(
            worker_id=worker_id,
            agent_run_repository=AsyncMock(),
            transaction_manager=RecordingTransactionManager(),
            processor=AsyncMock(),
            retry_policy=AgentRunRetryPolicy(
                base_delay_seconds=2.0,
                maximum_delay_seconds=60.0,
            ),
            lease_seconds=45.0,
            expire_pending_approvals=AsyncMock(),
            approval_expiration_batch_size=100,
        )


@pytest.mark.parametrize(
    "lease_seconds",
    [
        0.0,
        -1.0,
    ],
)
def test_worker_cycle_requires_positive_lease(
    lease_seconds: float,
) -> None:
    with pytest.raises(
        ValueError,
        match="lease_seconds must be greater than zero",
    ):
        RunAgentWorkerCycle(
            worker_id="worker-a",
            agent_run_repository=AsyncMock(),
            transaction_manager=RecordingTransactionManager(),
            processor=AsyncMock(),
            retry_policy=AgentRunRetryPolicy(
                base_delay_seconds=2.0,
                maximum_delay_seconds=60.0,
            ),
            lease_seconds=lease_seconds,
            expire_pending_approvals=AsyncMock(),
            approval_expiration_batch_size=100,
        )


def test_idle_result_rejects_agent_run_id() -> None:
    with pytest.raises(
        ValueError,
        match="idle worker cycle must not reference",
    ):
        WorkerCycleResult(
            outcome=WorkerCycleOutcome.IDLE,
            recovered_expired_run=False,
            agent_run_id=_RUN_ID,
        )


@pytest.mark.parametrize(
    "outcome",
    [
        WorkerCycleOutcome.PROCESSED,
        WorkerCycleOutcome.LEASE_LOST,
    ],
)
def test_non_idle_result_requires_agent_run_id(
    outcome: WorkerCycleOutcome,
) -> None:
    with pytest.raises(
        ValueError,
        match="non-idle worker cycle must reference",
    ):
        WorkerCycleResult(
            outcome=outcome,
            recovered_expired_run=False,
            agent_run_id=None,
        )
