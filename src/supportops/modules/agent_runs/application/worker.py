"""Application orchestration for one PostgreSQL worker cycle."""

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from uuid import UUID, uuid4

from supportops.core.transactions import TransactionManager
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
from supportops.modules.agent_runs.domain.recovery import (
    RecoverExpiredAgentRunCommand,
)
from supportops.modules.agent_runs.domain.repositories import (
    AgentRunRepository,
)
from supportops.modules.agent_runs.domain.transitions import (
    AgentRunTransitionResult,
)

UtcNowProvider = Callable[[], datetime]
UuidProvider = Callable[[], UUID]

_EXPIRED_LEASE_ERROR_CODE = "worker_lease_expired"
_EXPIRED_LEASE_ERROR_SUMMARY = "The worker lease expired before execution completed."


class WorkerCycleOutcome(StrEnum):
    """Observable result of one worker polling cycle."""

    IDLE = "idle"
    PROCESSED = "processed"
    LEASE_LOST = "lease_lost"


@dataclass(frozen=True, slots=True)
class WorkerCycleResult:
    """Result and maintenance activity produced by one worker cycle."""

    outcome: WorkerCycleOutcome
    recovered_expired_run: bool
    agent_run_id: UUID | None

    def __post_init__(self) -> None:
        if self.outcome is WorkerCycleOutcome.IDLE:
            if self.agent_run_id is not None:
                raise ValueError(
                    "An idle worker cycle must not reference an AgentRun.",
                )
            return

        if self.agent_run_id is None:
            raise ValueError(
                "A non-idle worker cycle must reference an AgentRun.",
            )


class RunAgentWorkerCycle:
    """Recover stale ownership and process at most one available AgentRun."""

    def __init__(
        self,
        *,
        worker_id: str,
        agent_run_repository: AgentRunRepository,
        transaction_manager: TransactionManager,
        processor: ProcessClaimedAgentRun,
        retry_policy: AgentRunRetryPolicy,
        lease_seconds: float,
        utc_now: UtcNowProvider | None = None,
        uuid_provider: UuidProvider | None = None,
    ) -> None:
        _validate_worker_id(worker_id)

        if lease_seconds <= 0:
            raise ValueError("lease_seconds must be greater than zero.")

        self._worker_id = worker_id
        self._agent_run_repository = agent_run_repository
        self._transaction_manager = transaction_manager
        self._processor = processor
        self._retry_policy = retry_policy
        self._lease_seconds = lease_seconds
        self._utc_now = utc_now or _utc_now
        self._uuid_provider = uuid_provider or uuid4

    async def execute(self) -> WorkerCycleResult:
        """Run one recovery, claim, and processing cycle."""

        recovered_expired_run = await self._recover_one_expired_run()
        claim = await self._claim_one_available_run()

        if claim is None:
            return WorkerCycleResult(
                outcome=WorkerCycleOutcome.IDLE,
                recovered_expired_run=recovered_expired_run,
                agent_run_id=None,
            )

        transition_result = await self._processor.execute(claim)

        outcome = (
            WorkerCycleOutcome.PROCESSED
            if transition_result is AgentRunTransitionResult.APPLIED
            else WorkerCycleOutcome.LEASE_LOST
        )

        return WorkerCycleResult(
            outcome=outcome,
            recovered_expired_run=recovered_expired_run,
            agent_run_id=claim.agent_run.id,
        )

    async def _recover_one_expired_run(self) -> bool:
        recovered_at = self._utc_now()

        retry_delay_seconds = self._retry_policy.delay_seconds_for_attempt(
            attempt_number=1,
        )

        async with self._transaction_manager.transaction():
            result = await self._agent_run_repository.recover_next_expired(
                RecoverExpiredAgentRunCommand(
                    recovered_at=recovered_at,
                    retry_delay_seconds=retry_delay_seconds,
                    error_code=_EXPIRED_LEASE_ERROR_CODE,
                    error_summary=_EXPIRED_LEASE_ERROR_SUMMARY,
                ),
            )

        return result is not None

    async def _claim_one_available_run(self) -> AgentRunClaim | None:
        claimed_at = self._utc_now()

        command = ClaimAgentRunCommand(
            worker_id=self._worker_id,
            lease_token=self._uuid_provider(),
            execution_request_id=self._uuid_provider(),
            claimed_at=claimed_at,
            lease_expires_at=claimed_at + timedelta(seconds=self._lease_seconds),
        )

        async with self._transaction_manager.transaction():
            return await self._agent_run_repository.claim_next_available(
                command,
            )


def _validate_worker_id(worker_id: str) -> None:
    if not worker_id:
        raise ValueError("worker_id is required.")

    if worker_id != worker_id.strip():
        raise ValueError(
            "worker_id must not contain surrounding whitespace.",
        )


def _utc_now() -> datetime:
    return datetime.now(UTC)
