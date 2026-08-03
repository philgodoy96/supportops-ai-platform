"""PostgreSQL repository for durable AgentRuns."""

from collections.abc import Sequence
from datetime import timedelta
from uuid import UUID

from sqlalchemy import and_, select
from sqlalchemy.ext.asyncio import AsyncSession

from supportops.modules.agent_runs.domain.claiming import (
    AgentRunClaim,
    ClaimAgentRunCommand,
)
from supportops.modules.agent_runs.domain.models import (
    AgentRun,
    AgentRunAttempt,
    AgentRunAttemptOutcome,
    AgentRunStatus,
)
from supportops.modules.agent_runs.domain.recovery import (
    ExpiredAgentRunDisposition,
    RecoverExpiredAgentRunCommand,
    RecoverExpiredAgentRunResult,
    bounded_exponential_retry_delay_seconds,
)
from supportops.modules.agent_runs.domain.repositories import (
    AgentRunRepository,
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

_FAILURE_BUDGET_OUTCOMES = {
    AgentRunAttemptOutcome.RETRYABLE_FAILURE,
    AgentRunAttemptOutcome.TIMED_OUT,
}


class SqlAlchemyAgentRunRepository(AgentRunRepository):
    """Persist AgentRuns through an active SQLAlchemy transaction."""

    def __init__(
        self,
        session: AsyncSession,
    ) -> None:
        self._session = session

    async def add(
        self,
        agent_run: AgentRun,
    ) -> None:
        """Add and flush an AgentRun inside the active transaction."""

        self._session.add(
            AgentRunRecord.from_domain(agent_run),
        )
        await self._session.flush()

    async def get(
        self,
        *,
        workspace_id: UUID,
        agent_run_id: UUID,
    ) -> AgentRun | None:
        """Return one workspace-scoped AgentRun when it exists."""

        statement = select(AgentRunRecord).where(
            AgentRunRecord.id == agent_run_id,
            AgentRunRecord.workspace_id == workspace_id,
        )
        result = await self._session.execute(statement)
        record = result.scalar_one_or_none()
        if record is None:
            return None
        return record.to_domain()

    async def list_attempts(
        self,
        *,
        agent_run_id: UUID,
    ) -> Sequence[AgentRunAttempt]:
        """Return attempts ordered by attempt number ascending."""

        statement = (
            select(AgentRunAttemptRecord)
            .where(AgentRunAttemptRecord.agent_run_id == agent_run_id)
            .order_by(AgentRunAttemptRecord.attempt_number.asc())
        )
        result = await self._session.execute(statement)
        records = result.scalars().all()
        return tuple(record.to_domain() for record in records)

    async def claim_next_available(
        self,
        command: ClaimAgentRunCommand,
    ) -> AgentRunClaim | None:
        """Atomically claim the next eligible AgentRun, if available."""

        statement = (
            select(AgentRunRecord)
            .where(
                AgentRunRecord.status.in_(("queued", "retry_scheduled")),
                AgentRunRecord.available_at <= command.claimed_at,
                (AgentRunRecord.retryable_failure_count < AgentRunRecord.max_retryable_failures),
            )
            .order_by(
                AgentRunRecord.available_at.asc(),
                AgentRunRecord.created_at.asc(),
                AgentRunRecord.id.asc(),
            )
            .limit(1)
            .with_for_update(skip_locked=True)
        )
        result = await self._session.execute(statement)
        record = result.scalar_one_or_none()
        if record is None:
            return None

        attempt_number = record.attempt_count + 1
        record.status = "running"
        record.attempt_count = attempt_number
        record.lease_owner = command.worker_id
        record.lease_token = command.lease_token
        record.lease_expires_at = command.lease_expires_at
        if record.first_started_at is None:
            record.first_started_at = command.claimed_at
        record.updated_at = command.claimed_at

        attempt = AgentRunAttempt.start(
            agent_run_id=record.id,
            attempt_number=attempt_number,
            worker_id=command.worker_id,
            lease_token=command.lease_token,
            execution_request_id=command.execution_request_id,
            now=command.claimed_at,
        )
        self._session.add(AgentRunAttemptRecord.from_domain(attempt))
        await self._session.flush()

        return AgentRunClaim(
            agent_run=record.to_domain(),
            attempt=attempt,
        )

    async def mark_succeeded(
        self,
        command: CompleteAgentRunCommand,
    ) -> AgentRunTransitionResult:
        """Persist a successful fenced AgentRun transition."""

        run_statement = (
            select(AgentRunRecord)
            .where(
                and_(
                    AgentRunRecord.id == command.agent_run_id,
                    AgentRunRecord.status == AgentRunStatus.RUNNING.value,
                    AgentRunRecord.lease_token == command.lease_token,
                    AgentRunRecord.lease_expires_at > command.finished_at,
                )
            )
            .with_for_update()
        )
        run_result = await self._session.execute(run_statement)
        record = run_result.scalar_one_or_none()
        if record is None:
            return AgentRunTransitionResult.LEASE_LOST

        attempt = await self._load_active_attempt_for_lease(
            agent_run_id=command.agent_run_id,
            lease_token=command.lease_token,
        )
        if attempt is None:
            raise RuntimeError(
                "Active AgentRun attempt was not found for the current lease.",
            )

        attempt.finished_at = command.finished_at
        attempt.outcome = AgentRunAttemptOutcome.SUCCEEDED.value
        attempt.error_code = None
        attempt.error_summary = None

        record.status = AgentRunStatus.SUCCEEDED.value
        record.completed_at = command.finished_at
        record.lease_owner = None
        record.lease_token = None
        record.lease_expires_at = None
        record.last_error_code = None
        record.last_error_summary = None
        record.updated_at = command.finished_at

        await self._session.flush()
        return AgentRunTransitionResult.APPLIED

    async def mark_waiting_for_approval(
        self,
        command: WaitForApprovalAgentRunCommand,
    ) -> AgentRunTransitionResult:
        """Persist a fenced waiting-for-approval AgentRun transition."""

        run_statement = (
            select(AgentRunRecord)
            .where(
                and_(
                    AgentRunRecord.id == command.agent_run_id,
                    AgentRunRecord.status == AgentRunStatus.RUNNING.value,
                    AgentRunRecord.lease_token == command.lease_token,
                    AgentRunRecord.lease_expires_at > command.finished_at,
                )
            )
            .with_for_update()
        )
        run_result = await self._session.execute(run_statement)
        record = run_result.scalar_one_or_none()
        if record is None:
            return AgentRunTransitionResult.LEASE_LOST

        attempt = await self._load_active_attempt_for_lease(
            agent_run_id=command.agent_run_id,
            lease_token=command.lease_token,
        )
        if attempt is None:
            raise RuntimeError(
                "Active AgentRun attempt was not found for the current lease.",
            )

        if attempt.attempt_number != record.attempt_count:
            raise RuntimeError(
                "Active AgentRun attempt does not match the current attempt count.",
            )

        attempt.finished_at = command.finished_at
        attempt.outcome = AgentRunAttemptOutcome.AWAITING_APPROVAL.value
        attempt.error_code = None
        attempt.error_summary = None

        record.status = AgentRunStatus.WAITING_FOR_APPROVAL.value
        record.lease_owner = None
        record.lease_token = None
        record.lease_expires_at = None
        record.available_at = None
        record.completed_at = None
        record.last_error_code = None
        record.last_error_summary = None
        record.updated_at = command.finished_at

        await self._session.flush()
        return AgentRunTransitionResult.APPLIED

    async def requeue_waiting_for_approval(
        self,
        command: RequeueWaitingAgentRunCommand,
    ) -> AgentRunApprovalRequeueResult:
        """Requeue one AgentRun that is waiting for an approval decision."""

        run_statement = (
            select(AgentRunRecord)
            .where(
                and_(
                    AgentRunRecord.workspace_id == command.workspace_id,
                    AgentRunRecord.ticket_id == command.ticket_id,
                    AgentRunRecord.id == command.agent_run_id,
                )
            )
            .with_for_update()
        )
        run_result = await self._session.execute(run_statement)
        record = run_result.scalar_one_or_none()
        if record is None:
            return AgentRunApprovalRequeueResult.STATE_CONFLICT

        if (
            record.status != AgentRunStatus.WAITING_FOR_APPROVAL.value
            or record.lease_owner is not None
            or record.lease_token is not None
            or record.lease_expires_at is not None
            or record.completed_at is not None
            or record.last_error_code is not None
            or record.last_error_summary is not None
        ):
            return AgentRunApprovalRequeueResult.STATE_CONFLICT

        record.status = AgentRunStatus.QUEUED.value
        record.available_at = command.requeued_at
        record.updated_at = command.requeued_at

        await self._session.flush()
        return AgentRunApprovalRequeueResult.APPLIED

    async def record_failure(
        self,
        command: FailAgentRunCommand,
    ) -> AgentRunTransitionResult:
        """Persist a fenced AgentRun failure transition."""

        run_statement = (
            select(AgentRunRecord)
            .where(
                and_(
                    AgentRunRecord.id == command.agent_run_id,
                    AgentRunRecord.status == AgentRunStatus.RUNNING.value,
                    AgentRunRecord.lease_token == command.lease_token,
                    AgentRunRecord.lease_expires_at > command.finished_at,
                )
            )
            .with_for_update()
        )
        run_result = await self._session.execute(run_statement)
        record = run_result.scalar_one_or_none()
        if record is None:
            return AgentRunTransitionResult.LEASE_LOST

        attempt = await self._load_active_attempt_for_lease(
            agent_run_id=command.agent_run_id,
            lease_token=command.lease_token,
        )
        if attempt is None:
            raise RuntimeError(
                "Active AgentRun attempt was not found for the current lease.",
            )

        consumes_failure_budget = command.outcome in _FAILURE_BUDGET_OUTCOMES
        if consumes_failure_budget:
            prospective_failure_count = record.retryable_failure_count + 1
        else:
            prospective_failure_count = record.retryable_failure_count

        expected_disposition = _expected_failure_disposition(
            consumes_failure_budget=consumes_failure_budget,
            prospective_failure_count=prospective_failure_count,
            max_retryable_failures=record.max_retryable_failures,
        )
        if command.disposition is not expected_disposition:
            raise RuntimeError(
                "FailAgentRunCommand disposition does not match the "
                "prospective retryable failure budget.",
            )

        attempt.finished_at = command.finished_at
        attempt.outcome = command.outcome.value
        attempt.error_code = command.error_code
        attempt.error_summary = command.error_summary

        if consumes_failure_budget:
            record.retryable_failure_count = prospective_failure_count

        record.lease_owner = None
        record.lease_token = None
        record.lease_expires_at = None
        record.last_error_code = command.error_code
        record.last_error_summary = command.error_summary
        record.updated_at = command.finished_at

        if command.disposition is AgentRunFailureDisposition.RETRY_SCHEDULED:
            assert command.retry_available_at is not None
            record.status = AgentRunStatus.RETRY_SCHEDULED.value
            record.available_at = command.retry_available_at
            record.completed_at = None
        else:
            record.status = AgentRunStatus.FAILED.value
            record.completed_at = command.finished_at

        await self._session.flush()
        return AgentRunTransitionResult.APPLIED

    async def recover_next_expired(
        self,
        command: RecoverExpiredAgentRunCommand,
    ) -> RecoverExpiredAgentRunResult | None:
        """Atomically recover the next expired running AgentRun, if available."""

        statement = (
            select(AgentRunRecord)
            .where(
                AgentRunRecord.status == AgentRunStatus.RUNNING.value,
                AgentRunRecord.lease_expires_at.is_not(None),
                AgentRunRecord.lease_expires_at <= command.recovered_at,
            )
            .order_by(
                AgentRunRecord.lease_expires_at.asc(),
                AgentRunRecord.created_at.asc(),
                AgentRunRecord.id.asc(),
            )
            .limit(1)
            .with_for_update(skip_locked=True)
        )
        result = await self._session.execute(statement)
        record = result.scalar_one_or_none()
        if record is None:
            return None

        expired_lease_token = record.lease_token
        if expired_lease_token is None:
            raise RuntimeError(
                "Expired running AgentRun does not have a lease token.",
            )

        attempt = await self._load_active_attempt_for_lease(
            agent_run_id=record.id,
            lease_token=expired_lease_token,
        )
        if attempt is None:
            raise RuntimeError(
                "Active AgentRun attempt was not found for the expired lease.",
            )

        prospective_failure_count = record.retryable_failure_count + 1

        attempt.finished_at = command.recovered_at
        attempt.outcome = AgentRunAttemptOutcome.LEASE_EXPIRED.value
        attempt.error_code = command.error_code
        attempt.error_summary = command.error_summary

        record.retryable_failure_count = prospective_failure_count
        record.lease_owner = None
        record.lease_token = None
        record.lease_expires_at = None
        record.last_error_code = command.error_code
        record.last_error_summary = command.error_summary
        record.updated_at = command.recovered_at

        if prospective_failure_count < record.max_retryable_failures:
            retry_delay_seconds = bounded_exponential_retry_delay_seconds(
                failure_number=prospective_failure_count,
                base_delay_seconds=command.retry_base_delay_seconds,
                maximum_delay_seconds=(command.retry_maximum_delay_seconds),
            )
            record.status = AgentRunStatus.RETRY_SCHEDULED.value
            record.available_at = command.recovered_at + timedelta(
                seconds=retry_delay_seconds,
            )
            record.completed_at = None
            disposition = ExpiredAgentRunDisposition.RETRY_SCHEDULED
        else:
            record.status = AgentRunStatus.FAILED.value
            record.completed_at = command.recovered_at
            disposition = ExpiredAgentRunDisposition.FAILED

        await self._session.flush()
        return RecoverExpiredAgentRunResult(
            agent_run=record.to_domain(),
            expired_lease_token=expired_lease_token,
            disposition=disposition,
        )

    async def _load_active_attempt_for_lease(
        self,
        *,
        agent_run_id: UUID,
        lease_token: UUID,
    ) -> AgentRunAttemptRecord | None:
        """Load the unfinished attempt that matches the current lease."""

        statement = (
            select(AgentRunAttemptRecord)
            .where(
                and_(
                    AgentRunAttemptRecord.agent_run_id == agent_run_id,
                    AgentRunAttemptRecord.lease_token == lease_token,
                    AgentRunAttemptRecord.finished_at.is_(None),
                    AgentRunAttemptRecord.outcome.is_(None),
                )
            )
            .with_for_update()
        )
        result = await self._session.execute(statement)
        return result.scalar_one_or_none()


def _expected_failure_disposition(
    *,
    consumes_failure_budget: bool,
    prospective_failure_count: int,
    max_retryable_failures: int,
) -> AgentRunFailureDisposition:
    if not consumes_failure_budget:
        return AgentRunFailureDisposition.FAILED

    if prospective_failure_count < max_retryable_failures:
        return AgentRunFailureDisposition.RETRY_SCHEDULED

    return AgentRunFailureDisposition.FAILED
