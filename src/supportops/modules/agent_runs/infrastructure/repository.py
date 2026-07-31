"""PostgreSQL repository for durable AgentRuns."""

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from supportops.modules.agent_runs.domain.claiming import (
    AgentRunClaim,
    ClaimAgentRunCommand,
)
from supportops.modules.agent_runs.domain.models import (
    AgentRun,
    AgentRunAttempt,
)
from supportops.modules.agent_runs.domain.repositories import (
    AgentRunRepository,
)
from supportops.modules.agent_runs.infrastructure.models import (
    AgentRunAttemptRecord,
    AgentRunRecord,
)


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
                AgentRunRecord.attempt_count < AgentRunRecord.max_attempts,
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
