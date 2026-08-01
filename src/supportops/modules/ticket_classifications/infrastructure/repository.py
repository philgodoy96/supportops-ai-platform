"""PostgreSQL repositories for durable ticket classification."""

from collections.abc import Sequence
from uuid import UUID

from sqlalchemy import and_, select
from sqlalchemy.ext.asyncio import AsyncSession

from supportops.modules.agent_runs.domain.models import AgentRunStatus
from supportops.modules.agent_runs.infrastructure.models import (
    AgentRunAttemptRecord,
    AgentRunRecord,
)
from supportops.modules.ticket_classifications.application.persistence import (
    ClassificationPersistenceResult,
    PersistClassificationExecutionCommand,
)
from supportops.modules.ticket_classifications.domain.models import (
    LLMInvocation,
    TicketClassification,
)
from supportops.modules.ticket_classifications.infrastructure.models import (
    LLMInvocationRecord,
    TicketClassificationRecord,
)


class SqlAlchemyClassificationPersistenceRepository:
    """Persist classification aggregates through an active transaction."""

    def __init__(
        self,
        session: AsyncSession,
    ) -> None:
        self._session = session

    async def add(
        self,
        classification: TicketClassification,
    ) -> None:
        """Add and flush one accepted classification."""

        self._session.add(
            TicketClassificationRecord.from_domain(
                classification,
            ),
        )
        await self._session.flush()

    async def get_by_agent_run_id(
        self,
        *,
        workspace_id: UUID,
        agent_run_id: UUID,
    ) -> TicketClassification | None:
        """Return one workspace-scoped classification for an AgentRun."""

        statement = select(
            TicketClassificationRecord,
        ).where(
            TicketClassificationRecord.workspace_id == workspace_id,
            TicketClassificationRecord.agent_run_id == agent_run_id,
        )
        result = await self._session.execute(statement)
        record = result.scalar_one_or_none()

        if record is None:
            return None

        return record.to_domain()

    async def add_many(
        self,
        invocations: Sequence[LLMInvocation],
    ) -> None:
        """Add and flush logical invocation records."""

        if not invocations:
            return

        self._session.add_all(
            [
                LLMInvocationRecord.from_domain(
                    invocation,
                )
                for invocation in invocations
            ],
        )
        await self._session.flush()

    async def persist_fenced(
        self,
        command: PersistClassificationExecutionCommand,
    ) -> ClassificationPersistenceResult:
        """Persist one result only while the execution lease remains valid."""

        if not await self._lock_active_run(command):
            return ClassificationPersistenceResult.LEASE_LOST

        if not await self._lock_active_attempt(command):
            return ClassificationPersistenceResult.LEASE_LOST

        existing_classification = await self._load_classification_for_run(
            workspace_id=command.workspace_id,
            agent_run_id=command.agent_run_id,
        )
        if existing_classification is not None:
            return ClassificationPersistenceResult.ALREADY_CLASSIFIED

        existing_invocations = await self._load_existing_invocations(
            command,
        )

        missing_invocations: list[LLMInvocation] = []

        for invocation in command.invocations:
            existing = existing_invocations.get(
                invocation.invocation_sequence,
            )

            if existing is None:
                missing_invocations.append(invocation)
                continue

            if existing.to_domain() != invocation:
                raise RuntimeError(
                    "An invocation sequence is already persisted with different invocation data.",
                )

        if missing_invocations:
            self._session.add_all(
                [
                    LLMInvocationRecord.from_domain(
                        invocation,
                    )
                    for invocation in missing_invocations
                ],
            )
            await self._session.flush()

        if command.classification is not None:
            self._session.add(
                TicketClassificationRecord.from_domain(
                    command.classification,
                ),
            )
            await self._session.flush()

            return ClassificationPersistenceResult.APPLIED

        if missing_invocations:
            return ClassificationPersistenceResult.APPLIED

        return ClassificationPersistenceResult.ALREADY_RECORDED

    async def _lock_active_run(
        self,
        command: PersistClassificationExecutionCommand,
    ) -> bool:
        statement = (
            select(AgentRunRecord.id)
            .where(
                and_(
                    AgentRunRecord.id == command.agent_run_id,
                    AgentRunRecord.workspace_id == command.workspace_id,
                    AgentRunRecord.ticket_id == command.ticket_id,
                    AgentRunRecord.status == AgentRunStatus.RUNNING.value,
                    AgentRunRecord.lease_token == command.lease_token,
                    AgentRunRecord.lease_expires_at > command.persisted_at,
                ),
            )
            .with_for_update()
        )
        result = await self._session.execute(statement)

        return result.scalar_one_or_none() is not None

    async def _lock_active_attempt(
        self,
        command: PersistClassificationExecutionCommand,
    ) -> bool:
        statement = (
            select(AgentRunAttemptRecord.id)
            .where(
                and_(
                    AgentRunAttemptRecord.id == command.agent_run_attempt_id,
                    AgentRunAttemptRecord.agent_run_id == command.agent_run_id,
                    AgentRunAttemptRecord.lease_token == command.lease_token,
                    AgentRunAttemptRecord.finished_at.is_(None),
                    AgentRunAttemptRecord.outcome.is_(None),
                ),
            )
            .with_for_update()
        )
        result = await self._session.execute(statement)

        return result.scalar_one_or_none() is not None

    async def _load_classification_for_run(
        self,
        *,
        workspace_id: UUID,
        agent_run_id: UUID,
    ) -> TicketClassificationRecord | None:
        statement = (
            select(TicketClassificationRecord)
            .where(
                TicketClassificationRecord.workspace_id == workspace_id,
                TicketClassificationRecord.agent_run_id == agent_run_id,
            )
            .with_for_update()
        )
        result = await self._session.execute(statement)

        return result.scalar_one_or_none()

    async def _load_existing_invocations(
        self,
        command: PersistClassificationExecutionCommand,
    ) -> dict[int, LLMInvocationRecord]:
        sequences = tuple(invocation.invocation_sequence for invocation in command.invocations)
        statement = (
            select(LLMInvocationRecord)
            .where(
                LLMInvocationRecord.agent_run_attempt_id == command.agent_run_attempt_id,
                LLMInvocationRecord.invocation_sequence.in_(
                    sequences,
                ),
            )
            .with_for_update()
        )
        result = await self._session.execute(statement)
        records = result.scalars().all()

        return {record.invocation_sequence: record for record in records}
