"""PostgreSQL repositories for durable ticket classification."""

from collections.abc import Sequence
from datetime import datetime
from typing import cast
from uuid import UUID

from sqlalchemy import and_, literal, select, tuple_
from sqlalchemy.ext.asyncio import AsyncSession

from supportops.ai.schemas.ticket_classification import (
    TicketClassificationSchemaVersion,
)
from supportops.modules.agent_runs.domain.models import AgentRunStatus
from supportops.modules.agent_runs.infrastructure.models import (
    AgentRunAttemptRecord,
    AgentRunRecord,
)
from supportops.modules.ticket_classifications.application.persistence import (
    ClassificationPersistenceResult,
    PersistClassificationExecutionCommand,
)
from supportops.modules.ticket_classifications.domain.inspection import (
    AgentRunClassificationReference,
    LLMInvocationInspection,
)
from supportops.modules.ticket_classifications.domain.models import (
    LLMInvocation,
    TicketClassification,
)
from supportops.modules.ticket_classifications.domain.repositories import (
    TicketClassificationQueryRepository,
)
from supportops.modules.ticket_classifications.infrastructure.models import (
    LLMInvocationRecord,
    TicketClassificationRecord,
)


class SqlAlchemyTicketClassificationQueryRepository(
    TicketClassificationQueryRepository,
):
    """Read classification state through explicit workspace boundaries."""

    def __init__(
        self,
        session: AsyncSession,
    ) -> None:
        self._session = session

    async def get(
        self,
        *,
        workspace_id: UUID,
        classification_id: UUID,
    ) -> TicketClassification | None:
        """Return one classification only through its workspace boundary."""

        statement = select(
            TicketClassificationRecord,
        ).where(
            TicketClassificationRecord.workspace_id == workspace_id,
            TicketClassificationRecord.id == classification_id,
        )
        result = await self._session.execute(statement)
        record = result.scalar_one_or_none()

        if record is None:
            return None

        return record.to_domain()

    async def get_by_agent_run_id(
        self,
        *,
        workspace_id: UUID,
        agent_run_id: UUID,
    ) -> TicketClassification | None:
        """Return one accepted classification for a scoped AgentRun."""

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

    async def get_reference_by_agent_run_id(
        self,
        *,
        workspace_id: UUID,
        agent_run_id: UUID,
    ) -> AgentRunClassificationReference | None:
        """Return a lightweight classification reference for one AgentRun."""

        statement = select(
            TicketClassificationRecord.id,
            TicketClassificationRecord.schema_version,
            TicketClassificationRecord.created_at,
        ).where(
            TicketClassificationRecord.workspace_id == workspace_id,
            TicketClassificationRecord.agent_run_id == agent_run_id,
        )
        result = await self._session.execute(statement)
        row = result.one_or_none()

        if row is None:
            return None

        return AgentRunClassificationReference(
            id=row[0],
            schema_version=cast(
                TicketClassificationSchemaVersion,
                row[1],
            ),
            created_at=row[2],
        )

    async def list_by_ticket(
        self,
        *,
        workspace_id: UUID,
        ticket_id: UUID,
        limit: int,
        after_created_at: datetime | None = None,
        after_classification_id: UUID | None = None,
    ) -> Sequence[TicketClassification]:
        """List ticket classifications in deterministic descending order."""

        if limit < 1:
            raise ValueError(
                "Ticket classification list limit must be positive.",
            )

        if (after_created_at is None) != (after_classification_id is None):
            raise ValueError(
                "Ticket classification pagination position requires both timestamp and ID.",
            )

        statement = select(
            TicketClassificationRecord,
        ).where(
            TicketClassificationRecord.workspace_id == workspace_id,
            TicketClassificationRecord.ticket_id == ticket_id,
        )

        if after_created_at is not None and after_classification_id is not None:
            statement = statement.where(
                tuple_(
                    TicketClassificationRecord.created_at,
                    TicketClassificationRecord.id,
                )
                < tuple_(
                    literal(after_created_at),
                    literal(after_classification_id),
                ),
            )

        statement = statement.order_by(
            TicketClassificationRecord.created_at.desc(),
            TicketClassificationRecord.id.desc(),
        ).limit(limit)

        result = await self._session.execute(statement)

        return tuple(record.to_domain() for record in result.scalars().all())

    async def list_invocations_by_agent_run(
        self,
        *,
        workspace_id: UUID,
        agent_run_id: UUID,
    ) -> Sequence[LLMInvocationInspection]:
        """List safe invocation projections in attempt and sequence order."""

        statement = (
            select(
                LLMInvocationRecord,
                AgentRunAttemptRecord.attempt_number,
            )
            .join(
                AgentRunAttemptRecord,
                and_(
                    AgentRunAttemptRecord.id == LLMInvocationRecord.agent_run_attempt_id,
                    AgentRunAttemptRecord.agent_run_id == LLMInvocationRecord.agent_run_id,
                ),
            )
            .where(
                LLMInvocationRecord.workspace_id == workspace_id,
                LLMInvocationRecord.agent_run_id == agent_run_id,
            )
            .order_by(
                AgentRunAttemptRecord.attempt_number.asc(),
                LLMInvocationRecord.invocation_sequence.asc(),
            )
        )
        result = await self._session.execute(statement)

        return tuple(
            LLMInvocationInspection.from_domain(
                invocation=record.to_domain(),
                attempt_number=attempt_number,
            )
            for record, attempt_number in result.all()
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
