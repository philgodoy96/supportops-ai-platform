"""PostgreSQL repository for fenced recommendation persistence."""

from sqlalchemy import and_, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from supportops.modules.agent_runs.domain.models import (
    AgentRunStatus,
)
from supportops.modules.agent_runs.infrastructure.models import (
    AgentRunAttemptRecord,
    AgentRunRecord,
)
from supportops.modules.support_recommendations.application.persistence import (
    PersistSupportRecommendationCommand,
    SupportRecommendationExecutionRepository,
    SupportRecommendationPersistenceResult,
)
from supportops.modules.support_recommendations.infrastructure.models import (
    SupportRecommendationCitationRecord,
    SupportRecommendationRecord,
)
from supportops.modules.ticket_classifications.domain.models import (
    LLMInvocation,
)
from supportops.modules.ticket_classifications.infrastructure.models import (
    LLMInvocationRecord,
    TicketClassificationRecord,
)


class SqlAlchemySupportRecommendationExecutionRepository(SupportRecommendationExecutionRepository):
    """Persist recommendation aggregates through an active transaction."""

    def __init__(
        self,
        session: AsyncSession,
    ) -> None:
        self._session = session

    async def persist_fenced(
        self,
        command: PersistSupportRecommendationCommand,
    ) -> SupportRecommendationPersistenceResult:
        """Persist invocations and an optional recommendation atomically."""

        if not await self._lock_active_run(command):
            return SupportRecommendationPersistenceResult.LEASE_LOST

        if not await self._lock_active_attempt(command):
            return SupportRecommendationPersistenceResult.LEASE_LOST

        existing_recommendation = await self._load_recommendation_for_run(
            command=command,
        )

        if existing_recommendation is not None:
            return SupportRecommendationPersistenceResult.ALREADY_RECOMMENDED

        if command.recommendation is not None and not await self._lock_accepted_classification(
            command=command,
        ):
            raise RuntimeError("The accepted classification is not persisted for this AgentRun.")

        missing_invocations = await self._resolve_missing_invocations(
            command=command,
        )

        if missing_invocations:
            self._session.add_all(
                [LLMInvocationRecord.from_domain(invocation) for invocation in missing_invocations]
            )
            await self._session.flush()

        if command.recommendation is None:
            if missing_invocations:
                return SupportRecommendationPersistenceResult.APPLIED

            return SupportRecommendationPersistenceResult.ALREADY_RECORDED

        self._session.add(SupportRecommendationRecord.from_domain(command.recommendation))
        await self._session.flush()

        if command.citations:
            self._session.add_all(
                [
                    SupportRecommendationCitationRecord.from_domain(citation)
                    for citation in command.citations
                ]
            )
            await self._session.flush()

        return SupportRecommendationPersistenceResult.APPLIED

    async def _lock_active_run(
        self,
        command: PersistSupportRecommendationCommand,
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
                )
            )
            .with_for_update()
        )
        result = await self._session.execute(statement)

        return result.scalar_one_or_none() is not None

    async def _lock_active_attempt(
        self,
        command: PersistSupportRecommendationCommand,
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
                )
            )
            .with_for_update()
        )
        result = await self._session.execute(statement)

        return result.scalar_one_or_none() is not None

    async def _load_recommendation_for_run(
        self,
        *,
        command: PersistSupportRecommendationCommand,
    ) -> SupportRecommendationRecord | None:
        statement = (
            select(SupportRecommendationRecord)
            .where(
                SupportRecommendationRecord.workspace_id == command.workspace_id,
                SupportRecommendationRecord.ticket_id == command.ticket_id,
                SupportRecommendationRecord.agent_run_id == command.agent_run_id,
            )
            .with_for_update()
        )
        result = await self._session.execute(statement)

        return result.scalar_one_or_none()

    async def _lock_accepted_classification(
        self,
        *,
        command: PersistSupportRecommendationCommand,
    ) -> bool:
        recommendation = command.recommendation

        if recommendation is None:
            return False

        statement = (
            select(TicketClassificationRecord.id)
            .where(
                TicketClassificationRecord.id == recommendation.classification_id,
                TicketClassificationRecord.workspace_id == command.workspace_id,
                TicketClassificationRecord.ticket_id == command.ticket_id,
                TicketClassificationRecord.agent_run_id == command.agent_run_id,
            )
            .with_for_update()
        )
        result = await self._session.execute(statement)

        return result.scalar_one_or_none() is not None

    async def _resolve_missing_invocations(
        self,
        *,
        command: PersistSupportRecommendationCommand,
    ) -> tuple[LLMInvocation, ...]:
        invocation_ids = tuple(invocation.id for invocation in command.invocations)
        invocation_sequences = tuple(
            invocation.invocation_sequence for invocation in command.invocations
        )

        statement = (
            select(LLMInvocationRecord)
            .where(
                or_(
                    LLMInvocationRecord.id.in_(invocation_ids),
                    and_(
                        LLMInvocationRecord.agent_run_attempt_id == command.agent_run_attempt_id,
                        LLMInvocationRecord.invocation_sequence.in_(invocation_sequences),
                    ),
                )
            )
            .with_for_update()
        )
        result = await self._session.execute(statement)
        existing_records = tuple(result.scalars().all())

        records_by_id = {record.id: record for record in existing_records}
        records_by_sequence = {
            record.invocation_sequence: record
            for record in existing_records
            if (record.agent_run_attempt_id == command.agent_run_attempt_id)
        }

        missing_invocations: list[LLMInvocation] = []

        for invocation in command.invocations:
            record_by_id = records_by_id.get(invocation.id)
            record_by_sequence = records_by_sequence.get(invocation.invocation_sequence)

            if (
                record_by_id is not None
                and record_by_sequence is not None
                and record_by_id.id != record_by_sequence.id
            ):
                raise RuntimeError(
                    "An invocation identifier and sequence resolve to different persisted records."
                )

            existing_record = record_by_sequence or record_by_id

            if existing_record is None:
                missing_invocations.append(invocation)
                continue

            if existing_record.to_domain() != invocation:
                raise RuntimeError(
                    "An invocation identity is already persisted with different invocation data."
                )

        return tuple(missing_invocations)
