"""PostgreSQL reads for controlled-support inspection."""

from sqlalchemy import and_, select
from sqlalchemy.ext.asyncio import AsyncSession

from supportops.agent_tools.infrastructure.models import (
    AgentToolCallRecord,
)
from supportops.modules.agent_runs.infrastructure.models import (
    AgentRunAttemptRecord,
    AgentRunRecord,
)
from supportops.modules.controlled_support_inspection.application.repository import (
    ControlledSupportInspectionData,
    ControlledSupportInspectionIdentity,
    ControlledSupportInspectionRepository,
)
from supportops.modules.support_recommendations.infrastructure.models import (
    SupportRecommendationCitationRecord,
    SupportRecommendationRecord,
)
from supportops.modules.ticket_classifications.infrastructure.models import (
    LLMInvocationRecord,
    TicketClassificationRecord,
)


class SqlAlchemyControlledSupportInspectionRepository(ControlledSupportInspectionRepository):
    """Read one inspection through an active application transaction."""

    def __init__(
        self,
        session: AsyncSession,
    ) -> None:
        self._session = session

    async def get_inspection_data(
        self,
        identity: ControlledSupportInspectionIdentity,
    ) -> ControlledSupportInspectionData | None:
        """Load one exact workspace-, ticket-, and run-scoped snapshot."""

        agent_run_record = await self._load_agent_run(identity)

        if agent_run_record is None:
            return None

        attempts = await self._load_attempts(identity)
        classification = await self._load_classification(identity)
        tool_calls = await self._load_tool_calls(identity)
        llm_invocations = await self._load_invocations(identity)
        recommendation = await self._load_recommendation(identity)
        citations = (
            await self._load_citations(
                identity=identity,
                recommendation=recommendation,
            )
            if recommendation is not None
            else ()
        )

        return ControlledSupportInspectionData(
            agent_run=agent_run_record.to_domain(),
            attempts=tuple(record.to_domain() for record in attempts),
            classification=(classification.to_domain() if classification is not None else None),
            tool_calls=tuple(record.to_domain() for record in tool_calls),
            llm_invocations=tuple(record.to_domain() for record in llm_invocations),
            recommendation=(recommendation.to_domain() if recommendation is not None else None),
            citations=tuple(record.to_domain() for record in citations),
        )

    async def _load_agent_run(
        self,
        identity: ControlledSupportInspectionIdentity,
    ) -> AgentRunRecord | None:
        statement = select(AgentRunRecord).where(
            AgentRunRecord.workspace_id == identity.workspace_id,
            AgentRunRecord.ticket_id == identity.ticket_id,
            AgentRunRecord.id == identity.agent_run_id,
        )
        result = await self._session.execute(statement)

        return result.scalar_one_or_none()

    async def _load_attempts(
        self,
        identity: ControlledSupportInspectionIdentity,
    ) -> tuple[AgentRunAttemptRecord, ...]:
        statement = (
            select(AgentRunAttemptRecord)
            .where(AgentRunAttemptRecord.agent_run_id == identity.agent_run_id)
            .order_by(
                AgentRunAttemptRecord.attempt_number.asc(),
                AgentRunAttemptRecord.id.asc(),
            )
        )
        result = await self._session.execute(statement)

        return tuple(result.scalars().all())

    async def _load_classification(
        self,
        identity: ControlledSupportInspectionIdentity,
    ) -> TicketClassificationRecord | None:
        statement = select(TicketClassificationRecord).where(
            TicketClassificationRecord.workspace_id == identity.workspace_id,
            TicketClassificationRecord.ticket_id == identity.ticket_id,
            TicketClassificationRecord.agent_run_id == identity.agent_run_id,
        )
        result = await self._session.execute(statement)

        return result.scalar_one_or_none()

    async def _load_tool_calls(
        self,
        identity: ControlledSupportInspectionIdentity,
    ) -> tuple[AgentToolCallRecord, ...]:
        statement = (
            select(AgentToolCallRecord)
            .join(
                AgentRunAttemptRecord,
                and_(
                    AgentRunAttemptRecord.agent_run_id == AgentToolCallRecord.agent_run_id,
                    AgentRunAttemptRecord.id
                    == (AgentToolCallRecord.proposed_by_agent_run_attempt_id),
                ),
            )
            .where(
                AgentToolCallRecord.workspace_id == identity.workspace_id,
                AgentToolCallRecord.ticket_id == identity.ticket_id,
                AgentToolCallRecord.agent_run_id == identity.agent_run_id,
            )
            .order_by(
                AgentRunAttemptRecord.attempt_number.asc(),
                AgentToolCallRecord.sequence.asc(),
                AgentToolCallRecord.id.asc(),
            )
        )
        result = await self._session.execute(statement)

        return tuple(result.scalars().all())

    async def _load_invocations(
        self,
        identity: ControlledSupportInspectionIdentity,
    ) -> tuple[LLMInvocationRecord, ...]:
        statement = (
            select(LLMInvocationRecord)
            .join(
                AgentRunAttemptRecord,
                and_(
                    AgentRunAttemptRecord.agent_run_id == LLMInvocationRecord.agent_run_id,
                    AgentRunAttemptRecord.id == (LLMInvocationRecord.agent_run_attempt_id),
                ),
            )
            .where(
                LLMInvocationRecord.workspace_id == identity.workspace_id,
                LLMInvocationRecord.ticket_id == identity.ticket_id,
                LLMInvocationRecord.agent_run_id == identity.agent_run_id,
            )
            .order_by(
                AgentRunAttemptRecord.attempt_number.asc(),
                LLMInvocationRecord.invocation_sequence.asc(),
                LLMInvocationRecord.id.asc(),
            )
        )
        result = await self._session.execute(statement)

        return tuple(result.scalars().all())

    async def _load_recommendation(
        self,
        identity: ControlledSupportInspectionIdentity,
    ) -> SupportRecommendationRecord | None:
        statement = select(SupportRecommendationRecord).where(
            SupportRecommendationRecord.workspace_id == identity.workspace_id,
            SupportRecommendationRecord.ticket_id == identity.ticket_id,
            SupportRecommendationRecord.agent_run_id == identity.agent_run_id,
        )
        result = await self._session.execute(statement)

        return result.scalar_one_or_none()

    async def _load_citations(
        self,
        *,
        identity: ControlledSupportInspectionIdentity,
        recommendation: SupportRecommendationRecord,
    ) -> tuple[
        SupportRecommendationCitationRecord,
        ...,
    ]:
        statement = (
            select(SupportRecommendationCitationRecord)
            .join(
                SupportRecommendationRecord,
                and_(
                    SupportRecommendationRecord.workspace_id
                    == (SupportRecommendationCitationRecord.workspace_id),
                    SupportRecommendationRecord.id
                    == (SupportRecommendationCitationRecord.support_recommendation_id),
                ),
            )
            .where(
                SupportRecommendationCitationRecord.workspace_id == identity.workspace_id,
                SupportRecommendationCitationRecord.support_recommendation_id == recommendation.id,
                SupportRecommendationRecord.ticket_id == identity.ticket_id,
                SupportRecommendationRecord.agent_run_id == identity.agent_run_id,
            )
            .order_by(
                SupportRecommendationCitationRecord.ordinal.asc(),
                SupportRecommendationCitationRecord.id.asc(),
            )
        )
        result = await self._session.execute(statement)

        return tuple(result.scalars().all())
