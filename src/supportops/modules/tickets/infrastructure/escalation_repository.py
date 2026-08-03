"""SQLAlchemy repository for immutable ticket escalations."""

from uuid import UUID

from sqlalchemy import and_, literal, select, tuple_
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from supportops.agent_tools.infrastructure.grant_models import (
    SensitiveExecutionGrantRecord,
)
from supportops.agent_tools.infrastructure.models import (
    AgentToolCallRecord,
)
from supportops.modules.agent_runs.infrastructure.models import (
    AgentRunAttemptRecord,
    AgentRunRecord,
)
from supportops.modules.approvals.domain.models import (
    ApprovalRequestStatus,
)
from supportops.modules.approvals.infrastructure.models import (
    ApprovalRequestRecord,
)
from supportops.modules.tickets.domain.escalation import (
    TicketEscalation,
)
from supportops.modules.tickets.domain.escalation_repositories import (
    TicketEscalationConsistencyError,
    TicketEscalationListPage,
    TicketEscalationListQuery,
    TicketEscalationPageCursor,
    TicketEscalationPersistenceResult,
    TicketEscalationRepository,
)
from supportops.modules.tickets.infrastructure.escalation_models import (
    TicketEscalationRecord,
)
from supportops.modules.tickets.infrastructure.models import (
    TicketRecord,
)


class SqlAlchemyTicketEscalationRepository(TicketEscalationRepository):
    """Persist immutable escalations without owning transactions."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def persist(
        self,
        escalation: TicketEscalation,
    ) -> TicketEscalationPersistenceResult:
        """Insert or reuse one matching escalation."""

        existing = await self.get_by_approval_request_id(
            workspace_id=escalation.workspace_id,
            approval_request_id=escalation.approval_request_id,
        )
        if existing is not None:
            return _resolve_existing(existing, escalation)

        await self._verify_ticket(escalation)
        await self._verify_agent_run(escalation)
        await self._verify_execution_attempt(escalation)
        await self._verify_approval_request(escalation)
        await self._verify_agent_tool_call(escalation)
        await self._verify_execution_grant(escalation)

        try:
            async with self._session.begin_nested():
                self._session.add(
                    TicketEscalationRecord.from_domain(
                        escalation,
                    ),
                )
                await self._session.flush()
        except IntegrityError as error:
            existing = await self.get_by_approval_request_id(
                workspace_id=escalation.workspace_id,
                approval_request_id=(escalation.approval_request_id),
            )
            if existing is None:
                existing = await self.get_by_agent_tool_call_id(
                    workspace_id=escalation.workspace_id,
                    agent_tool_call_id=(escalation.agent_tool_call_id),
                )
            if existing is None:
                raise TicketEscalationConsistencyError(
                    "An escalation uniqueness conflict could not be resolved to a durable record.",
                ) from error
            return _resolve_existing(existing, escalation)

        return TicketEscalationPersistenceResult.APPLIED

    async def list_page(
        self,
        query: TicketEscalationListQuery,
    ) -> TicketEscalationListPage:
        """Return one workspace-scoped escalation page."""

        statement = select(TicketEscalationRecord).where(
            TicketEscalationRecord.workspace_id == query.workspace_id,
        )

        if query.ticket_id is not None:
            statement = statement.where(
                TicketEscalationRecord.ticket_id == query.ticket_id,
            )

        if query.cursor is not None:
            statement = statement.where(
                tuple_(
                    TicketEscalationRecord.created_at,
                    TicketEscalationRecord.id,
                )
                < tuple_(
                    literal(query.cursor.created_at),
                    literal(query.cursor.ticket_escalation_id),
                ),
            )

        statement = statement.order_by(
            TicketEscalationRecord.created_at.desc(),
            TicketEscalationRecord.id.desc(),
        ).limit(query.page_size + 1)
        result = await self._session.execute(statement)
        records = list(result.scalars().all())

        has_next_page = len(records) > query.page_size
        page_records = records[: query.page_size]
        items = tuple(record.to_domain() for record in page_records)

        next_cursor: TicketEscalationPageCursor | None = None
        if has_next_page and items:
            last_item = items[-1]
            next_cursor = TicketEscalationPageCursor(
                created_at=last_item.created_at,
                ticket_escalation_id=last_item.id,
            )

        return TicketEscalationListPage(
            items=items,
            next_cursor=next_cursor,
        )

    async def get_by_id(
        self,
        *,
        workspace_id: UUID,
        escalation_id: UUID,
    ) -> TicketEscalation | None:
        """Return one workspace-scoped escalation."""

        statement = select(TicketEscalationRecord).where(
            TicketEscalationRecord.workspace_id == workspace_id,
            TicketEscalationRecord.id == escalation_id,
        )
        record = await self._session.scalar(statement)
        return None if record is None else record.to_domain()

    async def get_by_approval_request_id(
        self,
        *,
        workspace_id: UUID,
        approval_request_id: UUID,
    ) -> TicketEscalation | None:
        """Return the escalation for one approval."""

        statement = select(TicketEscalationRecord).where(
            TicketEscalationRecord.workspace_id == workspace_id,
            TicketEscalationRecord.approval_request_id == approval_request_id,
        )
        record = await self._session.scalar(statement)
        return None if record is None else record.to_domain()

    async def get_by_agent_tool_call_id(
        self,
        *,
        workspace_id: UUID,
        agent_tool_call_id: UUID,
    ) -> TicketEscalation | None:
        """Return the escalation for one tool call."""

        statement = select(TicketEscalationRecord).where(
            TicketEscalationRecord.workspace_id == workspace_id,
            TicketEscalationRecord.agent_tool_call_id == agent_tool_call_id,
        )
        record = await self._session.scalar(statement)
        return None if record is None else record.to_domain()

    async def _verify_ticket(
        self,
        escalation: TicketEscalation,
    ) -> None:
        statement = select(TicketRecord).where(
            TicketRecord.workspace_id == escalation.workspace_id,
            TicketRecord.id == escalation.ticket_id,
        )
        record = await self._session.scalar(statement)
        if record is None:
            raise TicketEscalationConsistencyError(
                "The referenced Ticket does not exist in this workspace.",
            )

    async def _verify_agent_run(
        self,
        escalation: TicketEscalation,
    ) -> None:
        statement = select(AgentRunRecord).where(
            and_(
                AgentRunRecord.id == escalation.agent_run_id,
                AgentRunRecord.workspace_id == escalation.workspace_id,
                AgentRunRecord.ticket_id == escalation.ticket_id,
            )
        )
        record = await self._session.scalar(statement)
        if record is None:
            raise TicketEscalationConsistencyError(
                "The referenced AgentRun does not belong to this workspace and ticket.",
            )

    async def _verify_execution_attempt(
        self,
        escalation: TicketEscalation,
    ) -> None:
        statement = select(AgentRunAttemptRecord).where(
            and_(
                AgentRunAttemptRecord.id == escalation.executed_by_agent_run_attempt_id,
                AgentRunAttemptRecord.agent_run_id == escalation.agent_run_id,
            )
        )
        record = await self._session.scalar(statement)
        if record is None:
            raise TicketEscalationConsistencyError(
                "The referenced execution attempt does not belong to the AgentRun.",
            )

    async def _verify_approval_request(
        self,
        escalation: TicketEscalation,
    ) -> None:
        statement = select(ApprovalRequestRecord).where(
            ApprovalRequestRecord.workspace_id == escalation.workspace_id,
            ApprovalRequestRecord.id == escalation.approval_request_id,
        )
        record = await self._session.scalar(statement)
        if record is None:
            raise TicketEscalationConsistencyError(
                "The referenced ApprovalRequest does not exist in this workspace.",
            )

        approval = record.to_domain()
        if (
            approval.id != escalation.approval_request_id
            or approval.workspace_id != escalation.workspace_id
            or approval.ticket_id != escalation.ticket_id
            or approval.agent_run_id != escalation.agent_run_id
            or approval.agent_tool_call_id != escalation.agent_tool_call_id
            or approval.status is not ApprovalRequestStatus.APPROVED
        ):
            raise TicketEscalationConsistencyError(
                "The referenced ApprovalRequest does not match the escalation ownership.",
            )

    async def _verify_agent_tool_call(
        self,
        escalation: TicketEscalation,
    ) -> None:
        statement = select(AgentToolCallRecord).where(
            AgentToolCallRecord.workspace_id == escalation.workspace_id,
            AgentToolCallRecord.id == escalation.agent_tool_call_id,
        )
        record = await self._session.scalar(statement)
        if record is None:
            raise TicketEscalationConsistencyError(
                "The referenced AgentToolCall does not exist in this workspace.",
            )

        tool_call = record.to_domain()
        if (
            tool_call.id != escalation.agent_tool_call_id
            or tool_call.workspace_id != escalation.workspace_id
            or tool_call.ticket_id != escalation.ticket_id
            or tool_call.agent_run_id != escalation.agent_run_id
        ):
            raise TicketEscalationConsistencyError(
                "The referenced AgentToolCall does not match the escalation ownership.",
            )

    async def _verify_execution_grant(
        self,
        escalation: TicketEscalation,
    ) -> None:
        statement = select(SensitiveExecutionGrantRecord).where(
            SensitiveExecutionGrantRecord.workspace_id == escalation.workspace_id,
            SensitiveExecutionGrantRecord.approval_request_id == escalation.approval_request_id,
            SensitiveExecutionGrantRecord.agent_tool_call_id == escalation.agent_tool_call_id,
        )
        record = await self._session.scalar(statement)
        if record is None:
            raise TicketEscalationConsistencyError(
                "The referenced SensitiveExecutionGrant does not "
                "exist for this approval and tool call.",
            )

        grant = record.to_domain()
        granted_queue = grant.granted_input.get("target_queue")
        granted_reason = grant.granted_input.get("reason")
        if (
            grant.workspace_id != escalation.workspace_id
            or grant.ticket_id != escalation.ticket_id
            or grant.agent_run_id != escalation.agent_run_id
            or grant.executed_by_agent_run_attempt_id != escalation.executed_by_agent_run_attempt_id
            or grant.approval_request_id != escalation.approval_request_id
            or grant.agent_tool_call_id != escalation.agent_tool_call_id
            or grant.tool_name != "escalate_ticket"
            or grant.tool_version != 1
            or granted_queue != escalation.target_queue.value
            or granted_reason != escalation.reason
        ):
            raise TicketEscalationConsistencyError(
                "The referenced SensitiveExecutionGrant does not "
                "match the escalation authorization.",
            )


def _resolve_existing(
    existing: TicketEscalation,
    candidate: TicketEscalation,
) -> TicketEscalationPersistenceResult:
    if existing.matches_escalation(candidate):
        return TicketEscalationPersistenceResult.ALREADY_RECORDED
    raise TicketEscalationConsistencyError(
        "The existing ticket escalation conflicts with replay.",
    )
