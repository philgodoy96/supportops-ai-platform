"""PostgreSQL repository for durable approval-request persistence."""

from datetime import datetime
from uuid import UUID

from sqlalchemy import literal, select, tuple_
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from supportops.agent_tools.domain.audit import AgentToolCallStatus
from supportops.agent_tools.domain.contracts import ToolSafetyLevel
from supportops.agent_tools.infrastructure.models import AgentToolCallRecord
from supportops.modules.approvals.domain.models import (
    ApprovalRequest,
    ApprovalRequestStatus,
)
from supportops.modules.approvals.domain.repositories import (
    ApprovalRequestConsistencyError,
    ApprovalRequestListPage,
    ApprovalRequestListQuery,
    ApprovalRequestPageCursor,
    ApprovalRequestPersistenceResult,
    ApprovalRequestRepository,
)
from supportops.modules.approvals.infrastructure.models import (
    ApprovalRequestRecord,
)
from supportops.modules.ticket_classifications.infrastructure.models import (
    LLMInvocationRecord,
)


class SqlAlchemyApprovalRequestRepository(ApprovalRequestRepository):
    """Persist approval requests through an active transaction."""

    def __init__(
        self,
        session: AsyncSession,
    ) -> None:
        self._session = session

    async def persist_pending(
        self,
        approval_request: ApprovalRequest,
    ) -> ApprovalRequestPersistenceResult:
        """Persist one pending approval request idempotently."""

        if approval_request.status is not ApprovalRequestStatus.PENDING:
            raise ValueError("Only pending approval requests can be persisted.")

        existing = await self._load_for_update_by_tool_call(
            workspace_id=approval_request.workspace_id,
            agent_tool_call_id=approval_request.agent_tool_call_id,
        )

        if existing is not None:
            if existing.matches_pending_proposal(approval_request):
                return ApprovalRequestPersistenceResult.ALREADY_RECORDED

            raise ApprovalRequestConsistencyError(
                "An approval request is already persisted for this "
                "AgentToolCall with conflicting proposal data.",
            )

        await self._verify_agent_tool_call(approval_request)
        await self._verify_requesting_invocation(approval_request)

        try:
            async with self._session.begin_nested():
                self._session.add(
                    ApprovalRequestRecord.from_domain(approval_request),
                )
                await self._session.flush()
        except IntegrityError as error:
            reloaded = await self._load_for_update_by_tool_call(
                workspace_id=approval_request.workspace_id,
                agent_tool_call_id=approval_request.agent_tool_call_id,
            )

            if reloaded is not None and reloaded.matches_pending_proposal(
                approval_request,
            ):
                return ApprovalRequestPersistenceResult.ALREADY_RECORDED

            raise ApprovalRequestConsistencyError(
                "An approval request is already persisted for this "
                "AgentToolCall with conflicting proposal data.",
            ) from error

        return ApprovalRequestPersistenceResult.APPLIED

    async def list_page(
        self,
        query: ApprovalRequestListQuery,
    ) -> ApprovalRequestListPage:
        """Return one workspace-scoped approval page."""

        statement = select(ApprovalRequestRecord).where(
            ApprovalRequestRecord.workspace_id == query.workspace_id,
        )

        if query.status is not None:
            statement = statement.where(
                ApprovalRequestRecord.status == query.status.value,
            )

        if query.cursor is not None:
            statement = statement.where(
                tuple_(
                    ApprovalRequestRecord.created_at,
                    ApprovalRequestRecord.id,
                )
                < tuple_(
                    literal(query.cursor.created_at),
                    literal(query.cursor.approval_request_id),
                ),
            )

        statement = statement.order_by(
            ApprovalRequestRecord.created_at.desc(),
            ApprovalRequestRecord.id.desc(),
        ).limit(query.page_size + 1)
        result = await self._session.execute(statement)
        records = list(result.scalars().all())

        has_next_page = len(records) > query.page_size
        page_records = records[: query.page_size]
        items = tuple(record.to_domain() for record in page_records)

        next_cursor: ApprovalRequestPageCursor | None = None
        if has_next_page and items:
            last_item = items[-1]
            next_cursor = ApprovalRequestPageCursor(
                created_at=last_item.created_at,
                approval_request_id=last_item.id,
            )

        return ApprovalRequestListPage(
            items=items,
            next_cursor=next_cursor,
        )

    async def get_by_id(
        self,
        *,
        workspace_id: UUID,
        approval_request_id: UUID,
    ) -> ApprovalRequest | None:
        """Return one workspace-scoped approval request by ID."""

        statement = select(ApprovalRequestRecord).where(
            ApprovalRequestRecord.workspace_id == workspace_id,
            ApprovalRequestRecord.id == approval_request_id,
        )
        result = await self._session.execute(statement)
        record = result.scalar_one_or_none()

        if record is None:
            return None

        return record.to_domain()

    async def get_by_agent_tool_call_id(
        self,
        *,
        workspace_id: UUID,
        agent_tool_call_id: UUID,
    ) -> ApprovalRequest | None:
        """Return one workspace-scoped approval for a tool call."""

        statement = select(ApprovalRequestRecord).where(
            ApprovalRequestRecord.workspace_id == workspace_id,
            ApprovalRequestRecord.agent_tool_call_id == agent_tool_call_id,
        )
        result = await self._session.execute(statement)
        record = result.scalar_one_or_none()

        if record is None:
            return None

        return record.to_domain()

    async def list_by_agent_run(
        self,
        *,
        workspace_id: UUID,
        agent_run_id: UUID,
    ) -> tuple[ApprovalRequest, ...]:
        """Return workspace-scoped approvals for one AgentRun."""

        statement = (
            select(ApprovalRequestRecord)
            .where(
                ApprovalRequestRecord.workspace_id == workspace_id,
                ApprovalRequestRecord.agent_run_id == agent_run_id,
            )
            .order_by(
                ApprovalRequestRecord.created_at.asc(),
                ApprovalRequestRecord.id.asc(),
            )
        )
        result = await self._session.execute(statement)

        return tuple(record.to_domain() for record in result.scalars().all())

    async def get_by_id_for_update(
        self,
        *,
        workspace_id: UUID,
        approval_request_id: UUID,
    ) -> ApprovalRequest | None:
        """Lock and return one workspace-scoped approval request by ID."""

        statement = (
            select(ApprovalRequestRecord)
            .where(
                ApprovalRequestRecord.workspace_id == workspace_id,
                ApprovalRequestRecord.id == approval_request_id,
            )
            .with_for_update()
        )
        result = await self._session.execute(statement)
        record = result.scalar_one_or_none()

        if record is None:
            return None

        return record.to_domain()

    async def get_next_expired_pending_for_update(
        self,
        *,
        now: datetime,
    ) -> ApprovalRequest | None:
        """Lock the next overdue pending approval, if available."""

        statement = (
            select(ApprovalRequestRecord)
            .where(
                ApprovalRequestRecord.status == (ApprovalRequestStatus.PENDING.value),
                ApprovalRequestRecord.expires_at <= now,
            )
            .order_by(
                ApprovalRequestRecord.expires_at.asc(),
                ApprovalRequestRecord.id.asc(),
            )
            .limit(1)
            .with_for_update(skip_locked=True)
        )
        result = await self._session.execute(statement)
        record = result.scalar_one_or_none()

        if record is None:
            return None

        return record.to_domain()

    async def save(
        self,
        approval_request: ApprovalRequest,
    ) -> None:
        """Persist one terminal approval decision without committing."""

        statement = (
            select(ApprovalRequestRecord)
            .where(
                ApprovalRequestRecord.workspace_id == (approval_request.workspace_id),
                ApprovalRequestRecord.id == approval_request.id,
            )
            .with_for_update()
        )
        result = await self._session.execute(statement)
        record = result.scalar_one_or_none()

        if record is None:
            raise ApprovalRequestConsistencyError(
                "The approval request does not exist in this workspace.",
            )

        if (
            record.workspace_id != approval_request.workspace_id
            or record.ticket_id != approval_request.ticket_id
            or record.agent_run_id != approval_request.agent_run_id
            or record.agent_tool_call_id != approval_request.agent_tool_call_id
            or record.requested_by_llm_invocation_id
            != approval_request.requested_by_llm_invocation_id
            or record.tool_name != approval_request.tool_name
            or record.tool_version != approval_request.tool_version
            or record.safety_level != approval_request.safety_level.value
            or record.input_fingerprint != approval_request.input_fingerprint
            or dict(record.proposed_input) != dict(approval_request.proposed_input)
            or record.request_reason != approval_request.request_reason
            or record.expires_at != approval_request.expires_at
            or record.created_at != approval_request.created_at
        ):
            raise ApprovalRequestConsistencyError(
                "The approval request proposal data does not match the persisted immutable fields.",
            )

        record.status = approval_request.status.value
        record.decision_actor_reference = approval_request.decision_actor_reference
        record.decision_comment = approval_request.decision_comment
        record.decision_request_id = approval_request.decision_request_id
        record.decision_correlation_id = approval_request.decision_correlation_id
        record.decided_at = approval_request.decided_at
        record.updated_at = approval_request.updated_at

        await self._session.flush()

    async def _load_for_update_by_tool_call(
        self,
        *,
        workspace_id: UUID,
        agent_tool_call_id: UUID,
    ) -> ApprovalRequest | None:
        statement = (
            select(ApprovalRequestRecord)
            .where(
                ApprovalRequestRecord.workspace_id == workspace_id,
                ApprovalRequestRecord.agent_tool_call_id == agent_tool_call_id,
            )
            .with_for_update()
        )
        result = await self._session.execute(statement)
        record = result.scalar_one_or_none()

        if record is None:
            return None

        return record.to_domain()

    async def _verify_agent_tool_call(
        self,
        approval_request: ApprovalRequest,
    ) -> None:
        statement = select(AgentToolCallRecord).where(
            AgentToolCallRecord.id == approval_request.agent_tool_call_id,
            AgentToolCallRecord.workspace_id == approval_request.workspace_id,
        )
        result = await self._session.execute(statement)
        record = result.scalar_one_or_none()

        if record is None:
            raise ApprovalRequestConsistencyError(
                "The referenced AgentToolCall does not exist in this workspace.",
            )

        tool_call = record.to_domain()

        if (
            tool_call.id != approval_request.agent_tool_call_id
            or tool_call.workspace_id != approval_request.workspace_id
            or tool_call.ticket_id != approval_request.ticket_id
            or tool_call.agent_run_id != approval_request.agent_run_id
            or tool_call.status is not AgentToolCallStatus.PENDING_APPROVAL
            or tool_call.safety_level is not ToolSafetyLevel.SENSITIVE_WRITE
            or tool_call.tool_name != approval_request.tool_name
            or tool_call.tool_version != approval_request.tool_version
            or tool_call.input_fingerprint != approval_request.input_fingerprint
            or dict(tool_call.safe_input) != dict(approval_request.proposed_input)
        ):
            raise ApprovalRequestConsistencyError(
                "The referenced AgentToolCall does not match the approval proposal.",
            )

    async def _verify_requesting_invocation(
        self,
        approval_request: ApprovalRequest,
    ) -> None:
        statement = select(LLMInvocationRecord.id).where(
            LLMInvocationRecord.id == approval_request.requested_by_llm_invocation_id,
            LLMInvocationRecord.agent_run_id == approval_request.agent_run_id,
        )
        result = await self._session.execute(statement)

        if result.scalar_one_or_none() is None:
            raise ApprovalRequestConsistencyError(
                "The requesting LLM invocation does not belong to the AgentRun.",
            )
