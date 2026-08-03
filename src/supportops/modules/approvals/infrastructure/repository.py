"""PostgreSQL repository for durable approval-request persistence."""

from uuid import UUID

from sqlalchemy import select
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
