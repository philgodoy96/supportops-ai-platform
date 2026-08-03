"""SQLAlchemy repository for sensitive execution grants."""

from uuid import UUID

from sqlalchemy import and_, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from supportops.agent_tools.application.grant_persistence import (
    SensitiveExecutionGrantConsistencyError,
    SensitiveExecutionGrantPersistenceResult,
    SensitiveExecutionGrantRepository,
)
from supportops.agent_tools.domain.audit import AgentToolCallStatus
from supportops.agent_tools.domain.contracts import ToolSafetyLevel
from supportops.agent_tools.domain.grants import (
    SensitiveExecutionGrant,
)
from supportops.agent_tools.infrastructure.grant_models import (
    SensitiveExecutionGrantRecord,
)
from supportops.agent_tools.infrastructure.models import (
    AgentToolCallRecord,
)
from supportops.modules.agent_runs.domain.models import (
    AgentRunStatus,
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


class SqlAlchemySensitiveExecutionGrantRepository(SensitiveExecutionGrantRepository):
    """Persist immutable grants without owning the transaction."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def persist(
        self,
        grant: SensitiveExecutionGrant,
    ) -> SensitiveExecutionGrantPersistenceResult:
        """Insert or reuse one matching authorization."""

        existing = await self.get_by_approval_request_id(
            workspace_id=grant.workspace_id,
            approval_request_id=grant.approval_request_id,
        )
        if existing is not None:
            return _resolve_existing(existing, grant)

        await self._verify_approval_request(grant)
        await self._verify_agent_tool_call(grant)
        await self._verify_execution_attempt(grant)

        try:
            async with self._session.begin_nested():
                self._session.add(
                    SensitiveExecutionGrantRecord.from_domain(
                        grant,
                    ),
                )
                await self._session.flush()
        except IntegrityError as error:
            existing = await self.get_by_approval_request_id(
                workspace_id=grant.workspace_id,
                approval_request_id=grant.approval_request_id,
            )
            if existing is None:
                existing = await self.get_by_agent_tool_call_id(
                    workspace_id=grant.workspace_id,
                    agent_tool_call_id=grant.agent_tool_call_id,
                )
            if existing is None:
                raise SensitiveExecutionGrantConsistencyError(
                    "A grant uniqueness conflict could not be resolved to a durable record.",
                ) from error
            return _resolve_existing(existing, grant)

        return SensitiveExecutionGrantPersistenceResult.APPLIED

    async def get_by_id(
        self,
        *,
        workspace_id: UUID,
        grant_id: UUID,
    ) -> SensitiveExecutionGrant | None:
        """Return one workspace-scoped grant."""

        statement = select(SensitiveExecutionGrantRecord).where(
            SensitiveExecutionGrantRecord.workspace_id == workspace_id,
            SensitiveExecutionGrantRecord.id == grant_id,
        )
        record = await self._session.scalar(statement)
        return None if record is None else record.to_domain()

    async def get_by_approval_request_id(
        self,
        *,
        workspace_id: UUID,
        approval_request_id: UUID,
    ) -> SensitiveExecutionGrant | None:
        """Return the grant for one approval request."""

        statement = select(SensitiveExecutionGrantRecord).where(
            SensitiveExecutionGrantRecord.workspace_id == workspace_id,
            SensitiveExecutionGrantRecord.approval_request_id == approval_request_id,
        )
        record = await self._session.scalar(statement)
        return None if record is None else record.to_domain()

    async def get_by_agent_tool_call_id(
        self,
        *,
        workspace_id: UUID,
        agent_tool_call_id: UUID,
    ) -> SensitiveExecutionGrant | None:
        """Return the grant for one proposed tool call."""

        statement = select(SensitiveExecutionGrantRecord).where(
            SensitiveExecutionGrantRecord.workspace_id == workspace_id,
            SensitiveExecutionGrantRecord.agent_tool_call_id == agent_tool_call_id,
        )
        record = await self._session.scalar(statement)
        return None if record is None else record.to_domain()

    async def _verify_approval_request(
        self,
        grant: SensitiveExecutionGrant,
    ) -> None:
        statement = select(ApprovalRequestRecord).where(
            ApprovalRequestRecord.workspace_id == grant.workspace_id,
            ApprovalRequestRecord.id == grant.approval_request_id,
        )
        result = await self._session.execute(statement)
        record = result.scalar_one_or_none()

        if record is None:
            raise SensitiveExecutionGrantConsistencyError(
                "The referenced ApprovalRequest does not exist in this workspace.",
            )

        approval = record.to_domain()

        if (
            approval.id != grant.approval_request_id
            or approval.workspace_id != grant.workspace_id
            or approval.ticket_id != grant.ticket_id
            or approval.agent_run_id != grant.agent_run_id
            or approval.agent_tool_call_id != grant.agent_tool_call_id
            or approval.status is not ApprovalRequestStatus.APPROVED
            or approval.tool_name != grant.tool_name
            or approval.tool_version != grant.tool_version
            or approval.safety_level is not grant.safety_level
            or approval.input_fingerprint != grant.input_fingerprint
            or dict(approval.proposed_input) != dict(grant.granted_input)
            or approval.decision_actor_reference != grant.decision_actor_reference
            or approval.decision_request_id != grant.decision_request_id
            or approval.decision_correlation_id != grant.decision_correlation_id
            or approval.decided_at != grant.approved_at
        ):
            raise SensitiveExecutionGrantConsistencyError(
                "The referenced ApprovalRequest does not match the grant authorization.",
            )

    async def _verify_agent_tool_call(
        self,
        grant: SensitiveExecutionGrant,
    ) -> None:
        statement = select(AgentToolCallRecord).where(
            AgentToolCallRecord.workspace_id == grant.workspace_id,
            AgentToolCallRecord.id == grant.agent_tool_call_id,
        )
        result = await self._session.execute(statement)
        record = result.scalar_one_or_none()

        if record is None:
            raise SensitiveExecutionGrantConsistencyError(
                "The referenced AgentToolCall does not exist in this workspace.",
            )

        tool_call = record.to_domain()

        if (
            tool_call.id != grant.agent_tool_call_id
            or tool_call.workspace_id != grant.workspace_id
            or tool_call.ticket_id != grant.ticket_id
            or tool_call.agent_run_id != grant.agent_run_id
            or tool_call.status is not AgentToolCallStatus.PENDING_APPROVAL
            or tool_call.safety_level is not ToolSafetyLevel.SENSITIVE_WRITE
            or tool_call.tool_name != grant.tool_name
            or tool_call.tool_version != grant.tool_version
            or tool_call.input_fingerprint != grant.input_fingerprint
            or dict(tool_call.safe_input) != dict(grant.granted_input)
        ):
            raise SensitiveExecutionGrantConsistencyError(
                "The referenced AgentToolCall does not match the grant authorization.",
            )

    async def _verify_execution_attempt(
        self,
        grant: SensitiveExecutionGrant,
    ) -> None:
        run_statement = select(AgentRunRecord).where(
            and_(
                AgentRunRecord.id == grant.agent_run_id,
                AgentRunRecord.workspace_id == grant.workspace_id,
                AgentRunRecord.ticket_id == grant.ticket_id,
                AgentRunRecord.status == AgentRunStatus.RUNNING.value,
                AgentRunRecord.lease_token.is_not(None),
            )
        )
        run_result = await self._session.execute(run_statement)
        run_record = run_result.scalar_one_or_none()

        if run_record is None or run_record.lease_token is None:
            raise SensitiveExecutionGrantConsistencyError(
                "The AgentRun is not currently claimed for sensitive execution.",
            )

        attempt_statement = select(AgentRunAttemptRecord).where(
            and_(
                AgentRunAttemptRecord.id == grant.executed_by_agent_run_attempt_id,
                AgentRunAttemptRecord.agent_run_id == grant.agent_run_id,
                AgentRunAttemptRecord.lease_token == run_record.lease_token,
                AgentRunAttemptRecord.finished_at.is_(None),
                AgentRunAttemptRecord.outcome.is_(None),
            )
        )
        attempt_result = await self._session.execute(attempt_statement)

        if attempt_result.scalar_one_or_none() is None:
            raise SensitiveExecutionGrantConsistencyError(
                "The executing AgentRunAttempt is not the current "
                "claimed resume attempt for the AgentRun.",
            )


def _resolve_existing(
    existing: SensitiveExecutionGrant,
    candidate: SensitiveExecutionGrant,
) -> SensitiveExecutionGrantPersistenceResult:
    if existing.matches_authorization(candidate):
        return SensitiveExecutionGrantPersistenceResult.ALREADY_RECORDED
    raise SensitiveExecutionGrantConsistencyError(
        "The existing sensitive execution grant conflicts with the requested authorization.",
    )
