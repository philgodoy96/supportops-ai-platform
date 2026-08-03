"""PostgreSQL repository for fenced tool-call lifecycle persistence."""

from uuid import UUID

from sqlalchemy import and_, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from supportops.agent_tools.application.persistence import (
    AgentToolCallExecutionRepository,
    AgentToolCallPersistenceResult,
    PersistAgentToolCallCommand,
)
from supportops.agent_tools.domain.audit import AgentToolCall
from supportops.agent_tools.domain.contracts import ToolSafetyLevel
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


class SqlAlchemyAgentToolCallExecutionRepository(AgentToolCallExecutionRepository):
    """Persist tool-call lifecycle records through an active transaction."""

    def __init__(
        self,
        session: AsyncSession,
    ) -> None:
        self._session = session

    async def persist_fenced(
        self,
        command: PersistAgentToolCallCommand,
    ) -> AgentToolCallPersistenceResult:
        """Persist a tool-call lifecycle record while the lease is valid."""

        if not await self._lock_active_run(command):
            return AgentToolCallPersistenceResult.LEASE_LOST

        if not await self._lock_active_attempt(command):
            return AgentToolCallPersistenceResult.LEASE_LOST

        existing_records = await self._load_existing_identities(
            attempt_id=command.agent_run_attempt_id,
            sequence=command.tool_call.sequence,
            provider_tool_call_id=(command.tool_call.provider_tool_call_id),
        )

        for existing_record in existing_records:
            if existing_record.sequence == command.tool_call.sequence:
                if existing_record.to_domain() == command.tool_call:
                    return AgentToolCallPersistenceResult.ALREADY_RECORDED

                raise RuntimeError(
                    "A tool-call sequence is already persisted with different audit data."
                )

            if (
                command.tool_call.provider_tool_call_id is not None
                and existing_record.provider_tool_call_id == command.tool_call.provider_tool_call_id
            ):
                raise RuntimeError(
                    "A provider tool-call identifier is already "
                    "persisted for another tool-call sequence."
                )

        if command.tool_call.safety_level is ToolSafetyLevel.SENSITIVE_WRITE:
            sensitive_result = await self._resolve_sensitive_proposal_replay(
                command.tool_call,
            )
            if sensitive_result is not None:
                return sensitive_result

        self._session.add(AgentToolCallRecord.from_domain(command.tool_call))
        await self._session.flush()

        return AgentToolCallPersistenceResult.APPLIED

    async def _resolve_sensitive_proposal_replay(
        self,
        tool_call: AgentToolCall,
    ) -> AgentToolCallPersistenceResult | None:
        statement = (
            select(AgentToolCallRecord)
            .where(
                AgentToolCallRecord.agent_run_id == tool_call.agent_run_id,
                AgentToolCallRecord.tool_name == tool_call.tool_name,
                AgentToolCallRecord.tool_version == tool_call.tool_version,
                AgentToolCallRecord.input_fingerprint == (tool_call.input_fingerprint),
                AgentToolCallRecord.safety_level == (ToolSafetyLevel.SENSITIVE_WRITE.value),
            )
            .with_for_update()
        )
        result = await self._session.execute(statement)
        existing_record = result.scalar_one_or_none()

        if existing_record is None:
            return None

        existing = existing_record.to_domain()

        if (
            existing.workspace_id == tool_call.workspace_id
            and existing.ticket_id == tool_call.ticket_id
            and existing.tool_name == tool_call.tool_name
            and existing.tool_version == tool_call.tool_version
            and existing.input_fingerprint == tool_call.input_fingerprint
            and dict(existing.safe_input) == dict(tool_call.safe_input)
            and existing.safety_level is ToolSafetyLevel.SENSITIVE_WRITE
        ):
            return AgentToolCallPersistenceResult.ALREADY_RECORDED

        raise RuntimeError(
            "A sensitive tool-call proposal identity is already persisted with "
            "conflicting ownership or input data."
        )

    async def _lock_active_run(
        self,
        command: PersistAgentToolCallCommand,
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
        command: PersistAgentToolCallCommand,
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

    async def _load_existing_identities(
        self,
        *,
        attempt_id: UUID,
        sequence: int,
        provider_tool_call_id: str | None,
    ) -> tuple[AgentToolCallRecord, ...]:
        identity_predicates = [
            AgentToolCallRecord.sequence == sequence,
        ]

        if provider_tool_call_id is not None:
            identity_predicates.append(
                AgentToolCallRecord.provider_tool_call_id == provider_tool_call_id
            )

        statement = (
            select(AgentToolCallRecord)
            .where(
                AgentToolCallRecord.proposed_by_agent_run_attempt_id == attempt_id,
                or_(*identity_predicates),
            )
            .order_by(AgentToolCallRecord.sequence.asc())
            .with_for_update()
        )
        result = await self._session.execute(statement)

        return tuple(result.scalars().all())
