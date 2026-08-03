"""Idempotent execution service for approved sensitive tools."""

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from typing import Protocol
from uuid import UUID, uuid4

from supportops.agent_tools.application.grant_persistence import (
    SensitiveExecutionGrantPersistenceResult,
    SensitiveExecutionGrantRepository,
)
from supportops.agent_tools.domain.audit import (
    AgentToolCall,
    AgentToolCallStatus,
)
from supportops.agent_tools.domain.grants import (
    SensitiveExecutionGrant,
)
from supportops.agent_tools.tools.escalate_ticket import (
    EscalateTicketInput,
    EscalateTicketOutput,
)
from supportops.core.transactions import TransactionManager
from supportops.modules.agent_runs.application.execution import (
    AgentRunExecutionContext,
)
from supportops.modules.approvals.domain.models import (
    ApprovalRequest,
    ApprovalRequestStatus,
)
from supportops.modules.approvals.domain.repositories import (
    ApprovalRequestRepository,
)
from supportops.modules.tickets.domain.escalation import (
    TicketEscalation,
)
from supportops.modules.tickets.domain.escalation_repositories import (
    TicketEscalationPersistenceResult,
    TicketEscalationRepository,
)

type UtcNowProvider = Callable[[], datetime]
type UuidFactory = Callable[[], UUID]


class SensitiveExecutionStatus(StrEnum):
    """Outcome of one idempotent sensitive execution."""

    APPLIED = "applied"
    ALREADY_RECORDED = "already_recorded"


@dataclass(frozen=True, slots=True)
class SensitiveExecutionResult:
    """Durable outcome of one approved sensitive execution."""

    status: SensitiveExecutionStatus
    grant: SensitiveExecutionGrant
    escalation: TicketEscalation
    output: EscalateTicketOutput


class SensitiveExecutionConsistencyError(RuntimeError):
    """Raised when durable sensitive state conflicts with replay."""


class SensitiveToolCallExecutionRepository(Protocol):
    """Persistence boundary used by granted sensitive execution."""

    async def get_by_id_for_update(
        self,
        *,
        workspace_id: UUID,
        agent_tool_call_id: UUID,
    ) -> AgentToolCall | None:
        """Lock and return one workspace-scoped tool call."""

        ...

    async def save_granted_execution_success(
        self,
        *,
        tool_call: AgentToolCall,
    ) -> None:
        """Persist one granted sensitive execution success."""

        ...


class ExecuteApprovedTicketEscalation:
    """Authorize and persist one internal escalation exactly once."""

    def __init__(
        self,
        *,
        transaction_manager: TransactionManager,
        approval_request_repository: ApprovalRequestRepository,
        tool_call_repository: SensitiveToolCallExecutionRepository,
        grant_repository: SensitiveExecutionGrantRepository,
        escalation_repository: TicketEscalationRepository,
        utc_now: UtcNowProvider | None = None,
        uuid_factory: UuidFactory = uuid4,
    ) -> None:
        self._transaction_manager = transaction_manager
        self._approval_request_repository = approval_request_repository
        self._tool_call_repository = tool_call_repository
        self._grant_repository = grant_repository
        self._escalation_repository = escalation_repository
        self._utc_now = utc_now or _utc_now
        self._uuid_factory = uuid_factory

    async def execute(
        self,
        *,
        context: AgentRunExecutionContext,
        approval_request_id: UUID,
        agent_tool_call_id: UUID,
    ) -> SensitiveExecutionResult:
        """Execute one grant-backed escalation in a short transaction."""

        executed_at = self._utc_now()

        async with self._transaction_manager.transaction():
            approval = await self._approval_request_repository.get_by_id_for_update(
                workspace_id=context.agent_run.workspace_id,
                approval_request_id=approval_request_id,
            )
            if approval is None:
                raise SensitiveExecutionConsistencyError(
                    "Approved request was not found.",
                )
            if approval.status is not ApprovalRequestStatus.APPROVED:
                raise SensitiveExecutionConsistencyError(
                    "Sensitive execution requires an approved request.",
                )

            tool_call = await self._tool_call_repository.get_by_id_for_update(
                workspace_id=context.agent_run.workspace_id,
                agent_tool_call_id=agent_tool_call_id,
            )
            if tool_call is None:
                raise SensitiveExecutionConsistencyError(
                    "Sensitive tool call was not found.",
                )

            _validate_context_and_approval(
                context=context,
                approval=approval,
                tool_call=tool_call,
            )

            if tool_call.status is AgentToolCallStatus.SUCCEEDED:
                existing_grant = await self._grant_repository.get_by_agent_tool_call_id(
                    workspace_id=(context.agent_run.workspace_id),
                    agent_tool_call_id=tool_call.id,
                )
                existing_escalation = await self._escalation_repository.get_by_agent_tool_call_id(
                    workspace_id=(context.agent_run.workspace_id),
                    agent_tool_call_id=tool_call.id,
                )
                if existing_grant is None or existing_escalation is None:
                    raise SensitiveExecutionConsistencyError(
                        "Completed sensitive execution is missing "
                        "its durable authorization or escalation.",
                    )
                return SensitiveExecutionResult(
                    status=(SensitiveExecutionStatus.ALREADY_RECORDED),
                    grant=existing_grant,
                    escalation=existing_escalation,
                    output=_to_output(existing_escalation),
                )

            if tool_call.status is not AgentToolCallStatus.PENDING_APPROVAL:
                raise SensitiveExecutionConsistencyError(
                    "Sensitive tool call is not executable.",
                )

            grant = SensitiveExecutionGrant.create(
                approval_request=approval,
                tool_call=tool_call,
                executed_by_agent_run_attempt_id=context.attempt.id,
                created_at=executed_at,
                grant_id=self._uuid_factory(),
            )
            grant_result = await self._grant_repository.persist(
                grant,
            )
            if grant_result is SensitiveExecutionGrantPersistenceResult.ALREADY_RECORDED:
                durable_grant = await self._grant_repository.get_by_agent_tool_call_id(
                    workspace_id=grant.workspace_id,
                    agent_tool_call_id=grant.agent_tool_call_id,
                )
                if durable_grant is None:
                    raise SensitiveExecutionConsistencyError(
                        "Recorded grant could not be loaded.",
                    )
                grant = durable_grant

            input_data = EscalateTicketInput.model_validate(
                dict(grant.granted_input),
            )
            escalation = TicketEscalation.create_from_grant(
                grant=grant,
                input_data=input_data,
                created_at=executed_at,
                escalation_id=self._uuid_factory(),
            )
            escalation_result = await self._escalation_repository.persist(
                escalation,
            )
            if escalation_result is TicketEscalationPersistenceResult.ALREADY_RECORDED:
                durable_escalation = await self._escalation_repository.get_by_agent_tool_call_id(
                    workspace_id=escalation.workspace_id,
                    agent_tool_call_id=(escalation.agent_tool_call_id),
                )
                if durable_escalation is None:
                    raise SensitiveExecutionConsistencyError(
                        "Recorded escalation could not be loaded.",
                    )
                escalation = durable_escalation

            completed_tool_call = tool_call.complete_granted_execution_success(
                executed_by_agent_run_attempt_id=(context.attempt.id),
                execution_started_at=executed_at,
                finished_at=executed_at,
                safe_output=_to_output(
                    escalation,
                ).model_dump(mode="json"),
            )
            await self._tool_call_repository.save_granted_execution_success(
                tool_call=completed_tool_call,
            )

        status = (
            SensitiveExecutionStatus.APPLIED
            if (
                grant_result is SensitiveExecutionGrantPersistenceResult.APPLIED
                and escalation_result is TicketEscalationPersistenceResult.APPLIED
            )
            else SensitiveExecutionStatus.ALREADY_RECORDED
        )
        return SensitiveExecutionResult(
            status=status,
            grant=grant,
            escalation=escalation,
            output=_to_output(escalation),
        )


def _validate_context_and_approval(
    *,
    context: AgentRunExecutionContext,
    approval: ApprovalRequest,
    tool_call: AgentToolCall,
) -> None:
    checks = (
        approval.workspace_id == context.agent_run.workspace_id,
        approval.ticket_id == context.ticket.id,
        approval.agent_run_id == context.agent_run.id,
        approval.agent_tool_call_id == tool_call.id,
        tool_call.workspace_id == context.agent_run.workspace_id,
        tool_call.ticket_id == context.ticket.id,
        tool_call.agent_run_id == context.agent_run.id,
        approval.tool_name == tool_call.tool_name,
        approval.tool_version == tool_call.tool_version,
        approval.input_fingerprint == tool_call.input_fingerprint,
        dict(approval.proposed_input) == dict(tool_call.safe_input),
    )
    if not all(checks):
        raise SensitiveExecutionConsistencyError(
            "Approval, tool call, and AgentRun context do not match.",
        )


def _to_output(
    escalation: TicketEscalation,
) -> EscalateTicketOutput:
    return EscalateTicketOutput(
        escalation_id=escalation.id,
        ticket_id=escalation.ticket_id,
        target_queue=escalation.target_queue,
        status="escalated",
    )


def _utc_now() -> datetime:
    return datetime.now(UTC)
