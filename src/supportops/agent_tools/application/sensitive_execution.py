"""Idempotent execution service for approved sensitive tools."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import AbstractContextManager
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from time import perf_counter
from typing import Final, Protocol
from uuid import UUID, uuid4

from supportops.agent_tools.application.grant_persistence import (
    SensitiveExecutionGrantPersistenceResult,
    SensitiveExecutionGrantRepository,
)
from supportops.agent_tools.domain.audit import (
    AgentToolCall,
    AgentToolCallStatus,
)
from supportops.agent_tools.domain.contracts import ToolSafetyLevel
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
from supportops.observability.context import (
    current_observation_context,
    current_trace_context,
)
from supportops.observability.contracts import (
    ObservabilityClient,
    ObservationScope,
)
from supportops.observability.identity import agent_run_trace_identity
from supportops.observability.models import (
    EventObservation,
    JsonValue,
    ObservationAttributes,
    ObservationStatus,
    ObservationType,
    ObservationUpdate,
)
from supportops.observability.noop import NoOpObservabilityClient

type UtcNowProvider = Callable[[], datetime]
type UuidFactory = Callable[[], UUID]

_OBSERVATION_NAME: Final = "tool.execute"
_ESCALATION_EVENT_NAME: Final = "ticket.escalated"
_UNEXPECTED_FAILURE_CODE: Final = "tool_execution_unexpected_failure"

_OBSERVATION_METADATA_KEYS: Final = frozenset(
    {
        "tool_name",
        "tool_safety",
        "requires_approval",
        "agent_run_id",
        "agent_run_attempt_id",
        "tool_call_id",
        "workspace_id",
        "ticket_id",
        "execution_request_id",
        "correlation_id",
        "status",
        "tool_outcome",
        "error_code",
        "latency_ms",
        "idempotent_replay",
    }
)
_OBSERVATION_METADATA_PATHS: Final = frozenset((key,) for key in _OBSERVATION_METADATA_KEYS)
_ESCALATION_EVENT_METADATA_KEYS: Final = frozenset(
    {
        "escalation_id",
        "agent_run_id",
        "agent_run_attempt_id",
        "workspace_id",
        "ticket_id",
        "approval_request_id",
        "agent_tool_call_id",
        "tool_name",
        "target_queue",
        "status",
        "escalation_outcome",
        "idempotent_replay",
    }
)
_ESCALATION_EVENT_METADATA_PATHS: Final = frozenset(
    (key,) for key in _ESCALATION_EVENT_METADATA_KEYS
)


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
        observability_client: ObservabilityClient | None = None,
    ) -> None:
        self._transaction_manager = transaction_manager
        self._approval_request_repository = approval_request_repository
        self._tool_call_repository = tool_call_repository
        self._grant_repository = grant_repository
        self._escalation_repository = escalation_repository
        self._utc_now = utc_now or _utc_now
        self._uuid_factory = uuid_factory
        self._observability_client = observability_client or NoOpObservabilityClient()

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

            observation = _SafeToolObservation(
                client=self._observability_client,
                attributes=_start_attributes(
                    context=context,
                    tool_call=tool_call,
                ),
            )
            observation.start()
            started_monotonic = perf_counter()

            try:
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
                    durable_escalation = (
                        await self._escalation_repository.get_by_agent_tool_call_id(
                            workspace_id=escalation.workspace_id,
                            agent_tool_call_id=(escalation.agent_tool_call_id),
                        )
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
                result = SensitiveExecutionResult(
                    status=status,
                    grant=grant,
                    escalation=escalation,
                    output=_to_output(escalation),
                )
                observation.update(
                    _completion_update(
                        result=result,
                        latency_ms=_elapsed_milliseconds(started_monotonic),
                    )
                )
                if result.status is SensitiveExecutionStatus.APPLIED:
                    observation.record_escalation_event(
                        context=context,
                        result=result,
                    )
                return result
            except Exception:
                observation.update(
                    _unexpected_failure_update(
                        latency_ms=_elapsed_milliseconds(started_monotonic),
                    )
                )
                raise
            finally:
                observation.close()


class _SafeToolObservation:
    """Isolate observability failures from sensitive execution behavior."""

    def __init__(
        self,
        *,
        client: ObservabilityClient,
        attributes: ObservationAttributes,
    ) -> None:
        self._client = client
        self._attributes = attributes
        self._manager: AbstractContextManager[ObservationScope] | None = None
        self._scope: ObservationScope | None = None

    def start(self) -> None:
        try:
            self._manager = self._client.start_observation(self._attributes)
            self._scope = self._manager.__enter__()
        except Exception:
            self._manager = None
            self._scope = None

    def update(self, update: ObservationUpdate | None) -> None:
        if self._scope is None or update is None:
            return

        try:
            self._scope.update(update)
        except Exception:
            return

    def record_escalation_event(
        self,
        *,
        context: AgentRunExecutionContext,
        result: SensitiveExecutionResult,
    ) -> None:
        event = _ticket_escalated_event(result=result)
        if event is None:
            return

        if self._scope is not None:
            try:
                self._scope.record_event(event)
                return
            except Exception:
                pass

        _safe_record_ticket_escalated(
            client=self._client,
            context=context,
            event=event,
        )

    def close(self) -> None:
        if self._manager is None:
            return

        try:
            self._manager.__exit__(None, None, None)
        except Exception:
            return
        finally:
            self._manager = None
            self._scope = None


def _start_attributes(
    *,
    context: AgentRunExecutionContext,
    tool_call: AgentToolCall,
) -> ObservationAttributes:
    try:
        return ObservationAttributes(
            name=_OBSERVATION_NAME,
            observation_type=ObservationType.TOOL,
            metadata=_start_metadata(
                context=context,
                tool_call=tool_call,
            ),
            metadata_paths=_OBSERVATION_METADATA_PATHS,
            input_data=None,
            input_paths=frozenset(),
            output_paths=frozenset(),
        )
    except Exception:
        return ObservationAttributes(
            name=_OBSERVATION_NAME,
            observation_type=ObservationType.TOOL,
            input_data=None,
            input_paths=frozenset(),
            output_paths=frozenset(),
        )


def _start_metadata(
    *,
    context: AgentRunExecutionContext,
    tool_call: AgentToolCall,
) -> dict[str, JsonValue]:
    metadata: dict[str, JsonValue] = {
        "tool_name": tool_call.tool_name,
        "tool_safety": tool_call.safety_level.value,
        "requires_approval": _requires_approval(tool_call.safety_level),
        "agent_run_id": str(context.agent_run.id),
        "agent_run_attempt_id": str(context.attempt.id),
        "tool_call_id": str(tool_call.id),
        "workspace_id": str(context.agent_run.workspace_id),
        "ticket_id": str(context.ticket.id),
    }

    correlation_id = getattr(context.agent_run, "correlation_id", None)
    if correlation_id is not None:
        metadata["correlation_id"] = str(correlation_id)

    execution_request_id = getattr(context.attempt, "execution_request_id", None)
    if execution_request_id is not None:
        metadata["execution_request_id"] = str(execution_request_id)

    return metadata


def _completion_update(
    *,
    result: SensitiveExecutionResult,
    latency_ms: int,
) -> ObservationUpdate | None:
    try:
        if result.status is SensitiveExecutionStatus.ALREADY_RECORDED:
            return ObservationUpdate(
                status=ObservationStatus.OK,
                metadata={
                    "status": ObservationStatus.OK.value,
                    "tool_outcome": "already_recorded",
                    "latency_ms": latency_ms,
                    "idempotent_replay": True,
                },
            )

        return ObservationUpdate(
            status=ObservationStatus.OK,
            metadata={
                "status": ObservationStatus.OK.value,
                "tool_outcome": "succeeded",
                "latency_ms": latency_ms,
            },
        )
    except Exception:
        return None


def _unexpected_failure_update(
    *,
    latency_ms: int,
) -> ObservationUpdate | None:
    try:
        return ObservationUpdate(
            status=ObservationStatus.ERROR,
            metadata={
                "status": ObservationStatus.ERROR.value,
                "tool_outcome": "unexpected_failure",
                "error_code": _UNEXPECTED_FAILURE_CODE,
                "latency_ms": latency_ms,
            },
            error_code=_UNEXPECTED_FAILURE_CODE,
        )
    except Exception:
        return None


def _ticket_escalated_event(
    *,
    result: SensitiveExecutionResult,
) -> EventObservation | None:
    try:
        escalation = result.escalation
        return EventObservation(
            name=_ESCALATION_EVENT_NAME,
            status=ObservationStatus.OK,
            metadata={
                "escalation_id": str(escalation.id),
                "agent_run_id": str(escalation.agent_run_id),
                "agent_run_attempt_id": str(
                    escalation.executed_by_agent_run_attempt_id,
                ),
                "workspace_id": str(escalation.workspace_id),
                "ticket_id": str(escalation.ticket_id),
                "approval_request_id": str(escalation.approval_request_id),
                "agent_tool_call_id": str(escalation.agent_tool_call_id),
                "tool_name": "escalate_ticket",
                "target_queue": escalation.target_queue.value,
                "status": result.output.status,
                "escalation_outcome": "applied",
                "idempotent_replay": False,
            },
            metadata_paths=_ESCALATION_EVENT_METADATA_PATHS,
        )
    except Exception:
        return None


def _safe_record_ticket_escalated(
    *,
    client: ObservabilityClient,
    context: AgentRunExecutionContext,
    event: EventObservation,
) -> None:
    try:
        if current_observation_context() is not None or current_trace_context() is not None:
            client.record_event(event)
            return

        client.record_trace_event(
            identity=agent_run_trace_identity(
                agent_run_id=context.agent_run.id,
                ticket_id=context.ticket.id,
            ),
            event=event,
        )
    except Exception:
        return


def _requires_approval(safety_level: ToolSafetyLevel) -> bool:
    return safety_level is not ToolSafetyLevel.READ_ONLY


def _elapsed_milliseconds(started_at: float) -> int:
    return max(0, round((perf_counter() - started_at) * 1000))


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
