"""Transactional approval decision and expiration services."""

from contextlib import suppress
from datetime import datetime
from typing import Protocol
from uuid import UUID

from supportops.agent_tools.domain.audit import (
    AgentToolCall,
    AgentToolCallStatus,
)
from supportops.agent_tools.domain.contracts import ToolSafetyLevel
from supportops.core.transactions import TransactionManager
from supportops.modules.agent_runs.domain.models import AgentRunStatus
from supportops.modules.agent_runs.domain.repositories import (
    AgentRunRepository,
)
from supportops.modules.agent_runs.domain.transitions import (
    AgentRunApprovalRequeueResult,
    RequeueWaitingAgentRunCommand,
)
from supportops.modules.approvals.application.errors import (
    ApprovalDecisionConflictError,
    ApprovalRequestExpiredError,
    ApprovalRequestNotFoundError,
    ApprovalRunStateConflictError,
    ApprovalToolCallStateConflictError,
)
from supportops.modules.approvals.application.models import (
    ApprovalDecisionResult,
    ApprovalExpirationBatchResult,
    ApproveApprovalRequestCommand,
    ExpirePendingApprovalRequestsCommand,
    RejectApprovalRequestCommand,
)
from supportops.modules.approvals.domain.models import (
    ApprovalRequest,
    ApprovalRequestStatus,
)
from supportops.modules.approvals.domain.repositories import (
    ApprovalRequestRepository,
)
from supportops.observability.contracts import ObservabilityClient
from supportops.observability.identity import agent_run_trace_identity
from supportops.observability.models import (
    EventObservation,
    FieldPaths,
    ObservationStatus,
)
from supportops.observability.noop import NoOpObservabilityClient

_APPROVAL_DECISION_EVENT_METADATA_PATHS: FieldPaths = frozenset(
    {
        ("approval_request_id",),
        ("agent_run_id",),
        ("workspace_id",),
        ("ticket_id",),
        ("approval_status",),
        ("tool_name",),
    }
)

_APPROVAL_EXPIRED_EVENT_METADATA_PATHS: FieldPaths = frozenset(
    {
        ("approval_request_id",),
        ("agent_run_id",),
        ("workspace_id",),
        ("ticket_id",),
        ("approval_status",),
        ("tool_name",),
    }
)

_RESUME_SCHEDULED_EVENT_METADATA_PATHS: FieldPaths = frozenset(
    {
        ("approval_request_id",),
        ("agent_run_id",),
        ("workspace_id",),
        ("ticket_id",),
        ("agent_run_status",),
        ("available_at",),
    }
)


class ApprovalAgentToolCallRepository(Protocol):
    """Tool-call operations required by approval decisions."""

    async def get_by_id_for_update(
        self,
        *,
        workspace_id: UUID,
        agent_tool_call_id: UUID,
    ) -> AgentToolCall | None:
        """Lock and return one workspace-scoped tool call."""

        ...

    async def save_approval_outcome(
        self,
        tool_call: AgentToolCall,
    ) -> None:
        """Persist a rejected or expired non-executed tool call."""

        ...


class DecideApprovalRequest:
    """Persist immutable approval decisions and schedule graph resume."""

    def __init__(
        self,
        *,
        transaction_manager: TransactionManager,
        approval_request_repository: ApprovalRequestRepository,
        agent_run_repository: AgentRunRepository,
        agent_tool_call_repository: ApprovalAgentToolCallRepository,
        observability_client: ObservabilityClient | None = None,
    ) -> None:
        self._transaction_manager = transaction_manager
        self._approval_request_repository = approval_request_repository
        self._agent_run_repository = agent_run_repository
        self._agent_tool_call_repository = agent_tool_call_repository
        self._observability_client = observability_client or NoOpObservabilityClient()

    async def approve(
        self,
        command: ApproveApprovalRequestCommand,
    ) -> ApprovalDecisionResult:
        """Approve one pending request or return its identical decision."""

        return await self._decide(
            command=command,
            target_status=ApprovalRequestStatus.APPROVED,
        )

    async def reject(
        self,
        command: RejectApprovalRequestCommand,
    ) -> ApprovalDecisionResult:
        """Reject one pending request or return its identical decision."""

        return await self._decide(
            command=command,
            target_status=ApprovalRequestStatus.REJECTED,
        )

    async def _decide(
        self,
        *,
        command: ApproveApprovalRequestCommand | RejectApprovalRequestCommand,
        target_status: ApprovalRequestStatus,
    ) -> ApprovalDecisionResult:
        expired_request: ApprovalRequest | None = None
        result: ApprovalDecisionResult | None = None
        resume_scheduled_for: ApprovalRequest | None = None
        resume_available_at: datetime | None = None

        async with self._transaction_manager.transaction():
            approval_request = await self._approval_request_repository.get_by_id_for_update(
                workspace_id=command.workspace_id,
                approval_request_id=command.approval_request_id,
            )
            if approval_request is None:
                raise ApprovalRequestNotFoundError(
                    command.approval_request_id,
                )

            if approval_request.status is not ApprovalRequestStatus.PENDING:
                result = self._resolve_existing_decision(
                    approval_request=approval_request,
                    target_status=target_status,
                    actor_reference=command.actor_reference,
                    comment=command.comment,
                )
            elif command.decided_at >= approval_request.expires_at:
                expired_request = await self._expire_locked(
                    approval_request=approval_request,
                    decided_at=command.decided_at,
                )
                resume_scheduled_for = expired_request
                resume_available_at = command.decided_at
            else:
                decided_request = self._build_decision(
                    approval_request=approval_request,
                    command=command,
                    target_status=target_status,
                )
                await self._requeue_waiting_run(
                    approval_request=approval_request,
                    requeued_at=command.decided_at,
                )
                tool_call = await self._load_pending_tool_call(
                    approval_request,
                )
                if target_status is ApprovalRequestStatus.REJECTED:
                    rejected_tool_call = tool_call.reject_for_approval(
                        decided_at=command.decided_at,
                    )
                    await self._agent_tool_call_repository.save_approval_outcome(rejected_tool_call)

                await self._approval_request_repository.save(
                    decided_request,
                )
                result = ApprovalDecisionResult(
                    approval_request=decided_request,
                    idempotent=False,
                )
                resume_scheduled_for = decided_request
                resume_available_at = command.decided_at

        if expired_request is not None:
            _safe_record_approval_expired(
                client=self._observability_client,
                approval_request=expired_request,
            )
            if resume_scheduled_for is not None and resume_available_at is not None:
                _safe_record_resume_scheduled(
                    client=self._observability_client,
                    approval_request=resume_scheduled_for,
                    available_at=resume_available_at,
                )
            raise ApprovalRequestExpiredError(expired_request)
        if result is None:
            raise RuntimeError(
                "Approval decision completed without a result.",
            )
        if not result.idempotent:
            _safe_record_approval_decision(
                client=self._observability_client,
                approval_request=result.approval_request,
            )
            if resume_scheduled_for is not None and resume_available_at is not None:
                _safe_record_resume_scheduled(
                    client=self._observability_client,
                    approval_request=resume_scheduled_for,
                    available_at=resume_available_at,
                )
        return result

    def _build_decision(
        self,
        *,
        approval_request: ApprovalRequest,
        command: ApproveApprovalRequestCommand | RejectApprovalRequestCommand,
        target_status: ApprovalRequestStatus,
    ) -> ApprovalRequest:
        if target_status is ApprovalRequestStatus.APPROVED:
            if not isinstance(command, ApproveApprovalRequestCommand):
                raise TypeError("An approval command is required.")
            return approval_request.approve(
                actor_reference=command.actor_reference,
                comment=command.comment,
                request_id=command.request_id,
                correlation_id=command.correlation_id,
                decided_at=command.decided_at,
            )

        if not isinstance(command, RejectApprovalRequestCommand):
            raise TypeError("A rejection command is required.")
        return approval_request.reject(
            actor_reference=command.actor_reference,
            comment=command.comment,
            request_id=command.request_id,
            correlation_id=command.correlation_id,
            decided_at=command.decided_at,
        )

    def _resolve_existing_decision(
        self,
        *,
        approval_request: ApprovalRequest,
        target_status: ApprovalRequestStatus,
        actor_reference: str,
        comment: str | None,
    ) -> ApprovalDecisionResult:
        if (
            approval_request.status is target_status
            and approval_request.decision_actor_reference == actor_reference
            and approval_request.decision_comment == comment
        ):
            return ApprovalDecisionResult(
                approval_request=approval_request,
                idempotent=True,
            )

        if approval_request.status is ApprovalRequestStatus.EXPIRED:
            raise ApprovalRequestExpiredError(approval_request)

        raise ApprovalDecisionConflictError(approval_request.id)

    async def _expire_locked(
        self,
        *,
        approval_request: ApprovalRequest,
        decided_at: datetime,
    ) -> ApprovalRequest:
        expired_request = approval_request.expire(
            decided_at=decided_at,
        )
        await self._requeue_waiting_run(
            approval_request=approval_request,
            requeued_at=decided_at,
        )
        tool_call = await self._load_pending_tool_call(
            approval_request,
        )
        expired_tool_call = tool_call.expire_for_approval(
            decided_at=decided_at,
        )
        await self._agent_tool_call_repository.save_approval_outcome(
            expired_tool_call,
        )
        await self._approval_request_repository.save(expired_request)
        return expired_request

    async def _requeue_waiting_run(
        self,
        *,
        approval_request: ApprovalRequest,
        requeued_at: datetime,
    ) -> None:
        result = await self._agent_run_repository.requeue_waiting_for_approval(
            RequeueWaitingAgentRunCommand(
                workspace_id=approval_request.workspace_id,
                ticket_id=approval_request.ticket_id,
                agent_run_id=approval_request.agent_run_id,
                requeued_at=requeued_at,
            ),
        )
        if result is not AgentRunApprovalRequeueResult.APPLIED:
            raise ApprovalRunStateConflictError(
                approval_request.agent_run_id,
            )

    async def _load_pending_tool_call(
        self,
        approval_request: ApprovalRequest,
    ) -> AgentToolCall:
        tool_call = await self._agent_tool_call_repository.get_by_id_for_update(
            workspace_id=approval_request.workspace_id,
            agent_tool_call_id=(approval_request.agent_tool_call_id),
        )
        if tool_call is None:
            raise ApprovalToolCallStateConflictError(
                approval_request.agent_tool_call_id,
            )

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
            raise ApprovalToolCallStateConflictError(
                approval_request.agent_tool_call_id,
            )

        return tool_call


class ExpirePendingApprovalRequests:
    """Expire a bounded number of overdue pending approvals."""

    def __init__(
        self,
        *,
        transaction_manager: TransactionManager,
        approval_request_repository: ApprovalRequestRepository,
        agent_run_repository: AgentRunRepository,
        agent_tool_call_repository: ApprovalAgentToolCallRepository,
        observability_client: ObservabilityClient | None = None,
    ) -> None:
        self._transaction_manager = transaction_manager
        self._approval_request_repository = approval_request_repository
        self._agent_run_repository = agent_run_repository
        self._agent_tool_call_repository = agent_tool_call_repository
        self._observability_client = observability_client or NoOpObservabilityClient()

    async def execute(
        self,
        command: ExpirePendingApprovalRequestsCommand,
    ) -> ApprovalExpirationBatchResult:
        """Expire pending requests using bounded SKIP LOCKED work."""

        expired_ids: list[UUID] = []

        for _ in range(command.batch_size):
            expired_id = await self._expire_next(command)
            if expired_id is None:
                break
            expired_ids.append(expired_id)

        return ApprovalExpirationBatchResult(
            approval_request_ids=tuple(expired_ids),
        )

    async def _expire_next(
        self,
        command: ExpirePendingApprovalRequestsCommand,
    ) -> UUID | None:
        expired_request: ApprovalRequest | None = None

        async with self._transaction_manager.transaction():
            approval_request = (
                await self._approval_request_repository.get_next_expired_pending_for_update(
                    now=command.now,
                )
            )
            if approval_request is None:
                return None

            expired_request = approval_request.expire(
                decided_at=command.now,
            )

            requeue_result = await self._agent_run_repository.requeue_waiting_for_approval(
                RequeueWaitingAgentRunCommand(
                    workspace_id=(approval_request.workspace_id),
                    ticket_id=approval_request.ticket_id,
                    agent_run_id=approval_request.agent_run_id,
                    requeued_at=command.now,
                ),
            )
            if requeue_result is not AgentRunApprovalRequeueResult.APPLIED:
                raise ApprovalRunStateConflictError(
                    approval_request.agent_run_id,
                )

            tool_call = await self._agent_tool_call_repository.get_by_id_for_update(
                workspace_id=(approval_request.workspace_id),
                agent_tool_call_id=(approval_request.agent_tool_call_id),
            )
            if (
                tool_call is None
                or tool_call.status is not AgentToolCallStatus.PENDING_APPROVAL
                or tool_call.safety_level is not ToolSafetyLevel.SENSITIVE_WRITE
                or tool_call.agent_run_id != approval_request.agent_run_id
                or tool_call.input_fingerprint != approval_request.input_fingerprint
            ):
                raise ApprovalToolCallStateConflictError(
                    approval_request.agent_tool_call_id,
                )

            await self._agent_tool_call_repository.save_approval_outcome(
                tool_call.expire_for_approval(
                    decided_at=command.now,
                ),
            )
            await self._approval_request_repository.save(
                expired_request,
            )

        _safe_record_approval_expired(
            client=self._observability_client,
            approval_request=expired_request,
        )
        _safe_record_resume_scheduled(
            client=self._observability_client,
            approval_request=expired_request,
            available_at=command.now,
        )
        return expired_request.id


def _safe_record_approval_decision(
    *,
    client: ObservabilityClient,
    approval_request: ApprovalRequest,
) -> None:
    if approval_request.status is ApprovalRequestStatus.APPROVED:
        event_name = "approval.approved"
    elif approval_request.status is ApprovalRequestStatus.REJECTED:
        event_name = "approval.rejected"
    else:
        return

    with suppress(Exception):
        client.record_trace_event(
            identity=agent_run_trace_identity(
                agent_run_id=approval_request.agent_run_id,
                ticket_id=approval_request.ticket_id,
            ),
            event=EventObservation(
                name=event_name,
                status=ObservationStatus.OK,
                metadata={
                    "approval_request_id": str(approval_request.id),
                    "agent_run_id": str(approval_request.agent_run_id),
                    "workspace_id": str(approval_request.workspace_id),
                    "ticket_id": str(approval_request.ticket_id),
                    "approval_status": approval_request.status.value,
                    "tool_name": approval_request.tool_name,
                },
                metadata_paths=_APPROVAL_DECISION_EVENT_METADATA_PATHS,
            ),
        )


def _safe_record_approval_expired(
    *,
    client: ObservabilityClient,
    approval_request: ApprovalRequest,
) -> None:
    with suppress(Exception):
        client.record_trace_event(
            identity=agent_run_trace_identity(
                agent_run_id=approval_request.agent_run_id,
                ticket_id=approval_request.ticket_id,
            ),
            event=EventObservation(
                name="approval.expired",
                status=ObservationStatus.OK,
                metadata={
                    "approval_request_id": str(approval_request.id),
                    "agent_run_id": str(approval_request.agent_run_id),
                    "workspace_id": str(approval_request.workspace_id),
                    "ticket_id": str(approval_request.ticket_id),
                    "approval_status": approval_request.status.value,
                    "tool_name": approval_request.tool_name,
                },
                metadata_paths=_APPROVAL_EXPIRED_EVENT_METADATA_PATHS,
            ),
        )


def _safe_record_resume_scheduled(
    *,
    client: ObservabilityClient,
    approval_request: ApprovalRequest,
    available_at: datetime,
) -> None:
    with suppress(Exception):
        client.record_trace_event(
            identity=agent_run_trace_identity(
                agent_run_id=approval_request.agent_run_id,
                ticket_id=approval_request.ticket_id,
            ),
            event=EventObservation(
                name="workflow.resume_scheduled",
                status=ObservationStatus.OK,
                metadata={
                    "approval_request_id": str(approval_request.id),
                    "agent_run_id": str(approval_request.agent_run_id),
                    "workspace_id": str(approval_request.workspace_id),
                    "ticket_id": str(approval_request.ticket_id),
                    "agent_run_status": AgentRunStatus.QUEUED.value,
                    "available_at": available_at.isoformat(),
                },
                metadata_paths=_RESUME_SCHEDULED_EVENT_METADATA_PATHS,
            ),
        )
