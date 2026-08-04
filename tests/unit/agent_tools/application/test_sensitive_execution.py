"""Unit tests for approved sensitive execution."""

from __future__ import annotations

from contextlib import AbstractContextManager, asynccontextmanager
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace, TracebackType
from typing import Any, Literal
from unittest.mock import AsyncMock, call
from uuid import UUID, uuid4

import pytest

from supportops.agent_tools.application.grant_persistence import (
    SensitiveExecutionGrantConsistencyError,
    SensitiveExecutionGrantPersistenceResult,
)
from supportops.agent_tools.application.sensitive_execution import (
    ExecuteApprovedTicketEscalation,
    SensitiveExecutionConsistencyError,
    SensitiveExecutionStatus,
)
from supportops.agent_tools.domain.audit import AgentToolCall
from supportops.agent_tools.domain.grants import SensitiveExecutionGrant
from supportops.agent_tools.tools.escalate_ticket import (
    EscalateTicketInput,
    TicketEscalationTargetQueue,
)
from supportops.modules.approvals.domain.models import ApprovalRequest
from supportops.modules.tickets.domain.escalation import TicketEscalation
from supportops.modules.tickets.domain.escalation_repositories import (
    TicketEscalationConsistencyError,
    TicketEscalationPersistenceResult,
)
from supportops.observability.context import (
    ActiveObservationContext,
    current_observation_context,
    observation_context_scope,
)
from supportops.observability.contracts import TraceScope
from supportops.observability.models import (
    EventObservation,
    ObservabilityProvider,
    ObservationAttributes,
    ObservationStatus,
    ObservationType,
    ObservationUpdate,
    TraceAttributes,
)

_NOW = datetime(2026, 8, 3, 19, 0, tzinfo=UTC)


class FakeTransactionManager:
    def __init__(self) -> None:
        self.entered = 0

    @asynccontextmanager
    async def transaction(self) -> Any:
        self.entered += 1
        yield


def _records() -> tuple[Any, AgentToolCall, ApprovalRequest]:
    workspace_id = uuid4()
    ticket_id = uuid4()
    agent_run_id = uuid4()
    attempt_id = uuid4()
    tool_call = AgentToolCall.propose_for_approval(
        workspace_id=workspace_id,
        ticket_id=ticket_id,
        agent_run_id=agent_run_id,
        proposed_by_agent_run_attempt_id=uuid4(),
        sequence=1,
        provider_tool_call_id="call-1",
        tool_name="escalate_ticket",
        tool_version=1,
        input_fingerprint="a" * 64,
        safe_input={
            "target_queue": "engineering_support",
            "reason": "A product defect requires review.",
        },
        proposed_at=_NOW,
    )
    approval = ApprovalRequest.create_pending(
        tool_call=tool_call,
        requested_by_llm_invocation_id=uuid4(),
        request_reason="A product defect requires review.",
        expires_at=_NOW + timedelta(days=1),
        now=_NOW,
    ).approve(
        actor_reference="operator:alice",
        comment=None,
        request_id=uuid4(),
        correlation_id=uuid4(),
        decided_at=_NOW + timedelta(minutes=5),
    )
    context = SimpleNamespace(
        agent_run=SimpleNamespace(
            id=agent_run_id,
            workspace_id=workspace_id,
        ),
        ticket=SimpleNamespace(id=ticket_id),
        attempt=SimpleNamespace(id=attempt_id),
    )
    return context, tool_call, approval


def _grant(
    approval: ApprovalRequest,
    tool_call: AgentToolCall,
    *,
    executed_by_agent_run_attempt_id: UUID,
    grant_id: UUID | None = None,
) -> SensitiveExecutionGrant:
    return SensitiveExecutionGrant.create(
        approval_request=approval,
        tool_call=tool_call,
        executed_by_agent_run_attempt_id=executed_by_agent_run_attempt_id,
        created_at=_NOW + timedelta(minutes=6),
        grant_id=grant_id or uuid4(),
    )


def _escalation(
    grant: SensitiveExecutionGrant,
    *,
    escalation_id: UUID | None = None,
) -> TicketEscalation:
    return TicketEscalation.create_from_grant(
        grant=grant,
        input_data=EscalateTicketInput.model_validate(
            dict(grant.granted_input),
        ),
        created_at=_NOW + timedelta(minutes=6),
        escalation_id=escalation_id or uuid4(),
    )


def _executor(
    *,
    context: Any,
    tool_call: AgentToolCall,
    approval: ApprovalRequest,
    grant_repository: Any,
    escalation_repository: Any,
    tool_repository: Any | None = None,
    transaction_manager: FakeTransactionManager | None = None,
    observability_client: object | None = None,
) -> ExecuteApprovedTicketEscalation:
    return ExecuteApprovedTicketEscalation(
        transaction_manager=transaction_manager or FakeTransactionManager(),
        approval_request_repository=SimpleNamespace(
            get_by_id_for_update=AsyncMock(
                return_value=approval,
            ),
        ),
        tool_call_repository=tool_repository
        or SimpleNamespace(
            get_by_id_for_update=AsyncMock(
                return_value=tool_call,
            ),
            save_granted_execution_success=AsyncMock(),
        ),
        grant_repository=grant_repository,
        escalation_repository=escalation_repository,
        utc_now=lambda: _NOW + timedelta(minutes=6),
        uuid_factory=uuid4,
        observability_client=observability_client,  # type: ignore[arg-type]
    )


@pytest.mark.asyncio
async def test_execution_persists_grant_escalation_and_success() -> None:
    context, tool_call, approval = _records()
    transaction_manager = FakeTransactionManager()
    grant_repository = SimpleNamespace(
        persist=AsyncMock(
            return_value=(SensitiveExecutionGrantPersistenceResult.APPLIED),
        ),
        get_by_agent_tool_call_id=AsyncMock(),
    )
    escalation_repository = SimpleNamespace(
        persist=AsyncMock(
            return_value=TicketEscalationPersistenceResult.APPLIED,
        ),
        get_by_agent_tool_call_id=AsyncMock(),
    )
    tool_repository = SimpleNamespace(
        get_by_id_for_update=AsyncMock(
            return_value=tool_call,
        ),
        save_granted_execution_success=AsyncMock(),
    )
    executor = _executor(
        context=context,
        tool_call=tool_call,
        approval=approval,
        grant_repository=grant_repository,
        escalation_repository=escalation_repository,
        tool_repository=tool_repository,
        transaction_manager=transaction_manager,
    )

    result = await executor.execute(
        context=context,
        approval_request_id=approval.id,
        agent_tool_call_id=tool_call.id,
    )

    assert result.status is SensitiveExecutionStatus.APPLIED
    assert result.output.status == "escalated"
    assert set(result.output.model_dump(mode="json")) == {
        "escalation_id",
        "ticket_id",
        "target_queue",
        "status",
    }
    assert transaction_manager.entered == 1
    grant_repository.persist.assert_awaited_once()
    escalation_repository.persist.assert_awaited_once()
    tool_repository.save_granted_execution_success.assert_awaited_once()
    assert tool_repository.get_by_id_for_update.await_args_list == [
        call(
            workspace_id=context.agent_run.workspace_id,
            agent_tool_call_id=tool_call.id,
        ),
    ]


@pytest.mark.asyncio
async def test_rejected_approval_never_executes() -> None:
    context, tool_call, _approval = _records()
    rejected = ApprovalRequest.create_pending(
        tool_call=tool_call,
        requested_by_llm_invocation_id=uuid4(),
        request_reason="A product defect requires review.",
        expires_at=_NOW + timedelta(days=1),
        now=_NOW,
    ).reject(
        actor_reference="operator:alice",
        comment="Do not escalate.",
        request_id=uuid4(),
        correlation_id=uuid4(),
        decided_at=_NOW + timedelta(minutes=5),
    )
    grant_repository = SimpleNamespace(
        persist=AsyncMock(),
        get_by_agent_tool_call_id=AsyncMock(),
    )
    executor = _executor(
        context=context,
        tool_call=tool_call,
        approval=rejected,
        grant_repository=grant_repository,
        escalation_repository=SimpleNamespace(
            persist=AsyncMock(),
            get_by_agent_tool_call_id=AsyncMock(),
        ),
    )

    with pytest.raises(
        SensitiveExecutionConsistencyError,
        match="approved",
    ):
        await executor.execute(
            context=context,
            approval_request_id=rejected.id,
            agent_tool_call_id=tool_call.id,
        )

    grant_repository.persist.assert_not_awaited()


@pytest.mark.asyncio
async def test_pending_approval_never_executes() -> None:
    context, tool_call, _approval = _records()
    pending = ApprovalRequest.create_pending(
        tool_call=tool_call,
        requested_by_llm_invocation_id=uuid4(),
        request_reason="A product defect requires review.",
        expires_at=_NOW + timedelta(days=1),
        now=_NOW,
    )
    grant_repository = SimpleNamespace(
        persist=AsyncMock(),
        get_by_agent_tool_call_id=AsyncMock(),
    )
    executor = _executor(
        context=context,
        tool_call=tool_call,
        approval=pending,
        grant_repository=grant_repository,
        escalation_repository=SimpleNamespace(
            persist=AsyncMock(),
            get_by_agent_tool_call_id=AsyncMock(),
        ),
    )

    with pytest.raises(
        SensitiveExecutionConsistencyError,
        match="approved",
    ):
        await executor.execute(
            context=context,
            approval_request_id=pending.id,
            agent_tool_call_id=tool_call.id,
        )

    grant_repository.persist.assert_not_awaited()


@pytest.mark.asyncio
async def test_ownership_mismatch_fails_before_grant() -> None:
    context, tool_call, approval = _records()
    context.ticket.id = uuid4()
    grant_repository = SimpleNamespace(
        persist=AsyncMock(),
        get_by_agent_tool_call_id=AsyncMock(),
    )
    executor = _executor(
        context=context,
        tool_call=tool_call,
        approval=approval,
        grant_repository=grant_repository,
        escalation_repository=SimpleNamespace(
            persist=AsyncMock(),
            get_by_agent_tool_call_id=AsyncMock(),
        ),
    )

    with pytest.raises(
        SensitiveExecutionConsistencyError,
        match="do not match",
    ):
        await executor.execute(
            context=context,
            approval_request_id=approval.id,
            agent_tool_call_id=tool_call.id,
        )

    grant_repository.persist.assert_not_awaited()


@pytest.mark.asyncio
async def test_grant_conflict_fails_closed() -> None:
    context, tool_call, approval = _records()
    grant_repository = SimpleNamespace(
        persist=AsyncMock(
            side_effect=SensitiveExecutionGrantConsistencyError(
                "conflicting grant",
            ),
        ),
        get_by_agent_tool_call_id=AsyncMock(),
    )
    escalation_repository = SimpleNamespace(
        persist=AsyncMock(),
        get_by_agent_tool_call_id=AsyncMock(),
    )
    executor = _executor(
        context=context,
        tool_call=tool_call,
        approval=approval,
        grant_repository=grant_repository,
        escalation_repository=escalation_repository,
    )

    with pytest.raises(SensitiveExecutionGrantConsistencyError):
        await executor.execute(
            context=context,
            approval_request_id=approval.id,
            agent_tool_call_id=tool_call.id,
        )

    escalation_repository.persist.assert_not_awaited()


@pytest.mark.asyncio
async def test_escalation_conflict_fails_closed() -> None:
    context, tool_call, approval = _records()
    grant_repository = SimpleNamespace(
        persist=AsyncMock(
            return_value=(SensitiveExecutionGrantPersistenceResult.APPLIED),
        ),
        get_by_agent_tool_call_id=AsyncMock(),
    )
    escalation_repository = SimpleNamespace(
        persist=AsyncMock(
            side_effect=TicketEscalationConsistencyError(
                "conflicting escalation",
            ),
        ),
        get_by_agent_tool_call_id=AsyncMock(),
    )
    tool_repository = SimpleNamespace(
        get_by_id_for_update=AsyncMock(return_value=tool_call),
        save_granted_execution_success=AsyncMock(),
    )
    executor = _executor(
        context=context,
        tool_call=tool_call,
        approval=approval,
        grant_repository=grant_repository,
        escalation_repository=escalation_repository,
        tool_repository=tool_repository,
    )

    with pytest.raises(TicketEscalationConsistencyError):
        await executor.execute(
            context=context,
            approval_request_id=approval.id,
            agent_tool_call_id=tool_call.id,
        )

    tool_repository.save_granted_execution_success.assert_not_awaited()


@pytest.mark.asyncio
async def test_grant_replay_missing_escalation_recovers() -> None:
    context, tool_call, approval = _records()
    durable_grant = _grant(
        approval,
        tool_call,
        executed_by_agent_run_attempt_id=context.attempt.id,
    )
    grant_repository = SimpleNamespace(
        persist=AsyncMock(
            return_value=(SensitiveExecutionGrantPersistenceResult.ALREADY_RECORDED),
        ),
        get_by_agent_tool_call_id=AsyncMock(
            return_value=durable_grant,
        ),
    )
    escalation_repository = SimpleNamespace(
        persist=AsyncMock(
            return_value=TicketEscalationPersistenceResult.APPLIED,
        ),
        get_by_agent_tool_call_id=AsyncMock(),
    )
    tool_repository = SimpleNamespace(
        get_by_id_for_update=AsyncMock(return_value=tool_call),
        save_granted_execution_success=AsyncMock(),
    )
    executor = _executor(
        context=context,
        tool_call=tool_call,
        approval=approval,
        grant_repository=grant_repository,
        escalation_repository=escalation_repository,
        tool_repository=tool_repository,
    )

    result = await executor.execute(
        context=context,
        approval_request_id=approval.id,
        agent_tool_call_id=tool_call.id,
    )

    assert result.status is SensitiveExecutionStatus.ALREADY_RECORDED
    assert result.grant.id == durable_grant.id
    escalation_repository.persist.assert_awaited_once()
    tool_repository.save_granted_execution_success.assert_awaited_once()


@pytest.mark.asyncio
async def test_grant_and_escalation_replay_completes_pending_tool() -> None:
    context, tool_call, approval = _records()
    durable_grant = _grant(
        approval,
        tool_call,
        executed_by_agent_run_attempt_id=context.attempt.id,
    )
    durable_escalation = _escalation(durable_grant)
    grant_repository = SimpleNamespace(
        persist=AsyncMock(
            return_value=(SensitiveExecutionGrantPersistenceResult.ALREADY_RECORDED),
        ),
        get_by_agent_tool_call_id=AsyncMock(
            return_value=durable_grant,
        ),
    )
    escalation_repository = SimpleNamespace(
        persist=AsyncMock(
            return_value=(TicketEscalationPersistenceResult.ALREADY_RECORDED),
        ),
        get_by_agent_tool_call_id=AsyncMock(
            return_value=durable_escalation,
        ),
    )
    tool_repository = SimpleNamespace(
        get_by_id_for_update=AsyncMock(return_value=tool_call),
        save_granted_execution_success=AsyncMock(),
    )
    executor = _executor(
        context=context,
        tool_call=tool_call,
        approval=approval,
        grant_repository=grant_repository,
        escalation_repository=escalation_repository,
        tool_repository=tool_repository,
    )

    result = await executor.execute(
        context=context,
        approval_request_id=approval.id,
        agent_tool_call_id=tool_call.id,
    )

    assert result.status is SensitiveExecutionStatus.ALREADY_RECORDED
    assert result.escalation.id == durable_escalation.id
    tool_repository.save_granted_execution_success.assert_awaited_once()


@pytest.mark.asyncio
async def test_succeeded_replay_returns_existing() -> None:
    context, tool_call, approval = _records()
    durable_grant = _grant(
        approval,
        tool_call,
        executed_by_agent_run_attempt_id=context.attempt.id,
    )
    durable_escalation = _escalation(durable_grant)
    succeeded = tool_call.complete_granted_execution_success(
        executed_by_agent_run_attempt_id=context.attempt.id,
        execution_started_at=_NOW + timedelta(minutes=6),
        finished_at=_NOW + timedelta(minutes=6),
        safe_output={
            "escalation_id": str(durable_escalation.id),
            "ticket_id": str(tool_call.ticket_id),
            "target_queue": (TicketEscalationTargetQueue.ENGINEERING_SUPPORT.value),
            "status": "escalated",
        },
    )
    grant_repository = SimpleNamespace(
        persist=AsyncMock(),
        get_by_agent_tool_call_id=AsyncMock(
            return_value=durable_grant,
        ),
    )
    escalation_repository = SimpleNamespace(
        persist=AsyncMock(),
        get_by_agent_tool_call_id=AsyncMock(
            return_value=durable_escalation,
        ),
    )
    tool_repository = SimpleNamespace(
        get_by_id_for_update=AsyncMock(return_value=succeeded),
        save_granted_execution_success=AsyncMock(),
    )
    executor = _executor(
        context=context,
        tool_call=succeeded,
        approval=approval,
        grant_repository=grant_repository,
        escalation_repository=escalation_repository,
        tool_repository=tool_repository,
    )

    result = await executor.execute(
        context=context,
        approval_request_id=approval.id,
        agent_tool_call_id=tool_call.id,
    )

    assert result.status is SensitiveExecutionStatus.ALREADY_RECORDED
    assert result.grant.id == durable_grant.id
    assert result.escalation.id == durable_escalation.id
    grant_repository.persist.assert_not_awaited()
    escalation_repository.persist.assert_not_awaited()
    tool_repository.save_granted_execution_success.assert_not_awaited()


@pytest.mark.asyncio
async def test_succeeded_without_grant_fails_closed() -> None:
    context, tool_call, approval = _records()
    durable_grant = _grant(
        approval,
        tool_call,
        executed_by_agent_run_attempt_id=context.attempt.id,
    )
    durable_escalation = _escalation(durable_grant)
    succeeded = tool_call.complete_granted_execution_success(
        executed_by_agent_run_attempt_id=context.attempt.id,
        execution_started_at=_NOW + timedelta(minutes=6),
        finished_at=_NOW + timedelta(minutes=6),
        safe_output={
            "escalation_id": str(durable_escalation.id),
            "ticket_id": str(tool_call.ticket_id),
            "target_queue": (TicketEscalationTargetQueue.ENGINEERING_SUPPORT.value),
            "status": "escalated",
        },
    )
    grant_repository = SimpleNamespace(
        persist=AsyncMock(),
        get_by_agent_tool_call_id=AsyncMock(return_value=None),
    )
    executor = _executor(
        context=context,
        tool_call=succeeded,
        approval=approval,
        grant_repository=grant_repository,
        escalation_repository=SimpleNamespace(
            persist=AsyncMock(),
            get_by_agent_tool_call_id=AsyncMock(
                return_value=durable_escalation,
            ),
        ),
        tool_repository=SimpleNamespace(
            get_by_id_for_update=AsyncMock(return_value=succeeded),
            save_granted_execution_success=AsyncMock(),
        ),
    )

    with pytest.raises(
        SensitiveExecutionConsistencyError,
        match="missing",
    ):
        await executor.execute(
            context=context,
            approval_request_id=approval.id,
            agent_tool_call_id=tool_call.id,
        )


@pytest.mark.asyncio
async def test_succeeded_without_escalation_fails_closed() -> None:
    context, tool_call, approval = _records()
    durable_grant = _grant(
        approval,
        tool_call,
        executed_by_agent_run_attempt_id=context.attempt.id,
    )
    succeeded = tool_call.complete_granted_execution_success(
        executed_by_agent_run_attempt_id=context.attempt.id,
        execution_started_at=_NOW + timedelta(minutes=6),
        finished_at=_NOW + timedelta(minutes=6),
        safe_output={
            "escalation_id": str(uuid4()),
            "ticket_id": str(tool_call.ticket_id),
            "target_queue": (TicketEscalationTargetQueue.ENGINEERING_SUPPORT.value),
            "status": "escalated",
        },
    )
    executor = _executor(
        context=context,
        tool_call=succeeded,
        approval=approval,
        grant_repository=SimpleNamespace(
            persist=AsyncMock(),
            get_by_agent_tool_call_id=AsyncMock(
                return_value=durable_grant,
            ),
        ),
        escalation_repository=SimpleNamespace(
            persist=AsyncMock(),
            get_by_agent_tool_call_id=AsyncMock(return_value=None),
        ),
        tool_repository=SimpleNamespace(
            get_by_id_for_update=AsyncMock(return_value=succeeded),
            save_granted_execution_success=AsyncMock(),
        ),
    )

    with pytest.raises(
        SensitiveExecutionConsistencyError,
        match="missing",
    ):
        await executor.execute(
            context=context,
            approval_request_id=approval.id,
            agent_tool_call_id=tool_call.id,
        )


@pytest.mark.asyncio
async def test_expired_approval_never_creates_grant() -> None:
    context, tool_call, _approval = _records()
    expired = ApprovalRequest.create_pending(
        tool_call=tool_call,
        requested_by_llm_invocation_id=uuid4(),
        request_reason="A product defect requires review.",
        expires_at=_NOW + timedelta(days=1),
        now=_NOW,
    ).expire(
        decided_at=_NOW + timedelta(days=1),
    )
    grant_repository = SimpleNamespace(
        persist=AsyncMock(),
        get_by_agent_tool_call_id=AsyncMock(),
    )
    executor = _executor(
        context=context,
        tool_call=tool_call,
        approval=expired,
        grant_repository=grant_repository,
        escalation_repository=SimpleNamespace(
            persist=AsyncMock(),
            get_by_agent_tool_call_id=AsyncMock(),
        ),
    )

    with pytest.raises(
        SensitiveExecutionConsistencyError,
        match="approved",
    ):
        await executor.execute(
            context=context,
            approval_request_id=expired.id,
            agent_tool_call_id=tool_call.id,
        )

    grant_repository.persist.assert_not_awaited()


@pytest.mark.asyncio
async def test_execution_locks_approval_before_tool_call() -> None:
    context, tool_call, approval = _records()
    order: list[str] = []

    async def lock_approval(**kwargs: object) -> ApprovalRequest:
        del kwargs
        order.append("approval")
        return approval

    async def lock_tool(**kwargs: object) -> AgentToolCall:
        del kwargs
        order.append("tool_call")
        return tool_call

    async def persist_grant(grant: object) -> SensitiveExecutionGrantPersistenceResult:
        del grant
        order.append("grant")
        return SensitiveExecutionGrantPersistenceResult.APPLIED

    async def persist_escalation(
        escalation: object,
    ) -> TicketEscalationPersistenceResult:
        del escalation
        order.append("escalation")
        return TicketEscalationPersistenceResult.APPLIED

    async def save_success(*, tool_call: AgentToolCall) -> None:
        del tool_call
        order.append("tool_success")

    executor = ExecuteApprovedTicketEscalation(
        transaction_manager=FakeTransactionManager(),
        approval_request_repository=SimpleNamespace(
            get_by_id_for_update=AsyncMock(side_effect=lock_approval),
        ),
        tool_call_repository=SimpleNamespace(
            get_by_id_for_update=AsyncMock(side_effect=lock_tool),
            save_granted_execution_success=AsyncMock(
                side_effect=save_success,
            ),
        ),
        grant_repository=SimpleNamespace(
            persist=AsyncMock(side_effect=persist_grant),
            get_by_agent_tool_call_id=AsyncMock(),
        ),
        escalation_repository=SimpleNamespace(
            persist=AsyncMock(side_effect=persist_escalation),
            get_by_agent_tool_call_id=AsyncMock(),
        ),
        utc_now=lambda: _NOW + timedelta(minutes=6),
        uuid_factory=uuid4,
    )

    result = await executor.execute(
        context=context,
        approval_request_id=approval.id,
        agent_tool_call_id=tool_call.id,
    )

    assert result.status is SensitiveExecutionStatus.APPLIED
    assert order == [
        "approval",
        "tool_call",
        "grant",
        "escalation",
        "tool_success",
    ]


@pytest.mark.asyncio
async def test_exact_replay_returns_already_recorded() -> None:
    context, tool_call, approval = _records()
    durable_grant = _grant(
        approval,
        tool_call,
        executed_by_agent_run_attempt_id=context.attempt.id,
    )
    durable_escalation = _escalation(durable_grant)
    succeeded = tool_call.complete_granted_execution_success(
        executed_by_agent_run_attempt_id=context.attempt.id,
        execution_started_at=_NOW + timedelta(minutes=6),
        finished_at=_NOW + timedelta(minutes=6),
        safe_output={
            "escalation_id": str(durable_escalation.id),
            "ticket_id": str(durable_escalation.ticket_id),
            "target_queue": "engineering_support",
            "status": "escalated",
        },
    )
    grant_repository = SimpleNamespace(
        persist=AsyncMock(),
        get_by_agent_tool_call_id=AsyncMock(
            return_value=durable_grant,
        ),
    )
    escalation_repository = SimpleNamespace(
        persist=AsyncMock(),
        get_by_agent_tool_call_id=AsyncMock(
            return_value=durable_escalation,
        ),
    )
    tool_repository = SimpleNamespace(
        get_by_id_for_update=AsyncMock(return_value=succeeded),
        save_granted_execution_success=AsyncMock(),
    )
    executor = _executor(
        context=context,
        tool_call=succeeded,
        approval=approval,
        grant_repository=grant_repository,
        escalation_repository=escalation_repository,
        tool_repository=tool_repository,
    )

    result = await executor.execute(
        context=context,
        approval_request_id=approval.id,
        agent_tool_call_id=tool_call.id,
    )

    assert result.status is (SensitiveExecutionStatus.ALREADY_RECORDED)
    assert result.grant.id == durable_grant.id
    assert result.escalation.id == durable_escalation.id
    grant_repository.persist.assert_not_awaited()
    escalation_repository.persist.assert_not_awaited()
    tool_repository.save_granted_execution_success.assert_not_awaited()


class RecordingObservationScope:
    def __init__(
        self,
        *,
        attributes: ObservationAttributes,
        fail_update: bool = False,
    ) -> None:
        self.attributes = attributes
        self._fail_update = fail_update
        self.updates: list[ObservationUpdate] = []

    @property
    def observation_id(self) -> str | None:
        return "sensitive-tool-observation-1"

    def update(self, update: ObservationUpdate) -> None:
        if self._fail_update:
            raise RuntimeError("synthetic update failure")

        self.updates.append(update)

    def start_observation(
        self,
        attributes: ObservationAttributes,
    ) -> AbstractContextManager[RecordingObservationScope]:
        del attributes
        raise AssertionError("Nested observations are not expected.")

    def record_event(self, event: EventObservation) -> None:
        del event
        raise AssertionError("Events are not expected.")


class RecordingObservationManager(AbstractContextManager[RecordingObservationScope]):
    def __init__(
        self,
        *,
        scope: RecordingObservationScope,
        fail_enter: bool = False,
        fail_exit: bool = False,
    ) -> None:
        self._scope = scope
        self._fail_enter = fail_enter
        self._fail_exit = fail_exit
        self.exit_calls = 0
        self._context_manager = observation_context_scope(
            ActiveObservationContext(
                name=scope.attributes.name,
                observation_id=scope.observation_id,
            )
        )

    def __enter__(self) -> RecordingObservationScope:
        if self._fail_enter:
            raise RuntimeError("synthetic enter failure")

        self._context_manager.__enter__()
        return self._scope

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> Literal[False]:
        self.exit_calls += 1
        self._context_manager.__exit__(exc_type, exc, traceback)

        if self._fail_exit:
            raise RuntimeError("synthetic exit failure")

        return False


class RecordingObservabilityClient:
    def __init__(
        self,
        *,
        fail_start: bool = False,
        fail_enter: bool = False,
        fail_update: bool = False,
        fail_exit: bool = False,
    ) -> None:
        self._fail_start = fail_start
        self._fail_enter = fail_enter
        self._fail_update = fail_update
        self._fail_exit = fail_exit
        self.attributes: list[ObservationAttributes] = []
        self.scopes: list[RecordingObservationScope] = []
        self.managers: list[RecordingObservationManager] = []
        self.parent_observation_names: list[str | None] = []

    @property
    def provider(self) -> ObservabilityProvider:
        return ObservabilityProvider.NOOP

    @property
    def enabled(self) -> bool:
        return True

    def start_trace(
        self,
        attributes: TraceAttributes,
    ) -> AbstractContextManager[TraceScope]:
        del attributes
        raise AssertionError("Tool tracing must not create roots.")

    def start_observation(
        self,
        attributes: ObservationAttributes,
    ) -> AbstractContextManager[RecordingObservationScope]:
        if self._fail_start:
            raise RuntimeError("synthetic start failure")

        parent = current_observation_context()
        self.parent_observation_names.append(
            None if parent is None else parent.name,
        )
        scope = RecordingObservationScope(
            attributes=attributes,
            fail_update=self._fail_update,
        )
        manager = RecordingObservationManager(
            scope=scope,
            fail_enter=self._fail_enter,
            fail_exit=self._fail_exit,
        )
        self.attributes.append(attributes)
        self.scopes.append(scope)
        self.managers.append(manager)
        return manager

    def record_event(self, event: EventObservation) -> None:
        del event
        raise AssertionError("Tool tracing must not emit events.")

    def record_trace_event(self, *, identity: object, event: EventObservation) -> None:
        del identity, event
        raise AssertionError("Tool tracing must not emit events.")

    def flush(self) -> None:
        return None

    def shutdown(self) -> None:
        return None


def _applied_repositories() -> tuple[Any, Any, Any]:
    grant_repository = SimpleNamespace(
        persist=AsyncMock(
            return_value=(SensitiveExecutionGrantPersistenceResult.APPLIED),
        ),
        get_by_agent_tool_call_id=AsyncMock(),
    )
    escalation_repository = SimpleNamespace(
        persist=AsyncMock(
            return_value=TicketEscalationPersistenceResult.APPLIED,
        ),
        get_by_agent_tool_call_id=AsyncMock(),
    )
    return grant_repository, escalation_repository, None


@pytest.mark.asyncio
async def test_records_one_tool_observation_for_approved_sensitive_execution() -> None:
    context, tool_call, approval = _records()
    context.agent_run.correlation_id = uuid4()
    context.attempt.execution_request_id = uuid4()
    grant_repository, escalation_repository, _ = _applied_repositories()
    tool_repository = SimpleNamespace(
        get_by_id_for_update=AsyncMock(return_value=tool_call),
        save_granted_execution_success=AsyncMock(),
    )
    observability = RecordingObservabilityClient()
    executor = _executor(
        context=context,
        tool_call=tool_call,
        approval=approval,
        grant_repository=grant_repository,
        escalation_repository=escalation_repository,
        tool_repository=tool_repository,
        observability_client=observability,
    )

    result = await executor.execute(
        context=context,
        approval_request_id=approval.id,
        agent_tool_call_id=tool_call.id,
    )

    assert result.status is SensitiveExecutionStatus.APPLIED
    assert len(observability.attributes) == 1
    attributes = observability.attributes[0]
    assert attributes.name == "tool.execute"
    assert attributes.observation_type is ObservationType.TOOL
    assert attributes.metadata["tool_name"] == "escalate_ticket"
    assert attributes.metadata["tool_safety"] == "sensitive_write"
    assert attributes.metadata["requires_approval"] is True
    assert attributes.metadata["tool_call_id"] == str(tool_call.id)
    assert attributes.metadata["correlation_id"] == str(
        context.agent_run.correlation_id,
    )
    assert attributes.metadata["execution_request_id"] == str(
        context.attempt.execution_request_id,
    )
    assert "lease_token" not in attributes.metadata
    assert "execution_grant" not in attributes.metadata
    assert "granted_input" not in attributes.metadata
    assert attributes.input_data is None
    update = observability.scopes[0].updates[0]
    assert update.status is ObservationStatus.OK
    assert update.metadata["tool_outcome"] == "succeeded"
    assert update.output_data is None
    tool_repository.save_granted_execution_success.assert_awaited_once()
    assert current_observation_context() is None


@pytest.mark.asyncio
async def test_already_recorded_before_execution_creates_no_tool_observation() -> None:
    context, tool_call, approval = _records()
    durable_grant = _grant(
        approval,
        tool_call,
        executed_by_agent_run_attempt_id=context.attempt.id,
    )
    durable_escalation = _escalation(durable_grant)
    succeeded = tool_call.complete_granted_execution_success(
        executed_by_agent_run_attempt_id=context.attempt.id,
        execution_started_at=_NOW + timedelta(minutes=6),
        finished_at=_NOW + timedelta(minutes=6),
        safe_output={
            "escalation_id": str(durable_escalation.id),
            "ticket_id": str(durable_escalation.ticket_id),
            "target_queue": "engineering_support",
            "status": "escalated",
        },
    )
    observability = RecordingObservabilityClient()
    executor = _executor(
        context=context,
        tool_call=succeeded,
        approval=approval,
        grant_repository=SimpleNamespace(
            persist=AsyncMock(),
            get_by_agent_tool_call_id=AsyncMock(return_value=durable_grant),
        ),
        escalation_repository=SimpleNamespace(
            persist=AsyncMock(),
            get_by_agent_tool_call_id=AsyncMock(
                return_value=durable_escalation,
            ),
        ),
        tool_repository=SimpleNamespace(
            get_by_id_for_update=AsyncMock(return_value=succeeded),
            save_granted_execution_success=AsyncMock(),
        ),
        observability_client=observability,
    )

    result = await executor.execute(
        context=context,
        approval_request_id=approval.id,
        agent_tool_call_id=tool_call.id,
    )

    assert result.status is SensitiveExecutionStatus.ALREADY_RECORDED
    assert observability.attributes == []


@pytest.mark.asyncio
async def test_idempotent_in_boundary_records_already_recorded_outcome() -> None:
    context, tool_call, approval = _records()
    durable_grant = _grant(
        approval,
        tool_call,
        executed_by_agent_run_attempt_id=context.attempt.id,
    )
    durable_escalation = _escalation(durable_grant)
    observability = RecordingObservabilityClient()
    tool_repository = SimpleNamespace(
        get_by_id_for_update=AsyncMock(return_value=tool_call),
        save_granted_execution_success=AsyncMock(),
    )
    executor = _executor(
        context=context,
        tool_call=tool_call,
        approval=approval,
        grant_repository=SimpleNamespace(
            persist=AsyncMock(
                return_value=(SensitiveExecutionGrantPersistenceResult.ALREADY_RECORDED),
            ),
            get_by_agent_tool_call_id=AsyncMock(return_value=durable_grant),
        ),
        escalation_repository=SimpleNamespace(
            persist=AsyncMock(
                return_value=(TicketEscalationPersistenceResult.ALREADY_RECORDED),
            ),
            get_by_agent_tool_call_id=AsyncMock(
                return_value=durable_escalation,
            ),
        ),
        tool_repository=tool_repository,
        observability_client=observability,
    )

    result = await executor.execute(
        context=context,
        approval_request_id=approval.id,
        agent_tool_call_id=tool_call.id,
    )

    assert result.status is SensitiveExecutionStatus.ALREADY_RECORDED
    assert len(observability.attributes) == 1
    update = observability.scopes[0].updates[0]
    assert update.status is ObservationStatus.OK
    assert update.metadata["tool_outcome"] == "already_recorded"
    assert update.metadata["idempotent_replay"] is True
    tool_repository.save_granted_execution_success.assert_awaited_once()


@pytest.mark.asyncio
async def test_sensitive_consistency_failure_inside_boundary_is_error() -> None:
    context, tool_call, approval = _records()
    observability = RecordingObservabilityClient()
    expected = SensitiveExecutionConsistencyError("Recorded grant could not be loaded.")
    executor = _executor(
        context=context,
        tool_call=tool_call,
        approval=approval,
        grant_repository=SimpleNamespace(
            persist=AsyncMock(
                return_value=(SensitiveExecutionGrantPersistenceResult.ALREADY_RECORDED),
            ),
            get_by_agent_tool_call_id=AsyncMock(return_value=None),
        ),
        escalation_repository=SimpleNamespace(
            persist=AsyncMock(),
            get_by_agent_tool_call_id=AsyncMock(),
        ),
        observability_client=observability,
    )

    with pytest.raises(SensitiveExecutionConsistencyError) as raised:
        await executor.execute(
            context=context,
            approval_request_id=approval.id,
            agent_tool_call_id=tool_call.id,
        )

    assert type(raised.value) is SensitiveExecutionConsistencyError
    assert str(raised.value) == str(expected)
    assert len(observability.attributes) == 1
    update = observability.scopes[0].updates[0]
    assert update.status is ObservationStatus.ERROR
    assert update.metadata["tool_outcome"] == "unexpected_failure"
    assert update.metadata["error_code"] == ("tool_execution_unexpected_failure")
    assert current_observation_context() is None


@pytest.mark.asyncio
async def test_proposal_waiting_paths_are_outside_tool_observation_boundary() -> None:
    """Approval proposal and wait paths never enter this executor boundary."""

    observability = RecordingObservabilityClient()
    context, tool_call, _approval = _records()
    pending_approval = ApprovalRequest.create_pending(
        tool_call=tool_call,
        requested_by_llm_invocation_id=uuid4(),
        request_reason="A product defect requires review.",
        expires_at=_NOW + timedelta(days=1),
        now=_NOW,
    )
    executor = _executor(
        context=context,
        tool_call=tool_call,
        approval=pending_approval,
        grant_repository=SimpleNamespace(
            persist=AsyncMock(),
            get_by_agent_tool_call_id=AsyncMock(),
        ),
        escalation_repository=SimpleNamespace(
            persist=AsyncMock(),
            get_by_agent_tool_call_id=AsyncMock(),
        ),
        observability_client=observability,
    )

    with pytest.raises(SensitiveExecutionConsistencyError):
        await executor.execute(
            context=context,
            approval_request_id=pending_approval.id,
            agent_tool_call_id=tool_call.id,
        )

    assert observability.attributes == []


@pytest.mark.asyncio
async def test_sensitive_observation_omits_approval_and_escalation_content() -> None:
    context, tool_call, approval = _records()
    grant_repository, escalation_repository, _ = _applied_repositories()
    observability = RecordingObservabilityClient()
    executor = _executor(
        context=context,
        tool_call=tool_call,
        approval=approval,
        grant_repository=grant_repository,
        escalation_repository=escalation_repository,
        observability_client=observability,
    )

    await executor.execute(
        context=context,
        approval_request_id=approval.id,
        agent_tool_call_id=tool_call.id,
    )

    exported = repr(observability.attributes[0]) + repr(observability.scopes[0].updates[0])
    assert "A product defect requires review." not in exported
    assert "proposed_input" not in exported
    assert "approval_comment" not in exported
    assert "escalation_reason" not in exported
    assert "engineering_support" not in exported


@pytest.mark.asyncio
async def test_sensitive_observability_failures_fail_open() -> None:
    context, tool_call, approval = _records()
    grant_repository, escalation_repository, _ = _applied_repositories()

    for kwargs in (
        {"fail_start": True},
        {"fail_enter": True},
        {"fail_update": True},
        {"fail_exit": True},
    ):
        tool_repository = SimpleNamespace(
            get_by_id_for_update=AsyncMock(return_value=tool_call),
            save_granted_execution_success=AsyncMock(),
        )
        result = await _executor(
            context=context,
            tool_call=tool_call,
            approval=approval,
            grant_repository=grant_repository,
            escalation_repository=escalation_repository,
            tool_repository=tool_repository,
            observability_client=RecordingObservabilityClient(**kwargs),
        ).execute(
            context=context,
            approval_request_id=approval.id,
            agent_tool_call_id=tool_call.id,
        )

        assert result.status is SensitiveExecutionStatus.APPLIED
        tool_repository.save_granted_execution_success.assert_awaited_once()
        assert current_observation_context() is None


@pytest.mark.asyncio
async def test_sensitive_tool_observation_nests_under_active_parent() -> None:
    context, tool_call, approval = _records()
    grant_repository, escalation_repository, _ = _applied_repositories()
    observability = RecordingObservabilityClient()
    executor = _executor(
        context=context,
        tool_call=tool_call,
        approval=approval,
        grant_repository=grant_repository,
        escalation_repository=escalation_repository,
        observability_client=observability,
    )

    with observation_context_scope(
        ActiveObservationContext(
            name="graph-node.execute_sensitive_tool",
            observation_id="node-1",
        )
    ):
        await executor.execute(
            context=context,
            approval_request_id=approval.id,
            agent_tool_call_id=tool_call.id,
        )

    assert observability.parent_observation_names == ["graph-node.execute_sensitive_tool"]
    assert current_observation_context() is None
