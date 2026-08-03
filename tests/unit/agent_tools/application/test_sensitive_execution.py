"""Unit tests for approved sensitive execution."""

from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any
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
