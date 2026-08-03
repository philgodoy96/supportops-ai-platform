"""Unit tests for transactional approval application services."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from supportops.agent_tools.domain.audit import AgentToolCallStatus
from supportops.agent_tools.domain.contracts import ToolSafetyLevel
from supportops.modules.agent_runs.domain.transitions import (
    AgentRunApprovalRequeueResult,
)
from supportops.modules.approvals.application.errors import (
    ApprovalDecisionConflictError,
    ApprovalRequestExpiredError,
    ApprovalRequestNotFoundError,
)
from supportops.modules.approvals.application.models import (
    ApproveApprovalRequestCommand,
    ExpirePendingApprovalRequestsCommand,
    RejectApprovalRequestCommand,
)
from supportops.modules.approvals.application.services import (
    DecideApprovalRequest,
    ExpirePendingApprovalRequests,
)
from supportops.modules.approvals.domain.models import (
    ApprovalRequestStatus,
)

_NOW = datetime(2026, 8, 2, 22, 45, tzinfo=UTC)


class FakeTransactionManager:
    """Minimal transaction manager for application-service tests."""

    @asynccontextmanager
    async def transaction(self) -> AsyncIterator[None]:
        yield


def _pending_approval(*, expired: bool = False) -> Any:
    expires_at = _NOW - timedelta(seconds=1) if expired else _NOW + timedelta(hours=1)
    approval = SimpleNamespace(
        id=uuid4(),
        workspace_id=uuid4(),
        ticket_id=uuid4(),
        agent_run_id=uuid4(),
        agent_tool_call_id=uuid4(),
        status=ApprovalRequestStatus.PENDING,
        tool_name="escalate_ticket",
        tool_version=1,
        input_fingerprint="a" * 64,
        proposed_input={
            "target_queue": "security",
            "reason": "Potential security incident.",
        },
        expires_at=expires_at,
        decision_actor_reference=None,
        decision_comment=None,
    )

    def build_transition(**changes: Any) -> Any:
        values = dict(approval.__dict__)
        values.pop("approve", None)
        values.pop("reject", None)
        values.pop("expire", None)
        values.update(changes)
        return SimpleNamespace(**values)

    approval.approve = lambda **kwargs: build_transition(
        status=ApprovalRequestStatus.APPROVED,
        decision_actor_reference=kwargs["actor_reference"],
        decision_comment=kwargs["comment"],
    )
    approval.reject = lambda **kwargs: build_transition(
        status=ApprovalRequestStatus.REJECTED,
        decision_actor_reference=kwargs["actor_reference"],
        decision_comment=kwargs["comment"],
    )
    approval.expire = lambda **kwargs: build_transition(
        status=ApprovalRequestStatus.EXPIRED,
        decision_actor_reference="system:approval-expiration",
        decision_comment=None,
    )
    return approval


def _pending_tool_call(approval: Any) -> Any:
    return SimpleNamespace(
        id=approval.agent_tool_call_id,
        workspace_id=approval.workspace_id,
        ticket_id=approval.ticket_id,
        agent_run_id=approval.agent_run_id,
        status=AgentToolCallStatus.PENDING_APPROVAL,
        safety_level=ToolSafetyLevel.SENSITIVE_WRITE,
        tool_name=approval.tool_name,
        tool_version=approval.tool_version,
        input_fingerprint=approval.input_fingerprint,
        safe_input=dict(approval.proposed_input),
        reject_for_approval=lambda **kwargs: SimpleNamespace(
            status=AgentToolCallStatus.REJECTED,
        ),
        expire_for_approval=lambda **kwargs: SimpleNamespace(
            status=AgentToolCallStatus.EXPIRED,
        ),
    )


def _service_dependencies(approval: Any) -> tuple[Any, Any, Any]:
    approval_repository = SimpleNamespace(
        get_by_id_for_update=AsyncMock(return_value=approval),
        get_next_expired_pending_for_update=AsyncMock(
            return_value=None,
        ),
        save=AsyncMock(),
    )
    agent_run_repository = SimpleNamespace(
        requeue_waiting_for_approval=AsyncMock(
            return_value=AgentRunApprovalRequeueResult.APPLIED,
        ),
    )
    tool_call_repository = SimpleNamespace(
        get_by_id_for_update=AsyncMock(
            return_value=_pending_tool_call(approval),
        ),
        save_approval_outcome=AsyncMock(),
    )
    return (
        approval_repository,
        agent_run_repository,
        tool_call_repository,
    )


@pytest.mark.asyncio
async def test_approve_persists_decision_and_requeues_run() -> None:
    approval = _pending_approval()
    (
        approval_repository,
        agent_run_repository,
        tool_call_repository,
    ) = _service_dependencies(approval)
    service = DecideApprovalRequest(
        transaction_manager=FakeTransactionManager(),
        approval_request_repository=approval_repository,
        agent_run_repository=agent_run_repository,
        agent_tool_call_repository=tool_call_repository,
    )

    result = await service.approve(
        ApproveApprovalRequestCommand(
            workspace_id=approval.workspace_id,
            approval_request_id=approval.id,
            actor_reference="operator:alice",
            comment=None,
            request_id=uuid4(),
            correlation_id=uuid4(),
            decided_at=_NOW,
        ),
    )

    assert result.idempotent is False
    assert result.approval_request.status is ApprovalRequestStatus.APPROVED
    agent_run_repository.requeue_waiting_for_approval.assert_awaited_once()
    approval_repository.save.assert_awaited_once()
    tool_call_repository.save_approval_outcome.assert_not_awaited()


@pytest.mark.asyncio
async def test_reject_finalizes_tool_call_without_execution() -> None:
    approval = _pending_approval()
    (
        approval_repository,
        agent_run_repository,
        tool_call_repository,
    ) = _service_dependencies(approval)
    service = DecideApprovalRequest(
        transaction_manager=FakeTransactionManager(),
        approval_request_repository=approval_repository,
        agent_run_repository=agent_run_repository,
        agent_tool_call_repository=tool_call_repository,
    )

    result = await service.reject(
        RejectApprovalRequestCommand(
            workspace_id=approval.workspace_id,
            approval_request_id=approval.id,
            actor_reference="operator:alice",
            comment="Escalation is not required.",
            request_id=uuid4(),
            correlation_id=uuid4(),
            decided_at=_NOW,
        ),
    )

    assert result.approval_request.status is ApprovalRequestStatus.REJECTED
    tool_call_repository.save_approval_outcome.assert_awaited_once()


@pytest.mark.asyncio
async def test_missing_request_raises_not_found() -> None:
    approval = _pending_approval()
    (
        approval_repository,
        agent_run_repository,
        tool_call_repository,
    ) = _service_dependencies(approval)
    approval_repository.get_by_id_for_update.return_value = None
    service = DecideApprovalRequest(
        transaction_manager=FakeTransactionManager(),
        approval_request_repository=approval_repository,
        agent_run_repository=agent_run_repository,
        agent_tool_call_repository=tool_call_repository,
    )

    with pytest.raises(ApprovalRequestNotFoundError):
        await service.approve(
            ApproveApprovalRequestCommand(
                workspace_id=approval.workspace_id,
                approval_request_id=approval.id,
                actor_reference="operator:alice",
                comment=None,
                request_id=uuid4(),
                correlation_id=uuid4(),
                decided_at=_NOW,
            ),
        )


@pytest.mark.asyncio
async def test_conflicting_terminal_decision_is_rejected() -> None:
    approval = _pending_approval()
    approval.status = ApprovalRequestStatus.REJECTED
    approval.decision_actor_reference = "operator:bob"
    approval.decision_comment = "Rejected."
    (
        approval_repository,
        agent_run_repository,
        tool_call_repository,
    ) = _service_dependencies(approval)
    service = DecideApprovalRequest(
        transaction_manager=FakeTransactionManager(),
        approval_request_repository=approval_repository,
        agent_run_repository=agent_run_repository,
        agent_tool_call_repository=tool_call_repository,
    )

    with pytest.raises(ApprovalDecisionConflictError):
        await service.approve(
            ApproveApprovalRequestCommand(
                workspace_id=approval.workspace_id,
                approval_request_id=approval.id,
                actor_reference="operator:alice",
                comment=None,
                request_id=uuid4(),
                correlation_id=uuid4(),
                decided_at=_NOW,
            ),
        )


@pytest.mark.asyncio
async def test_overdue_decision_expires_then_raises() -> None:
    approval = _pending_approval(expired=True)
    (
        approval_repository,
        agent_run_repository,
        tool_call_repository,
    ) = _service_dependencies(approval)
    service = DecideApprovalRequest(
        transaction_manager=FakeTransactionManager(),
        approval_request_repository=approval_repository,
        agent_run_repository=agent_run_repository,
        agent_tool_call_repository=tool_call_repository,
    )

    with pytest.raises(ApprovalRequestExpiredError):
        await service.approve(
            ApproveApprovalRequestCommand(
                workspace_id=approval.workspace_id,
                approval_request_id=approval.id,
                actor_reference="operator:alice",
                comment=None,
                request_id=uuid4(),
                correlation_id=uuid4(),
                decided_at=_NOW,
            ),
        )

    approval_repository.save.assert_awaited_once()
    tool_call_repository.save_approval_outcome.assert_awaited_once()
    agent_run_repository.requeue_waiting_for_approval.assert_awaited_once()


@pytest.mark.asyncio
async def test_expiration_batch_is_bounded() -> None:
    first = _pending_approval(expired=True)
    second = _pending_approval(expired=True)
    approval_repository = SimpleNamespace(
        get_next_expired_pending_for_update=AsyncMock(
            side_effect=[first, second],
        ),
        save=AsyncMock(),
    )
    agent_run_repository = SimpleNamespace(
        requeue_waiting_for_approval=AsyncMock(
            return_value=AgentRunApprovalRequeueResult.APPLIED,
        ),
    )
    tool_call_repository = SimpleNamespace(
        get_by_id_for_update=AsyncMock(
            side_effect=[
                _pending_tool_call(first),
                _pending_tool_call(second),
            ],
        ),
        save_approval_outcome=AsyncMock(),
    )
    service = ExpirePendingApprovalRequests(
        transaction_manager=FakeTransactionManager(),
        approval_request_repository=approval_repository,
        agent_run_repository=agent_run_repository,
        agent_tool_call_repository=tool_call_repository,
    )

    result = await service.execute(
        ExpirePendingApprovalRequestsCommand(
            now=_NOW,
            batch_size=2,
        ),
    )

    assert result.expired_count == 2
    assert result.approval_request_ids == (first.id, second.id)
