"""Unit tests for transactional approval application services."""

from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any
from uuid import UUID, uuid4

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
    ApprovalRunStateConflictError,
    ApprovalToolCallStateConflictError,
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
_WORKSPACE_ID = UUID("aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa")
_APPROVAL_ID = UUID("bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb")
_TICKET_ID = UUID("cccccccc-cccc-4ccc-8ccc-cccccccccccc")
_AGENT_RUN_ID = UUID("dddddddd-dddd-4ddd-8ddd-dddddddddddd")
_TOOL_CALL_ID = UUID("eeeeeeee-eeee-4eee-8eee-eeeeeeeeeeee")
_REQUEST_ID = UUID("ffffffff-ffff-4fff-8fff-ffffffffffff")
_CORRELATION_ID = UUID("11111111-1111-4111-8111-111111111111")


class RecordingTransactionManager:
    """Record transaction enter/exit order and exception exits."""

    def __init__(self, operations: list[str]) -> None:
        self.operations = operations
        self.entries = 0
        self.clean_exits = 0
        self.exception_exits: list[BaseException] = []

    @asynccontextmanager
    async def transaction(self) -> AsyncIterator[None]:
        self.entries += 1
        self.operations.append("transaction_enter")
        try:
            yield
        except BaseException as exc:
            self.operations.append("transaction_exit_exception")
            self.exception_exits.append(exc)
            raise
        else:
            self.operations.append("transaction_exit")
            self.clean_exits += 1


def _pending_approval(*, expired: bool = False) -> Any:
    expires_at = _NOW - timedelta(seconds=1) if expired else _NOW + timedelta(hours=1)
    approval = SimpleNamespace(
        id=_APPROVAL_ID,
        workspace_id=_WORKSPACE_ID,
        ticket_id=_TICKET_ID,
        agent_run_id=_AGENT_RUN_ID,
        agent_tool_call_id=_TOOL_CALL_ID,
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
        decision_request_id=None,
        decision_correlation_id=None,
        decided_at=None,
    )

    def build_transition(**changes: Any) -> Any:
        values = {
            key: value
            for key, value in approval.__dict__.items()
            if key not in {"approve", "reject", "expire"}
        }
        values.update(changes)
        return SimpleNamespace(**values)

    approval.approve = lambda **kwargs: build_transition(
        status=ApprovalRequestStatus.APPROVED,
        decision_actor_reference=kwargs["actor_reference"],
        decision_comment=kwargs["comment"],
        decision_request_id=kwargs["request_id"],
        decision_correlation_id=kwargs["correlation_id"],
        decided_at=kwargs["decided_at"],
    )
    approval.reject = lambda **kwargs: build_transition(
        status=ApprovalRequestStatus.REJECTED,
        decision_actor_reference=kwargs["actor_reference"],
        decision_comment=kwargs["comment"],
        decision_request_id=kwargs["request_id"],
        decision_correlation_id=kwargs["correlation_id"],
        decided_at=kwargs["decided_at"],
    )
    approval.expire = lambda **kwargs: build_transition(
        status=ApprovalRequestStatus.EXPIRED,
        decision_actor_reference="system:approval-expiration",
        decision_comment=None,
        decision_request_id=None,
        decision_correlation_id=None,
        decided_at=kwargs["decided_at"],
    )
    return approval


def _pending_tool_call(
    approval: Any,
    *,
    mutate: Callable[[Any], None] | None = None,
) -> Any:
    tool_call = SimpleNamespace(
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
            decided_at=kwargs["decided_at"],
        ),
        expire_for_approval=lambda **kwargs: SimpleNamespace(
            status=AgentToolCallStatus.EXPIRED,
            decided_at=kwargs["decided_at"],
        ),
    )
    if mutate is not None:
        mutate(tool_call)
    return tool_call


def _recording_dependencies(
    approval: Any,
    *,
    operations: list[str] | None = None,
    tool_call: Any | None = ...,
    requeue_result: AgentRunApprovalRequeueResult = (AgentRunApprovalRequeueResult.APPLIED),
    approval_lookup: Any | None = ...,
) -> tuple[Any, Any, Any, RecordingTransactionManager]:
    ops = operations if operations is not None else []
    transaction_manager = RecordingTransactionManager(ops)
    resolved_lookup = approval if approval_lookup is ... else approval_lookup
    if tool_call is ...:
        resolved_tool_call: Any | None = _pending_tool_call(approval)
    else:
        resolved_tool_call = tool_call

    async def get_approval(**kwargs: Any) -> Any:
        del kwargs
        ops.append("approval.get_by_id_for_update")
        return resolved_lookup

    async def save_approval(request: Any) -> None:
        del request
        ops.append("approval.save")

    async def get_next_expired(**kwargs: Any) -> Any:
        del kwargs
        ops.append("approval.get_next_expired_pending_for_update")
        return None

    async def requeue(command: Any) -> AgentRunApprovalRequeueResult:
        del command
        ops.append("agent_run.requeue_waiting_for_approval")
        return requeue_result

    async def get_tool_call(**kwargs: Any) -> Any:
        del kwargs
        ops.append("tool_call.get_by_id_for_update")
        return resolved_tool_call

    async def save_tool_outcome(tool_call_value: Any) -> None:
        del tool_call_value
        ops.append("tool_call.save_approval_outcome")

    approval_repository = SimpleNamespace(
        get_by_id_for_update=get_approval,
        get_next_expired_pending_for_update=get_next_expired,
        save=save_approval,
    )
    agent_run_repository = SimpleNamespace(
        requeue_waiting_for_approval=requeue,
    )
    tool_call_repository = SimpleNamespace(
        get_by_id_for_update=get_tool_call,
        save_approval_outcome=save_tool_outcome,
    )
    return (
        approval_repository,
        agent_run_repository,
        tool_call_repository,
        transaction_manager,
    )


def _decide_service(
    approval: Any,
    *,
    operations: list[str] | None = None,
    tool_call: Any | None = ...,
    requeue_result: AgentRunApprovalRequeueResult = (AgentRunApprovalRequeueResult.APPLIED),
    approval_lookup: Any | None = ...,
) -> tuple[DecideApprovalRequest, list[str], RecordingTransactionManager]:
    ops = operations if operations is not None else []
    (
        approval_repository,
        agent_run_repository,
        tool_call_repository,
        transaction_manager,
    ) = _recording_dependencies(
        approval,
        operations=ops,
        tool_call=tool_call,
        requeue_result=requeue_result,
        approval_lookup=approval_lookup,
    )
    service = DecideApprovalRequest(
        transaction_manager=transaction_manager,
        approval_request_repository=approval_repository,
        agent_run_repository=agent_run_repository,
        agent_tool_call_repository=tool_call_repository,
    )
    return service, ops, transaction_manager


def _approve_command(
    *,
    workspace_id: UUID = _WORKSPACE_ID,
    approval_request_id: UUID = _APPROVAL_ID,
    actor_reference: str = "operator:alice",
    comment: str | None = None,
    request_id: UUID = _REQUEST_ID,
    correlation_id: UUID = _CORRELATION_ID,
    decided_at: datetime = _NOW,
) -> ApproveApprovalRequestCommand:
    return ApproveApprovalRequestCommand(
        workspace_id=workspace_id,
        approval_request_id=approval_request_id,
        actor_reference=actor_reference,
        comment=comment,
        request_id=request_id,
        correlation_id=correlation_id,
        decided_at=decided_at,
    )


def _reject_command(
    *,
    workspace_id: UUID = _WORKSPACE_ID,
    approval_request_id: UUID = _APPROVAL_ID,
    actor_reference: str = "operator:alice",
    comment: str = "Escalation is not required.",
    request_id: UUID = _REQUEST_ID,
    correlation_id: UUID = _CORRELATION_ID,
    decided_at: datetime = _NOW,
) -> RejectApprovalRequestCommand:
    return RejectApprovalRequestCommand(
        workspace_id=workspace_id,
        approval_request_id=approval_request_id,
        actor_reference=actor_reference,
        comment=comment,
        request_id=request_id,
        correlation_id=correlation_id,
        decided_at=decided_at,
    )


@pytest.mark.asyncio
async def test_approve_persists_decision_and_requeues_run() -> None:
    approval = _pending_approval()
    service, ops, transaction_manager = _decide_service(approval)

    result = await service.approve(_approve_command())

    assert result.idempotent is False
    assert result.approval_request.status is ApprovalRequestStatus.APPROVED
    assert ops == [
        "transaction_enter",
        "approval.get_by_id_for_update",
        "agent_run.requeue_waiting_for_approval",
        "tool_call.get_by_id_for_update",
        "approval.save",
        "transaction_exit",
    ]
    assert transaction_manager.clean_exits == 1
    assert transaction_manager.exception_exits == []


@pytest.mark.asyncio
async def test_reject_finalizes_tool_call_without_execution() -> None:
    approval = _pending_approval()
    service, ops, transaction_manager = _decide_service(approval)

    result = await service.reject(_reject_command())

    assert result.idempotent is False
    assert result.approval_request.status is ApprovalRequestStatus.REJECTED
    assert ops == [
        "transaction_enter",
        "approval.get_by_id_for_update",
        "agent_run.requeue_waiting_for_approval",
        "tool_call.get_by_id_for_update",
        "tool_call.save_approval_outcome",
        "approval.save",
        "transaction_exit",
    ]
    assert transaction_manager.clean_exits == 1


@pytest.mark.asyncio
async def test_approve_exact_retry_is_idempotent_without_side_effects() -> None:
    pending = _pending_approval()
    store: dict[str, Any] = {"current": pending}
    ops: list[str] = []
    transaction_manager = RecordingTransactionManager(ops)

    async def get_approval(**kwargs: Any) -> Any:
        del kwargs
        ops.append("approval.get_by_id_for_update")
        return store["current"]

    async def save_approval(request: Any) -> None:
        ops.append("approval.save")
        store["current"] = request

    async def requeue(command: Any) -> AgentRunApprovalRequeueResult:
        del command
        ops.append("agent_run.requeue_waiting_for_approval")
        return AgentRunApprovalRequeueResult.APPLIED

    async def get_tool_call(**kwargs: Any) -> Any:
        del kwargs
        ops.append("tool_call.get_by_id_for_update")
        return _pending_tool_call(pending)

    async def save_tool_outcome(tool_call_value: Any) -> None:
        del tool_call_value
        ops.append("tool_call.save_approval_outcome")

    service = DecideApprovalRequest(
        transaction_manager=transaction_manager,
        approval_request_repository=SimpleNamespace(
            get_by_id_for_update=get_approval,
            save=save_approval,
        ),
        agent_run_repository=SimpleNamespace(
            requeue_waiting_for_approval=requeue,
        ),
        agent_tool_call_repository=SimpleNamespace(
            get_by_id_for_update=get_tool_call,
            save_approval_outcome=save_tool_outcome,
        ),
    )

    first = await service.approve(_approve_command())
    assert first.idempotent is False
    assert store["current"].status is ApprovalRequestStatus.APPROVED

    retry_ops_start = len(ops)
    second = await service.approve(_approve_command())

    assert second.idempotent is True
    assert second.approval_request.status is ApprovalRequestStatus.APPROVED
    assert second.approval_request.decision_actor_reference == "operator:alice"
    assert second.approval_request.decision_comment is None
    retry_ops = ops[retry_ops_start:]
    assert retry_ops == [
        "transaction_enter",
        "approval.get_by_id_for_update",
        "transaction_exit",
    ]
    assert "agent_run.requeue_waiting_for_approval" not in retry_ops
    assert "approval.save" not in retry_ops
    assert "tool_call.save_approval_outcome" not in retry_ops
    assert "tool_call.get_by_id_for_update" not in retry_ops


@pytest.mark.asyncio
async def test_approve_conflicts_on_different_actor_comment_or_reject() -> None:
    approved = _pending_approval()
    approved.status = ApprovalRequestStatus.APPROVED
    approved.decision_actor_reference = "operator:alice"
    approved.decision_comment = None

    service, _, transaction_manager = _decide_service(
        approved,
        approval_lookup=approved,
    )

    with pytest.raises(ApprovalDecisionConflictError):
        await service.approve(_approve_command(actor_reference="operator:bob"))

    with pytest.raises(ApprovalDecisionConflictError):
        await service.approve(_approve_command(comment="Different."))

    with pytest.raises(ApprovalDecisionConflictError):
        await service.reject(_reject_command())

    assert transaction_manager.exception_exits
    assert all(
        isinstance(exc, ApprovalDecisionConflictError)
        for exc in transaction_manager.exception_exits
    )


@pytest.mark.asyncio
async def test_reject_exact_retry_is_idempotent_without_side_effects() -> None:
    pending = _pending_approval()
    store: dict[str, Any] = {"current": pending}
    ops: list[str] = []
    transaction_manager = RecordingTransactionManager(ops)

    async def get_approval(**kwargs: Any) -> Any:
        del kwargs
        ops.append("approval.get_by_id_for_update")
        return store["current"]

    async def save_approval(request: Any) -> None:
        ops.append("approval.save")
        store["current"] = request

    async def requeue(command: Any) -> AgentRunApprovalRequeueResult:
        del command
        ops.append("agent_run.requeue_waiting_for_approval")
        return AgentRunApprovalRequeueResult.APPLIED

    async def get_tool_call(**kwargs: Any) -> Any:
        del kwargs
        ops.append("tool_call.get_by_id_for_update")
        return _pending_tool_call(pending)

    async def save_tool_outcome(tool_call_value: Any) -> None:
        del tool_call_value
        ops.append("tool_call.save_approval_outcome")

    service = DecideApprovalRequest(
        transaction_manager=transaction_manager,
        approval_request_repository=SimpleNamespace(
            get_by_id_for_update=get_approval,
            save=save_approval,
        ),
        agent_run_repository=SimpleNamespace(
            requeue_waiting_for_approval=requeue,
        ),
        agent_tool_call_repository=SimpleNamespace(
            get_by_id_for_update=get_tool_call,
            save_approval_outcome=save_tool_outcome,
        ),
    )

    first = await service.reject(_reject_command())
    assert first.idempotent is False

    retry_ops_start = len(ops)
    second = await service.reject(_reject_command())

    assert second.idempotent is True
    retry_ops = ops[retry_ops_start:]
    assert retry_ops == [
        "transaction_enter",
        "approval.get_by_id_for_update",
        "transaction_exit",
    ]
    assert "agent_run.requeue_waiting_for_approval" not in retry_ops
    assert "tool_call.save_approval_outcome" not in retry_ops
    assert "approval.save" not in retry_ops


@pytest.mark.asyncio
async def test_reject_conflicts_on_different_actor_comment_or_approve() -> None:
    rejected = _pending_approval()
    rejected.status = ApprovalRequestStatus.REJECTED
    rejected.decision_actor_reference = "operator:alice"
    rejected.decision_comment = "Escalation is not required."

    service, _, _ = _decide_service(rejected, approval_lookup=rejected)

    with pytest.raises(ApprovalDecisionConflictError):
        await service.reject(_reject_command(actor_reference="operator:bob"))

    with pytest.raises(ApprovalDecisionConflictError):
        await service.reject(_reject_command(comment="Other reason."))

    with pytest.raises(ApprovalDecisionConflictError):
        await service.approve(_approve_command())


@pytest.mark.asyncio
async def test_terminal_expired_request_raises_without_writes() -> None:
    expired = _pending_approval()
    expired.status = ApprovalRequestStatus.EXPIRED
    expired.decision_actor_reference = "system:approval-expiration"
    expired.decided_at = _NOW

    service, ops, transaction_manager = _decide_service(
        expired,
        approval_lookup=expired,
    )

    with pytest.raises(ApprovalRequestExpiredError):
        await service.approve(_approve_command())

    assert ops == [
        "transaction_enter",
        "approval.get_by_id_for_update",
        "transaction_exit_exception",
    ]
    assert "approval.save" not in ops
    assert "tool_call.save_approval_outcome" not in ops
    assert "agent_run.requeue_waiting_for_approval" not in ops
    assert isinstance(
        transaction_manager.exception_exits[0],
        ApprovalRequestExpiredError,
    )


@pytest.mark.asyncio
async def test_missing_request_raises_not_found() -> None:
    approval = _pending_approval()
    service, ops, transaction_manager = _decide_service(
        approval,
        approval_lookup=None,
    )

    with pytest.raises(ApprovalRequestNotFoundError) as exc_info:
        await service.approve(_approve_command())

    assert exc_info.value.approval_request_id == _APPROVAL_ID
    assert "approval.save" not in ops
    assert isinstance(
        transaction_manager.exception_exits[0],
        ApprovalRequestNotFoundError,
    )


@pytest.mark.asyncio
async def test_overdue_decision_expires_then_raises_after_transaction_exit() -> None:
    approval = _pending_approval(expired=True)
    ops: list[str] = []
    service, _, transaction_manager = _decide_service(approval, operations=ops)

    with pytest.raises(ApprovalRequestExpiredError) as exc_info:
        await service.approve(_approve_command())

    assert ops == [
        "transaction_enter",
        "approval.get_by_id_for_update",
        "agent_run.requeue_waiting_for_approval",
        "tool_call.get_by_id_for_update",
        "tool_call.save_approval_outcome",
        "approval.save",
        "transaction_exit",
    ]
    assert transaction_manager.clean_exits == 1
    assert transaction_manager.exception_exits == []
    assert exc_info.value.approval_request.status is (ApprovalRequestStatus.EXPIRED)


@pytest.mark.parametrize(
    ("conflict_factory", "expected_error"),
    [
        (
            lambda: {
                "requeue_result": AgentRunApprovalRequeueResult.STATE_CONFLICT,
            },
            ApprovalRunStateConflictError,
        ),
        (
            lambda: {"tool_call": None},
            ApprovalToolCallStateConflictError,
        ),
        (
            lambda: {
                "tool_call": _pending_tool_call(
                    _pending_approval(),
                    mutate=lambda tool: setattr(
                        tool,
                        "workspace_id",
                        uuid4(),
                    ),
                ),
            },
            ApprovalToolCallStateConflictError,
        ),
        (
            lambda: {
                "tool_call": _pending_tool_call(
                    _pending_approval(),
                    mutate=lambda tool: setattr(tool, "ticket_id", uuid4()),
                ),
            },
            ApprovalToolCallStateConflictError,
        ),
        (
            lambda: {
                "tool_call": _pending_tool_call(
                    _pending_approval(),
                    mutate=lambda tool: setattr(tool, "agent_run_id", uuid4()),
                ),
            },
            ApprovalToolCallStateConflictError,
        ),
        (
            lambda: {
                "tool_call": _pending_tool_call(
                    _pending_approval(),
                    mutate=lambda tool: setattr(
                        tool,
                        "status",
                        AgentToolCallStatus.SUCCEEDED,
                    ),
                ),
            },
            ApprovalToolCallStateConflictError,
        ),
        (
            lambda: {
                "tool_call": _pending_tool_call(
                    _pending_approval(),
                    mutate=lambda tool: setattr(
                        tool,
                        "safety_level",
                        ToolSafetyLevel.READ_ONLY,
                    ),
                ),
            },
            ApprovalToolCallStateConflictError,
        ),
        (
            lambda: {
                "tool_call": _pending_tool_call(
                    _pending_approval(),
                    mutate=lambda tool: setattr(
                        tool,
                        "tool_name",
                        "create_escalation",
                    ),
                ),
            },
            ApprovalToolCallStateConflictError,
        ),
        (
            lambda: {
                "tool_call": _pending_tool_call(
                    _pending_approval(),
                    mutate=lambda tool: setattr(tool, "tool_version", 2),
                ),
            },
            ApprovalToolCallStateConflictError,
        ),
        (
            lambda: {
                "tool_call": _pending_tool_call(
                    _pending_approval(),
                    mutate=lambda tool: setattr(
                        tool,
                        "input_fingerprint",
                        "b" * 64,
                    ),
                ),
            },
            ApprovalToolCallStateConflictError,
        ),
        (
            lambda: {
                "tool_call": _pending_tool_call(
                    _pending_approval(),
                    mutate=lambda tool: setattr(
                        tool,
                        "safe_input",
                        {"target_queue": "other"},
                    ),
                ),
            },
            ApprovalToolCallStateConflictError,
        ),
    ],
)
@pytest.mark.asyncio
async def test_decision_conflicts_roll_back_without_persisting(
    conflict_factory: Callable[[], dict[str, Any]],
    expected_error: type[Exception],
) -> None:
    approval = _pending_approval()
    conflict_kwargs = conflict_factory()
    ops: list[str] = []
    service, _, transaction_manager = _decide_service(
        approval,
        operations=ops,
        **conflict_kwargs,
    )

    with pytest.raises(expected_error):
        await service.approve(_approve_command())

    assert "approval.save" not in ops
    assert "tool_call.save_approval_outcome" not in ops
    assert ops[-1] == "transaction_exit_exception"
    assert isinstance(transaction_manager.exception_exits[0], expected_error)


@pytest.mark.asyncio
async def test_expiration_batch_empty_queue_returns_zero() -> None:
    ops: list[str] = []
    transaction_manager = RecordingTransactionManager(ops)

    async def get_next(**kwargs: Any) -> None:
        del kwargs
        ops.append("approval.get_next_expired_pending_for_update")
        return None

    async def fail_save(request: Any) -> None:
        del request
        raise AssertionError("approval.save must not be called")

    async def fail_requeue(command: Any) -> AgentRunApprovalRequeueResult:
        del command
        raise AssertionError("requeue must not be called")

    async def fail_tool_lock(**kwargs: Any) -> Any:
        del kwargs
        raise AssertionError("tool lock must not be called")

    async def fail_tool_save(tool: Any) -> None:
        del tool
        raise AssertionError("tool save must not be called")

    service = ExpirePendingApprovalRequests(
        transaction_manager=transaction_manager,
        approval_request_repository=SimpleNamespace(
            get_next_expired_pending_for_update=get_next,
            save=fail_save,
        ),
        agent_run_repository=SimpleNamespace(
            requeue_waiting_for_approval=fail_requeue,
        ),
        agent_tool_call_repository=SimpleNamespace(
            get_by_id_for_update=fail_tool_lock,
            save_approval_outcome=fail_tool_save,
        ),
    )

    result = await service.execute(
        ExpirePendingApprovalRequestsCommand(now=_NOW, batch_size=5),
    )

    assert result.expired_count == 0
    assert result.approval_request_ids == ()
    assert transaction_manager.entries == 1
    assert ops == [
        "transaction_enter",
        "approval.get_next_expired_pending_for_update",
        "transaction_exit",
    ]


@pytest.mark.asyncio
async def test_expiration_batch_is_bounded_and_ordered() -> None:
    first = _pending_approval(expired=True)
    first.id = UUID("10000000-0000-4000-8000-000000000001")
    second = _pending_approval(expired=True)
    second.id = UUID("20000000-0000-4000-8000-000000000002")
    third = _pending_approval(expired=True)
    third.id = UUID("30000000-0000-4000-8000-000000000003")
    queue = [first, second, third]
    ops: list[str] = []
    transaction_manager = RecordingTransactionManager(ops)
    save_count = {"approval": 0, "tool": 0, "requeue": 0}
    tool_lookup_index = {"value": 0}

    async def get_next(**kwargs: Any) -> Any:
        del kwargs
        ops.append("approval.get_next_expired_pending_for_update")
        if not queue:
            return None
        return queue.pop(0)

    async def save_approval(request: Any) -> None:
        del request
        ops.append("approval.save")
        save_count["approval"] += 1

    async def requeue(command: Any) -> AgentRunApprovalRequeueResult:
        del command
        ops.append("agent_run.requeue_waiting_for_approval")
        save_count["requeue"] += 1
        return AgentRunApprovalRequeueResult.APPLIED

    async def get_tool_ordered(**kwargs: Any) -> Any:
        del kwargs
        ops.append("tool_call.get_by_id_for_update")
        current = [first, second][tool_lookup_index["value"]]
        tool_lookup_index["value"] += 1
        return _pending_tool_call(current)

    async def save_tool(tool: Any) -> None:
        del tool
        ops.append("tool_call.save_approval_outcome")
        save_count["tool"] += 1

    service = ExpirePendingApprovalRequests(
        transaction_manager=transaction_manager,
        approval_request_repository=SimpleNamespace(
            get_next_expired_pending_for_update=get_next,
            save=save_approval,
        ),
        agent_run_repository=SimpleNamespace(
            requeue_waiting_for_approval=requeue,
        ),
        agent_tool_call_repository=SimpleNamespace(
            get_by_id_for_update=get_tool_ordered,
            save_approval_outcome=save_tool,
        ),
    )

    result = await service.execute(
        ExpirePendingApprovalRequestsCommand(now=_NOW, batch_size=2),
    )

    assert result.expired_count == 2
    assert result.approval_request_ids == (first.id, second.id)
    assert save_count == {"approval": 2, "tool": 2, "requeue": 2}
    assert transaction_manager.entries == 2
    assert transaction_manager.clean_exits == 2
    assert queue == [third]
    assert "claim" not in "".join(ops)
    assert "workflow" not in "".join(ops)


@pytest.mark.asyncio
async def test_expiration_batch_stops_when_repository_returns_none() -> None:
    first = _pending_approval(expired=True)
    first.id = UUID("10000000-0000-4000-8000-000000000001")
    ops: list[str] = []
    transaction_manager = RecordingTransactionManager(ops)
    responses: list[Any] = [first, None]

    async def get_next(**kwargs: Any) -> Any:
        del kwargs
        ops.append("approval.get_next_expired_pending_for_update")
        return responses.pop(0)

    async def save_approval(request: Any) -> None:
        del request
        ops.append("approval.save")

    async def requeue(command: Any) -> AgentRunApprovalRequeueResult:
        del command
        ops.append("agent_run.requeue_waiting_for_approval")
        return AgentRunApprovalRequeueResult.APPLIED

    async def get_tool(**kwargs: Any) -> Any:
        del kwargs
        ops.append("tool_call.get_by_id_for_update")
        return _pending_tool_call(first)

    async def save_tool(tool: Any) -> None:
        del tool
        ops.append("tool_call.save_approval_outcome")

    service = ExpirePendingApprovalRequests(
        transaction_manager=transaction_manager,
        approval_request_repository=SimpleNamespace(
            get_next_expired_pending_for_update=get_next,
            save=save_approval,
        ),
        agent_run_repository=SimpleNamespace(
            requeue_waiting_for_approval=requeue,
        ),
        agent_tool_call_repository=SimpleNamespace(
            get_by_id_for_update=get_tool,
            save_approval_outcome=save_tool,
        ),
    )

    result = await service.execute(
        ExpirePendingApprovalRequestsCommand(now=_NOW, batch_size=5),
    )

    assert result.approval_request_ids == (first.id,)
    assert transaction_manager.entries == 2


@pytest.mark.asyncio
async def test_expiration_batch_stops_on_row_conflict() -> None:
    first = _pending_approval(expired=True)
    first.id = UUID("10000000-0000-4000-8000-000000000001")
    second = _pending_approval(expired=True)
    second.id = UUID("20000000-0000-4000-8000-000000000002")
    ops: list[str] = []
    transaction_manager = RecordingTransactionManager(ops)
    queue = [first, second]
    saved_ids: list[UUID] = []

    async def get_next(**kwargs: Any) -> Any:
        del kwargs
        ops.append("approval.get_next_expired_pending_for_update")
        return queue.pop(0)

    async def save_approval(request: Any) -> None:
        ops.append("approval.save")
        saved_ids.append(request.id)

    async def requeue(command: Any) -> AgentRunApprovalRequeueResult:
        del command
        ops.append("agent_run.requeue_waiting_for_approval")
        return AgentRunApprovalRequeueResult.STATE_CONFLICT

    async def get_tool(**kwargs: Any) -> Any:
        del kwargs
        ops.append("tool_call.get_by_id_for_update")
        return _pending_tool_call(first)

    async def save_tool(tool: Any) -> None:
        del tool
        ops.append("tool_call.save_approval_outcome")

    service = ExpirePendingApprovalRequests(
        transaction_manager=transaction_manager,
        approval_request_repository=SimpleNamespace(
            get_next_expired_pending_for_update=get_next,
            save=save_approval,
        ),
        agent_run_repository=SimpleNamespace(
            requeue_waiting_for_approval=requeue,
        ),
        agent_tool_call_repository=SimpleNamespace(
            get_by_id_for_update=get_tool,
            save_approval_outcome=save_tool,
        ),
    )

    with pytest.raises(ApprovalRunStateConflictError):
        await service.execute(
            ExpirePendingApprovalRequestsCommand(now=_NOW, batch_size=5),
        )

    assert saved_ids == []
    assert "tool_call.save_approval_outcome" not in ops
    assert queue == [second]
    assert transaction_manager.entries == 1
    assert isinstance(
        transaction_manager.exception_exits[0],
        ApprovalRunStateConflictError,
    )
