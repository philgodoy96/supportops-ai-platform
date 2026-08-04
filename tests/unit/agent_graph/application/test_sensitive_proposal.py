"""Unit tests for durable sensitive proposal preparation."""

from collections.abc import AsyncIterator
from contextlib import AbstractContextManager, asynccontextmanager
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from supportops.agent_graph.application.sensitive_proposal import (
    SensitiveProposalCommand,
    SensitiveProposalConsistencyError,
    SensitiveProposalService,
)
from supportops.agent_tools.application.persistence import (
    AgentToolCallPersistenceResult,
)
from supportops.agent_tools.application.sensitive_bindings import (
    SensitiveToolRegistry,
)
from supportops.agent_tools.domain.audit import AgentToolCall
from supportops.agent_tools.domain.fingerprints import (
    create_tool_call_fingerprint,
)
from supportops.agent_tools.tools.escalate_ticket import (
    EscalateTicketInput,
    TicketEscalationTargetQueue,
    create_escalate_ticket_binding,
)
from supportops.modules.agent_runs.application.execution import (
    RetryableAgentRunExecutionError,
)
from supportops.modules.approvals.domain.models import ApprovalRequest
from supportops.modules.approvals.domain.repositories import (
    ApprovalRequestPersistenceResult,
)
from supportops.observability.models import (
    EventObservation,
    ObservabilityProvider,
)

_NOW = datetime(2026, 8, 3, 15, 0, tzinfo=UTC)


class FakeTransactionManager:
    @asynccontextmanager
    async def transaction(self) -> AsyncIterator[None]:
        yield


def _context() -> SimpleNamespace:
    workspace_id = uuid4()
    ticket_id = uuid4()
    agent_run_id = uuid4()
    attempt_id = uuid4()
    lease_token = uuid4()
    return SimpleNamespace(
        agent_run=SimpleNamespace(
            workspace_id=workspace_id,
            id=agent_run_id,
        ),
        ticket=SimpleNamespace(id=ticket_id),
        attempt=SimpleNamespace(
            id=attempt_id,
            lease_token=lease_token,
        ),
    )


def _command() -> SensitiveProposalCommand:
    return SensitiveProposalCommand(
        provider_tool_call_id="call-1",
        tool_name="escalate_ticket",
        tool_version=1,
        arguments=EscalateTicketInput(
            target_queue=(TicketEscalationTargetQueue.SECURITY_OPERATIONS),
            reason="Potential security incident.",
        ),
        requested_by_llm_invocation_id=uuid4(),
        sequence=1,
    )


@pytest.mark.asyncio
async def test_service_persists_tool_call_before_approval() -> None:
    order: list[str] = []

    async def persist_tool_call(command: object) -> AgentToolCallPersistenceResult:
        del command
        order.append("tool_call")
        return AgentToolCallPersistenceResult.APPLIED

    async def persist_approval(
        approval: object,
    ) -> ApprovalRequestPersistenceResult:
        del approval
        order.append("approval")
        return ApprovalRequestPersistenceResult.APPLIED

    tool_repository = SimpleNamespace(
        persist_fenced=AsyncMock(side_effect=persist_tool_call),
    )
    query_repository = SimpleNamespace(
        get_sensitive_by_identity=AsyncMock(),
    )
    approval_repository = SimpleNamespace(
        persist_pending=AsyncMock(side_effect=persist_approval),
        get_by_agent_tool_call_id=AsyncMock(),
    )
    ids = iter((uuid4(), uuid4()))
    service = SensitiveProposalService(
        transaction_manager=FakeTransactionManager(),
        sensitive_tool_registry=SensitiveToolRegistry(
            (create_escalate_ticket_binding(),),
        ),
        tool_call_execution_repository=tool_repository,
        tool_call_query_repository=query_repository,
        approval_request_repository=approval_repository,
        approval_ttl_seconds=86400,
        utc_now=lambda: _NOW,
        uuid_factory=lambda: next(ids),
    )

    outcome = await service.execute(
        context=cast(Any, _context()),
        command=_command(),
    )

    assert order == ["tool_call", "approval"]
    assert outcome.tool_call_created is True
    assert outcome.approval_request_created is True
    assert outcome.approval_request.agent_tool_call_id == (outcome.tool_call.id)
    assert outcome.approval_request.expires_at == (_NOW + timedelta(days=1))
    assert dict(outcome.approval_request.proposed_input) == {
        "target_queue": "security_operations",
        "reason": "Potential security incident.",
    }


@pytest.mark.asyncio
async def test_lease_loss_prevents_approval_persistence() -> None:
    approval_repository = SimpleNamespace(
        persist_pending=AsyncMock(),
        get_by_agent_tool_call_id=AsyncMock(),
    )
    service = SensitiveProposalService(
        transaction_manager=FakeTransactionManager(),
        sensitive_tool_registry=SensitiveToolRegistry(
            (create_escalate_ticket_binding(),),
        ),
        tool_call_execution_repository=SimpleNamespace(
            persist_fenced=AsyncMock(
                return_value=(AgentToolCallPersistenceResult.LEASE_LOST),
            ),
        ),
        tool_call_query_repository=SimpleNamespace(
            get_sensitive_by_identity=AsyncMock(),
        ),
        approval_request_repository=approval_repository,
        approval_ttl_seconds=86400,
        utc_now=lambda: _NOW,
    )

    with pytest.raises(
        RetryableAgentRunExecutionError,
        match="lease",
    ):
        await service.execute(
            context=cast(Any, _context()),
            command=_command(),
        )

    approval_repository.persist_pending.assert_not_awaited()


def test_command_requires_uuid_invocation() -> None:
    with pytest.raises(TypeError, match="UUID"):
        SensitiveProposalCommand(
            provider_tool_call_id="call-1",
            tool_name="escalate_ticket",
            tool_version=1,
            arguments=EscalateTicketInput(
                target_queue=(TicketEscalationTargetQueue.SUPPORT_OPERATIONS),
                reason="Needs operational handling.",
            ),
            requested_by_llm_invocation_id=cast(Any, "not-a-uuid"),
            sequence=1,
        )


_ESCALATE_FINGERPRINT = "aa1034b4731d4b904e005cab2c4631a24e43b2dbe9612b9a1a191cea283cd08d"


def _existing_tool_call(context: SimpleNamespace) -> AgentToolCall:
    return AgentToolCall.propose_for_approval(
        workspace_id=context.agent_run.workspace_id,
        ticket_id=context.ticket.id,
        agent_run_id=context.agent_run.id,
        proposed_by_agent_run_attempt_id=context.attempt.id,
        sequence=1,
        provider_tool_call_id="call-1",
        tool_name="escalate_ticket",
        tool_version=1,
        input_fingerprint=_ESCALATE_FINGERPRINT,
        safe_input={
            "target_queue": "security_operations",
            "reason": "Potential security incident.",
        },
        proposed_at=_NOW - timedelta(minutes=5),
        tool_call_id=uuid4(),
    )


@pytest.mark.asyncio
async def test_replay_after_tool_call_reuses_original_records() -> None:
    context = _context()
    existing_tool = _existing_tool_call(context)
    existing_approval = ApprovalRequest.create_pending(
        tool_call=existing_tool,
        requested_by_llm_invocation_id=uuid4(),
        request_reason="Potential security incident.",
        expires_at=existing_tool.proposed_at + timedelta(days=1),
        now=existing_tool.proposed_at,
        approval_request_id=uuid4(),
    )
    order: list[str] = []

    async def persist_tool_call(command: object) -> AgentToolCallPersistenceResult:
        del command
        order.append("tool_call")
        return AgentToolCallPersistenceResult.ALREADY_RECORDED

    async def persist_approval(
        approval: object,
    ) -> ApprovalRequestPersistenceResult:
        del approval
        order.append("approval")
        return ApprovalRequestPersistenceResult.ALREADY_RECORDED

    service = SensitiveProposalService(
        transaction_manager=FakeTransactionManager(),
        sensitive_tool_registry=SensitiveToolRegistry(
            (create_escalate_ticket_binding(),),
        ),
        tool_call_execution_repository=SimpleNamespace(
            persist_fenced=AsyncMock(side_effect=persist_tool_call),
        ),
        tool_call_query_repository=SimpleNamespace(
            get_sensitive_by_identity=AsyncMock(
                return_value=existing_tool,
            ),
        ),
        approval_request_repository=SimpleNamespace(
            persist_pending=AsyncMock(side_effect=persist_approval),
            get_by_agent_tool_call_id=AsyncMock(
                return_value=existing_approval,
            ),
        ),
        approval_ttl_seconds=86400,
        utc_now=lambda: _NOW,
        uuid_factory=uuid4,
    )
    command = SensitiveProposalCommand(
        provider_tool_call_id="call-1",
        tool_name="escalate_ticket",
        tool_version=1,
        arguments=EscalateTicketInput(
            target_queue=(TicketEscalationTargetQueue.SECURITY_OPERATIONS),
            reason="Potential security incident.",
        ),
        requested_by_llm_invocation_id=(existing_approval.requested_by_llm_invocation_id),
        sequence=1,
    )

    outcome = await service.execute(
        context=cast(Any, context),
        command=command,
    )

    assert order == ["tool_call", "approval"]
    assert outcome.tool_call_created is False
    assert outcome.approval_request_created is False
    assert outcome.tool_call.id == existing_tool.id
    assert outcome.tool_call.proposed_by_agent_run_attempt_id == (
        existing_tool.proposed_by_agent_run_attempt_id
    )
    assert outcome.tool_call.provider_tool_call_id == (existing_tool.provider_tool_call_id)
    assert outcome.tool_call.sequence == existing_tool.sequence
    assert outcome.tool_call.proposed_at == existing_tool.proposed_at
    assert outcome.approval_request.id == existing_approval.id
    assert outcome.approval_request.created_at == (existing_approval.created_at)
    assert outcome.approval_request.expires_at == (existing_approval.expires_at)
    assert outcome.approval_request.agent_tool_call_id == (existing_tool.id)


@pytest.mark.asyncio
async def test_replay_after_tool_only_creates_approval_once() -> None:
    """Crash recovery A: tool persisted, approval not yet durable."""

    context = _context()
    existing_tool = _existing_tool_call(context)
    created_approval_ids: list[object] = []

    async def persist_approval(
        approval: object,
    ) -> ApprovalRequestPersistenceResult:
        created_approval_ids.append(approval)
        return ApprovalRequestPersistenceResult.APPLIED

    service = SensitiveProposalService(
        transaction_manager=FakeTransactionManager(),
        sensitive_tool_registry=SensitiveToolRegistry(
            (create_escalate_ticket_binding(),),
        ),
        tool_call_execution_repository=SimpleNamespace(
            persist_fenced=AsyncMock(
                return_value=(AgentToolCallPersistenceResult.ALREADY_RECORDED),
            ),
        ),
        tool_call_query_repository=SimpleNamespace(
            get_sensitive_by_identity=AsyncMock(
                return_value=existing_tool,
            ),
        ),
        approval_request_repository=SimpleNamespace(
            persist_pending=AsyncMock(side_effect=persist_approval),
            get_by_agent_tool_call_id=AsyncMock(),
        ),
        approval_ttl_seconds=86400,
        utc_now=lambda: _NOW,
        uuid_factory=uuid4,
    )

    outcome = await service.execute(
        context=cast(Any, context),
        command=_command(),
    )

    assert outcome.tool_call_created is False
    assert outcome.approval_request_created is True
    assert outcome.tool_call.id == existing_tool.id
    assert outcome.approval_request.agent_tool_call_id == (existing_tool.id)
    assert len(created_approval_ids) == 1


@pytest.mark.asyncio
async def test_conflicting_tool_replay_fails_closed() -> None:
    context = _context()
    existing_tool = _existing_tool_call(context)
    conflicting = AgentToolCall.propose_for_approval(
        workspace_id=context.agent_run.workspace_id,
        ticket_id=context.ticket.id,
        agent_run_id=context.agent_run.id,
        proposed_by_agent_run_attempt_id=context.attempt.id,
        sequence=1,
        provider_tool_call_id="call-1",
        tool_name="escalate_ticket",
        tool_version=1,
        input_fingerprint=_ESCALATE_FINGERPRINT,
        safe_input={
            "target_queue": "support_operations",
            "reason": "Different durable input.",
        },
        proposed_at=existing_tool.proposed_at,
        tool_call_id=existing_tool.id,
    )
    service = SensitiveProposalService(
        transaction_manager=FakeTransactionManager(),
        sensitive_tool_registry=SensitiveToolRegistry(
            (create_escalate_ticket_binding(),),
        ),
        tool_call_execution_repository=SimpleNamespace(
            persist_fenced=AsyncMock(
                return_value=(AgentToolCallPersistenceResult.ALREADY_RECORDED),
            ),
        ),
        tool_call_query_repository=SimpleNamespace(
            get_sensitive_by_identity=AsyncMock(
                return_value=conflicting,
            ),
        ),
        approval_request_repository=SimpleNamespace(
            persist_pending=AsyncMock(),
            get_by_agent_tool_call_id=AsyncMock(),
        ),
        approval_ttl_seconds=86400,
        utc_now=lambda: _NOW,
    )

    with pytest.raises(
        SensitiveProposalConsistencyError,
        match="conflicts",
    ):
        await service.execute(
            context=cast(Any, context),
            command=_command(),
        )


@pytest.mark.asyncio
async def test_conflicting_approval_replay_fails_closed() -> None:
    context = _context()
    existing_tool = _existing_tool_call(context)
    existing_approval = ApprovalRequest.create_pending(
        tool_call=existing_tool,
        requested_by_llm_invocation_id=uuid4(),
        request_reason="Different durable reason.",
        expires_at=existing_tool.proposed_at + timedelta(days=1),
        now=existing_tool.proposed_at,
    )
    service = SensitiveProposalService(
        transaction_manager=FakeTransactionManager(),
        sensitive_tool_registry=SensitiveToolRegistry(
            (create_escalate_ticket_binding(),),
        ),
        tool_call_execution_repository=SimpleNamespace(
            persist_fenced=AsyncMock(
                return_value=(AgentToolCallPersistenceResult.ALREADY_RECORDED),
            ),
        ),
        tool_call_query_repository=SimpleNamespace(
            get_sensitive_by_identity=AsyncMock(
                return_value=existing_tool,
            ),
        ),
        approval_request_repository=SimpleNamespace(
            persist_pending=AsyncMock(
                return_value=(ApprovalRequestPersistenceResult.ALREADY_RECORDED),
            ),
            get_by_agent_tool_call_id=AsyncMock(
                return_value=existing_approval,
            ),
        ),
        approval_ttl_seconds=86400,
        utc_now=lambda: _NOW,
    )

    with pytest.raises(
        SensitiveProposalConsistencyError,
        match="conflicts",
    ):
        await service.execute(
            context=cast(Any, context),
            command=_command(),
        )


def test_escalate_fingerprint_matches_fixture() -> None:
    binding = create_escalate_ticket_binding()
    arguments = EscalateTicketInput(
        target_queue=(TicketEscalationTargetQueue.SECURITY_OPERATIONS),
        reason="Potential security incident.",
    )

    assert (
        create_tool_call_fingerprint(
            definition=binding.definition,
            arguments=arguments,
        )
        == _ESCALATE_FINGERPRINT
    )


# ---------------------------------------------------------------------------
# Approval request observability (Commit 4 / PR C)
# ---------------------------------------------------------------------------


@dataclass
class _RecordingTraceEvent:
    identity: object
    event: EventObservation


class _RecordingObservabilityClient:
    provider = ObservabilityProvider.NOOP
    enabled = False

    def __init__(self, *, fail_record: bool = False) -> None:
        self.fail_record = fail_record
        self.trace_events: list[_RecordingTraceEvent] = []
        self.started_traces: list[object] = []

    def start_trace(self, attributes: object) -> AbstractContextManager[object]:
        self.started_traces.append(attributes)
        raise AssertionError("approval.requested must not open a fake root")

    def start_observation(self, attributes: object) -> AbstractContextManager[object]:
        del attributes
        raise AssertionError("approval.requested must not open observations")

    def record_event(self, event: object) -> None:
        del event

    def record_trace_event(self, *, identity: object, event: EventObservation) -> None:
        if self.fail_record:
            raise RuntimeError("record failed")
        self.trace_events.append(_RecordingTraceEvent(identity=identity, event=event))

    def flush(self) -> None:
        return None

    def shutdown(self) -> None:
        return None


@pytest.mark.asyncio
async def test_approval_requested_emitted_only_after_durable_creation() -> None:
    order: list[str] = []
    observability = _RecordingObservabilityClient()
    context = _context()

    async def persist_tool_call(command: object) -> AgentToolCallPersistenceResult:
        del command
        order.append("tool_call")
        assert observability.trace_events == []
        return AgentToolCallPersistenceResult.APPLIED

    async def persist_approval(
        approval: object,
    ) -> ApprovalRequestPersistenceResult:
        del approval
        order.append("approval")
        assert observability.trace_events == []
        return ApprovalRequestPersistenceResult.APPLIED

    ids = iter((uuid4(), uuid4()))
    service = SensitiveProposalService(
        transaction_manager=FakeTransactionManager(),
        sensitive_tool_registry=SensitiveToolRegistry(
            (create_escalate_ticket_binding(),),
        ),
        tool_call_execution_repository=SimpleNamespace(
            persist_fenced=AsyncMock(side_effect=persist_tool_call),
        ),
        tool_call_query_repository=SimpleNamespace(
            get_sensitive_by_identity=AsyncMock(),
        ),
        approval_request_repository=SimpleNamespace(
            persist_pending=AsyncMock(side_effect=persist_approval),
            get_by_agent_tool_call_id=AsyncMock(),
        ),
        approval_ttl_seconds=86400,
        utc_now=lambda: _NOW,
        uuid_factory=lambda: next(ids),
        observability_client=cast(Any, observability),
    )

    outcome = await service.execute(
        context=cast(Any, context),
        command=_command(),
    )

    assert order == ["tool_call", "approval"]
    assert outcome.approval_request_created is True
    assert len(observability.trace_events) == 1
    event = observability.trace_events[0]
    assert event.event.name == "approval.requested"
    assert event.identity.trace_seed == f"agent-run:{context.agent_run.id}"  # type: ignore[attr-defined]
    assert event.event.metadata["tool_name"] == "escalate_ticket"
    assert event.event.metadata["approval_status"] == "pending"
    assert "proposed_input" not in event.event.metadata
    assert "request_reason" not in event.event.metadata
    assert observability.started_traces == []


@pytest.mark.asyncio
async def test_replayed_approval_does_not_emit_duplicate_requested() -> None:
    observability = _RecordingObservabilityClient()
    context = _context()
    invocation_id = uuid4()
    existing_tool = AgentToolCall.propose_for_approval(
        workspace_id=context.agent_run.workspace_id,
        ticket_id=context.ticket.id,
        agent_run_id=context.agent_run.id,
        proposed_by_agent_run_attempt_id=context.attempt.id,
        sequence=1,
        provider_tool_call_id="call-1",
        tool_name="escalate_ticket",
        tool_version=1,
        input_fingerprint=_ESCALATE_FINGERPRINT,
        safe_input={
            "target_queue": "security_operations",
            "reason": "Potential security incident.",
        },
        proposed_at=_NOW,
    )
    existing_approval = ApprovalRequest.create_pending(
        tool_call=existing_tool,
        requested_by_llm_invocation_id=invocation_id,
        request_reason="Potential security incident.",
        expires_at=_NOW + timedelta(days=1),
        now=_NOW,
    )
    service = SensitiveProposalService(
        transaction_manager=FakeTransactionManager(),
        sensitive_tool_registry=SensitiveToolRegistry(
            (create_escalate_ticket_binding(),),
        ),
        tool_call_execution_repository=SimpleNamespace(
            persist_fenced=AsyncMock(
                return_value=AgentToolCallPersistenceResult.ALREADY_RECORDED,
            ),
        ),
        tool_call_query_repository=SimpleNamespace(
            get_sensitive_by_identity=AsyncMock(return_value=existing_tool),
        ),
        approval_request_repository=SimpleNamespace(
            persist_pending=AsyncMock(
                return_value=ApprovalRequestPersistenceResult.ALREADY_RECORDED,
            ),
            get_by_agent_tool_call_id=AsyncMock(return_value=existing_approval),
        ),
        approval_ttl_seconds=86400,
        utc_now=lambda: _NOW,
        observability_client=cast(Any, observability),
    )

    outcome = await service.execute(
        context=cast(Any, context),
        command=SensitiveProposalCommand(
            provider_tool_call_id="call-1",
            tool_name="escalate_ticket",
            tool_version=1,
            arguments=EscalateTicketInput(
                target_queue=(TicketEscalationTargetQueue.SECURITY_OPERATIONS),
                reason="Potential security incident.",
            ),
            requested_by_llm_invocation_id=invocation_id,
            sequence=1,
        ),
    )

    assert outcome.approval_request_created is False
    assert observability.trace_events == []


@pytest.mark.asyncio
async def test_approval_requested_event_failure_preserves_outcome() -> None:
    observability = _RecordingObservabilityClient(fail_record=True)
    ids = iter((uuid4(), uuid4()))
    service = SensitiveProposalService(
        transaction_manager=FakeTransactionManager(),
        sensitive_tool_registry=SensitiveToolRegistry(
            (create_escalate_ticket_binding(),),
        ),
        tool_call_execution_repository=SimpleNamespace(
            persist_fenced=AsyncMock(
                return_value=AgentToolCallPersistenceResult.APPLIED,
            ),
        ),
        tool_call_query_repository=SimpleNamespace(
            get_sensitive_by_identity=AsyncMock(),
        ),
        approval_request_repository=SimpleNamespace(
            persist_pending=AsyncMock(
                return_value=ApprovalRequestPersistenceResult.APPLIED,
            ),
            get_by_agent_tool_call_id=AsyncMock(),
        ),
        approval_ttl_seconds=86400,
        utc_now=lambda: _NOW,
        uuid_factory=lambda: next(ids),
        observability_client=cast(Any, observability),
    )

    outcome = await service.execute(
        context=cast(Any, _context()),
        command=_command(),
    )

    assert outcome.approval_request_created is True
