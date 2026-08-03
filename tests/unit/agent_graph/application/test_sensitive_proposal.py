"""Unit tests for durable sensitive proposal preparation."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from supportops.agent_graph.application.sensitive_proposal import (
    SensitiveProposalCommand,
    SensitiveProposalService,
)
from supportops.agent_tools.application.persistence import (
    AgentToolCallPersistenceResult,
)
from supportops.agent_tools.application.sensitive_bindings import (
    SensitiveToolRegistry,
)
from supportops.agent_tools.tools.escalate_ticket import (
    EscalateTicketInput,
    TicketEscalationTargetQueue,
    create_escalate_ticket_binding,
)
from supportops.modules.agent_runs.application.execution import (
    RetryableAgentRunExecutionError,
)
from supportops.modules.approvals.domain.repositories import (
    ApprovalRequestPersistenceResult,
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
