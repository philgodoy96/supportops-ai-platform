"""Integration tests for granted escalation execution."""

import asyncio
from uuid import UUID, uuid4

import pytest
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from supportops.agent_tools.application.sensitive_execution import (
    ExecuteApprovedTicketEscalation,
    SensitiveExecutionResult,
    SensitiveExecutionStatus,
)
from supportops.agent_tools.domain.audit import AgentToolCall, AgentToolCallStatus
from supportops.agent_tools.infrastructure.grant_models import (
    SensitiveExecutionGrantRecord,
)
from supportops.agent_tools.infrastructure.grant_repository import (
    SqlAlchemySensitiveExecutionGrantRepository,
)
from supportops.agent_tools.infrastructure.models import AgentToolCallRecord
from supportops.agent_tools.infrastructure.repository import (
    SqlAlchemyAgentToolCallExecutionRepository,
)
from supportops.infrastructure.postgresql.transaction import (
    SqlAlchemyTransactionManager,
)
from supportops.modules.agent_runs.application.execution import (
    AgentRunExecutionContext,
)
from supportops.modules.agent_runs.domain.claiming import AgentRunClaim
from supportops.modules.approvals.domain.models import ApprovalRequest
from supportops.modules.approvals.infrastructure.repository import (
    SqlAlchemyApprovalRequestRepository,
)
from supportops.modules.tickets.domain.models import Ticket
from supportops.modules.tickets.infrastructure.escalation_models import (
    TicketEscalationRecord,
)
from supportops.modules.tickets.infrastructure.escalation_repository import (
    SqlAlchemyTicketEscalationRepository,
)

ApprovedEscalationSeed = tuple[
    async_sessionmaker[AsyncSession],
    AgentRunClaim,
    Ticket,
    AgentToolCall,
    ApprovalRequest,
]


@pytest.mark.integration
async def test_granted_escalation_executes_once(
    approved_ticket_escalation_executor: ExecuteApprovedTicketEscalation,
    approved_ticket_escalation_context: AgentRunExecutionContext,
    approved_ticket_escalation_approval: ApprovalRequest,
    approved_ticket_escalation_tool_call: AgentToolCall,
    postgresql_session_factory: async_sessionmaker[AsyncSession],
) -> None:
    first = await approved_ticket_escalation_executor.execute(
        context=approved_ticket_escalation_context,
        approval_request_id=(approved_ticket_escalation_approval.id),
        agent_tool_call_id=(approved_ticket_escalation_tool_call.id),
    )
    second = await approved_ticket_escalation_executor.execute(
        context=approved_ticket_escalation_context,
        approval_request_id=(approved_ticket_escalation_approval.id),
        agent_tool_call_id=(approved_ticket_escalation_tool_call.id),
    )

    assert first.status is SensitiveExecutionStatus.APPLIED
    assert second.status is (SensitiveExecutionStatus.ALREADY_RECORDED)
    assert first.escalation.id == second.escalation.id
    assert first.grant.id == second.grant.id
    assert set(first.output.model_dump(mode="json")) == {
        "escalation_id",
        "ticket_id",
        "target_queue",
        "status",
    }

    async with postgresql_session_factory() as session:
        grant_count = await session.execute(
            select(func.count()).select_from(SensitiveExecutionGrantRecord)
        )
        escalation_count = await session.execute(
            select(func.count()).select_from(TicketEscalationRecord)
        )
        tool_record = (
            await session.execute(
                select(AgentToolCallRecord).where(
                    AgentToolCallRecord.id == (approved_ticket_escalation_tool_call.id),
                )
            )
        ).scalar_one()

    assert grant_count.scalar_one() == 1
    assert escalation_count.scalar_one() == 1
    assert tool_record.status == AgentToolCallStatus.SUCCEEDED.value
    assert tool_record.executed_by_agent_run_attempt_id == (
        approved_ticket_escalation_context.attempt.id
    )
    assert tool_record.proposed_by_agent_run_attempt_id == (
        approved_ticket_escalation_tool_call.proposed_by_agent_run_attempt_id
    )


@pytest.mark.integration
async def test_granted_escalation_concurrent_execution_converges(
    approved_ticket_escalation_seed: ApprovedEscalationSeed,
    approved_ticket_escalation_context: AgentRunExecutionContext,
    approved_ticket_escalation_approval: ApprovalRequest,
    approved_ticket_escalation_tool_call: AgentToolCall,
) -> None:
    session_factory, *_rest = approved_ticket_escalation_seed
    ready = 0
    ready_lock = asyncio.Lock()
    release = asyncio.Event()

    async def run_once() -> SensitiveExecutionResult:
        nonlocal ready

        async with session_factory() as session:
            executor = ExecuteApprovedTicketEscalation(
                transaction_manager=SqlAlchemyTransactionManager(session),
                approval_request_repository=(SqlAlchemyApprovalRequestRepository(session)),
                tool_call_repository=(SqlAlchemyAgentToolCallExecutionRepository(session)),
                grant_repository=(SqlAlchemySensitiveExecutionGrantRepository(session)),
                escalation_repository=(SqlAlchemyTicketEscalationRepository(session)),
            )
            async with ready_lock:
                ready += 1
                if ready == 2:
                    release.set()
            await release.wait()
            return await executor.execute(
                context=approved_ticket_escalation_context,
                approval_request_id=(approved_ticket_escalation_approval.id),
                agent_tool_call_id=(approved_ticket_escalation_tool_call.id),
            )

    first, second = await asyncio.gather(run_once(), run_once())
    statuses = {first.status, second.status}
    assert SensitiveExecutionStatus.APPLIED in statuses
    assert statuses <= {
        SensitiveExecutionStatus.APPLIED,
        SensitiveExecutionStatus.ALREADY_RECORDED,
    }
    assert first.grant.id == second.grant.id
    assert first.escalation.id == second.escalation.id

    async with session_factory() as session:
        grant_count = await session.execute(
            select(func.count()).select_from(SensitiveExecutionGrantRecord)
        )
        escalation_count = await session.execute(
            select(func.count()).select_from(TicketEscalationRecord)
        )

    assert grant_count.scalar_one() == 1
    assert escalation_count.scalar_one() == 1


@pytest.mark.integration
async def test_granted_escalation_rollback_leaves_no_partial_state(
    approved_ticket_escalation_seed: ApprovedEscalationSeed,
    approved_ticket_escalation_context: AgentRunExecutionContext,
    approved_ticket_escalation_approval: ApprovalRequest,
    approved_ticket_escalation_tool_call: AgentToolCall,
) -> None:
    session_factory, *_rest = approved_ticket_escalation_seed
    uuid_calls: dict[str, int] = {"count": 0}

    def uuid_factory() -> UUID:
        uuid_calls["count"] += 1
        if uuid_calls["count"] > 1:
            raise RuntimeError("force rollback")
        return uuid4()

    async with session_factory() as session:
        executor = ExecuteApprovedTicketEscalation(
            transaction_manager=SqlAlchemyTransactionManager(session),
            approval_request_repository=(SqlAlchemyApprovalRequestRepository(session)),
            tool_call_repository=(SqlAlchemyAgentToolCallExecutionRepository(session)),
            grant_repository=(SqlAlchemySensitiveExecutionGrantRepository(session)),
            escalation_repository=(SqlAlchemyTicketEscalationRepository(session)),
            uuid_factory=uuid_factory,
        )
        with pytest.raises(RuntimeError, match="force rollback"):
            await executor.execute(
                context=approved_ticket_escalation_context,
                approval_request_id=(approved_ticket_escalation_approval.id),
                agent_tool_call_id=(approved_ticket_escalation_tool_call.id),
            )

    async with session_factory() as session:
        grant_count = await session.execute(
            select(func.count()).select_from(SensitiveExecutionGrantRecord)
        )
        escalation_count = await session.execute(
            select(func.count()).select_from(TicketEscalationRecord)
        )
        tool_record = (
            await session.execute(
                select(AgentToolCallRecord).where(
                    AgentToolCallRecord.id == (approved_ticket_escalation_tool_call.id),
                )
            )
        ).scalar_one()

    assert grant_count.scalar_one() == 0
    assert escalation_count.scalar_one() == 0
    assert tool_record.status == AgentToolCallStatus.PENDING_APPROVAL.value
    assert tool_record.executed_by_agent_run_attempt_id is None
