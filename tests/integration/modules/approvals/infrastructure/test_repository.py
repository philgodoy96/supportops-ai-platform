"""Integration tests for durable approval-request persistence."""

import asyncio
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from uuid import UUID

import pytest
from pydantic import JsonValue
from sqlalchemy import func, select, text
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
)

from supportops.agent_tools.application.persistence import (
    AgentToolCallPersistenceResult,
    PersistAgentToolCallCommand,
)
from supportops.agent_tools.domain.audit import (
    AgentToolCall,
    AgentToolCallStatus,
)
from supportops.agent_tools.domain.contracts import ToolSafetyLevel
from supportops.agent_tools.infrastructure.repository import (
    SqlAlchemyAgentToolCallExecutionRepository,
)
from supportops.ai.gateway.errors import LLMErrorCode
from supportops.ai.gateway.results import LLMInvocationStatus
from supportops.ai.pricing.catalog import PRICING_CATALOG_VERSION
from supportops.ai.schemas.ticket_classification import (
    TICKET_CLASSIFICATION_SCHEMA_VERSION,
)
from supportops.infrastructure.postgresql.transaction import (
    SqlAlchemyTransactionManager,
)
from supportops.modules.agent_runs.domain.claiming import (
    AgentRunClaim,
    ClaimAgentRunCommand,
)
from supportops.modules.agent_runs.domain.models import (
    DETERMINISTIC_BASELINE_WORKFLOW_VERSION,
    AgentRun,
)
from supportops.modules.agent_runs.infrastructure.repository import (
    SqlAlchemyAgentRunRepository,
)
from supportops.modules.approvals.domain.models import ApprovalRequest
from supportops.modules.approvals.domain.repositories import (
    ApprovalRequestConsistencyError,
    ApprovalRequestPersistenceResult,
)
from supportops.modules.approvals.infrastructure.models import (
    ApprovalRequestRecord,
)
from supportops.modules.approvals.infrastructure.repository import (
    SqlAlchemyApprovalRequestRepository,
)
from supportops.modules.ticket_classifications.domain.models import (
    LLMInvocation,
)
from supportops.modules.ticket_classifications.infrastructure.models import (
    LLMInvocationRecord,
)
from supportops.modules.tickets.domain.models import Ticket
from supportops.modules.tickets.infrastructure.repository import (
    SqlAlchemyTicketRepository,
)
from supportops.modules.workspaces.domain.models import Workspace
from supportops.modules.workspaces.infrastructure.repository import (
    SqlAlchemyWorkspaceRepository,
)

pytestmark = pytest.mark.integration

_WORKSPACE_ID = UUID("10000000-0000-4000-8000-000000000001")
_OTHER_WORKSPACE_ID = UUID("10000000-0000-4000-8000-000000000091")
_TICKET_ID = UUID("20000000-0000-4000-8000-000000000002")
_AGENT_RUN_ID = UUID("30000000-0000-4000-8000-000000000003")
_LEASE_TOKEN = UUID("40000000-0000-4000-8000-000000000004")
_EXECUTION_REQUEST_ID = UUID("50000000-0000-4000-8000-000000000005")
_TOOL_CALL_ID = UUID("60000000-0000-4000-8000-000000000006")
_SECOND_TOOL_CALL_ID = UUID("70000000-0000-4000-8000-000000000007")
_THIRD_TOOL_CALL_ID = UUID("70000000-0000-4000-8000-000000000017")
_FOURTH_TOOL_CALL_ID = UUID("70000000-0000-4000-8000-000000000027")
_INVOCATION_ID = UUID("80000000-0000-4000-8000-000000000008")
_SECOND_INVOCATION_ID = UUID("80000000-0000-4000-8000-000000000018")
_THIRD_INVOCATION_ID = UUID("80000000-0000-4000-8000-000000000028")
_FOURTH_INVOCATION_ID = UUID("80000000-0000-4000-8000-000000000038")
_APPROVAL_REQUEST_ID = UUID("90000000-0000-4000-8000-000000000009")
_SECOND_APPROVAL_REQUEST_ID = UUID("90000000-0000-4000-8000-000000000019")
_THIRD_APPROVAL_REQUEST_ID = UUID("90000000-0000-4000-8000-000000000029")
_FOURTH_APPROVAL_REQUEST_ID = UUID("90000000-0000-4000-8000-000000000039")

_CREATED_AT = datetime(2026, 8, 2, 22, 0, tzinfo=UTC)
_CLAIMED_AT = _CREATED_AT + timedelta(minutes=1)
_LEASE_EXPIRES_AT = _CLAIMED_AT + timedelta(seconds=45)
_TOOL_PROPOSED_AT = _CLAIMED_AT + timedelta(seconds=1)
_INVOCATION_AT = _CLAIMED_AT + timedelta(seconds=2)
_APPROVAL_AT = _CLAIMED_AT + timedelta(seconds=3)
_EXPIRES_AT = _APPROVAL_AT + timedelta(hours=24)


async def _create_running_claim(
    session: AsyncSession,
) -> AgentRunClaim:
    workspace = Workspace(
        id=_WORKSPACE_ID,
        name="Approval Workspace",
        slug="approval-workspace",
        created_at=_CREATED_AT,
        updated_at=_CREATED_AT,
    )
    ticket = Ticket.create(
        ticket_id=_TICKET_ID,
        workspace_id=_WORKSPACE_ID,
        subject="Needs escalation approval",
        description=("The customer requested a policy-sensitive escalation."),
        external_reference=None,
        ingestion_request_id=UUID("81000000-0000-4000-8000-000000000008"),
        correlation_id=UUID("82000000-0000-4000-8000-000000000009"),
        now=_CREATED_AT,
    )
    agent_run = AgentRun.create_initial(
        agent_run_id=_AGENT_RUN_ID,
        workspace_id=_WORKSPACE_ID,
        ticket_id=_TICKET_ID,
        ingestion_request_id=ticket.ingestion_request_id,
        correlation_id=ticket.correlation_id,
        workflow_version=DETERMINISTIC_BASELINE_WORKFLOW_VERSION,
        max_retryable_failures=3,
        now=_CREATED_AT,
    )

    transaction_manager = SqlAlchemyTransactionManager(session)

    async with transaction_manager.transaction():
        await SqlAlchemyWorkspaceRepository(session).add(workspace)
        await SqlAlchemyTicketRepository(session).add(ticket)
        await SqlAlchemyAgentRunRepository(session).add(agent_run)

    async with transaction_manager.transaction():
        claim = await SqlAlchemyAgentRunRepository(session).claim_next_available(
            ClaimAgentRunCommand(
                worker_id="approval-worker-1",
                lease_token=_LEASE_TOKEN,
                execution_request_id=_EXECUTION_REQUEST_ID,
                claimed_at=_CLAIMED_AT,
                lease_expires_at=_LEASE_EXPIRES_AT,
            )
        )

    assert claim is not None
    return claim


def _pending_tool_call(
    claim: AgentRunClaim,
    *,
    tool_call_id: UUID = _TOOL_CALL_ID,
    sequence: int = 1,
    provider_tool_call_id: str = "provider-sensitive-1",
    tool_name: str = "escalate_ticket",
    tool_version: int = 1,
    input_fingerprint: str = "b" * 64,
    safe_input: dict[str, JsonValue] | None = None,
) -> AgentToolCall:
    return AgentToolCall.propose_for_approval(
        tool_call_id=tool_call_id,
        workspace_id=_WORKSPACE_ID,
        ticket_id=_TICKET_ID,
        agent_run_id=claim.agent_run.id,
        proposed_by_agent_run_attempt_id=claim.attempt.id,
        sequence=sequence,
        provider_tool_call_id=provider_tool_call_id,
        tool_name=tool_name,
        tool_version=tool_version,
        input_fingerprint=input_fingerprint,
        safe_input=safe_input or {"reason_code": "policy_required"},
        proposed_at=_TOOL_PROPOSED_AT,
    )


def _read_only_tool_call(claim: AgentRunClaim) -> AgentToolCall:
    return AgentToolCall.create_terminal(
        tool_call_id=_TOOL_CALL_ID,
        workspace_id=_WORKSPACE_ID,
        ticket_id=_TICKET_ID,
        agent_run_id=claim.agent_run.id,
        agent_run_attempt_id=claim.attempt.id,
        sequence=1,
        provider_tool_call_id="provider-readonly-1",
        tool_name="search_knowledge",
        tool_version=1,
        safety_level=ToolSafetyLevel.READ_ONLY,
        status=AgentToolCallStatus.SUCCEEDED,
        input_fingerprint="c" * 64,
        safe_input={"top_k": 5},
        safe_output={"result_count": 1},
        latency_ms=25,
        error_code=None,
        started_at=_TOOL_PROPOSED_AT,
        finished_at=_TOOL_PROPOSED_AT + timedelta(milliseconds=25),
    )


def _invocation(
    claim: AgentRunClaim,
    *,
    invocation_id: UUID = _INVOCATION_ID,
    sequence: int = 1,
) -> LLMInvocation:
    return LLMInvocation.create(
        invocation_id=invocation_id,
        workspace_id=_WORKSPACE_ID,
        ticket_id=_TICKET_ID,
        agent_run_id=claim.agent_run.id,
        agent_run_attempt_id=claim.attempt.id,
        invocation_sequence=sequence,
        status=LLMInvocationStatus.TIMED_OUT,
        provider="mock",
        model="mock-support-v1",
        provider_request_id=f"mock-request-{sequence}",
        prompt_id="controlled-support",
        prompt_version=1,
        prompt_content_hash="d" * 64,
        schema_version=TICKET_CLASSIFICATION_SCHEMA_VERSION,
        input_tokens=None,
        cached_input_tokens=None,
        output_tokens=None,
        reasoning_tokens=None,
        total_tokens=None,
        pricing_catalog_version=PRICING_CATALOG_VERSION,
        pricing_found=True,
        estimated_input_cost_usd=None,
        estimated_cached_input_cost_usd=None,
        estimated_output_cost_usd=None,
        estimated_total_cost_usd=None,
        latency_ms=12_000,
        error_code=LLMErrorCode.TIMEOUT,
        now=_INVOCATION_AT,
    )


async def _persist_tool_call(
    session: AsyncSession,
    claim: AgentRunClaim,
    tool_call: AgentToolCall,
) -> None:
    result = await SqlAlchemyAgentToolCallExecutionRepository(session).persist_fenced(
        PersistAgentToolCallCommand(
            workspace_id=_WORKSPACE_ID,
            ticket_id=_TICKET_ID,
            agent_run_id=claim.agent_run.id,
            agent_run_attempt_id=claim.attempt.id,
            lease_token=_LEASE_TOKEN,
            persisted_at=_APPROVAL_AT,
            tool_call=tool_call,
        )
    )
    assert result is AgentToolCallPersistenceResult.APPLIED


async def _persist_invocation(
    session: AsyncSession,
    invocation: LLMInvocation,
) -> None:
    session.add(LLMInvocationRecord.from_domain(invocation))
    await session.flush()


async def _seed_pending_context(
    session: AsyncSession,
    *,
    tool_call: AgentToolCall | None = None,
    invocation: LLMInvocation | None = None,
) -> tuple[AgentRunClaim, AgentToolCall, LLMInvocation]:
    claim = await _create_running_claim(session)
    pending_tool_call = tool_call or _pending_tool_call(claim)
    pending_invocation = invocation or _invocation(claim)

    async with SqlAlchemyTransactionManager(session).transaction():
        await _persist_invocation(session, pending_invocation)
        await _persist_tool_call(session, claim, pending_tool_call)

    return claim, pending_tool_call, pending_invocation


def _approval_request(
    tool_call: AgentToolCall,
    *,
    invocation_id: UUID = _INVOCATION_ID,
    approval_request_id: UUID = _APPROVAL_REQUEST_ID,
    request_reason: str = "Requires human review before escalation.",
    expires_at: datetime = _EXPIRES_AT,
    now: datetime = _APPROVAL_AT,
) -> ApprovalRequest:
    return ApprovalRequest.create_pending(
        tool_call=tool_call,
        requested_by_llm_invocation_id=invocation_id,
        request_reason=request_reason,
        expires_at=expires_at,
        approval_request_id=approval_request_id,
        now=now,
    )


async def _persist_approval(
    session: AsyncSession,
    approval_request: ApprovalRequest,
) -> ApprovalRequestPersistenceResult:
    repository = SqlAlchemyApprovalRequestRepository(session)

    async with SqlAlchemyTransactionManager(session).transaction():
        return await repository.persist_pending(approval_request)


async def _count_approvals(session: AsyncSession) -> int:
    result = await session.execute(
        select(func.count()).select_from(ApprovalRequestRecord),
    )
    return int(result.scalar_one())


async def test_persist_pending_applies(
    postgresql_session_factory: async_sessionmaker[AsyncSession],
    clean_business_tables: None,
) -> None:
    async with postgresql_session_factory() as session:
        _, tool_call, _ = await _seed_pending_context(session)
        approval = _approval_request(tool_call)

        result = await _persist_approval(session, approval)

    assert result is ApprovalRequestPersistenceResult.APPLIED

    async with postgresql_session_factory() as session:
        loaded = await SqlAlchemyApprovalRequestRepository(session).get_by_id(
            workspace_id=_WORKSPACE_ID,
            approval_request_id=_APPROVAL_REQUEST_ID,
        )

    assert loaded == approval


async def test_identical_replay_is_already_recorded(
    postgresql_session_factory: async_sessionmaker[AsyncSession],
    clean_business_tables: None,
) -> None:
    async with postgresql_session_factory() as session:
        _, tool_call, _ = await _seed_pending_context(session)
        approval = _approval_request(tool_call)

        first = await _persist_approval(session, approval)
        second = await _persist_approval(
            session,
            _approval_request(
                tool_call,
                approval_request_id=_SECOND_APPROVAL_REQUEST_ID,
                now=_APPROVAL_AT + timedelta(seconds=5),
            ),
        )

    assert first is ApprovalRequestPersistenceResult.APPLIED
    assert second is ApprovalRequestPersistenceResult.ALREADY_RECORDED

    async with postgresql_session_factory() as session:
        assert await _count_approvals(session) == 1


async def test_conflicting_replay_raises_consistency_error(
    postgresql_session_factory: async_sessionmaker[AsyncSession],
    clean_business_tables: None,
) -> None:
    async with postgresql_session_factory() as session:
        _, tool_call, _ = await _seed_pending_context(session)
        await _persist_approval(session, _approval_request(tool_call))

        with pytest.raises(ApprovalRequestConsistencyError):
            await _persist_approval(
                session,
                _approval_request(
                    tool_call,
                    approval_request_id=_SECOND_APPROVAL_REQUEST_ID,
                    request_reason="Different reason.",
                ),
            )


async def test_different_workspace_cannot_load_record(
    postgresql_session_factory: async_sessionmaker[AsyncSession],
    clean_business_tables: None,
) -> None:
    async with postgresql_session_factory() as session:
        _, tool_call, _ = await _seed_pending_context(session)
        await _persist_approval(session, _approval_request(tool_call))

        repository = SqlAlchemyApprovalRequestRepository(session)
        assert (
            await repository.get_by_id(
                workspace_id=_OTHER_WORKSPACE_ID,
                approval_request_id=_APPROVAL_REQUEST_ID,
            )
            is None
        )
        assert (
            await repository.get_by_agent_tool_call_id(
                workspace_id=_OTHER_WORKSPACE_ID,
                agent_tool_call_id=_TOOL_CALL_ID,
            )
            is None
        )
        assert (
            await repository.list_by_agent_run(
                workspace_id=_OTHER_WORKSPACE_ID,
                agent_run_id=_AGENT_RUN_ID,
            )
            == ()
        )


async def test_get_by_id_and_tool_call_id(
    postgresql_session_factory: async_sessionmaker[AsyncSession],
    clean_business_tables: None,
) -> None:
    async with postgresql_session_factory() as session:
        _, tool_call, _ = await _seed_pending_context(session)
        approval = _approval_request(tool_call)
        await _persist_approval(session, approval)

        repository = SqlAlchemyApprovalRequestRepository(session)
        by_id = await repository.get_by_id(
            workspace_id=_WORKSPACE_ID,
            approval_request_id=_APPROVAL_REQUEST_ID,
        )
        by_tool_call = await repository.get_by_agent_tool_call_id(
            workspace_id=_WORKSPACE_ID,
            agent_tool_call_id=_TOOL_CALL_ID,
        )

    assert by_id == approval
    assert by_tool_call == approval


async def test_list_by_agent_run_is_deterministic(
    postgresql_session_factory: async_sessionmaker[AsyncSession],
    clean_business_tables: None,
) -> None:
    async with postgresql_session_factory() as session:
        claim = await _create_running_claim(session)
        first_tool_call = _pending_tool_call(claim)
        second_tool_call = _pending_tool_call(
            claim,
            tool_call_id=_SECOND_TOOL_CALL_ID,
            sequence=2,
            provider_tool_call_id="provider-sensitive-2",
            input_fingerprint="e" * 64,
            safe_input={"reason_code": "alternate"},
        )
        first_invocation = _invocation(claim)
        second_invocation = _invocation(
            claim,
            invocation_id=_SECOND_INVOCATION_ID,
            sequence=2,
        )

        async with SqlAlchemyTransactionManager(session).transaction():
            await _persist_invocation(session, first_invocation)
            await _persist_invocation(session, second_invocation)
            await _persist_tool_call(session, claim, first_tool_call)
            await _persist_tool_call(session, claim, second_tool_call)

        first_approval = _approval_request(
            first_tool_call,
            now=_APPROVAL_AT + timedelta(seconds=2),
        )
        second_approval = _approval_request(
            second_tool_call,
            invocation_id=_SECOND_INVOCATION_ID,
            approval_request_id=_SECOND_APPROVAL_REQUEST_ID,
            now=_APPROVAL_AT + timedelta(seconds=1),
            request_reason="Alternate escalation path.",
        )

        await _persist_approval(session, first_approval)
        await _persist_approval(session, second_approval)

        listed = await SqlAlchemyApprovalRequestRepository(session).list_by_agent_run(
            workspace_id=_WORKSPACE_ID,
            agent_run_id=_AGENT_RUN_ID,
        )

    assert [item.id for item in listed] == [
        _SECOND_APPROVAL_REQUEST_ID,
        _APPROVAL_REQUEST_ID,
    ]


async def test_missing_tool_call_is_rejected(
    postgresql_session_factory: async_sessionmaker[AsyncSession],
    clean_business_tables: None,
) -> None:
    async with postgresql_session_factory() as session:
        claim = await _create_running_claim(session)
        tool_call = _pending_tool_call(claim)
        invocation = _invocation(claim)

        async with SqlAlchemyTransactionManager(session).transaction():
            await _persist_invocation(session, invocation)

        with pytest.raises(ApprovalRequestConsistencyError, match="AgentToolCall"):
            await _persist_approval(session, _approval_request(tool_call))


async def test_non_pending_tool_call_is_rejected(
    postgresql_session_factory: async_sessionmaker[AsyncSession],
    clean_business_tables: None,
) -> None:
    async with postgresql_session_factory() as session:
        claim = await _create_running_claim(session)
        pending = _pending_tool_call(claim)
        invocation = _invocation(claim)

        async with SqlAlchemyTransactionManager(session).transaction():
            await _persist_invocation(session, invocation)
            await _persist_tool_call(session, claim, pending)

        rejected = pending.reject_for_approval(
            decided_at=_APPROVAL_AT + timedelta(minutes=1),
        )

        async with SqlAlchemyTransactionManager(session).transaction():
            await session.execute(
                text(
                    """
                    UPDATE agent_tool_calls
                    SET status = :status,
                        finished_at = :finished_at
                    WHERE id = :tool_call_id
                    """
                ),
                {
                    "status": rejected.status.value,
                    "finished_at": rejected.finished_at,
                    "tool_call_id": rejected.id,
                },
            )

        with pytest.raises(ApprovalRequestConsistencyError, match="does not match"):
            await _persist_approval(session, _approval_request(pending))


async def test_read_only_tool_call_is_rejected(
    postgresql_session_factory: async_sessionmaker[AsyncSession],
    clean_business_tables: None,
) -> None:
    async with postgresql_session_factory() as session:
        claim = await _create_running_claim(session)
        read_only = _read_only_tool_call(claim)
        invocation = _invocation(claim)
        proposal_shaped = _pending_tool_call(
            claim,
            tool_call_id=read_only.id,
            input_fingerprint="f" * 64,
        )

        async with SqlAlchemyTransactionManager(session).transaction():
            await _persist_invocation(session, invocation)
            await _persist_tool_call(session, claim, read_only)

        with pytest.raises(ApprovalRequestConsistencyError, match="does not match"):
            await _persist_approval(session, _approval_request(proposal_shaped))


async def test_mismatched_tool_identity_is_rejected(
    postgresql_session_factory: async_sessionmaker[AsyncSession],
    clean_business_tables: None,
) -> None:
    async with postgresql_session_factory() as session:
        _, tool_call, _ = await _seed_pending_context(session)
        approval = _approval_request(tool_call)
        mismatched = replace(approval, tool_name="create_escalation")

        with pytest.raises(ApprovalRequestConsistencyError, match="does not match"):
            await _persist_approval(session, mismatched)


async def test_missing_or_mismatched_invocation_is_rejected(
    postgresql_session_factory: async_sessionmaker[AsyncSession],
    clean_business_tables: None,
) -> None:
    async with postgresql_session_factory() as session:
        claim = await _create_running_claim(session)
        tool_call = _pending_tool_call(claim)

        async with SqlAlchemyTransactionManager(session).transaction():
            await _persist_tool_call(session, claim, tool_call)

        with pytest.raises(
            ApprovalRequestConsistencyError,
            match="LLM invocation",
        ):
            await _persist_approval(session, _approval_request(tool_call))

        async with SqlAlchemyTransactionManager(session).transaction():
            await _persist_invocation(session, _invocation(claim))

        with pytest.raises(
            ApprovalRequestConsistencyError,
            match="LLM invocation",
        ):
            await _persist_approval(
                session,
                _approval_request(
                    tool_call,
                    invocation_id=_SECOND_INVOCATION_ID,
                ),
            )


async def test_concurrent_identical_persistence_converges(
    postgresql_session_factory: async_sessionmaker[AsyncSession],
    clean_business_tables: None,
) -> None:
    async with postgresql_session_factory() as setup_session:
        _, tool_call, _ = await _seed_pending_context(setup_session)

    ready = 0
    ready_lock = asyncio.Lock()
    release_inserts = asyncio.Event()
    first_approval = _approval_request(tool_call)
    second_approval = _approval_request(
        tool_call,
        approval_request_id=_SECOND_APPROVAL_REQUEST_ID,
        now=_APPROVAL_AT + timedelta(seconds=1),
    )

    async def persist(
        approval: ApprovalRequest,
    ) -> ApprovalRequestPersistenceResult:
        nonlocal ready

        async with postgresql_session_factory() as session:
            repository = SqlAlchemyApprovalRequestRepository(session)

            async with ready_lock:
                ready += 1
                if ready == 2:
                    release_inserts.set()

            await release_inserts.wait()

            async with SqlAlchemyTransactionManager(session).transaction():
                return await repository.persist_pending(approval)

    results = await asyncio.gather(
        persist(first_approval),
        persist(second_approval),
    )

    assert set(results) == {
        ApprovalRequestPersistenceResult.APPLIED,
        ApprovalRequestPersistenceResult.ALREADY_RECORDED,
    }

    async with postgresql_session_factory() as session:
        assert await _count_approvals(session) == 1


async def test_caller_transaction_controls_commit_and_rollback(
    postgresql_session_factory: async_sessionmaker[AsyncSession],
    clean_business_tables: None,
) -> None:
    async with postgresql_session_factory() as session:
        _, tool_call, _ = await _seed_pending_context(session)
        approval = _approval_request(tool_call)

    async with postgresql_session_factory() as session:
        repository = SqlAlchemyApprovalRequestRepository(session)

        with pytest.raises(RuntimeError, match="force rollback"):
            async with SqlAlchemyTransactionManager(session).transaction():
                result = await repository.persist_pending(approval)
                assert result is ApprovalRequestPersistenceResult.APPLIED
                raise RuntimeError("force rollback")

    async with postgresql_session_factory() as session:
        assert await _count_approvals(session) == 0

    async with postgresql_session_factory() as session:
        repository = SqlAlchemyApprovalRequestRepository(session)
        async with SqlAlchemyTransactionManager(session).transaction():
            result = await repository.persist_pending(approval)
            assert result is ApprovalRequestPersistenceResult.APPLIED

    async with postgresql_session_factory() as session:
        assert await _count_approvals(session) == 1


async def test_get_by_id_for_update_is_workspace_scoped(
    postgresql_session_factory: async_sessionmaker[AsyncSession],
    clean_business_tables: None,
) -> None:
    async with postgresql_session_factory() as session:
        _, tool_call, _ = await _seed_pending_context(session)
        approval = _approval_request(tool_call)
        await _persist_approval(session, approval)

        repository = SqlAlchemyApprovalRequestRepository(session)
        async with SqlAlchemyTransactionManager(session).transaction():
            locked = await repository.get_by_id_for_update(
                workspace_id=_WORKSPACE_ID,
                approval_request_id=_APPROVAL_REQUEST_ID,
            )
            cross_workspace = await repository.get_by_id_for_update(
                workspace_id=_OTHER_WORKSPACE_ID,
                approval_request_id=_APPROVAL_REQUEST_ID,
            )

    assert locked == approval
    assert cross_workspace is None


async def test_get_next_expired_pending_for_update_orders_and_skips_locked(
    postgresql_session_factory: async_sessionmaker[AsyncSession],
    clean_business_tables: None,
) -> None:
    now = _APPROVAL_AT + timedelta(hours=25)

    async with postgresql_session_factory() as setup_session:
        claim = await _create_running_claim(setup_session)
        first_tool_call = _pending_tool_call(claim)
        second_tool_call = _pending_tool_call(
            claim,
            tool_call_id=_SECOND_TOOL_CALL_ID,
            sequence=2,
            provider_tool_call_id="provider-sensitive-2",
            input_fingerprint="e" * 64,
            safe_input={"reason_code": "alternate"},
        )
        first_invocation = _invocation(claim)
        second_invocation = _invocation(
            claim,
            invocation_id=_SECOND_INVOCATION_ID,
            sequence=2,
        )

        async with SqlAlchemyTransactionManager(setup_session).transaction():
            await _persist_invocation(setup_session, first_invocation)
            await _persist_invocation(setup_session, second_invocation)
            await _persist_tool_call(setup_session, claim, first_tool_call)
            await _persist_tool_call(setup_session, claim, second_tool_call)

        earlier = _approval_request(
            first_tool_call,
            expires_at=_APPROVAL_AT + timedelta(hours=1),
        )
        later = _approval_request(
            second_tool_call,
            invocation_id=_SECOND_INVOCATION_ID,
            approval_request_id=_SECOND_APPROVAL_REQUEST_ID,
            expires_at=_APPROVAL_AT + timedelta(hours=2),
            request_reason="Alternate escalation path.",
        )
        await _persist_approval(setup_session, earlier)
        await _persist_approval(setup_session, later)

    async with postgresql_session_factory() as first_session:
        first_repository = SqlAlchemyApprovalRequestRepository(first_session)
        async with SqlAlchemyTransactionManager(first_session).transaction():
            first = await first_repository.get_next_expired_pending_for_update(
                now=now,
            )
            assert first is not None
            assert first.id == _APPROVAL_REQUEST_ID

            async with postgresql_session_factory() as second_session:
                second_repository = SqlAlchemyApprovalRequestRepository(
                    second_session,
                )
                async with SqlAlchemyTransactionManager(
                    second_session,
                ).transaction():
                    second = await second_repository.get_next_expired_pending_for_update(
                        now=now,
                    )
                    assert second is not None
                    assert second.id == _SECOND_APPROVAL_REQUEST_ID


async def test_save_updates_decision_fields_only(
    postgresql_session_factory: async_sessionmaker[AsyncSession],
    clean_business_tables: None,
) -> None:
    async with postgresql_session_factory() as session:
        _, tool_call, _ = await _seed_pending_context(session)
        pending = _approval_request(tool_call)
        await _persist_approval(session, pending)

        decided_at = _APPROVAL_AT + timedelta(minutes=5)
        approved = pending.approve(
            actor_reference="operator:alice",
            comment="Looks good.",
            request_id=UUID("88888888-8888-4888-8888-888888888888"),
            correlation_id=UUID("99999999-9999-4999-8999-999999999999"),
            decided_at=decided_at,
        )

        repository = SqlAlchemyApprovalRequestRepository(session)
        async with SqlAlchemyTransactionManager(session).transaction():
            await repository.save(approved)

        loaded = await repository.get_by_id(
            workspace_id=_WORKSPACE_ID,
            approval_request_id=_APPROVAL_REQUEST_ID,
        )

    assert loaded is not None
    assert loaded.status is approved.status
    assert loaded.decision_actor_reference == "operator:alice"
    assert loaded.decision_comment == "Looks good."
    assert loaded.decided_at == decided_at
    assert loaded.updated_at == decided_at
    assert loaded.proposed_input == pending.proposed_input
    assert loaded.request_reason == pending.request_reason
    assert loaded.expires_at == pending.expires_at
    assert loaded.created_at == pending.created_at


async def test_save_rejects_immutable_mismatch(
    postgresql_session_factory: async_sessionmaker[AsyncSession],
    clean_business_tables: None,
) -> None:
    async with postgresql_session_factory() as session:
        _, tool_call, _ = await _seed_pending_context(session)
        pending = _approval_request(tool_call)
        await _persist_approval(session, pending)

        decided_at = _APPROVAL_AT + timedelta(minutes=5)
        approved = pending.approve(
            actor_reference="operator:alice",
            comment=None,
            request_id=UUID("88888888-8888-4888-8888-888888888888"),
            correlation_id=UUID("99999999-9999-4999-8999-999999999999"),
            decided_at=decided_at,
        )
        mismatched = replace(approved, request_reason="Mutated reason.")

        repository = SqlAlchemyApprovalRequestRepository(session)
        with pytest.raises(
            ApprovalRequestConsistencyError,
            match="immutable",
        ):
            async with SqlAlchemyTransactionManager(session).transaction():
                await repository.save(mismatched)


async def test_save_does_not_commit(
    postgresql_session_factory: async_sessionmaker[AsyncSession],
    clean_business_tables: None,
) -> None:
    async with postgresql_session_factory() as session:
        _, tool_call, _ = await _seed_pending_context(session)
        pending = _approval_request(tool_call)
        await _persist_approval(session, pending)

        decided_at = _APPROVAL_AT + timedelta(minutes=5)
        approved = pending.approve(
            actor_reference="operator:alice",
            comment=None,
            request_id=UUID("88888888-8888-4888-8888-888888888888"),
            correlation_id=UUID("99999999-9999-4999-8999-999999999999"),
            decided_at=decided_at,
        )

        repository = SqlAlchemyApprovalRequestRepository(session)
        with pytest.raises(RuntimeError, match="force rollback"):
            async with SqlAlchemyTransactionManager(session).transaction():
                await repository.save(approved)
                raise RuntimeError("force rollback")

    async with postgresql_session_factory() as session:
        loaded = await SqlAlchemyApprovalRequestRepository(session).get_by_id(
            workspace_id=_WORKSPACE_ID,
            approval_request_id=_APPROVAL_REQUEST_ID,
        )

    assert loaded is not None
    assert loaded.status.value == "pending"
    assert loaded.decided_at is None


async def test_get_next_expired_pending_ignores_future_and_terminal_rows(
    postgresql_session_factory: async_sessionmaker[AsyncSession],
    clean_business_tables: None,
) -> None:
    now = _APPROVAL_AT + timedelta(hours=25)
    overdue_expires_at = _APPROVAL_AT + timedelta(hours=1)
    decision_ids = (
        (
            UUID("88888888-8888-4888-8888-888888888881"),
            UUID("99999999-9999-4999-8999-999999999991"),
        ),
        (
            UUID("88888888-8888-4888-8888-888888888882"),
            UUID("99999999-9999-4999-8999-999999999992"),
        ),
    )

    async with postgresql_session_factory() as setup_session:
        claim = await _create_running_claim(setup_session)
        tool_specs = (
            (_TOOL_CALL_ID, 1, "provider-sensitive-1", "b" * 64, "one"),
            (_SECOND_TOOL_CALL_ID, 2, "provider-sensitive-2", "e" * 64, "two"),
            (_THIRD_TOOL_CALL_ID, 3, "provider-sensitive-3", "f" * 64, "three"),
            (_FOURTH_TOOL_CALL_ID, 4, "provider-sensitive-4", "c" * 64, "four"),
        )
        invocation_ids = (
            _INVOCATION_ID,
            _SECOND_INVOCATION_ID,
            _THIRD_INVOCATION_ID,
            _FOURTH_INVOCATION_ID,
        )
        approval_ids = (
            _APPROVAL_REQUEST_ID,
            _SECOND_APPROVAL_REQUEST_ID,
            _THIRD_APPROVAL_REQUEST_ID,
            _FOURTH_APPROVAL_REQUEST_ID,
        )

        tool_calls = []
        for index, (
            tool_call_id,
            sequence,
            provider_id,
            fingerprint,
            reason,
        ) in enumerate(tool_specs):
            tool_call = _pending_tool_call(
                claim,
                tool_call_id=tool_call_id,
                sequence=sequence,
                provider_tool_call_id=provider_id,
                input_fingerprint=fingerprint,
                safe_input={"reason_code": reason},
            )
            tool_calls.append(tool_call)
            invocation = _invocation(
                claim,
                invocation_id=invocation_ids[index],
                sequence=sequence,
            )
            async with SqlAlchemyTransactionManager(setup_session).transaction():
                await _persist_invocation(setup_session, invocation)
                await _persist_tool_call(setup_session, claim, tool_call)

        approved = _approval_request(
            tool_calls[0],
            approval_request_id=approval_ids[0],
            expires_at=overdue_expires_at,
        )
        rejected = _approval_request(
            tool_calls[1],
            invocation_id=invocation_ids[1],
            approval_request_id=approval_ids[1],
            expires_at=overdue_expires_at,
            request_reason="Rejected proposal.",
        )
        expired = _approval_request(
            tool_calls[2],
            invocation_id=invocation_ids[2],
            approval_request_id=approval_ids[2],
            expires_at=overdue_expires_at,
            request_reason="Expired proposal.",
        )
        future = _approval_request(
            tool_calls[3],
            invocation_id=invocation_ids[3],
            approval_request_id=approval_ids[3],
            expires_at=_APPROVAL_AT + timedelta(hours=48),
            request_reason="Still within TTL.",
        )

        for pending in (approved, rejected, expired, future):
            await _persist_approval(setup_session, pending)

        repository = SqlAlchemyApprovalRequestRepository(setup_session)
        decided_at = _APPROVAL_AT + timedelta(minutes=5)
        async with SqlAlchemyTransactionManager(setup_session).transaction():
            await repository.save(
                approved.approve(
                    actor_reference="operator:alice",
                    comment=None,
                    request_id=decision_ids[0][0],
                    correlation_id=decision_ids[0][1],
                    decided_at=decided_at,
                ),
            )
            await repository.save(
                rejected.reject(
                    actor_reference="operator:bob",
                    comment="Not warranted.",
                    request_id=decision_ids[1][0],
                    correlation_id=decision_ids[1][1],
                    decided_at=decided_at,
                ),
            )
            await repository.save(
                expired.expire(decided_at=overdue_expires_at),
            )

    async with postgresql_session_factory() as session:
        repository = SqlAlchemyApprovalRequestRepository(session)
        async with SqlAlchemyTransactionManager(session).transaction():
            selected = await repository.get_next_expired_pending_for_update(
                now=now,
            )

    assert selected is None


async def test_get_next_expired_pending_orders_by_expires_at_then_id(
    postgresql_session_factory: async_sessionmaker[AsyncSession],
    clean_business_tables: None,
) -> None:
    now = _APPROVAL_AT + timedelta(hours=25)
    shared_expiry = _APPROVAL_AT + timedelta(hours=1)

    async with postgresql_session_factory() as setup_session:
        claim = await _create_running_claim(setup_session)
        first_tool_call = _pending_tool_call(claim)
        second_tool_call = _pending_tool_call(
            claim,
            tool_call_id=_SECOND_TOOL_CALL_ID,
            sequence=2,
            provider_tool_call_id="provider-sensitive-2",
            input_fingerprint="e" * 64,
            safe_input={"reason_code": "alternate"},
        )
        first_invocation = _invocation(claim)
        second_invocation = _invocation(
            claim,
            invocation_id=_SECOND_INVOCATION_ID,
            sequence=2,
        )

        async with SqlAlchemyTransactionManager(setup_session).transaction():
            await _persist_invocation(setup_session, first_invocation)
            await _persist_invocation(setup_session, second_invocation)
            await _persist_tool_call(setup_session, claim, first_tool_call)
            await _persist_tool_call(setup_session, claim, second_tool_call)

        higher_id = _approval_request(
            first_tool_call,
            approval_request_id=_THIRD_APPROVAL_REQUEST_ID,
            expires_at=shared_expiry,
        )
        lower_id = _approval_request(
            second_tool_call,
            invocation_id=_SECOND_INVOCATION_ID,
            approval_request_id=_SECOND_APPROVAL_REQUEST_ID,
            expires_at=shared_expiry,
            request_reason="Alternate escalation path.",
        )
        await _persist_approval(setup_session, higher_id)
        await _persist_approval(setup_session, lower_id)

    async with postgresql_session_factory() as session:
        repository = SqlAlchemyApprovalRequestRepository(session)
        async with SqlAlchemyTransactionManager(session).transaction():
            first = await repository.get_next_expired_pending_for_update(
                now=now,
            )

    assert first is not None
    assert first.id == _SECOND_APPROVAL_REQUEST_ID


async def test_concurrent_conflicting_persist_pending_fails_closed(
    postgresql_session_factory: async_sessionmaker[AsyncSession],
    clean_business_tables: None,
) -> None:
    async with postgresql_session_factory() as setup_session:
        _, tool_call, _ = await _seed_pending_context(setup_session)

    ready = 0
    ready_lock = asyncio.Lock()
    release_inserts = asyncio.Event()
    first_approval = _approval_request(tool_call)
    conflicting = _approval_request(
        tool_call,
        approval_request_id=_SECOND_APPROVAL_REQUEST_ID,
        request_reason="Conflicting proposal reason.",
        now=_APPROVAL_AT + timedelta(seconds=1),
    )
    outcomes: list[object] = []

    async def persist(
        approval: ApprovalRequest,
    ) -> ApprovalRequestPersistenceResult | Exception:
        nonlocal ready

        async with postgresql_session_factory() as session:
            repository = SqlAlchemyApprovalRequestRepository(session)

            async with ready_lock:
                ready += 1
                if ready == 2:
                    release_inserts.set()

            await release_inserts.wait()

            try:
                async with SqlAlchemyTransactionManager(session).transaction():
                    result = await repository.persist_pending(approval)
                # Session must remain usable after the conflict/savepoint path.
                await session.execute(select(func.count()).select_from(ApprovalRequestRecord))
                return result
            except Exception as exc:
                await session.execute(select(func.count()).select_from(ApprovalRequestRecord))
                return exc

    outcomes = list(
        await asyncio.gather(
            persist(first_approval),
            persist(conflicting),
        ),
    )

    applied = [item for item in outcomes if item is ApprovalRequestPersistenceResult.APPLIED]
    conflicts = [item for item in outcomes if isinstance(item, ApprovalRequestConsistencyError)]
    assert len(applied) == 1
    assert len(conflicts) == 1

    async with postgresql_session_factory() as session:
        assert await _count_approvals(session) == 1
        first_loaded = await SqlAlchemyApprovalRequestRepository(session).get_by_id(
            workspace_id=_WORKSPACE_ID,
            approval_request_id=_APPROVAL_REQUEST_ID,
        )
        second_loaded = await SqlAlchemyApprovalRequestRepository(session).get_by_id(
            workspace_id=_WORKSPACE_ID,
            approval_request_id=_SECOND_APPROVAL_REQUEST_ID,
        )

    winners = [row for row in (first_loaded, second_loaded) if row is not None]
    assert len(winners) == 1
    assert winners[0].request_reason in {
        first_approval.request_reason,
        conflicting.request_reason,
    }
