"""Integration tests for sensitive execution grant persistence."""

import asyncio
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from uuid import UUID, uuid4

import pytest
from pydantic import JsonValue
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
)

from supportops.agent_tools.application.grant_persistence import (
    SensitiveExecutionGrantConsistencyError,
    SensitiveExecutionGrantPersistenceResult,
)
from supportops.agent_tools.application.persistence import (
    AgentToolCallPersistenceResult,
    PersistAgentToolCallCommand,
)
from supportops.agent_tools.domain.audit import AgentToolCall
from supportops.agent_tools.domain.grants import SensitiveExecutionGrant
from supportops.agent_tools.infrastructure.grant_models import (
    SensitiveExecutionGrantRecord,
)
from supportops.agent_tools.infrastructure.grant_repository import (
    SqlAlchemySensitiveExecutionGrantRepository,
)
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
    ApprovalRequestPersistenceResult,
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
_INVOCATION_ID = UUID("80000000-0000-4000-8000-000000000008")
_APPROVAL_REQUEST_ID = UUID("90000000-0000-4000-8000-000000000009")
_SECOND_APPROVAL_REQUEST_ID = UUID("90000000-0000-4000-8000-000000000019")
_GRANT_ID = UUID("a0000000-0000-4000-8000-00000000000a")
_SECOND_GRANT_ID = UUID("a0000000-0000-4000-8000-00000000000b")
_DECISION_REQUEST_ID = UUID("b0000000-0000-4000-8000-00000000000b")
_DECISION_CORRELATION_ID = UUID("c0000000-0000-4000-8000-00000000000c")

_CREATED_AT = datetime(2026, 8, 3, 18, 0, tzinfo=UTC)
_CLAIMED_AT = _CREATED_AT + timedelta(minutes=1)
_LEASE_EXPIRES_AT = _CLAIMED_AT + timedelta(seconds=45)
_TOOL_PROPOSED_AT = _CLAIMED_AT + timedelta(seconds=1)
_INVOCATION_AT = _CLAIMED_AT + timedelta(seconds=2)
_APPROVAL_AT = _CLAIMED_AT + timedelta(seconds=3)
_EXPIRES_AT = _APPROVAL_AT + timedelta(hours=24)
_DECIDED_AT = _APPROVAL_AT + timedelta(minutes=5)
_GRANT_CREATED_AT = _DECIDED_AT + timedelta(minutes=1)


async def _create_running_claim(
    session: AsyncSession,
) -> AgentRunClaim:
    workspace = Workspace(
        id=_WORKSPACE_ID,
        name="Grant Workspace",
        slug="grant-workspace",
        created_at=_CREATED_AT,
        updated_at=_CREATED_AT,
    )
    ticket = Ticket.create(
        ticket_id=_TICKET_ID,
        workspace_id=_WORKSPACE_ID,
        subject="Needs approved escalation",
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
                worker_id="grant-worker-1",
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
        tool_name="escalate_ticket",
        tool_version=1,
        input_fingerprint=input_fingerprint,
        safe_input=safe_input
        or {
            "target_queue": "engineering_support",
            "reason": "A product defect requires review.",
        },
        proposed_at=_TOOL_PROPOSED_AT,
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


async def _seed_approved_context(
    session: AsyncSession,
) -> tuple[AgentRunClaim, AgentToolCall, ApprovalRequest]:
    claim = await _create_running_claim(session)
    tool_call = _pending_tool_call(claim)
    invocation = _invocation(claim)
    pending = ApprovalRequest.create_pending(
        tool_call=tool_call,
        requested_by_llm_invocation_id=_INVOCATION_ID,
        request_reason="Requires human review before escalation.",
        expires_at=_EXPIRES_AT,
        approval_request_id=_APPROVAL_REQUEST_ID,
        now=_APPROVAL_AT,
    )
    approved = pending.approve(
        actor_reference="operator:alice",
        comment=None,
        request_id=_DECISION_REQUEST_ID,
        correlation_id=_DECISION_CORRELATION_ID,
        decided_at=_DECIDED_AT,
    )

    async with SqlAlchemyTransactionManager(session).transaction():
        await _persist_invocation(session, invocation)
        await _persist_tool_call(session, claim, tool_call)
        result = await SqlAlchemyApprovalRequestRepository(session).persist_pending(
            pending,
        )
        assert result is ApprovalRequestPersistenceResult.APPLIED
        await SqlAlchemyApprovalRequestRepository(session).save(approved)

    return claim, tool_call, approved


def _grant(
    claim: AgentRunClaim,
    tool_call: AgentToolCall,
    approval: ApprovalRequest,
    *,
    grant_id: UUID = _GRANT_ID,
    executed_by_agent_run_attempt_id: UUID | None = None,
    created_at: datetime = _GRANT_CREATED_AT,
) -> SensitiveExecutionGrant:
    return SensitiveExecutionGrant.create(
        approval_request=approval,
        tool_call=tool_call,
        executed_by_agent_run_attempt_id=(executed_by_agent_run_attempt_id or claim.attempt.id),
        created_at=created_at,
        grant_id=grant_id,
    )


async def _persist_grant(
    session: AsyncSession,
    grant: SensitiveExecutionGrant,
) -> SensitiveExecutionGrantPersistenceResult:
    repository = SqlAlchemySensitiveExecutionGrantRepository(session)
    async with SqlAlchemyTransactionManager(session).transaction():
        return await repository.persist(grant)


async def _count_grants(session: AsyncSession) -> int:
    result = await session.execute(select(func.count()).select_from(SensitiveExecutionGrantRecord))
    return int(result.scalar_one())


async def test_persist_grant_applies_and_is_idempotent(
    postgresql_session_factory: async_sessionmaker[AsyncSession],
    clean_business_tables: None,
) -> None:
    async with postgresql_session_factory() as session:
        claim, tool_call, approval = await _seed_approved_context(session)
        grant = _grant(claim, tool_call, approval)

        first = await _persist_grant(session, grant)
        second = await _persist_grant(session, grant)

    assert first is SensitiveExecutionGrantPersistenceResult.APPLIED
    assert second is SensitiveExecutionGrantPersistenceResult.ALREADY_RECORDED

    async with postgresql_session_factory() as session:
        assert await _count_grants(session) == 1


async def test_workspace_scope_hides_grant(
    postgresql_session_factory: async_sessionmaker[AsyncSession],
    clean_business_tables: None,
) -> None:
    async with postgresql_session_factory() as session:
        claim, tool_call, approval = await _seed_approved_context(session)
        grant = _grant(claim, tool_call, approval)
        await _persist_grant(session, grant)

        repository = SqlAlchemySensitiveExecutionGrantRepository(session)
        result = await repository.get_by_id(
            workspace_id=_OTHER_WORKSPACE_ID,
            grant_id=grant.id,
        )
        by_approval = await repository.get_by_approval_request_id(
            workspace_id=_OTHER_WORKSPACE_ID,
            approval_request_id=approval.id,
        )
        by_tool_call = await repository.get_by_agent_tool_call_id(
            workspace_id=_OTHER_WORKSPACE_ID,
            agent_tool_call_id=tool_call.id,
        )

    assert result is None
    assert by_approval is None
    assert by_tool_call is None


async def test_get_helpers_return_workspace_scoped_grant(
    postgresql_session_factory: async_sessionmaker[AsyncSession],
    clean_business_tables: None,
) -> None:
    async with postgresql_session_factory() as session:
        claim, tool_call, approval = await _seed_approved_context(session)
        grant = _grant(claim, tool_call, approval)
        await _persist_grant(session, grant)

        repository = SqlAlchemySensitiveExecutionGrantRepository(session)
        by_id = await repository.get_by_id(
            workspace_id=_WORKSPACE_ID,
            grant_id=grant.id,
        )
        by_approval = await repository.get_by_approval_request_id(
            workspace_id=_WORKSPACE_ID,
            approval_request_id=approval.id,
        )
        by_tool_call = await repository.get_by_agent_tool_call_id(
            workspace_id=_WORKSPACE_ID,
            agent_tool_call_id=tool_call.id,
        )

    assert by_id == grant
    assert by_approval == grant
    assert by_tool_call == grant


async def test_conflicting_replay_fails_closed(
    postgresql_session_factory: async_sessionmaker[AsyncSession],
    clean_business_tables: None,
) -> None:
    async with postgresql_session_factory() as session:
        claim, tool_call, approval = await _seed_approved_context(session)
        grant = _grant(claim, tool_call, approval)
        await _persist_grant(session, grant)

        conflicting = replace(
            grant,
            id=_SECOND_GRANT_ID,
            executed_by_agent_run_attempt_id=uuid4(),
            created_at=_GRANT_CREATED_AT + timedelta(seconds=1),
        )

        with pytest.raises(SensitiveExecutionGrantConsistencyError):
            await _persist_grant(session, conflicting)

        assert await _count_grants(session) == 1


async def test_missing_approval_is_rejected(
    postgresql_session_factory: async_sessionmaker[AsyncSession],
    clean_business_tables: None,
) -> None:
    async with postgresql_session_factory() as session:
        claim, tool_call, approval = await _seed_approved_context(session)
        grant = _grant(claim, tool_call, approval)
        missing_approval = replace(grant, approval_request_id=uuid4())

        with pytest.raises(
            SensitiveExecutionGrantConsistencyError,
            match="ApprovalRequest",
        ):
            await _persist_grant(session, missing_approval)


async def test_non_approved_approval_is_rejected(
    postgresql_session_factory: async_sessionmaker[AsyncSession],
    clean_business_tables: None,
) -> None:
    async with postgresql_session_factory() as session:
        claim = await _create_running_claim(session)
        tool_call = _pending_tool_call(claim)
        invocation = _invocation(claim)
        pending = ApprovalRequest.create_pending(
            tool_call=tool_call,
            requested_by_llm_invocation_id=_INVOCATION_ID,
            request_reason="Requires human review before escalation.",
            expires_at=_EXPIRES_AT,
            approval_request_id=_APPROVAL_REQUEST_ID,
            now=_APPROVAL_AT,
        )

        async with SqlAlchemyTransactionManager(session).transaction():
            await _persist_invocation(session, invocation)
            await _persist_tool_call(session, claim, tool_call)
            await SqlAlchemyApprovalRequestRepository(session).persist_pending(
                pending,
            )

        # Domain create requires approved status, so craft a candidate grant
        # against durable pending approval state via an approved-shaped object
        # that still points at the pending durable row.
        approved_shape = pending.approve(
            actor_reference="operator:alice",
            comment=None,
            request_id=_DECISION_REQUEST_ID,
            correlation_id=_DECISION_CORRELATION_ID,
            decided_at=_DECIDED_AT,
        )
        grant = _grant(claim, tool_call, approved_shape)

        with pytest.raises(
            SensitiveExecutionGrantConsistencyError,
            match="ApprovalRequest",
        ):
            await _persist_grant(session, grant)


async def test_non_pending_tool_call_is_rejected(
    postgresql_session_factory: async_sessionmaker[AsyncSession],
    clean_business_tables: None,
) -> None:
    async with postgresql_session_factory() as session:
        claim, tool_call, approval = await _seed_approved_context(session)
        grant = _grant(claim, tool_call, approval)
        rejected = tool_call.reject_for_approval(decided_at=_GRANT_CREATED_AT)

        async with SqlAlchemyTransactionManager(session).transaction():
            await SqlAlchemyAgentToolCallExecutionRepository(session).save_approval_outcome(
                rejected
            )

        with pytest.raises(
            SensitiveExecutionGrantConsistencyError,
            match="AgentToolCall",
        ):
            await _persist_grant(session, grant)


async def test_wrong_execution_attempt_ownership_is_rejected(
    postgresql_session_factory: async_sessionmaker[AsyncSession],
    clean_business_tables: None,
) -> None:
    async with postgresql_session_factory() as session:
        claim, tool_call, approval = await _seed_approved_context(session)
        grant = _grant(
            claim,
            tool_call,
            approval,
            executed_by_agent_run_attempt_id=uuid4(),
        )

        with pytest.raises(
            SensitiveExecutionGrantConsistencyError,
            match="AgentRunAttempt",
        ):
            await _persist_grant(session, grant)


async def test_caller_controls_commit_and_rollback(
    postgresql_session_factory: async_sessionmaker[AsyncSession],
    clean_business_tables: None,
) -> None:
    async with postgresql_session_factory() as session:
        claim, tool_call, approval = await _seed_approved_context(session)
        grant = _grant(claim, tool_call, approval)

    async with postgresql_session_factory() as session:
        repository = SqlAlchemySensitiveExecutionGrantRepository(session)

        with pytest.raises(RuntimeError, match="force rollback"):
            async with SqlAlchemyTransactionManager(session).transaction():
                result = await repository.persist(grant)
                assert result is SensitiveExecutionGrantPersistenceResult.APPLIED
                raise RuntimeError("force rollback")

    async with postgresql_session_factory() as session:
        assert await _count_grants(session) == 0

    async with postgresql_session_factory() as session:
        result = await _persist_grant(session, grant)
        assert result is SensitiveExecutionGrantPersistenceResult.APPLIED

    async with postgresql_session_factory() as session:
        assert await _count_grants(session) == 1


async def test_concurrent_identical_persistence_converges(
    postgresql_session_factory: async_sessionmaker[AsyncSession],
    clean_business_tables: None,
) -> None:
    async with postgresql_session_factory() as setup_session:
        claim, tool_call, approval = await _seed_approved_context(setup_session)
        first_grant = _grant(claim, tool_call, approval)
        second_grant = replace(
            first_grant,
            id=_SECOND_GRANT_ID,
            created_at=_GRANT_CREATED_AT + timedelta(seconds=1),
        )

    ready = 0
    ready_lock = asyncio.Lock()
    release_inserts = asyncio.Event()

    async def persist(
        grant: SensitiveExecutionGrant,
    ) -> SensitiveExecutionGrantPersistenceResult:
        nonlocal ready

        async with postgresql_session_factory() as session:
            repository = SqlAlchemySensitiveExecutionGrantRepository(session)

            async with ready_lock:
                ready += 1
                if ready == 2:
                    release_inserts.set()

            await release_inserts.wait()

            async with SqlAlchemyTransactionManager(session).transaction():
                return await repository.persist(grant)

    results = await asyncio.gather(
        persist(first_grant),
        persist(second_grant),
    )

    assert set(results) == {
        SensitiveExecutionGrantPersistenceResult.APPLIED,
        SensitiveExecutionGrantPersistenceResult.ALREADY_RECORDED,
    }

    async with postgresql_session_factory() as session:
        assert await _count_grants(session) == 1


async def test_concurrent_conflicting_persistence_yields_one_row_and_conflict(
    postgresql_session_factory: async_sessionmaker[AsyncSession],
    clean_business_tables: None,
) -> None:
    async with postgresql_session_factory() as setup_session:
        claim, tool_call, approval = await _seed_approved_context(setup_session)
        first_grant = _grant(claim, tool_call, approval)
        conflicting = replace(
            first_grant,
            id=_SECOND_GRANT_ID,
            decision_actor_reference="operator:bob",
            created_at=_GRANT_CREATED_AT + timedelta(seconds=1),
        )

    ready = 0
    ready_lock = asyncio.Lock()
    release_inserts = asyncio.Event()

    async def persist(
        grant: SensitiveExecutionGrant,
    ) -> SensitiveExecutionGrantPersistenceResult | Exception:
        nonlocal ready

        async with postgresql_session_factory() as session:
            repository = SqlAlchemySensitiveExecutionGrantRepository(session)

            async with ready_lock:
                ready += 1
                if ready == 2:
                    release_inserts.set()

            await release_inserts.wait()

            try:
                async with SqlAlchemyTransactionManager(session).transaction():
                    result = await repository.persist(grant)
                # Session must remain usable after the conflict/savepoint path.
                await session.execute(
                    select(func.count()).select_from(SensitiveExecutionGrantRecord)
                )
                return result
            except Exception as exc:
                await session.execute(
                    select(func.count()).select_from(SensitiveExecutionGrantRecord)
                )
                return exc

    outcomes = list(
        await asyncio.gather(
            persist(first_grant),
            persist(conflicting),
        ),
    )

    applied = [
        item for item in outcomes if item is SensitiveExecutionGrantPersistenceResult.APPLIED
    ]
    conflicts = [
        item for item in outcomes if isinstance(item, SensitiveExecutionGrantConsistencyError)
    ]
    assert len(applied) == 1
    assert len(conflicts) == 1

    async with postgresql_session_factory() as session:
        assert await _count_grants(session) == 1


async def test_savepoint_path_leaves_session_usable(
    postgresql_session_factory: async_sessionmaker[AsyncSession],
    clean_business_tables: None,
) -> None:
    async with postgresql_session_factory() as session:
        claim, tool_call, approval = await _seed_approved_context(session)
        grant = _grant(claim, tool_call, approval)
        await _persist_grant(session, grant)

        repository = SqlAlchemySensitiveExecutionGrantRepository(session)
        async with SqlAlchemyTransactionManager(session).transaction():
            replay = await repository.persist(grant)
            assert replay is (SensitiveExecutionGrantPersistenceResult.ALREADY_RECORDED)
            count = await session.execute(
                select(func.count()).select_from(SensitiveExecutionGrantRecord)
            )
            assert int(count.scalar_one()) == 1
