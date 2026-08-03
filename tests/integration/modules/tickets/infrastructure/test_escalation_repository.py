"""Integration tests for ticket escalation persistence."""

import asyncio
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from uuid import UUID, uuid4

import pytest
from pydantic import JsonValue
from sqlalchemy import func, select, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
)

from supportops.agent_tools.application.grant_persistence import (
    SensitiveExecutionGrantPersistenceResult,
)
from supportops.agent_tools.application.persistence import (
    AgentToolCallPersistenceResult,
    PersistAgentToolCallCommand,
)
from supportops.agent_tools.domain.audit import AgentToolCall
from supportops.agent_tools.domain.grants import SensitiveExecutionGrant
from supportops.agent_tools.infrastructure.grant_repository import (
    SqlAlchemySensitiveExecutionGrantRepository,
)
from supportops.agent_tools.infrastructure.repository import (
    SqlAlchemyAgentToolCallExecutionRepository,
)
from supportops.agent_tools.tools.escalate_ticket import (
    EscalateTicketInput,
    TicketEscalationTargetQueue,
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
from supportops.modules.tickets.domain.escalation import (
    TicketEscalation,
)
from supportops.modules.tickets.domain.escalation_repositories import (
    TicketEscalationConsistencyError,
    TicketEscalationPersistenceResult,
)
from supportops.modules.tickets.domain.models import Ticket, TicketStatus
from supportops.modules.tickets.infrastructure.escalation_models import (
    TicketEscalationRecord,
)
from supportops.modules.tickets.infrastructure.escalation_repository import (
    SqlAlchemyTicketEscalationRepository,
)
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
_INVOCATION_ID = UUID("80000000-0000-4000-8000-000000000008")
_APPROVAL_REQUEST_ID = UUID("90000000-0000-4000-8000-000000000009")
_GRANT_ID = UUID("a0000000-0000-4000-8000-00000000000a")
_ESCALATION_ID = UUID("d0000000-0000-4000-8000-00000000000d")
_SECOND_ESCALATION_ID = UUID("d0000000-0000-4000-8000-00000000000e")
_DECISION_REQUEST_ID = UUID("b0000000-0000-4000-8000-00000000000b")
_DECISION_CORRELATION_ID = UUID("c0000000-0000-4000-8000-00000000000c")

_CREATED_AT = datetime(2026, 8, 3, 18, 30, tzinfo=UTC)
_CLAIMED_AT = _CREATED_AT + timedelta(minutes=1)
_LEASE_EXPIRES_AT = _CLAIMED_AT + timedelta(seconds=45)
_TOOL_PROPOSED_AT = _CLAIMED_AT + timedelta(seconds=1)
_INVOCATION_AT = _CLAIMED_AT + timedelta(seconds=2)
_APPROVAL_AT = _CLAIMED_AT + timedelta(seconds=3)
_EXPIRES_AT = _APPROVAL_AT + timedelta(hours=24)
_DECIDED_AT = _APPROVAL_AT + timedelta(minutes=5)
_GRANT_CREATED_AT = _DECIDED_AT + timedelta(minutes=1)
_ESCALATION_CREATED_AT = _GRANT_CREATED_AT + timedelta(minutes=1)


async def _create_running_claim(
    session: AsyncSession,
) -> AgentRunClaim:
    workspace = Workspace(
        id=_WORKSPACE_ID,
        name="Escalation Workspace",
        slug="escalation-workspace",
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
                worker_id="escalation-worker-1",
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


async def _seed_granted_context(
    session: AsyncSession,
) -> tuple[AgentRunClaim, AgentToolCall, ApprovalRequest, SensitiveExecutionGrant]:
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
    grant = SensitiveExecutionGrant.create(
        approval_request=approved,
        tool_call=tool_call,
        executed_by_agent_run_attempt_id=claim.attempt.id,
        created_at=_GRANT_CREATED_AT,
        grant_id=_GRANT_ID,
    )

    async with SqlAlchemyTransactionManager(session).transaction():
        await _persist_invocation(session, invocation)
        await _persist_tool_call(session, claim, tool_call)
        result = await SqlAlchemyApprovalRequestRepository(session).persist_pending(
            pending,
        )
        assert result is ApprovalRequestPersistenceResult.APPLIED
        await SqlAlchemyApprovalRequestRepository(session).save(approved)
        grant_result = await SqlAlchemySensitiveExecutionGrantRepository(session).persist(grant)
        assert grant_result is SensitiveExecutionGrantPersistenceResult.APPLIED

    return claim, tool_call, approved, grant


def _escalation(
    grant: SensitiveExecutionGrant,
    *,
    escalation_id: UUID = _ESCALATION_ID,
    created_at: datetime = _ESCALATION_CREATED_AT,
) -> TicketEscalation:
    return TicketEscalation.create_from_grant(
        grant=grant,
        input_data=EscalateTicketInput.model_validate(
            dict(grant.granted_input),
        ),
        created_at=created_at,
        escalation_id=escalation_id,
    )


async def _persist_escalation(
    session: AsyncSession,
    escalation: TicketEscalation,
) -> TicketEscalationPersistenceResult:
    repository = SqlAlchemyTicketEscalationRepository(session)
    async with SqlAlchemyTransactionManager(session).transaction():
        return await repository.persist(escalation)


async def _count_escalations(session: AsyncSession) -> int:
    result = await session.execute(select(func.count()).select_from(TicketEscalationRecord))
    return int(result.scalar_one())


async def test_persist_escalation_applies_and_is_idempotent(
    postgresql_session_factory: async_sessionmaker[AsyncSession],
    clean_business_tables: None,
) -> None:
    async with postgresql_session_factory() as session:
        _, _, _, grant = await _seed_granted_context(session)
        escalation = _escalation(grant)

        first = await _persist_escalation(session, escalation)
        second = await _persist_escalation(session, escalation)

    assert first is TicketEscalationPersistenceResult.APPLIED
    assert second is TicketEscalationPersistenceResult.ALREADY_RECORDED

    async with postgresql_session_factory() as session:
        assert await _count_escalations(session) == 1


async def test_workspace_scope_hides_escalation(
    postgresql_session_factory: async_sessionmaker[AsyncSession],
    clean_business_tables: None,
) -> None:
    async with postgresql_session_factory() as session:
        _, tool_call, approval, grant = await _seed_granted_context(session)
        escalation = _escalation(grant)
        await _persist_escalation(session, escalation)

        repository = SqlAlchemyTicketEscalationRepository(session)
        result = await repository.get_by_id(
            workspace_id=_OTHER_WORKSPACE_ID,
            escalation_id=escalation.id,
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


async def test_get_helpers_return_workspace_scoped_escalation(
    postgresql_session_factory: async_sessionmaker[AsyncSession],
    clean_business_tables: None,
) -> None:
    async with postgresql_session_factory() as session:
        _, tool_call, approval, grant = await _seed_granted_context(session)
        escalation = _escalation(grant)
        await _persist_escalation(session, escalation)

        repository = SqlAlchemyTicketEscalationRepository(session)
        by_id = await repository.get_by_id(
            workspace_id=_WORKSPACE_ID,
            escalation_id=escalation.id,
        )
        by_approval = await repository.get_by_approval_request_id(
            workspace_id=_WORKSPACE_ID,
            approval_request_id=approval.id,
        )
        by_tool_call = await repository.get_by_agent_tool_call_id(
            workspace_id=_WORKSPACE_ID,
            agent_tool_call_id=tool_call.id,
        )

    assert by_id == escalation
    assert by_approval == escalation
    assert by_tool_call == escalation


async def test_conflicting_replay_fails_closed(
    postgresql_session_factory: async_sessionmaker[AsyncSession],
    clean_business_tables: None,
) -> None:
    async with postgresql_session_factory() as session:
        _, _, _, grant = await _seed_granted_context(session)
        escalation = _escalation(grant)
        await _persist_escalation(session, escalation)

        conflicting = replace(
            escalation,
            id=_SECOND_ESCALATION_ID,
            executed_by_agent_run_attempt_id=uuid4(),
            created_at=_ESCALATION_CREATED_AT + timedelta(seconds=1),
        )

        with pytest.raises(TicketEscalationConsistencyError):
            await _persist_escalation(session, conflicting)

        assert await _count_escalations(session) == 1


async def test_caller_controls_commit_and_rollback(
    postgresql_session_factory: async_sessionmaker[AsyncSession],
    clean_business_tables: None,
) -> None:
    async with postgresql_session_factory() as session:
        _, _, _, grant = await _seed_granted_context(session)
        escalation = _escalation(grant)

    async with postgresql_session_factory() as session:
        repository = SqlAlchemyTicketEscalationRepository(session)

        with pytest.raises(RuntimeError, match="force rollback"):
            async with SqlAlchemyTransactionManager(session).transaction():
                result = await repository.persist(escalation)
                assert result is TicketEscalationPersistenceResult.APPLIED
                raise RuntimeError("force rollback")

    async with postgresql_session_factory() as session:
        assert await _count_escalations(session) == 0

    async with postgresql_session_factory() as session:
        result = await _persist_escalation(session, escalation)
        assert result is TicketEscalationPersistenceResult.APPLIED

    async with postgresql_session_factory() as session:
        assert await _count_escalations(session) == 1


async def test_concurrent_identical_persistence_converges(
    postgresql_session_factory: async_sessionmaker[AsyncSession],
    clean_business_tables: None,
) -> None:
    async with postgresql_session_factory() as setup_session:
        _, _, _, grant = await _seed_granted_context(setup_session)
        first = _escalation(grant)
        second = replace(
            first,
            id=_SECOND_ESCALATION_ID,
            created_at=_ESCALATION_CREATED_AT + timedelta(seconds=1),
        )

    ready = 0
    ready_lock = asyncio.Lock()
    release_inserts = asyncio.Event()

    async def persist(
        escalation: TicketEscalation,
    ) -> TicketEscalationPersistenceResult:
        nonlocal ready

        async with postgresql_session_factory() as session:
            repository = SqlAlchemyTicketEscalationRepository(session)

            async with ready_lock:
                ready += 1
                if ready == 2:
                    release_inserts.set()

            await release_inserts.wait()

            async with SqlAlchemyTransactionManager(session).transaction():
                return await repository.persist(escalation)

    results = await asyncio.gather(
        persist(first),
        persist(second),
    )

    assert set(results) == {
        TicketEscalationPersistenceResult.APPLIED,
        TicketEscalationPersistenceResult.ALREADY_RECORDED,
    }

    async with postgresql_session_factory() as session:
        assert await _count_escalations(session) == 1


async def test_concurrent_conflicting_persistence_yields_one_row_and_conflict(
    postgresql_session_factory: async_sessionmaker[AsyncSession],
    clean_business_tables: None,
) -> None:
    async with postgresql_session_factory() as setup_session:
        _, _, _, grant = await _seed_granted_context(setup_session)
        first = _escalation(grant)
        conflicting = replace(
            first,
            id=_SECOND_ESCALATION_ID,
            target_queue=TicketEscalationTargetQueue.SUPPORT_OPERATIONS,
            reason="A different escalation reason.",
            created_at=_ESCALATION_CREATED_AT + timedelta(seconds=1),
        )

    ready = 0
    ready_lock = asyncio.Lock()
    release_inserts = asyncio.Event()

    async def persist(
        escalation: TicketEscalation,
    ) -> TicketEscalationPersistenceResult | Exception:
        nonlocal ready

        async with postgresql_session_factory() as session:
            repository = SqlAlchemyTicketEscalationRepository(session)

            async with ready_lock:
                ready += 1
                if ready == 2:
                    release_inserts.set()

            await release_inserts.wait()

            try:
                async with SqlAlchemyTransactionManager(session).transaction():
                    result = await repository.persist(escalation)
                await session.execute(select(func.count()).select_from(TicketEscalationRecord))
                return result
            except Exception as exc:
                await session.execute(select(func.count()).select_from(TicketEscalationRecord))
                return exc

    outcomes = list(
        await asyncio.gather(
            persist(first),
            persist(conflicting),
        ),
    )

    applied = [item for item in outcomes if item is TicketEscalationPersistenceResult.APPLIED]
    conflicts = [item for item in outcomes if isinstance(item, TicketEscalationConsistencyError)]
    # Conflicting content may fail closed before insert when grant input
    # validation rejects the mismatched candidate, or after unique conflict.
    assert len(applied) == 1
    assert len(conflicts) == 1

    async with postgresql_session_factory() as session:
        assert await _count_escalations(session) == 1


async def test_savepoint_path_leaves_session_usable(
    postgresql_session_factory: async_sessionmaker[AsyncSession],
    clean_business_tables: None,
) -> None:
    async with postgresql_session_factory() as session:
        _, _, _, grant = await _seed_granted_context(session)
        escalation = _escalation(grant)
        await _persist_escalation(session, escalation)

        repository = SqlAlchemyTicketEscalationRepository(session)
        async with SqlAlchemyTransactionManager(session).transaction():
            replay = await repository.persist(escalation)
            assert replay is (TicketEscalationPersistenceResult.ALREADY_RECORDED)
            count = await session.execute(select(func.count()).select_from(TicketEscalationRecord))
            assert int(count.scalar_one()) == 1


async def test_ticket_remains_open_after_escalation_persist(
    postgresql_session_factory: async_sessionmaker[AsyncSession],
    clean_business_tables: None,
) -> None:
    async with postgresql_session_factory() as session:
        _, _, _, grant = await _seed_granted_context(session)
        escalation = _escalation(grant)
        await _persist_escalation(session, escalation)

        ticket = await SqlAlchemyTicketRepository(session).get(
            workspace_id=_WORKSPACE_ID,
            ticket_id=_TICKET_ID,
        )

    assert ticket is not None
    assert ticket.status is TicketStatus.OPEN


async def test_fk_rejects_wrong_ticket_ownership(
    postgresql_session_factory: async_sessionmaker[AsyncSession],
    clean_business_tables: None,
) -> None:
    async with postgresql_session_factory() as session:
        _, _, _, grant = await _seed_granted_context(session)
        escalation = _escalation(grant)

        async with SqlAlchemyTransactionManager(session).transaction():
            with pytest.raises(IntegrityError):
                await session.execute(
                    text(
                        """
                        INSERT INTO ticket_escalations (
                            id, workspace_id, ticket_id, agent_run_id,
                            executed_by_agent_run_attempt_id,
                            approval_request_id, agent_tool_call_id,
                            target_queue, reason, created_at
                        ) VALUES (
                            :id, :workspace_id, :ticket_id, :agent_run_id,
                            :attempt_id, :approval_id, :tool_call_id,
                            :target_queue, :reason, :created_at
                        )
                        """
                    ),
                    {
                        "id": uuid4(),
                        "workspace_id": _WORKSPACE_ID,
                        "ticket_id": uuid4(),
                        "agent_run_id": escalation.agent_run_id,
                        "attempt_id": (escalation.executed_by_agent_run_attempt_id),
                        "approval_id": escalation.approval_request_id,
                        "tool_call_id": escalation.agent_tool_call_id,
                        "target_queue": escalation.target_queue.value,
                        "reason": escalation.reason,
                        "created_at": escalation.created_at,
                    },
                )


async def test_fk_rejects_wrong_agent_run(
    postgresql_session_factory: async_sessionmaker[AsyncSession],
    clean_business_tables: None,
) -> None:
    async with postgresql_session_factory() as session:
        _, _, _, grant = await _seed_granted_context(session)
        escalation = _escalation(grant)

        async with SqlAlchemyTransactionManager(session).transaction():
            with pytest.raises(IntegrityError):
                await session.execute(
                    text(
                        """
                        INSERT INTO ticket_escalations (
                            id, workspace_id, ticket_id, agent_run_id,
                            executed_by_agent_run_attempt_id,
                            approval_request_id, agent_tool_call_id,
                            target_queue, reason, created_at
                        ) VALUES (
                            :id, :workspace_id, :ticket_id, :agent_run_id,
                            :attempt_id, :approval_id, :tool_call_id,
                            :target_queue, :reason, :created_at
                        )
                        """
                    ),
                    {
                        "id": uuid4(),
                        "workspace_id": _WORKSPACE_ID,
                        "ticket_id": _TICKET_ID,
                        "agent_run_id": uuid4(),
                        "attempt_id": (escalation.executed_by_agent_run_attempt_id),
                        "approval_id": escalation.approval_request_id,
                        "tool_call_id": escalation.agent_tool_call_id,
                        "target_queue": escalation.target_queue.value,
                        "reason": escalation.reason,
                        "created_at": escalation.created_at,
                    },
                )


async def test_fk_rejects_wrong_attempt(
    postgresql_session_factory: async_sessionmaker[AsyncSession],
    clean_business_tables: None,
) -> None:
    async with postgresql_session_factory() as session:
        _, _, _, grant = await _seed_granted_context(session)
        escalation = _escalation(grant)

        async with SqlAlchemyTransactionManager(session).transaction():
            with pytest.raises(IntegrityError):
                await session.execute(
                    text(
                        """
                        INSERT INTO ticket_escalations (
                            id, workspace_id, ticket_id, agent_run_id,
                            executed_by_agent_run_attempt_id,
                            approval_request_id, agent_tool_call_id,
                            target_queue, reason, created_at
                        ) VALUES (
                            :id, :workspace_id, :ticket_id, :agent_run_id,
                            :attempt_id, :approval_id, :tool_call_id,
                            :target_queue, :reason, :created_at
                        )
                        """
                    ),
                    {
                        "id": uuid4(),
                        "workspace_id": _WORKSPACE_ID,
                        "ticket_id": _TICKET_ID,
                        "agent_run_id": escalation.agent_run_id,
                        "attempt_id": uuid4(),
                        "approval_id": escalation.approval_request_id,
                        "tool_call_id": escalation.agent_tool_call_id,
                        "target_queue": escalation.target_queue.value,
                        "reason": escalation.reason,
                        "created_at": escalation.created_at,
                    },
                )


async def test_fk_rejects_missing_approval(
    postgresql_session_factory: async_sessionmaker[AsyncSession],
    clean_business_tables: None,
) -> None:
    async with postgresql_session_factory() as session:
        _, _, _, grant = await _seed_granted_context(session)
        escalation = _escalation(grant)

        async with SqlAlchemyTransactionManager(session).transaction():
            with pytest.raises(IntegrityError):
                await session.execute(
                    text(
                        """
                        INSERT INTO ticket_escalations (
                            id, workspace_id, ticket_id, agent_run_id,
                            executed_by_agent_run_attempt_id,
                            approval_request_id, agent_tool_call_id,
                            target_queue, reason, created_at
                        ) VALUES (
                            :id, :workspace_id, :ticket_id, :agent_run_id,
                            :attempt_id, :approval_id, :tool_call_id,
                            :target_queue, :reason, :created_at
                        )
                        """
                    ),
                    {
                        "id": uuid4(),
                        "workspace_id": _WORKSPACE_ID,
                        "ticket_id": _TICKET_ID,
                        "agent_run_id": escalation.agent_run_id,
                        "attempt_id": (escalation.executed_by_agent_run_attempt_id),
                        "approval_id": uuid4(),
                        "tool_call_id": escalation.agent_tool_call_id,
                        "target_queue": escalation.target_queue.value,
                        "reason": escalation.reason,
                        "created_at": escalation.created_at,
                    },
                )


async def test_fk_rejects_missing_tool_call(
    postgresql_session_factory: async_sessionmaker[AsyncSession],
    clean_business_tables: None,
) -> None:
    async with postgresql_session_factory() as session:
        _, _, _, grant = await _seed_granted_context(session)
        escalation = _escalation(grant)

        async with SqlAlchemyTransactionManager(session).transaction():
            with pytest.raises(IntegrityError):
                await session.execute(
                    text(
                        """
                        INSERT INTO ticket_escalations (
                            id, workspace_id, ticket_id, agent_run_id,
                            executed_by_agent_run_attempt_id,
                            approval_request_id, agent_tool_call_id,
                            target_queue, reason, created_at
                        ) VALUES (
                            :id, :workspace_id, :ticket_id, :agent_run_id,
                            :attempt_id, :approval_id, :tool_call_id,
                            :target_queue, :reason, :created_at
                        )
                        """
                    ),
                    {
                        "id": uuid4(),
                        "workspace_id": _WORKSPACE_ID,
                        "ticket_id": _TICKET_ID,
                        "agent_run_id": escalation.agent_run_id,
                        "attempt_id": (escalation.executed_by_agent_run_attempt_id),
                        "approval_id": escalation.approval_request_id,
                        "tool_call_id": uuid4(),
                        "target_queue": escalation.target_queue.value,
                        "reason": escalation.reason,
                        "created_at": escalation.created_at,
                    },
                )
