"""Integration tests for fenced PostgreSQL tool-call persistence."""

import asyncio
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from uuid import UUID

import pytest
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
)

from supportops.agent_tools.application.persistence import (
    AgentToolCallPersistenceResult,
    PersistAgentToolCallCommand,
)
from supportops.agent_tools.application.queries import AgentToolCallLookup
from supportops.agent_tools.domain.audit import (
    AgentToolCall,
    AgentToolCallStatus,
)
from supportops.agent_tools.domain.contracts import (
    ToolSafetyLevel,
)
from supportops.agent_tools.infrastructure.models import (
    AgentToolCallRecord,
)
from supportops.agent_tools.infrastructure.query_repository import (
    SqlAlchemyAgentToolCallQueryRepository,
)
from supportops.agent_tools.infrastructure.repository import (
    SqlAlchemyAgentToolCallExecutionRepository,
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
_TICKET_ID = UUID("20000000-0000-4000-8000-000000000002")
_AGENT_RUN_ID = UUID("30000000-0000-4000-8000-000000000003")
_LEASE_TOKEN = UUID("40000000-0000-4000-8000-000000000004")
_EXECUTION_REQUEST_ID = UUID("50000000-0000-4000-8000-000000000005")
_TOOL_CALL_ID = UUID("60000000-0000-4000-8000-000000000006")
_SECOND_TOOL_CALL_ID = UUID("70000000-0000-4000-8000-000000000007")

_CREATED_AT = datetime(
    2026,
    8,
    2,
    18,
    0,
    tzinfo=UTC,
)
_CLAIMED_AT = _CREATED_AT + timedelta(minutes=1)
_LEASE_EXPIRES_AT = _CLAIMED_AT + timedelta(seconds=45)
_TOOL_STARTED_AT = _CLAIMED_AT + timedelta(seconds=1)
_TOOL_FINISHED_AT = _TOOL_STARTED_AT + timedelta(milliseconds=25)
_PERSISTED_AT = _TOOL_FINISHED_AT + timedelta(milliseconds=5)


async def _create_running_claim(
    session: AsyncSession,
) -> AgentRunClaim:
    workspace = Workspace(
        id=_WORKSPACE_ID,
        name="Controlled Support",
        slug="controlled-support",
        created_at=_CREATED_AT,
        updated_at=_CREATED_AT,
    )
    ticket = Ticket.create(
        ticket_id=_TICKET_ID,
        workspace_id=_WORKSPACE_ID,
        subject="Unable to reset account access",
        description=("The customer cannot complete the documented access-reset procedure."),
        external_reference=None,
        ingestion_request_id=UUID("81000000-0000-4000-8000-000000000008"),
        correlation_id=UUID("82000000-0000-4000-8000-000000000009"),
        now=_CREATED_AT,
    )
    agent_run = AgentRun.create_initial(
        agent_run_id=_AGENT_RUN_ID,
        workspace_id=_WORKSPACE_ID,
        ticket_id=_TICKET_ID,
        ingestion_request_id=(ticket.ingestion_request_id),
        correlation_id=ticket.correlation_id,
        workflow_version=(DETERMINISTIC_BASELINE_WORKFLOW_VERSION),
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
                worker_id="controlled-worker-1",
                lease_token=_LEASE_TOKEN,
                execution_request_id=(_EXECUTION_REQUEST_ID),
                claimed_at=_CLAIMED_AT,
                lease_expires_at=_LEASE_EXPIRES_AT,
            )
        )

    assert claim is not None

    return claim


def _tool_call(
    claim: AgentRunClaim,
    *,
    tool_call_id: UUID = _TOOL_CALL_ID,
    sequence: int = 1,
    provider_tool_call_id: str = ("provider-tool-call-1"),
    input_fingerprint: str = "a" * 64,
) -> AgentToolCall:
    return AgentToolCall.create_terminal(
        tool_call_id=tool_call_id,
        workspace_id=_WORKSPACE_ID,
        ticket_id=_TICKET_ID,
        agent_run_id=claim.agent_run.id,
        agent_run_attempt_id=claim.attempt.id,
        sequence=sequence,
        provider_tool_call_id=provider_tool_call_id,
        tool_name="search_knowledge",
        tool_version=1,
        safety_level=ToolSafetyLevel.READ_ONLY,
        status=AgentToolCallStatus.SUCCEEDED,
        input_fingerprint=input_fingerprint,
        safe_input={
            "top_k": 5,
            "document_ids": None,
        },
        safe_output={
            "result_count": 1,
            "chunk_ids": ["83000000-0000-4000-8000-000000000010"],
        },
        latency_ms=25,
        error_code=None,
        started_at=_TOOL_STARTED_AT,
        finished_at=_TOOL_FINISHED_AT,
    )


def _command(
    claim: AgentRunClaim,
    *,
    tool_call: AgentToolCall | None = None,
    lease_token: UUID = _LEASE_TOKEN,
    persisted_at: datetime = _PERSISTED_AT,
) -> PersistAgentToolCallCommand:
    return PersistAgentToolCallCommand(
        workspace_id=_WORKSPACE_ID,
        ticket_id=_TICKET_ID,
        agent_run_id=claim.agent_run.id,
        agent_run_attempt_id=claim.attempt.id,
        lease_token=lease_token,
        persisted_at=persisted_at,
        tool_call=tool_call or _tool_call(claim),
    )


async def _persist(
    session: AsyncSession,
    command: PersistAgentToolCallCommand,
) -> AgentToolCallPersistenceResult:
    repository = SqlAlchemyAgentToolCallExecutionRepository(session)

    async with SqlAlchemyTransactionManager(session).transaction():
        return await repository.persist_fenced(command)


async def _load_records(
    session: AsyncSession,
) -> tuple[AgentToolCallRecord, ...]:
    result = await session.execute(
        select(AgentToolCallRecord).order_by(AgentToolCallRecord.sequence.asc())
    )

    return tuple(result.scalars().all())


async def _count_records(
    session: AsyncSession,
) -> int:
    result = await session.execute(select(func.count()).select_from(AgentToolCallRecord))

    return int(result.scalar_one())


async def test_persists_terminal_tool_call_under_active_lease(
    postgresql_session_factory: async_sessionmaker[AsyncSession],
    clean_business_tables: None,
) -> None:
    async with postgresql_session_factory() as session:
        claim = await _create_running_claim(session)

        result = await _persist(
            session,
            _command(claim),
        )

    assert result is AgentToolCallPersistenceResult.APPLIED

    async with postgresql_session_factory() as session:
        records = await _load_records(session)

    assert len(records) == 1
    assert records[0].to_domain() == _tool_call(claim)


async def test_identical_replay_is_idempotent(
    postgresql_session_factory: async_sessionmaker[AsyncSession],
    clean_business_tables: None,
) -> None:
    async with postgresql_session_factory() as session:
        claim = await _create_running_claim(session)
        command = _command(claim)

        first_result = await _persist(
            session,
            command,
        )
        second_result = await _persist(
            session,
            command,
        )

    assert first_result is (AgentToolCallPersistenceResult.APPLIED)
    assert second_result is (AgentToolCallPersistenceResult.ALREADY_RECORDED)

    async with postgresql_session_factory() as session:
        assert await _count_records(session) == 1


async def test_rejects_conflicting_replay_for_same_sequence(
    postgresql_session_factory: async_sessionmaker[AsyncSession],
    clean_business_tables: None,
) -> None:
    async with postgresql_session_factory() as session:
        claim = await _create_running_claim(session)
        original_call = _tool_call(claim)

        await _persist(
            session,
            _command(
                claim,
                tool_call=original_call,
            ),
        )

        conflicting_call = replace(
            original_call,
            id=_SECOND_TOOL_CALL_ID,
            safe_output={
                "result_count": 0,
                "chunk_ids": [],
            },
        )

        with pytest.raises(
            RuntimeError,
            match=("sequence is already persisted with different audit data"),
        ):
            await _persist(
                session,
                _command(
                    claim,
                    tool_call=conflicting_call,
                ),
            )

    async with postgresql_session_factory() as session:
        assert await _count_records(session) == 1


async def test_rejects_provider_call_id_reuse_for_new_sequence(
    postgresql_session_factory: async_sessionmaker[AsyncSession],
    clean_business_tables: None,
) -> None:
    async with postgresql_session_factory() as session:
        claim = await _create_running_claim(session)
        original_call = _tool_call(claim)

        await _persist(
            session,
            _command(
                claim,
                tool_call=original_call,
            ),
        )

        conflicting_call = _tool_call(
            claim,
            tool_call_id=_SECOND_TOOL_CALL_ID,
            sequence=2,
            provider_tool_call_id=(original_call.provider_tool_call_id or "provider-tool-call-1"),
            input_fingerprint="b" * 64,
        )

        with pytest.raises(
            RuntimeError,
            match=("provider tool-call identifier is already persisted"),
        ):
            await _persist(
                session,
                _command(
                    claim,
                    tool_call=conflicting_call,
                ),
            )

    async with postgresql_session_factory() as session:
        assert await _count_records(session) == 1


async def test_returns_lease_lost_for_wrong_token(
    postgresql_session_factory: async_sessionmaker[AsyncSession],
    clean_business_tables: None,
) -> None:
    async with postgresql_session_factory() as session:
        claim = await _create_running_claim(session)

        result = await _persist(
            session,
            _command(
                claim,
                lease_token=UUID("90000000-0000-4000-8000-000000000011"),
            ),
        )

    assert result is (AgentToolCallPersistenceResult.LEASE_LOST)

    async with postgresql_session_factory() as session:
        assert await _count_records(session) == 0


async def test_returns_lease_lost_after_expiration(
    postgresql_session_factory: async_sessionmaker[AsyncSession],
    clean_business_tables: None,
) -> None:
    async with postgresql_session_factory() as session:
        claim = await _create_running_claim(session)

        result = await _persist(
            session,
            _command(
                claim,
                persisted_at=_LEASE_EXPIRES_AT,
            ),
        )

    assert result is (AgentToolCallPersistenceResult.LEASE_LOST)

    async with postgresql_session_factory() as session:
        assert await _count_records(session) == 0


async def test_loads_terminal_audit_by_exact_attempt_sequence(
    postgresql_session_factory: async_sessionmaker[AsyncSession],
    clean_business_tables: None,
) -> None:
    async with postgresql_session_factory() as session:
        claim = await _create_running_claim(session)

        await _persist(
            session,
            _command(claim),
        )

        query = AgentToolCallLookup(
            workspace_id=_WORKSPACE_ID,
            ticket_id=_TICKET_ID,
            agent_run_id=claim.agent_run.id,
            proposed_by_agent_run_attempt_id=claim.attempt.id,
            sequence=1,
        )

        async with SqlAlchemyTransactionManager(session).transaction():
            loaded = await SqlAlchemyAgentToolCallQueryRepository(
                session
            ).get_by_proposal_attempt_sequence(query)

    assert loaded == _tool_call(claim)


async def test_returns_none_for_cross_workspace_tool_audit_lookup(
    postgresql_session_factory: async_sessionmaker[AsyncSession],
    clean_business_tables: None,
) -> None:
    async with postgresql_session_factory() as session:
        claim = await _create_running_claim(session)

        await _persist(
            session,
            _command(claim),
        )

        query = AgentToolCallLookup(
            workspace_id=UUID("a0000000-0000-4000-8000-000000000012"),
            ticket_id=_TICKET_ID,
            agent_run_id=claim.agent_run.id,
            proposed_by_agent_run_attempt_id=claim.attempt.id,
            sequence=1,
        )

        async with SqlAlchemyTransactionManager(session).transaction():
            loaded = await SqlAlchemyAgentToolCallQueryRepository(
                session
            ).get_by_proposal_attempt_sequence(query)

    assert loaded is None


async def test_returns_none_for_missing_tool_call_sequence(
    postgresql_session_factory: async_sessionmaker[AsyncSession],
    clean_business_tables: None,
) -> None:
    async with postgresql_session_factory() as session:
        claim = await _create_running_claim(session)

        await _persist(
            session,
            _command(claim),
        )

        query = AgentToolCallLookup(
            workspace_id=_WORKSPACE_ID,
            ticket_id=_TICKET_ID,
            agent_run_id=claim.agent_run.id,
            proposed_by_agent_run_attempt_id=claim.attempt.id,
            sequence=2,
        )

        async with SqlAlchemyTransactionManager(session).transaction():
            loaded = await SqlAlchemyAgentToolCallQueryRepository(
                session
            ).get_by_proposal_attempt_sequence(query)

    assert loaded is None


def _pending_sensitive_tool_call(
    claim: AgentRunClaim,
    *,
    tool_call_id: UUID = _TOOL_CALL_ID,
    sequence: int = 1,
    provider_tool_call_id: str = "provider-sensitive-1",
    input_fingerprint: str = "b" * 64,
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
        safe_input={"reason_code": "policy_required"},
        proposed_at=_TOOL_STARTED_AT,
    )


async def test_persists_pending_sensitive_proposal(
    postgresql_session_factory: async_sessionmaker[AsyncSession],
    clean_business_tables: None,
) -> None:
    async with postgresql_session_factory() as session:
        claim = await _create_running_claim(session)
        tool_call = _pending_sensitive_tool_call(claim)

        result = await _persist(
            session,
            _command(claim, tool_call=tool_call),
        )

    assert result is AgentToolCallPersistenceResult.APPLIED

    async with postgresql_session_factory() as session:
        records = await _load_records(session)

    assert len(records) == 1
    assert records[0].to_domain() == tool_call
    assert records[0].executed_by_agent_run_attempt_id is None


async def test_sensitive_proposal_replay_returns_already_recorded(
    postgresql_session_factory: async_sessionmaker[AsyncSession],
    clean_business_tables: None,
) -> None:
    async with postgresql_session_factory() as session:
        claim = await _create_running_claim(session)
        first = _pending_sensitive_tool_call(claim)
        proposal_attempt_id = claim.attempt.id

        first_result = await _persist(
            session,
            _command(claim, tool_call=first),
        )

        replay = AgentToolCall.propose_for_approval(
            tool_call_id=_SECOND_TOOL_CALL_ID,
            workspace_id=_WORKSPACE_ID,
            ticket_id=_TICKET_ID,
            agent_run_id=claim.agent_run.id,
            proposed_by_agent_run_attempt_id=claim.attempt.id,
            sequence=2,
            provider_tool_call_id="provider-sensitive-2",
            tool_name="escalate_ticket",
            tool_version=1,
            input_fingerprint="b" * 64,
            safe_input={"reason_code": "policy_required"},
            proposed_at=_TOOL_STARTED_AT + timedelta(seconds=1),
        )

        second_result = await _persist(
            session,
            _command(
                claim,
                tool_call=replay,
                persisted_at=_TOOL_STARTED_AT + timedelta(seconds=2),
            ),
        )

    assert first_result is AgentToolCallPersistenceResult.APPLIED
    assert second_result is AgentToolCallPersistenceResult.ALREADY_RECORDED

    async with postgresql_session_factory() as session:
        records = await _load_records(session)

    assert len(records) == 1
    assert records[0].id == first.id
    assert records[0].proposed_by_agent_run_attempt_id == proposal_attempt_id
    assert records[0].provider_tool_call_id == "provider-sensitive-1"


async def test_conflicting_sensitive_proposal_raises(
    postgresql_session_factory: async_sessionmaker[AsyncSession],
    clean_business_tables: None,
) -> None:
    async with postgresql_session_factory() as session:
        claim = await _create_running_claim(session)
        first = _pending_sensitive_tool_call(claim)

        await _persist(
            session,
            _command(claim, tool_call=first),
        )

        conflicting = AgentToolCall.propose_for_approval(
            tool_call_id=_SECOND_TOOL_CALL_ID,
            workspace_id=_WORKSPACE_ID,
            ticket_id=_TICKET_ID,
            agent_run_id=claim.agent_run.id,
            proposed_by_agent_run_attempt_id=claim.attempt.id,
            sequence=2,
            provider_tool_call_id="provider-sensitive-2",
            tool_name="escalate_ticket",
            tool_version=1,
            input_fingerprint="b" * 64,
            safe_input={"reason_code": "different_reason"},
            proposed_at=_TOOL_STARTED_AT + timedelta(seconds=1),
        )

        with pytest.raises(
            RuntimeError,
            match="conflicting ownership or input data",
        ):
            await _persist(
                session,
                _command(
                    claim,
                    tool_call=conflicting,
                    persisted_at=_TOOL_STARTED_AT + timedelta(seconds=2),
                ),
            )


async def test_save_approval_outcome_rejects_pending_to_rejected(
    postgresql_session_factory: async_sessionmaker[AsyncSession],
    clean_business_tables: None,
) -> None:
    decided_at = _TOOL_STARTED_AT + timedelta(minutes=1)

    async with postgresql_session_factory() as session:
        claim = await _create_running_claim(session)
        tool_call = _pending_sensitive_tool_call(claim)
        await _persist(session, _command(claim, tool_call=tool_call))

        rejected = tool_call.reject_for_approval(decided_at=decided_at)
        repository = SqlAlchemyAgentToolCallExecutionRepository(session)
        async with SqlAlchemyTransactionManager(session).transaction():
            await repository.save_approval_outcome(rejected)

            loaded = await repository.get_by_id_for_update(
                workspace_id=_WORKSPACE_ID,
                agent_tool_call_id=tool_call.id,
            )

    assert loaded is not None
    assert loaded.status is AgentToolCallStatus.REJECTED
    assert loaded.finished_at == decided_at
    assert loaded.tool_name == tool_call.tool_name
    assert loaded.tool_version == tool_call.tool_version
    assert loaded.input_fingerprint == tool_call.input_fingerprint
    assert dict(loaded.safe_input) == dict(tool_call.safe_input)
    assert loaded.proposed_by_agent_run_attempt_id == (tool_call.proposed_by_agent_run_attempt_id)
    assert loaded.executed_by_agent_run_attempt_id is None
    assert loaded.proposed_at == tool_call.proposed_at
    assert loaded.execution_started_at is None
    assert loaded.safe_output is None
    assert loaded.latency_ms is None
    assert loaded.error_code is None


async def test_save_approval_outcome_expires_pending_proposal(
    postgresql_session_factory: async_sessionmaker[AsyncSession],
    clean_business_tables: None,
) -> None:
    decided_at = _TOOL_STARTED_AT + timedelta(hours=24)

    async with postgresql_session_factory() as session:
        claim = await _create_running_claim(session)
        tool_call = _pending_sensitive_tool_call(claim)
        await _persist(session, _command(claim, tool_call=tool_call))

        expired = tool_call.expire_for_approval(decided_at=decided_at)
        repository = SqlAlchemyAgentToolCallExecutionRepository(session)
        async with SqlAlchemyTransactionManager(session).transaction():
            await repository.save_approval_outcome(expired)

            loaded = await repository.get_by_id_for_update(
                workspace_id=_WORKSPACE_ID,
                agent_tool_call_id=tool_call.id,
            )

    assert loaded is not None
    assert loaded.status is AgentToolCallStatus.EXPIRED
    assert loaded.finished_at == decided_at
    assert loaded.tool_name == tool_call.tool_name
    assert loaded.input_fingerprint == tool_call.input_fingerprint
    assert dict(loaded.safe_input) == dict(tool_call.safe_input)
    assert loaded.proposed_by_agent_run_attempt_id == (tool_call.proposed_by_agent_run_attempt_id)
    assert loaded.executed_by_agent_run_attempt_id is None
    assert loaded.proposed_at == tool_call.proposed_at
    assert loaded.execution_started_at is None
    assert loaded.safe_output is None


async def test_get_by_id_for_update_is_workspace_scoped(
    postgresql_session_factory: async_sessionmaker[AsyncSession],
    clean_business_tables: None,
) -> None:
    async with postgresql_session_factory() as session:
        claim = await _create_running_claim(session)
        tool_call = _pending_sensitive_tool_call(claim)
        await _persist(session, _command(claim, tool_call=tool_call))

        repository = SqlAlchemyAgentToolCallExecutionRepository(session)
        async with SqlAlchemyTransactionManager(session).transaction():
            locked = await repository.get_by_id_for_update(
                workspace_id=_WORKSPACE_ID,
                agent_tool_call_id=tool_call.id,
            )
            cross_workspace = await repository.get_by_id_for_update(
                workspace_id=UUID("aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"),
                agent_tool_call_id=tool_call.id,
            )

    assert locked == tool_call
    assert cross_workspace is None


async def test_save_approval_outcome_rejects_non_pending(
    postgresql_session_factory: async_sessionmaker[AsyncSession],
    clean_business_tables: None,
) -> None:
    async with postgresql_session_factory() as session:
        claim = await _create_running_claim(session)
        tool_call = _pending_sensitive_tool_call(claim)
        await _persist(session, _command(claim, tool_call=tool_call))

        rejected = tool_call.reject_for_approval(
            decided_at=_TOOL_STARTED_AT + timedelta(minutes=1),
        )
        repository = SqlAlchemyAgentToolCallExecutionRepository(session)
        async with SqlAlchemyTransactionManager(session).transaction():
            await repository.save_approval_outcome(rejected)

        with pytest.raises(RuntimeError, match="pending_approval"):
            async with SqlAlchemyTransactionManager(session).transaction():
                await repository.save_approval_outcome(rejected)


async def test_save_approval_outcome_rejects_immutable_mismatch(
    postgresql_session_factory: async_sessionmaker[AsyncSession],
    clean_business_tables: None,
) -> None:
    async with postgresql_session_factory() as session:
        claim = await _create_running_claim(session)
        tool_call = _pending_sensitive_tool_call(claim)
        await _persist(session, _command(claim, tool_call=tool_call))

        rejected = replace(
            tool_call.reject_for_approval(
                decided_at=_TOOL_STARTED_AT + timedelta(minutes=1),
            ),
            tool_name="create_escalation",
        )
        repository = SqlAlchemyAgentToolCallExecutionRepository(session)

        with pytest.raises(RuntimeError, match="immutable"):
            async with SqlAlchemyTransactionManager(session).transaction():
                await repository.save_approval_outcome(rejected)


async def test_concurrent_reject_and_expire_produce_one_terminal_state(
    postgresql_session_factory: async_sessionmaker[AsyncSession],
    clean_business_tables: None,
) -> None:
    decided_at = _TOOL_STARTED_AT + timedelta(minutes=1)

    async with postgresql_session_factory() as setup_session:
        claim = await _create_running_claim(setup_session)
        tool_call = _pending_sensitive_tool_call(claim)
        await _persist(setup_session, _command(claim, tool_call=tool_call))

    ready = 0
    ready_lock = asyncio.Lock()
    release = asyncio.Event()
    outcomes: list[object] = []

    async def finalize(kind: str) -> str | Exception:
        nonlocal ready

        async with postgresql_session_factory() as session:
            repository = SqlAlchemyAgentToolCallExecutionRepository(session)
            terminal = (
                tool_call.reject_for_approval(decided_at=decided_at)
                if kind == "reject"
                else tool_call.expire_for_approval(decided_at=decided_at)
            )

            async with ready_lock:
                ready += 1
                if ready == 2:
                    release.set()
            await release.wait()

            try:
                async with SqlAlchemyTransactionManager(session).transaction():
                    await repository.save_approval_outcome(terminal)
                return kind
            except Exception as exc:
                return exc

    outcomes = list(
        await asyncio.gather(
            finalize("reject"),
            finalize("expire"),
        ),
    )

    successes = [item for item in outcomes if item in {"reject", "expire"}]
    failures = [item for item in outcomes if isinstance(item, Exception)]
    assert len(successes) == 1
    assert len(failures) == 1

    async with postgresql_session_factory() as session:
        repository = SqlAlchemyAgentToolCallExecutionRepository(session)
        async with SqlAlchemyTransactionManager(session).transaction():
            loaded = await repository.get_by_id_for_update(
                workspace_id=_WORKSPACE_ID,
                agent_tool_call_id=tool_call.id,
            )

    assert loaded is not None
    assert loaded.status in {
        AgentToolCallStatus.REJECTED,
        AgentToolCallStatus.EXPIRED,
    }
    assert loaded.finished_at == decided_at

    conflicting = (
        tool_call.expire_for_approval(decided_at=decided_at + timedelta(minutes=1))
        if loaded.status is AgentToolCallStatus.REJECTED
        else tool_call.reject_for_approval(
            decided_at=decided_at + timedelta(minutes=1),
        )
    )
    async with postgresql_session_factory() as session:
        repository = SqlAlchemyAgentToolCallExecutionRepository(session)
        with pytest.raises(RuntimeError, match="pending_approval"):
            async with SqlAlchemyTransactionManager(session).transaction():
                await repository.save_approval_outcome(conflicting)

    async with postgresql_session_factory() as session:
        repository = SqlAlchemyAgentToolCallExecutionRepository(session)
        async with SqlAlchemyTransactionManager(session).transaction():
            final = await repository.get_by_id_for_update(
                workspace_id=_WORKSPACE_ID,
                agent_tool_call_id=tool_call.id,
            )

    assert final is not None
    assert final.status is loaded.status
    assert final.finished_at == decided_at


async def test_save_granted_execution_success_applies(
    postgresql_session_factory: async_sessionmaker[AsyncSession],
    clean_business_tables: None,
) -> None:
    executed_at = _TOOL_STARTED_AT + timedelta(minutes=5)
    resume_attempt_id: UUID

    async with postgresql_session_factory() as session:
        claim = await _create_running_claim(session)
        resume_attempt_id = claim.attempt.id
        tool_call = _pending_sensitive_tool_call(claim)
        await _persist(session, _command(claim, tool_call=tool_call))

        completed = tool_call.complete_granted_execution_success(
            executed_by_agent_run_attempt_id=resume_attempt_id,
            execution_started_at=executed_at,
            finished_at=executed_at,
            safe_output={
                "escalation_id": str(_SECOND_TOOL_CALL_ID),
                "ticket_id": str(_TICKET_ID),
                "target_queue": "engineering_support",
                "status": "escalated",
            },
        )
        repository = SqlAlchemyAgentToolCallExecutionRepository(session)
        async with SqlAlchemyTransactionManager(session).transaction():
            await repository.save_granted_execution_success(
                tool_call=completed,
            )
            loaded = await repository.get_by_id_for_update(
                workspace_id=_WORKSPACE_ID,
                agent_tool_call_id=tool_call.id,
            )

    assert loaded is not None
    assert loaded.status is AgentToolCallStatus.SUCCEEDED
    assert loaded.executed_by_agent_run_attempt_id == (resume_attempt_id)
    assert loaded.proposed_by_agent_run_attempt_id == (tool_call.proposed_by_agent_run_attempt_id)
    assert loaded.latency_ms == 0
    assert loaded.error_code is None
    assert dict(loaded.safe_output or {})["status"] == "escalated"
    assert loaded.execution_started_at == executed_at
    assert loaded.finished_at == executed_at


async def test_save_granted_execution_success_rejects_repeat(
    postgresql_session_factory: async_sessionmaker[AsyncSession],
    clean_business_tables: None,
) -> None:
    executed_at = _TOOL_STARTED_AT + timedelta(minutes=5)

    async with postgresql_session_factory() as session:
        claim = await _create_running_claim(session)
        tool_call = _pending_sensitive_tool_call(claim)
        await _persist(session, _command(claim, tool_call=tool_call))
        completed = tool_call.complete_granted_execution_success(
            executed_by_agent_run_attempt_id=claim.attempt.id,
            execution_started_at=executed_at,
            finished_at=executed_at,
            safe_output={
                "escalation_id": str(_SECOND_TOOL_CALL_ID),
                "ticket_id": str(_TICKET_ID),
                "target_queue": "engineering_support",
                "status": "escalated",
            },
        )
        repository = SqlAlchemyAgentToolCallExecutionRepository(session)
        async with SqlAlchemyTransactionManager(session).transaction():
            await repository.save_granted_execution_success(
                tool_call=completed,
            )

        with pytest.raises(RuntimeError, match="pending_approval"):
            async with SqlAlchemyTransactionManager(session).transaction():
                await repository.save_granted_execution_success(
                    tool_call=completed,
                )


async def test_save_granted_execution_success_is_workspace_scoped(
    postgresql_session_factory: async_sessionmaker[AsyncSession],
    clean_business_tables: None,
) -> None:
    executed_at = _TOOL_STARTED_AT + timedelta(minutes=5)

    async with postgresql_session_factory() as session:
        claim = await _create_running_claim(session)
        tool_call = _pending_sensitive_tool_call(claim)
        await _persist(session, _command(claim, tool_call=tool_call))
        completed = tool_call.complete_granted_execution_success(
            executed_by_agent_run_attempt_id=claim.attempt.id,
            execution_started_at=executed_at,
            finished_at=executed_at,
            safe_output={
                "escalation_id": str(_SECOND_TOOL_CALL_ID),
                "ticket_id": str(_TICKET_ID),
                "target_queue": "engineering_support",
                "status": "escalated",
            },
        )
        wrong_workspace = replace(
            completed,
            workspace_id=UUID("10000000-0000-4000-8000-000000000099"),
        )
        repository = SqlAlchemyAgentToolCallExecutionRepository(session)
        with pytest.raises(RuntimeError, match="does not exist"):
            async with SqlAlchemyTransactionManager(session).transaction():
                await repository.save_granted_execution_success(
                    tool_call=wrong_workspace,
                )


async def test_save_granted_execution_success_rejects_immutable_mismatch(
    postgresql_session_factory: async_sessionmaker[AsyncSession],
    clean_business_tables: None,
) -> None:
    executed_at = _TOOL_STARTED_AT + timedelta(minutes=5)

    async with postgresql_session_factory() as session:
        claim = await _create_running_claim(session)
        tool_call = _pending_sensitive_tool_call(claim)
        await _persist(session, _command(claim, tool_call=tool_call))
        completed = tool_call.complete_granted_execution_success(
            executed_by_agent_run_attempt_id=claim.attempt.id,
            execution_started_at=executed_at,
            finished_at=executed_at,
            safe_output={
                "escalation_id": str(_SECOND_TOOL_CALL_ID),
                "ticket_id": str(_TICKET_ID),
                "target_queue": "engineering_support",
                "status": "escalated",
            },
        )
        mismatched = replace(
            completed,
            input_fingerprint="c" * 64,
        )
        repository = SqlAlchemyAgentToolCallExecutionRepository(session)
        with pytest.raises(RuntimeError, match="proposal identity"):
            async with SqlAlchemyTransactionManager(session).transaction():
                await repository.save_granted_execution_success(
                    tool_call=mismatched,
                )


async def test_save_granted_execution_success_caller_owns_commit(
    postgresql_session_factory: async_sessionmaker[AsyncSession],
    clean_business_tables: None,
) -> None:
    executed_at = _TOOL_STARTED_AT + timedelta(minutes=5)

    async with postgresql_session_factory() as session:
        claim = await _create_running_claim(session)
        tool_call = _pending_sensitive_tool_call(claim)
        await _persist(session, _command(claim, tool_call=tool_call))
        completed = tool_call.complete_granted_execution_success(
            executed_by_agent_run_attempt_id=claim.attempt.id,
            execution_started_at=executed_at,
            finished_at=executed_at,
            safe_output={
                "escalation_id": str(_SECOND_TOOL_CALL_ID),
                "ticket_id": str(_TICKET_ID),
                "target_queue": "engineering_support",
                "status": "escalated",
            },
        )
        repository = SqlAlchemyAgentToolCallExecutionRepository(session)
        with pytest.raises(RuntimeError, match="force rollback"):
            async with SqlAlchemyTransactionManager(session).transaction():
                await repository.save_granted_execution_success(
                    tool_call=completed,
                )
                raise RuntimeError("force rollback")

        async with SqlAlchemyTransactionManager(session).transaction():
            loaded = await repository.get_by_id_for_update(
                workspace_id=_WORKSPACE_ID,
                agent_tool_call_id=tool_call.id,
            )

    assert loaded is not None
    assert loaded.status is AgentToolCallStatus.PENDING_APPROVAL
    assert loaded.executed_by_agent_run_attempt_id is None
