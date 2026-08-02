"""Integration tests for fenced PostgreSQL tool-call persistence."""

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
        max_attempts=3,
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
    return AgentToolCall.create(
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
            agent_run_attempt_id=claim.attempt.id,
            sequence=1,
        )

        async with SqlAlchemyTransactionManager(session).transaction():
            loaded = await SqlAlchemyAgentToolCallQueryRepository(session).get_by_attempt_sequence(
                query
            )

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
            agent_run_attempt_id=claim.attempt.id,
            sequence=1,
        )

        async with SqlAlchemyTransactionManager(session).transaction():
            loaded = await SqlAlchemyAgentToolCallQueryRepository(session).get_by_attempt_sequence(
                query
            )

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
            agent_run_attempt_id=claim.attempt.id,
            sequence=2,
        )

        async with SqlAlchemyTransactionManager(session).transaction():
            loaded = await SqlAlchemyAgentToolCallQueryRepository(session).get_by_attempt_sequence(
                query
            )

    assert loaded is None
