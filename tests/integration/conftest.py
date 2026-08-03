"""Shared fixtures for infrastructure integration tests."""

from __future__ import annotations

import asyncio
import subprocess
import sys
from collections.abc import AsyncIterator, Callable, Iterator, Mapping
from datetime import UTC, datetime, timedelta
from uuid import UUID

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from sqlalchemy import text
from sqlalchemy.exc import ProgrammingError
from sqlalchemy.ext.asyncio import (
    AsyncConnection,
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
)

from supportops.agent_tools.application.persistence import (
    AgentToolCallPersistenceResult,
    PersistAgentToolCallCommand,
)
from supportops.agent_tools.application.sensitive_execution import (
    ExecuteApprovedTicketEscalation,
)
from supportops.agent_tools.domain.audit import AgentToolCall
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
from supportops.api.application import create_application
from supportops.core.settings import Settings
from supportops.infrastructure.postgresql import (
    create_postgresql_engine,
    create_postgresql_session_factory,
    dispose_postgresql_engine,
)
from supportops.infrastructure.postgresql.transaction import (
    SqlAlchemyTransactionManager,
)
from supportops.modules.agent_runs.application.execution import (
    AgentRunExecutionContext,
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

if sys.platform == "win32":

    def pytest_asyncio_loop_factories(
        config: object,
        item: object,
    ) -> Mapping[str, Callable[[], asyncio.AbstractEventLoop]]:
        """Force SelectorEventLoop so Psycopg can run under Windows asyncio."""

        del config, item
        return {"selector": asyncio.SelectorEventLoop}


# Session-level advisory lock shared by every integration test that mutates the
# local PostgreSQL database. Concurrent pytest processes otherwise race on
# cleanup and flake with foreign-key / unique violations.
_INTEGRATION_DATABASE_LOCK_KEY = 742_891_305

# Citations RESTRICT-reference knowledge_document_chunks, so clear them first.
BUSINESS_DATA_DELETE_STATEMENTS: tuple[str, ...] = (
    "DELETE FROM support_recommendation_citations",
    "DELETE FROM support_recommendations",
    "DELETE FROM ticket_escalations",
    "DELETE FROM sensitive_execution_grants",
    "DELETE FROM approval_requests",
    "DELETE FROM agent_tool_calls",
    "DELETE FROM knowledge_document_chunks",
    "UPDATE knowledge_documents SET active_version_id = NULL",
    "DELETE FROM knowledge_document_versions",
    "DELETE FROM knowledge_documents",
    "DELETE FROM ticket_classifications",
    "DELETE FROM llm_invocations",
    "DELETE FROM agent_run_attempts",
    "DELETE FROM agent_runs",
    "DELETE FROM tickets",
    "DELETE FROM workspaces",
)


def run_alembic_upgrade_head() -> None:
    """Apply Alembic migrations through head or raise on failure."""

    result = subprocess.run(
        [sys.executable, "-m", "alembic", "upgrade", "head"],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Failed to upgrade integration database to Alembic head:\n{result.stderr}"
        )


async def clear_integration_business_data(
    connection: AsyncConnection,
) -> None:
    """Delete all shared business rows in FK-safe order."""

    for statement in BUSINESS_DATA_DELETE_STATEMENTS:
        await connection.execute(text(statement))


@pytest.fixture
def integration_settings() -> Settings:
    """Load validated settings from the integration environment."""

    return Settings()


@pytest.fixture
def integration_application(
    integration_settings: Settings,
) -> FastAPI:
    """Create an application configured for live infrastructure."""

    return create_application(integration_settings)


@pytest.fixture
async def integration_client(
    integration_application: FastAPI,
) -> AsyncIterator[AsyncClient]:
    """Create an HTTP client with the real application lifecycle enabled."""

    async with (
        integration_application.router.lifespan_context(integration_application),
        AsyncClient(
            transport=ASGITransport(app=integration_application),
            base_url="http://test",
        ) as client,
    ):
        yield client


@pytest.fixture(scope="session")
def migrated_database() -> Iterator[None]:
    """Apply Alembic migrations to the shared local integration database."""

    try:
        run_alembic_upgrade_head()
    except RuntimeError as error:
        pytest.fail(str(error))

    yield None


@pytest.fixture
async def postgresql_engine(
    migrated_database: None,
    integration_settings: Settings,
) -> AsyncIterator[AsyncEngine]:
    """Create a disposable async engine for one integration test."""

    engine = create_postgresql_engine(integration_settings)

    try:
        yield engine
    finally:
        await dispose_postgresql_engine(engine)


@pytest.fixture
def postgresql_session_factory(
    postgresql_engine: AsyncEngine,
) -> async_sessionmaker[AsyncSession]:
    """Create a session factory bound to the test engine."""

    return create_postgresql_session_factory(postgresql_engine)


@pytest.fixture
async def postgresql_session(
    postgresql_session_factory: async_sessionmaker[AsyncSession],
) -> AsyncIterator[AsyncSession]:
    """Open one async session and roll back any leftover transaction."""

    async with postgresql_session_factory() as session:
        try:
            yield session
        finally:
            if session.in_transaction():
                await session.rollback()


@pytest.fixture
async def exclusive_integration_database(
    integration_settings: Settings,
) -> AsyncIterator[None]:
    """Hold the shared integration DB lock for schema-mutating tests."""

    engine = create_postgresql_engine(integration_settings)
    connection = await engine.connect()
    await connection.execute(
        text(f"SELECT pg_advisory_lock({_INTEGRATION_DATABASE_LOCK_KEY})"),
    )
    # Commit so CREATE INDEX CONCURRENTLY (checkpoint setup) is not blocked.
    await connection.commit()

    try:
        yield None
    finally:
        # Schema-mutating tests may leave the shared DB below head on failure
        # or interrupt; restore before releasing the lock so later fixtures
        # (e.g. clean_business_tables) do not hit missing relations.
        run_alembic_upgrade_head()
        await connection.execute(
            text(f"SELECT pg_advisory_unlock({_INTEGRATION_DATABASE_LOCK_KEY})"),
        )
        await connection.commit()
        await connection.close()
        await dispose_postgresql_engine(engine)


@pytest.fixture
async def clean_business_tables(
    postgresql_engine: AsyncEngine,
) -> AsyncIterator[None]:
    """Serialize and reset business rows around one integration test."""

    lock_connection = await postgresql_engine.connect()
    await lock_connection.execute(
        text(f"SELECT pg_advisory_lock({_INTEGRATION_DATABASE_LOCK_KEY})"),
    )

    async def cleanup() -> None:
        try:
            await clear_integration_business_data(lock_connection)
            await lock_connection.commit()
        except ProgrammingError:
            # A prior schema-mutating test/interrupt can leave head-shaped code
            # pointed at a DB missing newer relations such as approval_requests.
            await lock_connection.rollback()
            run_alembic_upgrade_head()
            await clear_integration_business_data(lock_connection)
            await lock_connection.commit()

    try:
        await cleanup()
        yield None
        await cleanup()
    finally:
        if lock_connection.in_transaction():
            await lock_connection.rollback()
        await lock_connection.execute(
            text(f"SELECT pg_advisory_unlock({_INTEGRATION_DATABASE_LOCK_KEY})"),
        )
        await lock_connection.commit()
        await lock_connection.close()


# --- Approved ticket-escalation execution fixtures ---

_APPROVED_ESCALATION_WORKSPACE_ID = UUID(
    "11000000-0000-4000-8000-000000000001",
)
_APPROVED_ESCALATION_TICKET_ID = UUID(
    "21000000-0000-4000-8000-000000000002",
)
_APPROVED_ESCALATION_AGENT_RUN_ID = UUID(
    "31000000-0000-4000-8000-000000000003",
)
_APPROVED_ESCALATION_LEASE_TOKEN = UUID(
    "41000000-0000-4000-8000-000000000004",
)
_APPROVED_ESCALATION_EXECUTION_REQUEST_ID = UUID(
    "51000000-0000-4000-8000-000000000005",
)
_APPROVED_ESCALATION_TOOL_CALL_ID = UUID(
    "61000000-0000-4000-8000-000000000006",
)
_APPROVED_ESCALATION_INVOCATION_ID = UUID(
    "81000000-0000-4000-8000-000000000008",
)
_APPROVED_ESCALATION_APPROVAL_REQUEST_ID = UUID(
    "91000000-0000-4000-8000-000000000009",
)
_APPROVED_ESCALATION_DECISION_REQUEST_ID = UUID(
    "b1000000-0000-4000-8000-00000000000b",
)
_APPROVED_ESCALATION_DECISION_CORRELATION_ID = UUID(
    "c1000000-0000-4000-8000-00000000000c",
)

_APPROVED_ESCALATION_CREATED_AT = datetime(
    2026,
    8,
    3,
    19,
    0,
    tzinfo=UTC,
)
_APPROVED_ESCALATION_CLAIMED_AT = _APPROVED_ESCALATION_CREATED_AT + timedelta(
    minutes=1,
)
_APPROVED_ESCALATION_LEASE_EXPIRES_AT = _APPROVED_ESCALATION_CLAIMED_AT + timedelta(
    seconds=45,
)
_APPROVED_ESCALATION_TOOL_PROPOSED_AT = _APPROVED_ESCALATION_CLAIMED_AT + timedelta(
    seconds=1,
)
_APPROVED_ESCALATION_INVOCATION_AT = _APPROVED_ESCALATION_CLAIMED_AT + timedelta(
    seconds=2,
)
_APPROVED_ESCALATION_APPROVAL_AT = _APPROVED_ESCALATION_CLAIMED_AT + timedelta(
    seconds=3,
)
_APPROVED_ESCALATION_EXPIRES_AT = _APPROVED_ESCALATION_APPROVAL_AT + timedelta(
    hours=24,
)
_APPROVED_ESCALATION_DECIDED_AT = _APPROVED_ESCALATION_APPROVAL_AT + timedelta(
    minutes=5,
)


async def _seed_approved_ticket_escalation(
    session: AsyncSession,
) -> tuple[AgentRunClaim, Ticket, AgentToolCall, ApprovalRequest]:
    workspace = Workspace(
        id=_APPROVED_ESCALATION_WORKSPACE_ID,
        name="Approved Escalation Workspace",
        slug="approved-escalation-workspace",
        created_at=_APPROVED_ESCALATION_CREATED_AT,
        updated_at=_APPROVED_ESCALATION_CREATED_AT,
    )
    ticket = Ticket.create(
        ticket_id=_APPROVED_ESCALATION_TICKET_ID,
        workspace_id=_APPROVED_ESCALATION_WORKSPACE_ID,
        subject="Needs approved escalation",
        description=("The customer requested a policy-sensitive escalation."),
        external_reference=None,
        ingestion_request_id=UUID("81100000-0000-4000-8000-000000000008"),
        correlation_id=UUID("82100000-0000-4000-8000-000000000009"),
        now=_APPROVED_ESCALATION_CREATED_AT,
    )
    agent_run = AgentRun.create_initial(
        agent_run_id=_APPROVED_ESCALATION_AGENT_RUN_ID,
        workspace_id=_APPROVED_ESCALATION_WORKSPACE_ID,
        ticket_id=_APPROVED_ESCALATION_TICKET_ID,
        ingestion_request_id=ticket.ingestion_request_id,
        correlation_id=ticket.correlation_id,
        workflow_version=DETERMINISTIC_BASELINE_WORKFLOW_VERSION,
        max_retryable_failures=3,
        now=_APPROVED_ESCALATION_CREATED_AT,
    )
    transaction_manager = SqlAlchemyTransactionManager(session)

    async with transaction_manager.transaction():
        await SqlAlchemyWorkspaceRepository(session).add(workspace)
        await SqlAlchemyTicketRepository(session).add(ticket)
        await SqlAlchemyAgentRunRepository(session).add(agent_run)

    async with transaction_manager.transaction():
        claim = await SqlAlchemyAgentRunRepository(session).claim_next_available(
            ClaimAgentRunCommand(
                worker_id="approved-escalation-worker-1",
                lease_token=_APPROVED_ESCALATION_LEASE_TOKEN,
                execution_request_id=(_APPROVED_ESCALATION_EXECUTION_REQUEST_ID),
                claimed_at=_APPROVED_ESCALATION_CLAIMED_AT,
                lease_expires_at=(_APPROVED_ESCALATION_LEASE_EXPIRES_AT),
            )
        )

    assert claim is not None

    tool_call = AgentToolCall.propose_for_approval(
        tool_call_id=_APPROVED_ESCALATION_TOOL_CALL_ID,
        workspace_id=_APPROVED_ESCALATION_WORKSPACE_ID,
        ticket_id=_APPROVED_ESCALATION_TICKET_ID,
        agent_run_id=claim.agent_run.id,
        proposed_by_agent_run_attempt_id=claim.attempt.id,
        sequence=1,
        provider_tool_call_id="approved-escalation-call-1",
        tool_name="escalate_ticket",
        tool_version=1,
        input_fingerprint="b" * 64,
        safe_input={
            "target_queue": "engineering_support",
            "reason": "A product defect requires review.",
        },
        proposed_at=_APPROVED_ESCALATION_TOOL_PROPOSED_AT,
    )
    invocation = LLMInvocation.create(
        invocation_id=_APPROVED_ESCALATION_INVOCATION_ID,
        workspace_id=_APPROVED_ESCALATION_WORKSPACE_ID,
        ticket_id=_APPROVED_ESCALATION_TICKET_ID,
        agent_run_id=claim.agent_run.id,
        agent_run_attempt_id=claim.attempt.id,
        invocation_sequence=1,
        status=LLMInvocationStatus.TIMED_OUT,
        provider="mock",
        model="mock-support-v1",
        provider_request_id="mock-request-1",
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
        now=_APPROVED_ESCALATION_INVOCATION_AT,
    )
    pending = ApprovalRequest.create_pending(
        tool_call=tool_call,
        requested_by_llm_invocation_id=(_APPROVED_ESCALATION_INVOCATION_ID),
        request_reason="Requires human review before escalation.",
        expires_at=_APPROVED_ESCALATION_EXPIRES_AT,
        approval_request_id=(_APPROVED_ESCALATION_APPROVAL_REQUEST_ID),
        now=_APPROVED_ESCALATION_APPROVAL_AT,
    )
    approved = pending.approve(
        actor_reference="operator:alice",
        comment=None,
        request_id=_APPROVED_ESCALATION_DECISION_REQUEST_ID,
        correlation_id=(_APPROVED_ESCALATION_DECISION_CORRELATION_ID),
        decided_at=_APPROVED_ESCALATION_DECIDED_AT,
    )

    async with transaction_manager.transaction():
        session.add(LLMInvocationRecord.from_domain(invocation))
        await session.flush()
        tool_result = await SqlAlchemyAgentToolCallExecutionRepository(
            session,
        ).persist_fenced(
            PersistAgentToolCallCommand(
                workspace_id=_APPROVED_ESCALATION_WORKSPACE_ID,
                ticket_id=_APPROVED_ESCALATION_TICKET_ID,
                agent_run_id=claim.agent_run.id,
                agent_run_attempt_id=claim.attempt.id,
                lease_token=_APPROVED_ESCALATION_LEASE_TOKEN,
                persisted_at=_APPROVED_ESCALATION_APPROVAL_AT,
                tool_call=tool_call,
            )
        )
        assert tool_result is AgentToolCallPersistenceResult.APPLIED
        approval_result = await SqlAlchemyApprovalRequestRepository(
            session,
        ).persist_pending(pending)
        assert approval_result is ApprovalRequestPersistenceResult.APPLIED
        await SqlAlchemyApprovalRequestRepository(session).save(approved)

    return claim, ticket, tool_call, approved


@pytest.fixture
async def approved_ticket_escalation_seed(
    postgresql_session_factory: async_sessionmaker[AsyncSession],
    clean_business_tables: None,
) -> AsyncIterator[
    tuple[
        async_sessionmaker[AsyncSession],
        AgentRunClaim,
        Ticket,
        AgentToolCall,
        ApprovalRequest,
    ]
]:
    """Seed one approved escalate_ticket proposal for execution tests."""

    async with postgresql_session_factory() as session:
        claim, ticket, tool_call, approval = await _seed_approved_ticket_escalation(
            session,
        )

    yield (
        postgresql_session_factory,
        claim,
        ticket,
        tool_call,
        approval,
    )


@pytest.fixture
def approved_ticket_escalation_tool_call(
    approved_ticket_escalation_seed: tuple[
        async_sessionmaker[AsyncSession],
        AgentRunClaim,
        Ticket,
        AgentToolCall,
        ApprovalRequest,
    ],
) -> AgentToolCall:
    """Return the pending approved-path AgentToolCall."""

    return approved_ticket_escalation_seed[3]


@pytest.fixture
def approved_ticket_escalation_approval(
    approved_ticket_escalation_seed: tuple[
        async_sessionmaker[AsyncSession],
        AgentRunClaim,
        Ticket,
        AgentToolCall,
        ApprovalRequest,
    ],
) -> ApprovalRequest:
    """Return the durable approved ApprovalRequest."""

    return approved_ticket_escalation_seed[4]


@pytest.fixture
def approved_ticket_escalation_context(
    approved_ticket_escalation_seed: tuple[
        async_sessionmaker[AsyncSession],
        AgentRunClaim,
        Ticket,
        AgentToolCall,
        ApprovalRequest,
    ],
) -> AgentRunExecutionContext:
    """Return resume AgentRun execution context for granted escalation."""

    _factory, claim, ticket, _tool_call, _approval = approved_ticket_escalation_seed
    return AgentRunExecutionContext(
        agent_run=claim.agent_run,
        attempt=claim.attempt,
        ticket=ticket,
    )


@pytest.fixture
def approved_ticket_escalation_executor(
    approved_ticket_escalation_seed: tuple[
        async_sessionmaker[AsyncSession],
        AgentRunClaim,
        Ticket,
        AgentToolCall,
        ApprovalRequest,
    ],
) -> ExecuteApprovedTicketEscalation:
    """Compose session-scoped granted escalation executor."""

    session_factory, *_rest = approved_ticket_escalation_seed

    class _SessionScopedExecutor:
        def __init__(self) -> None:
            self._session_factory = session_factory

        async def execute(
            self,
            *,
            context: AgentRunExecutionContext,
            approval_request_id: UUID,
            agent_tool_call_id: UUID,
        ) -> object:
            async with self._session_factory() as session:
                executor = ExecuteApprovedTicketEscalation(
                    transaction_manager=SqlAlchemyTransactionManager(
                        session,
                    ),
                    approval_request_repository=(SqlAlchemyApprovalRequestRepository(session)),
                    tool_call_repository=(SqlAlchemyAgentToolCallExecutionRepository(session)),
                    grant_repository=(SqlAlchemySensitiveExecutionGrantRepository(session)),
                    escalation_repository=(SqlAlchemyTicketEscalationRepository(session)),
                )
                return await executor.execute(
                    context=context,
                    approval_request_id=approval_request_id,
                    agent_tool_call_id=agent_tool_call_id,
                )

    return _SessionScopedExecutor()  # type: ignore[return-value]
