"""Integration tests for tool-call query repository lookups."""

from datetime import UTC, datetime, timedelta
from uuid import UUID

import pytest
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
)

from supportops.agent_tools.application.persistence import (
    PersistAgentToolCallCommand,
)
from supportops.agent_tools.application.queries import (
    SensitiveAgentToolCallLookup,
)
from supportops.agent_tools.domain.audit import AgentToolCall
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

_CREATED_AT = datetime(2026, 8, 2, 18, 0, tzinfo=UTC)
_CLAIMED_AT = _CREATED_AT + timedelta(minutes=1)
_LEASE_EXPIRES_AT = _CLAIMED_AT + timedelta(seconds=45)
_PROPOSED_AT = _CLAIMED_AT + timedelta(seconds=1)
_PERSISTED_AT = _PROPOSED_AT + timedelta(milliseconds=5)
_FINGERPRINT = "b" * 64


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
        description="The customer cannot complete access reset.",
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
                worker_id="controlled-worker-1",
                lease_token=_LEASE_TOKEN,
                execution_request_id=_EXECUTION_REQUEST_ID,
                claimed_at=_CLAIMED_AT,
                lease_expires_at=_LEASE_EXPIRES_AT,
            )
        )

    assert claim is not None
    return claim


async def test_loads_sensitive_proposal_by_identity(
    postgresql_session_factory: async_sessionmaker[AsyncSession],
    clean_business_tables: None,
) -> None:
    async with postgresql_session_factory() as session:
        claim = await _create_running_claim(session)
        tool_call = AgentToolCall.propose_for_approval(
            tool_call_id=_TOOL_CALL_ID,
            workspace_id=_WORKSPACE_ID,
            ticket_id=_TICKET_ID,
            agent_run_id=claim.agent_run.id,
            proposed_by_agent_run_attempt_id=claim.attempt.id,
            sequence=1,
            provider_tool_call_id="provider-sensitive-1",
            tool_name="escalate_ticket",
            tool_version=1,
            input_fingerprint=_FINGERPRINT,
            safe_input={"reason_code": "policy_required"},
            proposed_at=_PROPOSED_AT,
        )

        async with SqlAlchemyTransactionManager(session).transaction():
            await SqlAlchemyAgentToolCallExecutionRepository(session).persist_fenced(
                PersistAgentToolCallCommand(
                    workspace_id=_WORKSPACE_ID,
                    ticket_id=_TICKET_ID,
                    agent_run_id=claim.agent_run.id,
                    agent_run_attempt_id=claim.attempt.id,
                    lease_token=_LEASE_TOKEN,
                    persisted_at=_PERSISTED_AT,
                    tool_call=tool_call,
                )
            )

        async with SqlAlchemyTransactionManager(session).transaction():
            loaded = await SqlAlchemyAgentToolCallQueryRepository(
                session
            ).get_sensitive_by_identity(
                SensitiveAgentToolCallLookup(
                    workspace_id=_WORKSPACE_ID,
                    ticket_id=_TICKET_ID,
                    agent_run_id=claim.agent_run.id,
                    tool_name="escalate_ticket",
                    tool_version=1,
                    input_fingerprint=_FINGERPRINT,
                )
            )

    assert loaded == tool_call


async def test_sensitive_lookup_returns_none_for_cross_workspace(
    postgresql_session_factory: async_sessionmaker[AsyncSession],
    clean_business_tables: None,
) -> None:
    async with postgresql_session_factory() as session:
        claim = await _create_running_claim(session)
        tool_call = AgentToolCall.propose_for_approval(
            tool_call_id=_TOOL_CALL_ID,
            workspace_id=_WORKSPACE_ID,
            ticket_id=_TICKET_ID,
            agent_run_id=claim.agent_run.id,
            proposed_by_agent_run_attempt_id=claim.attempt.id,
            sequence=1,
            provider_tool_call_id="provider-sensitive-1",
            tool_name="escalate_ticket",
            tool_version=1,
            input_fingerprint=_FINGERPRINT,
            safe_input={"reason_code": "policy_required"},
            proposed_at=_PROPOSED_AT,
        )

        async with SqlAlchemyTransactionManager(session).transaction():
            await SqlAlchemyAgentToolCallExecutionRepository(session).persist_fenced(
                PersistAgentToolCallCommand(
                    workspace_id=_WORKSPACE_ID,
                    ticket_id=_TICKET_ID,
                    agent_run_id=claim.agent_run.id,
                    agent_run_attempt_id=claim.attempt.id,
                    lease_token=_LEASE_TOKEN,
                    persisted_at=_PERSISTED_AT,
                    tool_call=tool_call,
                )
            )

        async with SqlAlchemyTransactionManager(session).transaction():
            loaded = await SqlAlchemyAgentToolCallQueryRepository(
                session
            ).get_sensitive_by_identity(
                SensitiveAgentToolCallLookup(
                    workspace_id=UUID("a0000000-0000-4000-8000-000000000012"),
                    ticket_id=_TICKET_ID,
                    agent_run_id=claim.agent_run.id,
                    tool_name="escalate_ticket",
                    tool_version=1,
                    input_fingerprint=_FINGERPRINT,
                )
            )

    assert loaded is None
