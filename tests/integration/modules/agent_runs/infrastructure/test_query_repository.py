"""PostgreSQL integration tests for AgentRun inspection queries."""

from datetime import UTC, datetime, timedelta
from uuid import UUID

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from supportops.modules.agent_runs.domain.models import (
    AgentRun,
    AgentRunAttempt,
)
from supportops.modules.agent_runs.infrastructure.models import (
    AgentRunAttemptRecord,
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

_NOW = datetime(
    2026,
    7,
    31,
    20,
    0,
    tzinfo=UTC,
)
_WORKSPACE_ID = UUID(
    "032c8c87-57cc-4d14-bfbd-04968b4e8cd4",
)
_OTHER_WORKSPACE_ID = UUID(
    "db94eb06-c97d-47f8-9214-79558ba933c9",
)
_TICKET_ID = UUID(
    "38bb60fe-d2ea-4615-b499-91aa45069019",
)
_OTHER_TICKET_ID = UUID(
    "fb1af6fd-31fa-49fc-9ab0-095f06d57a24",
)
_AGENT_RUN_ID = UUID(
    "69184ef1-4d71-452e-8070-0b784c29368e",
)
_INGESTION_REQUEST_ID = UUID(
    "725eec8a-c504-4071-ac96-c78cc907f26c",
)
_CORRELATION_ID = UUID(
    "1038c98e-62fd-45df-9839-138f7105cb78",
)
_ATTEMPT_ONE_ID = UUID(
    "2b39f5b7-b2a4-48d0-b079-fdad286d5315",
)
_ATTEMPT_TWO_ID = UUID(
    "626e0940-cf3b-4b9f-ad49-98bce214469b",
)
_LEASE_TOKEN_ONE = UUID(
    "dd0ae456-3467-41db-93d1-a908f40e8365",
)
_LEASE_TOKEN_TWO = UUID(
    "b36000c4-62d7-4fe1-ad40-96872a245409",
)
_EXECUTION_REQUEST_ONE = UUID(
    "d1fa068f-2278-47a8-b3c9-39ccf91f0a5e",
)
_EXECUTION_REQUEST_TWO = UUID(
    "99988e91-f292-4ada-81b6-58551c96f02b",
)


async def persist_workspace_and_ticket(
    session: AsyncSession,
    *,
    ticket_id: UUID = _TICKET_ID,
    ingestion_request_id: UUID = _INGESTION_REQUEST_ID,
    correlation_id: UUID = _CORRELATION_ID,
) -> None:
    """Persist workspace and ticket required by AgentRun FK."""

    workspace = Workspace(
        id=_WORKSPACE_ID,
        name="Platform Support",
        slug="platform-support",
        created_at=_NOW,
        updated_at=_NOW,
    )
    ticket = Ticket.create(
        ticket_id=ticket_id,
        workspace_id=_WORKSPACE_ID,
        subject="Unable to access billing",
        description="The dashboard returns an access error.",
        external_reference=None,
        ingestion_request_id=ingestion_request_id,
        correlation_id=correlation_id,
        now=_NOW,
    )

    await SqlAlchemyWorkspaceRepository(session).add(workspace)
    await SqlAlchemyTicketRepository(session).add(ticket)


async def persist_ticket(
    session: AsyncSession,
    *,
    ticket_id: UUID,
    ingestion_request_id: UUID,
    correlation_id: UUID,
) -> None:
    """Persist an additional ticket in the shared workspace."""

    ticket = Ticket.create(
        ticket_id=ticket_id,
        workspace_id=_WORKSPACE_ID,
        subject=f"Ticket {ticket_id}",
        description="Deterministic query integration test.",
        external_reference=None,
        ingestion_request_id=ingestion_request_id,
        correlation_id=correlation_id,
        now=_NOW,
    )

    await SqlAlchemyTicketRepository(session).add(ticket)


def create_agent_run() -> AgentRun:
    """Create one deterministic queued AgentRun."""

    return AgentRun.create_initial(
        agent_run_id=_AGENT_RUN_ID,
        workspace_id=_WORKSPACE_ID,
        ticket_id=_TICKET_ID,
        ingestion_request_id=_INGESTION_REQUEST_ID,
        correlation_id=_CORRELATION_ID,
        max_attempts=3,
        now=_NOW,
    )


def create_attempt(
    *,
    attempt_id: UUID,
    attempt_number: int,
    lease_token: UUID,
    execution_request_id: UUID,
    started_at: datetime,
) -> AgentRunAttempt:
    """Create one deterministic AgentRun attempt."""

    return AgentRunAttempt.start(
        attempt_id=attempt_id,
        agent_run_id=_AGENT_RUN_ID,
        attempt_number=attempt_number,
        worker_id="worker-a",
        lease_token=lease_token,
        execution_request_id=execution_request_id,
        now=started_at,
    )


async def test_get_returns_workspace_scoped_agent_run(
    postgresql_session: AsyncSession,
) -> None:
    repository = SqlAlchemyAgentRunRepository(
        postgresql_session,
    )
    agent_run = create_agent_run()

    await persist_workspace_and_ticket(postgresql_session)
    await repository.add(agent_run)
    await postgresql_session.flush()

    result = await repository.get(
        workspace_id=_WORKSPACE_ID,
        agent_run_id=_AGENT_RUN_ID,
    )

    assert result == agent_run


async def test_get_returns_none_for_missing_agent_run(
    postgresql_session: AsyncSession,
) -> None:
    repository = SqlAlchemyAgentRunRepository(
        postgresql_session,
    )

    result = await repository.get(
        workspace_id=_WORKSPACE_ID,
        agent_run_id=_AGENT_RUN_ID,
    )

    assert result is None


async def test_get_hides_cross_workspace_agent_run(
    postgresql_session: AsyncSession,
) -> None:
    repository = SqlAlchemyAgentRunRepository(
        postgresql_session,
    )
    agent_run = create_agent_run()

    await persist_workspace_and_ticket(postgresql_session)
    await repository.add(agent_run)
    await postgresql_session.flush()

    result = await repository.get(
        workspace_id=_OTHER_WORKSPACE_ID,
        agent_run_id=_AGENT_RUN_ID,
    )

    assert result is None


async def test_list_attempts_returns_empty_tuple(
    postgresql_session: AsyncSession,
) -> None:
    repository = SqlAlchemyAgentRunRepository(
        postgresql_session,
    )
    agent_run = create_agent_run()

    await persist_workspace_and_ticket(postgresql_session)
    await repository.add(agent_run)
    await postgresql_session.flush()

    result = await repository.list_attempts(
        agent_run_id=_AGENT_RUN_ID,
    )

    assert result == ()


async def test_list_attempts_orders_by_attempt_number(
    postgresql_session: AsyncSession,
) -> None:
    repository = SqlAlchemyAgentRunRepository(
        postgresql_session,
    )
    agent_run = create_agent_run()

    await persist_workspace_and_ticket(postgresql_session)
    await repository.add(agent_run)
    await postgresql_session.flush()

    first_attempt = create_attempt(
        attempt_id=_ATTEMPT_ONE_ID,
        attempt_number=1,
        lease_token=_LEASE_TOKEN_ONE,
        execution_request_id=_EXECUTION_REQUEST_ONE,
        started_at=_NOW,
    )
    second_attempt = create_attempt(
        attempt_id=_ATTEMPT_TWO_ID,
        attempt_number=2,
        lease_token=_LEASE_TOKEN_TWO,
        execution_request_id=_EXECUTION_REQUEST_TWO,
        started_at=_NOW + timedelta(seconds=10),
    )

    postgresql_session.add_all(
        [
            AgentRunAttemptRecord.from_domain(
                second_attempt,
            ),
            AgentRunAttemptRecord.from_domain(
                first_attempt,
            ),
        ],
    )
    await postgresql_session.flush()

    result = await repository.list_attempts(
        agent_run_id=_AGENT_RUN_ID,
    )

    assert result == (
        first_attempt,
        second_attempt,
    )


async def test_list_attempts_only_returns_requested_run_history(
    postgresql_session: AsyncSession,
) -> None:
    repository = SqlAlchemyAgentRunRepository(
        postgresql_session,
    )
    agent_run = create_agent_run()
    other_ingestion_request_id = UUID(
        "447d8372-aedc-420d-b381-42ab7c995eb4",
    )
    other_correlation_id = UUID(
        "54fc11bc-bda3-4d78-b128-f59968def214",
    )

    await persist_workspace_and_ticket(postgresql_session)
    await persist_ticket(
        postgresql_session,
        ticket_id=_OTHER_TICKET_ID,
        ingestion_request_id=other_ingestion_request_id,
        correlation_id=other_correlation_id,
    )
    await repository.add(agent_run)
    await postgresql_session.flush()

    requested_attempt = create_attempt(
        attempt_id=_ATTEMPT_ONE_ID,
        attempt_number=1,
        lease_token=_LEASE_TOKEN_ONE,
        execution_request_id=_EXECUTION_REQUEST_ONE,
        started_at=_NOW,
    )

    other_run_id = UUID(
        "541b146a-a1e4-4117-8592-ac8ea0a5a4cc",
    )
    other_run = AgentRun.create_initial(
        agent_run_id=other_run_id,
        workspace_id=_WORKSPACE_ID,
        ticket_id=_OTHER_TICKET_ID,
        ingestion_request_id=other_ingestion_request_id,
        correlation_id=other_correlation_id,
        max_attempts=3,
        now=_NOW,
    )
    await repository.add(other_run)
    await postgresql_session.flush()

    other_attempt = AgentRunAttempt.start(
        attempt_id=_ATTEMPT_TWO_ID,
        agent_run_id=other_run_id,
        attempt_number=1,
        worker_id="worker-b",
        lease_token=_LEASE_TOKEN_TWO,
        execution_request_id=_EXECUTION_REQUEST_TWO,
        now=_NOW,
    )

    postgresql_session.add_all(
        [
            AgentRunAttemptRecord.from_domain(
                other_attempt,
            ),
            AgentRunAttemptRecord.from_domain(
                requested_attempt,
            ),
        ],
    )
    await postgresql_session.flush()

    result = await repository.list_attempts(
        agent_run_id=_AGENT_RUN_ID,
    )

    assert result == (requested_attempt,)
