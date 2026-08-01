"""Integration tests for transactional ticket intake scheduling."""

from datetime import UTC, datetime
from uuid import UUID

import pytest
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from supportops.application.ticket_intake import (
    CreateTicketWithInitialRun,
    TicketIntakeResult,
)
from supportops.infrastructure.postgresql.transaction import (
    SqlAlchemyTransactionManager,
)
from supportops.modules.agent_runs.domain.models import (
    DETERMINISTIC_BASELINE_WORKFLOW_VERSION,
    INITIAL_TICKET_PROCESSING_TRIGGER_KEY,
    INITIAL_TICKET_PROCESSING_WORKFLOW_NAME,
    TICKET_CLASSIFICATION_WORKFLOW_VERSION,
    AgentRun,
    AgentRunStatus,
)
from supportops.modules.agent_runs.infrastructure.models import (
    AgentRunRecord,
)
from supportops.modules.agent_runs.infrastructure.repository import (
    SqlAlchemyAgentRunRepository,
)
from supportops.modules.tickets.application.errors import (
    TicketExternalReferenceConflictApplicationError,
)
from supportops.modules.tickets.infrastructure.models import (
    TicketRecord,
)
from supportops.modules.tickets.infrastructure.repository import (
    SqlAlchemyTicketRepository,
)
from supportops.modules.workspaces.domain.models import Workspace
from supportops.modules.workspaces.infrastructure.repository import (
    SqlAlchemyWorkspaceRepository,
)

pytestmark = pytest.mark.integration

_WORKSPACE_ID = UUID(
    "032c8c87-57cc-4d14-bfbd-04968b4e8cd4",
)
_INGESTION_REQUEST_ID = UUID(
    "725eec8a-c504-4071-ac96-c78cc907f26c",
)
_CORRELATION_ID = UUID(
    "1038c98e-62fd-45df-9839-138f7105cb78",
)
_BASE_TIMESTAMP = datetime(
    2026,
    7,
    31,
    12,
    0,
    tzinfo=UTC,
)


class FailingAgentRunRepository(SqlAlchemyAgentRunRepository):
    """Fail after ticket persistence has already been flushed."""

    async def add(
        self,
        agent_run: AgentRun,
    ) -> None:
        raise RuntimeError("agent run insertion failed")


async def persist_workspace(
    session: AsyncSession,
) -> Workspace:
    """Persist the workspace used by ticket-intake tests."""

    workspace = Workspace(
        id=_WORKSPACE_ID,
        name="Platform Support",
        slug="platform-support",
        created_at=_BASE_TIMESTAMP,
        updated_at=_BASE_TIMESTAMP,
    )
    repository = SqlAlchemyWorkspaceRepository(session)
    transaction_manager = SqlAlchemyTransactionManager(session)

    async with transaction_manager.transaction():
        await repository.add(workspace)

    return workspace


def create_ticket_intake_service(
    session: AsyncSession,
    *,
    agent_run_repository: SqlAlchemyAgentRunRepository | None = None,
    workflow_version: str = TICKET_CLASSIFICATION_WORKFLOW_VERSION,
) -> CreateTicketWithInitialRun:
    """Compose ticket intake with one shared PostgreSQL session."""

    return CreateTicketWithInitialRun(
        workspace_repository=SqlAlchemyWorkspaceRepository(session),
        ticket_repository=SqlAlchemyTicketRepository(session),
        agent_run_repository=(agent_run_repository or SqlAlchemyAgentRunRepository(session)),
        transaction_manager=SqlAlchemyTransactionManager(session),
        workflow_version=workflow_version,
        max_attempts=3,
        utc_now=lambda: _BASE_TIMESTAMP,
    )


async def execute_ticket_intake(
    service: CreateTicketWithInitialRun,
    *,
    external_reference: str | None = "SUP-1042",
) -> TicketIntakeResult:
    """Execute the standard deterministic ticket intake."""

    return await service.execute(
        workspace_id=_WORKSPACE_ID,
        subject="Unable to access billing",
        description="The dashboard returns an access error.",
        external_reference=external_reference,
        ingestion_request_id=_INGESTION_REQUEST_ID,
        correlation_id=_CORRELATION_ID,
    )


async def count_tickets(
    session: AsyncSession,
) -> int:
    """Count persisted tickets."""

    result = await session.execute(
        select(func.count()).select_from(TicketRecord),
    )
    return result.scalar_one()


async def count_agent_runs(
    session: AsyncSession,
) -> int:
    """Count persisted AgentRuns."""

    result = await session.execute(
        select(func.count()).select_from(AgentRunRecord),
    )
    return result.scalar_one()


async def load_agent_run(
    session: AsyncSession,
    *,
    agent_run_id: UUID,
) -> AgentRunRecord:
    """Load one persisted AgentRun record."""

    record = await session.get(
        AgentRunRecord,
        agent_run_id,
    )

    assert record is not None
    return record


async def test_ticket_and_initial_agent_run_commit_together(
    postgresql_session: AsyncSession,
    clean_business_tables: None,
) -> None:
    await persist_workspace(postgresql_session)
    service = create_ticket_intake_service(postgresql_session)

    result = await execute_ticket_intake(service)

    assert await count_tickets(postgresql_session) == 1
    assert await count_agent_runs(postgresql_session) == 1

    persisted_ticket = await postgresql_session.get(
        TicketRecord,
        result.ticket.id,
    )
    persisted_run = await load_agent_run(
        postgresql_session,
        agent_run_id=result.processing_run.id,
    )

    assert persisted_ticket is not None
    assert persisted_run.ticket_id == persisted_ticket.id
    assert persisted_run.workspace_id == persisted_ticket.workspace_id


async def test_initial_agent_run_persists_approved_contract(
    postgresql_session: AsyncSession,
    clean_business_tables: None,
) -> None:
    await persist_workspace(postgresql_session)
    service = create_ticket_intake_service(postgresql_session)

    result = await execute_ticket_intake(service)

    persisted_run = await load_agent_run(
        postgresql_session,
        agent_run_id=result.processing_run.id,
    )

    assert persisted_run.status == AgentRunStatus.QUEUED.value
    assert persisted_run.workflow_name == INITIAL_TICKET_PROCESSING_WORKFLOW_NAME
    assert persisted_run.workflow_version == TICKET_CLASSIFICATION_WORKFLOW_VERSION
    assert persisted_run.trigger_key == INITIAL_TICKET_PROCESSING_TRIGGER_KEY
    assert persisted_run.available_at == _BASE_TIMESTAMP
    assert persisted_run.attempt_count == 0
    assert persisted_run.max_attempts == 3
    assert persisted_run.lease_owner is None
    assert persisted_run.lease_token is None
    assert persisted_run.lease_expires_at is None


async def test_initial_agent_run_persists_explicit_deterministic_baseline(
    postgresql_session: AsyncSession,
    clean_business_tables: None,
) -> None:
    await persist_workspace(postgresql_session)
    service = create_ticket_intake_service(
        postgresql_session,
        workflow_version=DETERMINISTIC_BASELINE_WORKFLOW_VERSION,
    )

    result = await execute_ticket_intake(service)

    persisted_run = await load_agent_run(
        postgresql_session,
        agent_run_id=result.processing_run.id,
    )

    assert persisted_run.workflow_version == DETERMINISTIC_BASELINE_WORKFLOW_VERSION


async def test_ticket_and_run_persist_matching_trace_identifiers(
    postgresql_session: AsyncSession,
    clean_business_tables: None,
) -> None:
    await persist_workspace(postgresql_session)
    service = create_ticket_intake_service(postgresql_session)

    result = await execute_ticket_intake(service)

    persisted_ticket = await postgresql_session.get(
        TicketRecord,
        result.ticket.id,
    )
    persisted_run = await load_agent_run(
        postgresql_session,
        agent_run_id=result.processing_run.id,
    )

    assert persisted_ticket is not None
    assert persisted_ticket.ingestion_request_id == _INGESTION_REQUEST_ID
    assert persisted_run.ingestion_request_id == persisted_ticket.ingestion_request_id
    assert persisted_ticket.correlation_id == _CORRELATION_ID
    assert persisted_run.correlation_id == persisted_ticket.correlation_id
    assert persisted_ticket.created_at == _BASE_TIMESTAMP
    assert persisted_run.created_at == persisted_ticket.created_at


async def test_agent_run_insertion_failure_rolls_back_ticket(
    postgresql_session: AsyncSession,
    clean_business_tables: None,
) -> None:
    await persist_workspace(postgresql_session)
    service = create_ticket_intake_service(
        postgresql_session,
        agent_run_repository=FailingAgentRunRepository(
            postgresql_session,
        ),
    )

    with pytest.raises(
        RuntimeError,
        match=r"agent run insertion failed",
    ):
        await execute_ticket_intake(service)

    assert await count_tickets(postgresql_session) == 0
    assert await count_agent_runs(postgresql_session) == 0


async def test_ticket_conflict_creates_no_additional_agent_run(
    postgresql_session: AsyncSession,
    clean_business_tables: None,
) -> None:
    await persist_workspace(postgresql_session)
    service = create_ticket_intake_service(postgresql_session)

    first_result = await execute_ticket_intake(service)

    with pytest.raises(
        TicketExternalReferenceConflictApplicationError,
        match=(
            r"Ticket external reference already exists "
            r"in the workspace\."
        ),
    ):
        await execute_ticket_intake(service)

    assert await count_tickets(postgresql_session) == 1
    assert await count_agent_runs(postgresql_session) == 1

    persisted_run = await load_agent_run(
        postgresql_session,
        agent_run_id=first_result.processing_run.id,
    )
    assert persisted_run.ticket_id == first_result.ticket.id


async def test_one_ticket_creates_one_initial_agent_run(
    postgresql_session: AsyncSession,
    clean_business_tables: None,
) -> None:
    await persist_workspace(postgresql_session)
    service = create_ticket_intake_service(postgresql_session)

    result = await execute_ticket_intake(
        service,
        external_reference=None,
    )

    query_result = await postgresql_session.execute(
        select(AgentRunRecord).where(
            AgentRunRecord.ticket_id == result.ticket.id,
            AgentRunRecord.trigger_key == INITIAL_TICKET_PROCESSING_TRIGGER_KEY,
        ),
    )
    persisted_runs = list(query_result.scalars())

    assert len(persisted_runs) == 1
    assert persisted_runs[0].id == result.processing_run.id
