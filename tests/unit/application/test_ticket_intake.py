"""Unit tests for transactional ticket intake scheduling."""

from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from uuid import UUID

import pytest

from supportops.application.ticket_intake import (
    CreateTicketWithInitialRun,
    TicketIntakeResult,
)
from supportops.modules.agent_runs.domain.claiming import (
    AgentRunClaim,
    ClaimAgentRunCommand,
)
from supportops.modules.agent_runs.domain.models import (
    DETERMINISTIC_BASELINE_WORKFLOW_VERSION,
    INITIAL_TICKET_PROCESSING_TRIGGER_KEY,
    INITIAL_TICKET_PROCESSING_WORKFLOW_NAME,
    AgentRun,
    AgentRunStatus,
)
from supportops.modules.tickets.application.errors import (
    TicketExternalReferenceConflictApplicationError,
)
from supportops.modules.tickets.domain.models import Ticket
from supportops.modules.tickets.domain.repositories import (
    TicketExternalReferenceConflictError,
)
from supportops.modules.workspaces.application.errors import (
    WorkspaceNotFoundError,
)
from supportops.modules.workspaces.domain.models import Workspace

_WORKSPACE_ID = UUID(
    "032c8c87-57cc-4d14-bfbd-04968b4e8cd4",
)
_REQUEST_ID = UUID(
    "725eec8a-c504-4071-ac96-c78cc907f26c",
)
_CORRELATION_ID = UUID(
    "1038c98e-62fd-45df-9839-138f7105cb78",
)
_TIMESTAMP = datetime(
    2026,
    7,
    31,
    12,
    0,
    tzinfo=UTC,
)


class FakeTransactionManager:
    """Record transaction completion and rollback."""

    def __init__(self) -> None:
        self.entered = False
        self.completed = False
        self.rolled_back = False

    @asynccontextmanager
    async def transaction(self) -> AsyncIterator[None]:
        self.entered = True

        try:
            yield
        except Exception:
            self.rolled_back = True
            raise
        else:
            self.completed = True


class FakeWorkspaceRepository:
    """In-memory workspace existence repository."""

    def __init__(
        self,
        *,
        workspace_exists: bool = True,
    ) -> None:
        self.workspace_exists = workspace_exists
        self.requested_workspace_id: UUID | None = None

    async def add(self, workspace: Workspace) -> None:
        raise AssertionError("add must not be called")

    async def get(
        self,
        workspace_id: UUID,
    ) -> Workspace | None:
        raise AssertionError("get must not be called")

    async def exists(
        self,
        workspace_id: UUID,
    ) -> bool:
        self.requested_workspace_id = workspace_id
        return self.workspace_exists


class FakeTicketRepository:
    """Record ticket insertion attempts."""

    def __init__(self) -> None:
        self.added_ticket: Ticket | None = None
        self.external_reference_conflict = False

    async def add(
        self,
        ticket: Ticket,
    ) -> None:
        if self.external_reference_conflict:
            raise TicketExternalReferenceConflictError(
                "duplicate external reference",
            )

        self.added_ticket = ticket

    async def get(
        self,
        workspace_id: UUID,
        ticket_id: UUID,
    ) -> Ticket | None:
        raise AssertionError("get must not be called")

    async def list(
        self,
        workspace_id: UUID,
        *,
        limit: int,
        after_created_at: datetime | None = None,
        after_ticket_id: UUID | None = None,
    ) -> Sequence[Ticket]:
        raise AssertionError("list must not be called")


class FakeAgentRunRepository:
    """Record durable AgentRun insertion attempts."""

    def __init__(self) -> None:
        self.added_run: AgentRun | None = None
        self.add_failure: Exception | None = None

    async def add(
        self,
        agent_run: AgentRun,
    ) -> None:
        if self.add_failure is not None:
            raise self.add_failure

        self.added_run = agent_run

    async def claim_next_available(
        self,
        command: ClaimAgentRunCommand,
    ) -> AgentRunClaim | None:
        raise AssertionError("claim_next_available must not be called")


def create_service(
    *,
    workspace_repository: FakeWorkspaceRepository | None = None,
    ticket_repository: FakeTicketRepository | None = None,
    agent_run_repository: FakeAgentRunRepository | None = None,
    transaction_manager: FakeTransactionManager | None = None,
    max_attempts: int = 3,
) -> tuple[
    CreateTicketWithInitialRun,
    FakeWorkspaceRepository,
    FakeTicketRepository,
    FakeAgentRunRepository,
    FakeTransactionManager,
]:
    workspace_repository = workspace_repository or FakeWorkspaceRepository()
    ticket_repository = ticket_repository or FakeTicketRepository()
    agent_run_repository = agent_run_repository or FakeAgentRunRepository()
    transaction_manager = transaction_manager or FakeTransactionManager()

    service = CreateTicketWithInitialRun(
        workspace_repository=workspace_repository,
        ticket_repository=ticket_repository,
        agent_run_repository=agent_run_repository,
        transaction_manager=transaction_manager,
        max_attempts=max_attempts,
        utc_now=lambda: _TIMESTAMP,
    )

    return (
        service,
        workspace_repository,
        ticket_repository,
        agent_run_repository,
        transaction_manager,
    )


async def execute_ticket_intake(
    service: CreateTicketWithInitialRun,
) -> TicketIntakeResult:
    return await service.execute(
        workspace_id=_WORKSPACE_ID,
        subject="  Unable to access billing  ",
        description="  The dashboard returns an access error.  ",
        external_reference="SUP-1042",
        ingestion_request_id=_REQUEST_ID,
        correlation_id=_CORRELATION_ID,
    )


async def test_ticket_intake_persists_ticket_and_run_atomically() -> None:
    (
        service,
        workspace_repository,
        ticket_repository,
        agent_run_repository,
        transaction_manager,
    ) = create_service()

    result = await execute_ticket_intake(service)

    assert workspace_repository.requested_workspace_id == _WORKSPACE_ID
    assert ticket_repository.added_ticket == result.ticket
    assert agent_run_repository.added_run == result.processing_run
    assert transaction_manager.entered
    assert transaction_manager.completed
    assert not transaction_manager.rolled_back


async def test_ticket_intake_creates_expected_initial_run() -> None:
    service, _, _, _, _ = create_service(max_attempts=4)

    result = await execute_ticket_intake(service)

    ticket = result.ticket
    processing_run = result.processing_run

    assert processing_run.workspace_id == ticket.workspace_id
    assert processing_run.ticket_id == ticket.id
    assert processing_run.workflow_name == INITIAL_TICKET_PROCESSING_WORKFLOW_NAME
    assert processing_run.workflow_version == DETERMINISTIC_BASELINE_WORKFLOW_VERSION
    assert processing_run.trigger_key == INITIAL_TICKET_PROCESSING_TRIGGER_KEY
    assert processing_run.status is AgentRunStatus.QUEUED
    assert processing_run.available_at == _TIMESTAMP
    assert processing_run.attempt_count == 0
    assert processing_run.max_attempts == 4


async def test_ticket_and_run_share_trace_identifiers_and_timestamp() -> None:
    service, _, _, _, _ = create_service()

    result = await execute_ticket_intake(service)

    ticket = result.ticket
    processing_run = result.processing_run

    assert ticket.ingestion_request_id == _REQUEST_ID
    assert processing_run.ingestion_request_id == _REQUEST_ID
    assert ticket.correlation_id == _CORRELATION_ID
    assert processing_run.correlation_id == _CORRELATION_ID
    assert ticket.created_at == _TIMESTAMP
    assert processing_run.created_at == _TIMESTAMP
    assert processing_run.available_at == ticket.created_at


async def test_ticket_intake_rejects_missing_workspace_before_inserts() -> None:
    workspace_repository = FakeWorkspaceRepository(
        workspace_exists=False,
    )
    (
        service,
        _,
        ticket_repository,
        agent_run_repository,
        transaction_manager,
    ) = create_service(
        workspace_repository=workspace_repository,
    )

    with pytest.raises(
        WorkspaceNotFoundError,
        match=r"Workspace was not found\.",
    ):
        await execute_ticket_intake(service)

    assert ticket_repository.added_ticket is None
    assert agent_run_repository.added_run is None
    assert transaction_manager.rolled_back
    assert not transaction_manager.completed


async def test_ticket_conflict_prevents_agent_run_insertion() -> None:
    ticket_repository = FakeTicketRepository()
    ticket_repository.external_reference_conflict = True

    (
        service,
        _,
        _,
        agent_run_repository,
        transaction_manager,
    ) = create_service(
        ticket_repository=ticket_repository,
    )

    with pytest.raises(
        TicketExternalReferenceConflictApplicationError,
        match=(
            r"Ticket external reference already exists "
            r"in the workspace\."
        ),
    ):
        await execute_ticket_intake(service)

    assert agent_run_repository.added_run is None
    assert transaction_manager.rolled_back
    assert not transaction_manager.completed


async def test_agent_run_insertion_failure_rolls_back_transaction() -> None:
    agent_run_repository = FakeAgentRunRepository()
    agent_run_repository.add_failure = RuntimeError(
        "agent run insertion failed",
    )

    (
        service,
        _,
        ticket_repository,
        _,
        transaction_manager,
    ) = create_service(
        agent_run_repository=agent_run_repository,
    )

    with pytest.raises(
        RuntimeError,
        match=r"agent run insertion failed",
    ):
        await execute_ticket_intake(service)

    assert ticket_repository.added_ticket is not None
    assert transaction_manager.rolled_back
    assert not transaction_manager.completed


def test_ticket_intake_rejects_invalid_max_attempts() -> None:
    with pytest.raises(
        ValueError,
        match=r"max_attempts must be at least one\.",
    ):
        CreateTicketWithInitialRun(
            workspace_repository=FakeWorkspaceRepository(),
            ticket_repository=FakeTicketRepository(),
            agent_run_repository=FakeAgentRunRepository(),
            transaction_manager=FakeTransactionManager(),
            max_attempts=0,
            utc_now=lambda: _TIMESTAMP,
        )
