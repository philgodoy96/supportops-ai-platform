"""Integration tests for the PostgreSQL ticket repository."""

import asyncio
from datetime import UTC, datetime, timedelta
from uuid import UUID

import pytest
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
)

from supportops.infrastructure.postgresql.transaction import (
    SqlAlchemyTransactionManager,
)
from supportops.modules.tickets.domain.models import Ticket
from supportops.modules.tickets.domain.repositories import (
    TicketExternalReferenceConflictError,
)
from supportops.modules.tickets.infrastructure.repository import (
    SqlAlchemyTicketRepository,
)
from supportops.modules.workspaces.domain.models import Workspace
from supportops.modules.workspaces.infrastructure.repository import (
    SqlAlchemyWorkspaceRepository,
)

pytestmark = pytest.mark.integration

_WORKSPACE_A_ID = UUID(
    "032c8c87-57cc-4d14-bfbd-04968b4e8cd4",
)
_WORKSPACE_B_ID = UUID(
    "4aefba3b-b57e-47d1-889e-bb28762fa1ed",
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


def create_workspace(
    *,
    workspace_id: UUID,
    name: str,
    slug: str,
) -> Workspace:
    """Create a deterministic workspace."""

    return Workspace(
        id=workspace_id,
        name=name,
        slug=slug,
        created_at=_BASE_TIMESTAMP,
        updated_at=_BASE_TIMESTAMP,
    )


def create_ticket(
    *,
    ticket_id: UUID,
    workspace_id: UUID = _WORKSPACE_A_ID,
    external_reference: str | None = None,
    created_at: datetime = _BASE_TIMESTAMP,
    subject: str = "Unable to access billing",
) -> Ticket:
    """Create a deterministic support ticket."""

    return Ticket.create(
        ticket_id=ticket_id,
        workspace_id=workspace_id,
        subject=subject,
        description="The dashboard returns an access error.",
        external_reference=external_reference,
        ingestion_request_id=_INGESTION_REQUEST_ID,
        correlation_id=_CORRELATION_ID,
        now=created_at,
    )


async def persist_workspaces(
    session: AsyncSession,
) -> tuple[Workspace, Workspace]:
    """Persist the two standard test workspaces."""

    repository = SqlAlchemyWorkspaceRepository(session)
    transaction_manager = SqlAlchemyTransactionManager(session)

    workspace_a = create_workspace(
        workspace_id=_WORKSPACE_A_ID,
        name="Platform Support",
        slug="platform-support",
    )
    workspace_b = create_workspace(
        workspace_id=_WORKSPACE_B_ID,
        name="Customer Success",
        slug="customer-success",
    )

    async with transaction_manager.transaction():
        await repository.add(workspace_a)
        await repository.add(workspace_b)

    return workspace_a, workspace_b


async def test_repository_adds_and_retrieves_ticket_with_workspace_scope(
    postgresql_session: AsyncSession,
    clean_business_tables: None,
) -> None:
    await persist_workspaces(postgresql_session)

    repository = SqlAlchemyTicketRepository(
        postgresql_session,
    )
    transaction_manager = SqlAlchemyTransactionManager(
        postgresql_session,
    )
    ticket = create_ticket(
        ticket_id=UUID(
            "f84d7304-8171-4842-a111-c3dbda2ff79b",
        ),
    )

    async with transaction_manager.transaction():
        await repository.add(ticket)

    assert (
        await repository.get(
            _WORKSPACE_A_ID,
            ticket.id,
        )
        == ticket
    )

    assert (
        await repository.get(
            _WORKSPACE_B_ID,
            ticket.id,
        )
        is None
    )


async def test_repository_returns_none_for_missing_ticket(
    postgresql_session: AsyncSession,
    clean_business_tables: None,
) -> None:
    await persist_workspaces(postgresql_session)

    repository = SqlAlchemyTicketRepository(
        postgresql_session,
    )

    assert (
        await repository.get(
            _WORKSPACE_A_ID,
            UUID("84e98bb4-7856-4689-b8c2-b9155a575e51"),
        )
        is None
    )


async def test_repository_translates_duplicate_external_reference(
    postgresql_session: AsyncSession,
    clean_business_tables: None,
) -> None:
    await persist_workspaces(postgresql_session)

    repository = SqlAlchemyTicketRepository(
        postgresql_session,
    )
    transaction_manager = SqlAlchemyTransactionManager(
        postgresql_session,
    )
    first_ticket = create_ticket(
        ticket_id=UUID(
            "f84d7304-8171-4842-a111-c3dbda2ff79b",
        ),
        external_reference="SUP-1042",
    )
    duplicate_ticket = create_ticket(
        ticket_id=UUID(
            "a8af84ac-d48c-4f12-80a4-297b04b199f2",
        ),
        external_reference="SUP-1042",
    )

    async with transaction_manager.transaction():
        await repository.add(first_ticket)

    with pytest.raises(
        TicketExternalReferenceConflictError,
        match=r"Ticket external reference already exists in the workspace\.",
    ):
        async with transaction_manager.transaction():
            await repository.add(duplicate_ticket)

    assert (
        await repository.get(
            _WORKSPACE_A_ID,
            first_ticket.id,
        )
        == first_ticket
    )
    assert (
        await repository.get(
            _WORKSPACE_A_ID,
            duplicate_ticket.id,
        )
        is None
    )


async def test_repository_allows_same_external_reference_across_workspaces(
    postgresql_session: AsyncSession,
    clean_business_tables: None,
) -> None:
    await persist_workspaces(postgresql_session)

    repository = SqlAlchemyTicketRepository(
        postgresql_session,
    )
    transaction_manager = SqlAlchemyTransactionManager(
        postgresql_session,
    )
    workspace_a_ticket = create_ticket(
        ticket_id=UUID(
            "f84d7304-8171-4842-a111-c3dbda2ff79b",
        ),
        workspace_id=_WORKSPACE_A_ID,
        external_reference="SUP-1042",
    )
    workspace_b_ticket = create_ticket(
        ticket_id=UUID(
            "a8af84ac-d48c-4f12-80a4-297b04b199f2",
        ),
        workspace_id=_WORKSPACE_B_ID,
        external_reference="SUP-1042",
    )

    async with transaction_manager.transaction():
        await repository.add(workspace_a_ticket)
        await repository.add(workspace_b_ticket)

    assert (
        await repository.get(
            _WORKSPACE_A_ID,
            workspace_a_ticket.id,
        )
        == workspace_a_ticket
    )
    assert (
        await repository.get(
            _WORKSPACE_B_ID,
            workspace_b_ticket.id,
        )
        == workspace_b_ticket
    )


async def test_repository_allows_multiple_null_external_references(
    postgresql_session: AsyncSession,
    clean_business_tables: None,
) -> None:
    await persist_workspaces(postgresql_session)

    repository = SqlAlchemyTicketRepository(
        postgresql_session,
    )
    transaction_manager = SqlAlchemyTransactionManager(
        postgresql_session,
    )
    first_ticket = create_ticket(
        ticket_id=UUID(
            "f84d7304-8171-4842-a111-c3dbda2ff79b",
        ),
    )
    second_ticket = create_ticket(
        ticket_id=UUID(
            "a8af84ac-d48c-4f12-80a4-297b04b199f2",
        ),
    )

    async with transaction_manager.transaction():
        await repository.add(first_ticket)
        await repository.add(second_ticket)

    tickets = await repository.list(
        _WORKSPACE_A_ID,
        limit=10,
    )

    assert {ticket.id for ticket in tickets} == {
        first_ticket.id,
        second_ticket.id,
    }


async def test_repository_relies_on_workspace_foreign_key(
    postgresql_session: AsyncSession,
    clean_business_tables: None,
) -> None:
    repository = SqlAlchemyTicketRepository(
        postgresql_session,
    )
    transaction_manager = SqlAlchemyTransactionManager(
        postgresql_session,
    )
    ticket = create_ticket(
        ticket_id=UUID(
            "f84d7304-8171-4842-a111-c3dbda2ff79b",
        ),
        workspace_id=UUID(
            "fc899d6f-4f50-4ad2-b7af-9cc438db0718",
        ),
    )

    with pytest.raises(IntegrityError):
        async with transaction_manager.transaction():
            await repository.add(ticket)

    assert (
        await repository.get(
            ticket.workspace_id,
            ticket.id,
        )
        is None
    )


async def test_repository_lists_only_workspace_tickets_in_stable_order(
    postgresql_session: AsyncSession,
    clean_business_tables: None,
) -> None:
    await persist_workspaces(postgresql_session)

    repository = SqlAlchemyTicketRepository(
        postgresql_session,
    )
    transaction_manager = SqlAlchemyTransactionManager(
        postgresql_session,
    )

    older_ticket = create_ticket(
        ticket_id=UUID(
            "00000000-0000-4000-8000-000000000001",
        ),
        created_at=_BASE_TIMESTAMP,
        subject="Older ticket",
    )
    same_time_lower_id = create_ticket(
        ticket_id=UUID(
            "00000000-0000-4000-8000-000000000002",
        ),
        created_at=_BASE_TIMESTAMP + timedelta(minutes=1),
        subject="Lower ID",
    )
    same_time_higher_id = create_ticket(
        ticket_id=UUID(
            "00000000-0000-4000-8000-000000000003",
        ),
        created_at=_BASE_TIMESTAMP + timedelta(minutes=1),
        subject="Higher ID",
    )
    other_workspace_ticket = create_ticket(
        ticket_id=UUID(
            "00000000-0000-4000-8000-000000000004",
        ),
        workspace_id=_WORKSPACE_B_ID,
        created_at=_BASE_TIMESTAMP + timedelta(minutes=2),
        subject="Other workspace",
    )

    async with transaction_manager.transaction():
        await repository.add(older_ticket)
        await repository.add(same_time_lower_id)
        await repository.add(same_time_higher_id)
        await repository.add(other_workspace_ticket)

    tickets = await repository.list(
        _WORKSPACE_A_ID,
        limit=10,
    )

    assert [ticket.id for ticket in tickets] == [
        same_time_higher_id.id,
        same_time_lower_id.id,
        older_ticket.id,
    ]


async def test_repository_uses_keyset_position_without_duplicates(
    postgresql_session: AsyncSession,
    clean_business_tables: None,
) -> None:
    await persist_workspaces(postgresql_session)

    repository = SqlAlchemyTicketRepository(
        postgresql_session,
    )
    transaction_manager = SqlAlchemyTransactionManager(
        postgresql_session,
    )

    tickets = [
        create_ticket(
            ticket_id=UUID(
                f"00000000-0000-4000-8000-{index:012d}",
            ),
            created_at=_BASE_TIMESTAMP
            + timedelta(
                minutes=index // 2,
            ),
            subject=f"Ticket {index}",
        )
        for index in range(1, 6)
    ]

    async with transaction_manager.transaction():
        for ticket in tickets:
            await repository.add(ticket)

    first_page = await repository.list(
        _WORKSPACE_A_ID,
        limit=2,
    )
    second_page = await repository.list(
        _WORKSPACE_A_ID,
        limit=2,
        after_created_at=first_page[-1].created_at,
        after_ticket_id=first_page[-1].id,
    )
    third_page = await repository.list(
        _WORKSPACE_A_ID,
        limit=2,
        after_created_at=second_page[-1].created_at,
        after_ticket_id=second_page[-1].id,
    )

    observed_ids = [
        ticket.id
        for ticket in [
            *first_page,
            *second_page,
            *third_page,
        ]
    ]

    assert len(observed_ids) == 5
    assert len(set(observed_ids)) == 5


@pytest.mark.parametrize(
    (
        "after_created_at",
        "after_ticket_id",
    ),
    [
        (
            _BASE_TIMESTAMP,
            None,
        ),
        (
            None,
            UUID(
                "00000000-0000-4000-8000-000000000001",
            ),
        ),
    ],
)
async def test_repository_rejects_partial_keyset_position(
    postgresql_session: AsyncSession,
    clean_business_tables: None,
    after_created_at: datetime | None,
    after_ticket_id: UUID | None,
) -> None:
    repository = SqlAlchemyTicketRepository(
        postgresql_session,
    )

    with pytest.raises(
        ValueError,
        match=r"Ticket pagination position requires both timestamp and ID\.",
    ):
        await repository.list(
            _WORKSPACE_A_ID,
            limit=20,
            after_created_at=after_created_at,
            after_ticket_id=after_ticket_id,
        )


async def test_repository_rejects_nonpositive_limit(
    postgresql_session: AsyncSession,
    clean_business_tables: None,
) -> None:
    repository = SqlAlchemyTicketRepository(
        postgresql_session,
    )

    with pytest.raises(
        ValueError,
        match=r"Ticket list limit must be positive\.",
    ):
        await repository.list(
            _WORKSPACE_A_ID,
            limit=0,
        )


async def test_database_constraint_serializes_concurrent_duplicate_references(
    postgresql_session_factory: async_sessionmaker[AsyncSession],
    clean_business_tables: None,
) -> None:
    async with postgresql_session_factory() as setup_session:
        await persist_workspaces(setup_session)

    ready = 0
    ready_lock = asyncio.Lock()
    release_inserts = asyncio.Event()

    first_ticket = create_ticket(
        ticket_id=UUID(
            "f84d7304-8171-4842-a111-c3dbda2ff79b",
        ),
        external_reference="CONCURRENT-1042",
    )
    second_ticket = create_ticket(
        ticket_id=UUID(
            "a8af84ac-d48c-4f12-80a4-297b04b199f2",
        ),
        external_reference="CONCURRENT-1042",
    )

    async def insert_ticket(
        ticket: Ticket,
    ) -> TicketExternalReferenceConflictError | None:
        nonlocal ready

        async with postgresql_session_factory() as session:
            repository = SqlAlchemyTicketRepository(session)
            transaction_manager = SqlAlchemyTransactionManager(
                session,
            )

            async with ready_lock:
                ready += 1

                if ready == 2:
                    release_inserts.set()

            await release_inserts.wait()

            try:
                async with transaction_manager.transaction():
                    await repository.add(ticket)
            except TicketExternalReferenceConflictError as error:
                return error

            return None

    results = await asyncio.gather(
        insert_ticket(first_ticket),
        insert_ticket(second_ticket),
    )

    conflicts = [
        result
        for result in results
        if isinstance(
            result,
            TicketExternalReferenceConflictError,
        )
    ]

    assert len(conflicts) == 1

    async with postgresql_session_factory() as verification_session:
        repository = SqlAlchemyTicketRepository(
            verification_session,
        )
        persisted_tickets = await repository.list(
            _WORKSPACE_A_ID,
            limit=10,
        )

    assert len(persisted_tickets) == 1
    assert persisted_tickets[0].external_reference == ("CONCURRENT-1042")
