"""Integration tests for the PostgreSQL document repository."""

from datetime import UTC, datetime, timedelta
from uuid import UUID

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from supportops.infrastructure.postgresql.transaction import (
    SqlAlchemyTransactionManager,
)
from supportops.modules.knowledge_documents.domain.models import (
    Document,
    DocumentMediaType,
    DocumentVersion,
    KnowledgeIndexProfile,
)
from supportops.modules.knowledge_documents.domain.repositories import (
    DocumentExternalReferenceConflictError,
)
from supportops.modules.knowledge_documents.infrastructure.repository import (
    SqlAlchemyDocumentRepository,
    SqlAlchemyDocumentVersionRepository,
)
from supportops.modules.workspaces.domain.models import Workspace
from supportops.modules.workspaces.infrastructure.repository import (
    SqlAlchemyWorkspaceRepository,
)

pytestmark = pytest.mark.integration

_WORKSPACE_A_ID = UUID("032c8c87-57cc-4d14-bfbd-04968b4e8cd4")
_WORKSPACE_B_ID = UUID("4aefba3b-b57e-47d1-889e-bb28762fa1ed")
_DOCUMENT_ID = UUID("276046a2-28ec-4cb1-8bb6-a2ff70f9064b")
_VERSION_ID = UUID("09036916-84cf-4a58-bdf4-09bc52716ec5")
_BASE_TIMESTAMP = datetime(2026, 8, 1, 22, 0, tzinfo=UTC)


def create_workspace(
    *,
    workspace_id: UUID,
    name: str,
    slug: str,
) -> Workspace:
    """Create one deterministic workspace."""

    return Workspace(
        id=workspace_id,
        name=name,
        slug=slug,
        created_at=_BASE_TIMESTAMP,
        updated_at=_BASE_TIMESTAMP,
    )


async def persist_workspaces(session: AsyncSession) -> None:
    """Persist the standard workspace pair."""

    repository = SqlAlchemyWorkspaceRepository(session)
    transaction_manager = SqlAlchemyTransactionManager(session)

    async with transaction_manager.transaction():
        await repository.add(
            create_workspace(
                workspace_id=_WORKSPACE_A_ID,
                name="Platform Support",
                slug="platform-support",
            )
        )
        await repository.add(
            create_workspace(
                workspace_id=_WORKSPACE_B_ID,
                name="Customer Success",
                slug="customer-success",
            )
        )


def create_document(
    *,
    document_id: UUID,
    workspace_id: UUID = _WORKSPACE_A_ID,
    external_reference: str | None = None,
    created_at: datetime = _BASE_TIMESTAMP,
    title: str = "Database Incident Runbook",
) -> Document:
    """Create one deterministic knowledge document."""

    return Document.create(
        document_id=document_id,
        workspace_id=workspace_id,
        title=title,
        external_reference=external_reference,
        now=created_at,
    )


def create_ready_version() -> DocumentVersion:
    """Create one ready version for activation tests."""

    pending = DocumentVersion.create_pending(
        workspace_id=_WORKSPACE_A_ID,
        document_id=_DOCUMENT_ID,
        document_version_id=_VERSION_ID,
        version_number=1,
        media_type=DocumentMediaType.TEXT_MARKDOWN,
        content="# Database incidents\nRestart the connection pool.\n",
        now=_BASE_TIMESTAMP,
    )
    profiled = pending.bind_index_profile(
        KnowledgeIndexProfile(
            chunking_strategy="markdown-token",
            chunking_version="v1",
            tokenizer_encoding="cl100k_base",
            embedding_provider="mock",
            embedding_model="mock-hashing-embedding-v1",
            embedding_dimensions=64,
            knowledge_collection="supportops-knowledge-mock-v1",
            knowledge_vector_name="dense",
        ),
        now=_BASE_TIMESTAMP,
    )
    return profiled.mark_ready(
        chunk_count=1,
        embedding_input_tokens=12,
        embedding_estimated_cost_usd=None,
        embedding_pricing_catalog_version=("supportops-embedding-pricing-2026-08-01"),
        indexed_at=_BASE_TIMESTAMP + timedelta(minutes=1),
    )


async def test_repository_adds_and_retrieves_document_with_workspace_scope(
    postgresql_session: AsyncSession,
    clean_business_tables: None,
) -> None:
    await persist_workspaces(postgresql_session)
    repository = SqlAlchemyDocumentRepository(postgresql_session)
    transaction_manager = SqlAlchemyTransactionManager(postgresql_session)
    document = create_document(document_id=_DOCUMENT_ID)

    async with transaction_manager.transaction():
        await repository.add(document)

    assert (
        await repository.get(
            workspace_id=_WORKSPACE_A_ID,
            document_id=document.id,
        )
        == document
    )
    assert (
        await repository.get(
            workspace_id=_WORKSPACE_B_ID,
            document_id=document.id,
        )
        is None
    )
    assert (
        await repository.get_for_update(
            workspace_id=_WORKSPACE_A_ID,
            document_id=document.id,
        )
        == document
    )
    await postgresql_session.commit()


async def test_repository_translates_duplicate_external_reference(
    postgresql_session: AsyncSession,
    clean_business_tables: None,
) -> None:
    await persist_workspaces(postgresql_session)
    repository = SqlAlchemyDocumentRepository(postgresql_session)
    transaction_manager = SqlAlchemyTransactionManager(postgresql_session)
    first = create_document(
        document_id=_DOCUMENT_ID,
        external_reference="runbook-database-incidents",
    )
    duplicate = create_document(
        document_id=UUID("db00b4aa-4c17-4bf5-a333-226af35069c8"),
        external_reference="runbook-database-incidents",
    )

    async with transaction_manager.transaction():
        await repository.add(first)

    with pytest.raises(
        DocumentExternalReferenceConflictError,
        match=(r"Document external reference already exists in the workspace\."),
    ):
        async with transaction_manager.transaction():
            await repository.add(duplicate)

    assert (
        await repository.get(
            workspace_id=_WORKSPACE_A_ID,
            document_id=first.id,
        )
        == first
    )
    assert (
        await repository.get(
            workspace_id=_WORKSPACE_A_ID,
            document_id=duplicate.id,
        )
        is None
    )


async def test_repository_allows_same_external_reference_across_workspaces(
    postgresql_session: AsyncSession,
    clean_business_tables: None,
) -> None:
    await persist_workspaces(postgresql_session)
    repository = SqlAlchemyDocumentRepository(postgresql_session)
    transaction_manager = SqlAlchemyTransactionManager(postgresql_session)
    first = create_document(
        document_id=_DOCUMENT_ID,
        external_reference="runbook-database-incidents",
    )
    second = create_document(
        document_id=UUID("db00b4aa-4c17-4bf5-a333-226af35069c8"),
        workspace_id=_WORKSPACE_B_ID,
        external_reference="runbook-database-incidents",
    )

    async with transaction_manager.transaction():
        await repository.add(first)
        await repository.add(second)

    assert (
        await repository.get(
            workspace_id=_WORKSPACE_A_ID,
            document_id=first.id,
        )
        == first
    )
    assert (
        await repository.get(
            workspace_id=_WORKSPACE_B_ID,
            document_id=second.id,
        )
        == second
    )


async def test_repository_lists_documents_in_stable_keyset_order(
    postgresql_session: AsyncSession,
    clean_business_tables: None,
) -> None:
    await persist_workspaces(postgresql_session)
    repository = SqlAlchemyDocumentRepository(postgresql_session)
    transaction_manager = SqlAlchemyTransactionManager(postgresql_session)
    documents = [
        create_document(
            document_id=UUID(f"00000000-0000-4000-8000-{index:012d}"),
            created_at=_BASE_TIMESTAMP + timedelta(minutes=index // 2),
            title=f"Runbook {index}",
        )
        for index in range(1, 6)
    ]
    other_workspace = create_document(
        document_id=UUID("00000000-0000-4000-8000-000000000099"),
        workspace_id=_WORKSPACE_B_ID,
        created_at=_BASE_TIMESTAMP + timedelta(hours=1),
    )

    async with transaction_manager.transaction():
        for document in [*documents, other_workspace]:
            await repository.add(document)

    first_page = await repository.list(
        _WORKSPACE_A_ID,
        limit=2,
    )
    second_page = await repository.list(
        _WORKSPACE_A_ID,
        limit=2,
        after_created_at=first_page[-1].created_at,
        after_document_id=first_page[-1].id,
    )
    third_page = await repository.list(
        _WORKSPACE_A_ID,
        limit=2,
        after_created_at=second_page[-1].created_at,
        after_document_id=second_page[-1].id,
    )
    observed = [*first_page, *second_page, *third_page]

    assert len(observed) == 5
    assert len({document.id for document in observed}) == 5
    assert all(document.workspace_id == _WORKSPACE_A_ID for document in observed)


async def test_repository_persists_explicit_ready_version_activation(
    postgresql_session: AsyncSession,
    clean_business_tables: None,
) -> None:
    await persist_workspaces(postgresql_session)
    document_repository = SqlAlchemyDocumentRepository(postgresql_session)
    version_repository = SqlAlchemyDocumentVersionRepository(postgresql_session)
    transaction_manager = SqlAlchemyTransactionManager(postgresql_session)
    document = create_document(document_id=_DOCUMENT_ID)
    ready_version = create_ready_version()

    async with transaction_manager.transaction():
        await document_repository.add(document)
        await version_repository.add(ready_version)

    activated = document.activate_version(
        ready_version,
        now=_BASE_TIMESTAMP + timedelta(minutes=2),
    )
    async with transaction_manager.transaction():
        locked = await document_repository.get_for_update(
            workspace_id=_WORKSPACE_A_ID,
            document_id=document.id,
        )
        assert locked == document
        await document_repository.update(activated)

    assert (
        await document_repository.get(
            workspace_id=_WORKSPACE_A_ID,
            document_id=document.id,
        )
        == activated
    )


@pytest.mark.parametrize(
    ("after_created_at", "after_document_id"),
    [
        (_BASE_TIMESTAMP, None),
        (
            None,
            UUID("00000000-0000-4000-8000-000000000001"),
        ),
    ],
)
async def test_repository_rejects_partial_document_keyset_position(
    postgresql_session: AsyncSession,
    clean_business_tables: None,
    after_created_at: datetime | None,
    after_document_id: UUID | None,
) -> None:
    repository = SqlAlchemyDocumentRepository(postgresql_session)

    with pytest.raises(
        ValueError,
        match=(r"Document pagination position requires both timestamp and ID\."),
    ):
        await repository.list(
            _WORKSPACE_A_ID,
            limit=20,
            after_created_at=after_created_at,
            after_document_id=after_document_id,
        )
