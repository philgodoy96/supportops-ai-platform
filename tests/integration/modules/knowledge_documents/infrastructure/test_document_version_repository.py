"""Integration tests for the PostgreSQL document-version repository."""

from dataclasses import replace
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from uuid import UUID

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from supportops.infrastructure.postgresql.transaction import (
    SqlAlchemyTransactionManager,
)
from supportops.modules.knowledge_documents.domain.content import (
    compute_content_sha256,
)
from supportops.modules.knowledge_documents.domain.models import (
    Document,
    DocumentMediaType,
    DocumentVersion,
    DocumentVersionStatus,
    KnowledgeIndexProfile,
)
from supportops.modules.knowledge_documents.domain.repositories import (
    DocumentVersionContentConflictError,
    DocumentVersionNumberConflictError,
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
_OTHER_DOCUMENT_ID = UUID("db00b4aa-4c17-4bf5-a333-226af35069c8")
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


async def persist_owners(session: AsyncSession) -> None:
    """Persist two workspaces and one document in each."""

    workspace_repository = SqlAlchemyWorkspaceRepository(session)
    document_repository = SqlAlchemyDocumentRepository(session)
    transaction_manager = SqlAlchemyTransactionManager(session)

    async with transaction_manager.transaction():
        await workspace_repository.add(
            create_workspace(
                workspace_id=_WORKSPACE_A_ID,
                name="Platform Support",
                slug="platform-support",
            )
        )
        await workspace_repository.add(
            create_workspace(
                workspace_id=_WORKSPACE_B_ID,
                name="Customer Success",
                slug="customer-success",
            )
        )
        await document_repository.add(
            Document.create(
                document_id=_DOCUMENT_ID,
                workspace_id=_WORKSPACE_A_ID,
                title="Database Incident Runbook",
                now=_BASE_TIMESTAMP,
            )
        )
        await document_repository.add(
            Document.create(
                document_id=_OTHER_DOCUMENT_ID,
                workspace_id=_WORKSPACE_B_ID,
                title="Billing Incident Runbook",
                now=_BASE_TIMESTAMP,
            )
        )


def create_version(
    *,
    version_id: UUID,
    document_id: UUID = _DOCUMENT_ID,
    workspace_id: UUID = _WORKSPACE_A_ID,
    version_number: int = 1,
    content: str = "# Database incidents\nRestart the pool.\n",
    created_at: datetime = _BASE_TIMESTAMP,
) -> DocumentVersion:
    """Create one deterministic pending version."""

    return DocumentVersion.create_pending(
        document_version_id=version_id,
        workspace_id=workspace_id,
        document_id=document_id,
        version_number=version_number,
        media_type=DocumentMediaType.TEXT_MARKDOWN,
        content=content,
        now=created_at,
    )


def create_profile() -> KnowledgeIndexProfile:
    """Return the approved deterministic mock profile."""

    return KnowledgeIndexProfile(
        chunking_strategy="markdown-token",
        chunking_version="v1",
        tokenizer_encoding="cl100k_base",
        embedding_provider="mock",
        embedding_model="mock-hashing-embedding-v1",
        embedding_dimensions=64,
        knowledge_collection="supportops-knowledge-mock-v1",
        knowledge_vector_name="dense",
    )


async def test_repository_adds_and_retrieves_owned_version(
    postgresql_session: AsyncSession,
    clean_business_tables: None,
) -> None:
    await persist_owners(postgresql_session)
    repository = SqlAlchemyDocumentVersionRepository(postgresql_session)
    transaction_manager = SqlAlchemyTransactionManager(postgresql_session)
    version = create_version(version_id=UUID("09036916-84cf-4a58-bdf4-09bc52716ec5"))

    async with transaction_manager.transaction():
        await repository.add(version)

    assert (
        await repository.get(
            workspace_id=_WORKSPACE_A_ID,
            document_id=_DOCUMENT_ID,
            document_version_id=version.id,
        )
        == version
    )
    assert (
        await repository.get(
            workspace_id=_WORKSPACE_B_ID,
            document_id=_DOCUMENT_ID,
            document_version_id=version.id,
        )
        is None
    )
    assert (
        await repository.get(
            workspace_id=_WORKSPACE_A_ID,
            document_id=_OTHER_DOCUMENT_ID,
            document_version_id=version.id,
        )
        is None
    )


async def test_repository_translates_duplicate_content_hash(
    postgresql_session: AsyncSession,
    clean_business_tables: None,
) -> None:
    await persist_owners(postgresql_session)
    repository = SqlAlchemyDocumentVersionRepository(postgresql_session)
    transaction_manager = SqlAlchemyTransactionManager(postgresql_session)
    first = create_version(
        version_id=UUID("09036916-84cf-4a58-bdf4-09bc52716ec5"),
        version_number=1,
    )
    duplicate = create_version(
        version_id=UUID("8ac21e0c-f869-4d84-a9e1-37d8074c9e54"),
        version_number=2,
    )

    async with transaction_manager.transaction():
        await repository.add(first)

    with pytest.raises(
        DocumentVersionContentConflictError,
        match=r"Document content already exists for this document\.",
    ):
        async with transaction_manager.transaction():
            await repository.add(duplicate)


async def test_repository_translates_duplicate_version_number(
    postgresql_session: AsyncSession,
    clean_business_tables: None,
) -> None:
    await persist_owners(postgresql_session)
    repository = SqlAlchemyDocumentVersionRepository(postgresql_session)
    transaction_manager = SqlAlchemyTransactionManager(postgresql_session)
    first = create_version(
        version_id=UUID("09036916-84cf-4a58-bdf4-09bc52716ec5"),
        version_number=1,
    )
    duplicate_number = create_version(
        version_id=UUID("8ac21e0c-f869-4d84-a9e1-37d8074c9e54"),
        version_number=1,
        content="# Database incidents\nEscalate to the database team.\n",
    )

    async with transaction_manager.transaction():
        await repository.add(first)

    with pytest.raises(
        DocumentVersionNumberConflictError,
        match=r"Document version number already exists\.",
    ):
        async with transaction_manager.transaction():
            await repository.add(duplicate_number)


async def test_repository_allows_same_content_in_different_documents(
    postgresql_session: AsyncSession,
    clean_business_tables: None,
) -> None:
    await persist_owners(postgresql_session)
    repository = SqlAlchemyDocumentVersionRepository(postgresql_session)
    transaction_manager = SqlAlchemyTransactionManager(postgresql_session)
    content = "# Shared procedure\nEscalate to operations.\n"
    first = create_version(
        version_id=UUID("09036916-84cf-4a58-bdf4-09bc52716ec5"),
        content=content,
    )
    second = create_version(
        version_id=UUID("8ac21e0c-f869-4d84-a9e1-37d8074c9e54"),
        workspace_id=_WORKSPACE_B_ID,
        document_id=_OTHER_DOCUMENT_ID,
        content=content,
    )

    async with transaction_manager.transaction():
        await repository.add(first)
        await repository.add(second)

    assert (
        await repository.get_by_content_hash(
            workspace_id=_WORKSPACE_A_ID,
            document_id=_DOCUMENT_ID,
            content_sha256=first.content_sha256,
        )
        == first
    )
    assert (
        await repository.get_by_content_hash(
            workspace_id=_WORKSPACE_B_ID,
            document_id=_OTHER_DOCUMENT_ID,
            content_sha256=second.content_sha256,
        )
        == second
    )


async def test_repository_updates_indexing_lifecycle_without_rewriting_source(
    postgresql_session: AsyncSession,
    clean_business_tables: None,
) -> None:
    await persist_owners(postgresql_session)
    repository = SqlAlchemyDocumentVersionRepository(postgresql_session)
    transaction_manager = SqlAlchemyTransactionManager(postgresql_session)
    pending = create_version(version_id=UUID("09036916-84cf-4a58-bdf4-09bc52716ec5"))

    async with transaction_manager.transaction():
        await repository.add(pending)

    profiled = pending.bind_index_profile(
        create_profile(),
        now=_BASE_TIMESTAMP + timedelta(minutes=1),
    )
    failed = profiled.mark_failed(
        error_code="embedding_timeout",
        chunk_count=2,
        now=_BASE_TIMESTAMP + timedelta(minutes=2),
    )
    async with transaction_manager.transaction():
        locked = await repository.get_for_update(
            workspace_id=_WORKSPACE_A_ID,
            document_id=_DOCUMENT_ID,
            document_version_id=pending.id,
        )
        assert locked == pending
        await repository.update(failed)

    async with transaction_manager.transaction():
        persisted_failed = await repository.get(
            workspace_id=_WORKSPACE_A_ID,
            document_id=_DOCUMENT_ID,
            document_version_id=pending.id,
        )
        assert persisted_failed == failed
        assert persisted_failed is not None
        assert persisted_failed.status is DocumentVersionStatus.FAILED

    ready = failed.prepare_retry(now=_BASE_TIMESTAMP + timedelta(minutes=3)).mark_ready(
        chunk_count=2,
        embedding_input_tokens=120,
        embedding_estimated_cost_usd=Decimal("0"),
        embedding_pricing_catalog_version=("supportops-embedding-pricing-2026-08-01"),
        indexed_at=_BASE_TIMESTAMP + timedelta(minutes=4),
    )
    async with transaction_manager.transaction():
        await repository.update(ready)

    assert (
        await repository.get(
            workspace_id=_WORKSPACE_A_ID,
            document_id=_DOCUMENT_ID,
            document_version_id=pending.id,
        )
        == ready
    )


async def test_repository_rejects_rewriting_immutable_source_fields(
    postgresql_session: AsyncSession,
    clean_business_tables: None,
) -> None:
    await persist_owners(postgresql_session)
    repository = SqlAlchemyDocumentVersionRepository(postgresql_session)
    transaction_manager = SqlAlchemyTransactionManager(postgresql_session)
    version = create_version(version_id=UUID("09036916-84cf-4a58-bdf4-09bc52716ec5"))

    async with transaction_manager.transaction():
        await repository.add(version)

    replacement_content = "# Rewritten source\nDo not persist this.\n"
    rewritten = replace(
        version,
        content=replacement_content,
        content_sha256=compute_content_sha256(replacement_content),
        updated_at=_BASE_TIMESTAMP + timedelta(minutes=1),
    )

    with pytest.raises(
        ValueError,
        match=(r"Immutable document version fields do not match persisted state\."),
    ):
        async with transaction_manager.transaction():
            await repository.update(rewritten)

    assert (
        await repository.get(
            workspace_id=_WORKSPACE_A_ID,
            document_id=_DOCUMENT_ID,
            document_version_id=version.id,
        )
        == version
    )


async def test_repository_returns_next_number_and_lists_versions_in_order(
    postgresql_session: AsyncSession,
    clean_business_tables: None,
) -> None:
    await persist_owners(postgresql_session)
    repository = SqlAlchemyDocumentVersionRepository(postgresql_session)
    document_repository = SqlAlchemyDocumentRepository(postgresql_session)
    transaction_manager = SqlAlchemyTransactionManager(postgresql_session)
    versions = [
        create_version(
            version_id=UUID(f"00000000-0000-4000-8000-{index:012d}"),
            version_number=index,
            content=f"# Version {index}\nUnique content {index}.\n",
            created_at=_BASE_TIMESTAMP + timedelta(minutes=index),
        )
        for index in range(1, 6)
    ]

    async with transaction_manager.transaction():
        locked = await document_repository.get_for_update(
            workspace_id=_WORKSPACE_A_ID,
            document_id=_DOCUMENT_ID,
        )
        assert locked is not None
        assert (
            await repository.next_version_number(
                workspace_id=_WORKSPACE_A_ID,
                document_id=_DOCUMENT_ID,
            )
            == 1
        )
        for version in versions:
            await repository.add(version)
        assert (
            await repository.next_version_number(
                workspace_id=_WORKSPACE_A_ID,
                document_id=_DOCUMENT_ID,
            )
            == 6
        )

    first_page = await repository.list(
        workspace_id=_WORKSPACE_A_ID,
        document_id=_DOCUMENT_ID,
        limit=2,
    )
    second_page = await repository.list(
        workspace_id=_WORKSPACE_A_ID,
        document_id=_DOCUMENT_ID,
        limit=2,
        after_version_number=first_page[-1].version_number,
        after_document_version_id=first_page[-1].id,
    )
    third_page = await repository.list(
        workspace_id=_WORKSPACE_A_ID,
        document_id=_DOCUMENT_ID,
        limit=2,
        after_version_number=second_page[-1].version_number,
        after_document_version_id=second_page[-1].id,
    )
    observed = [*first_page, *second_page, *third_page]

    assert [version.version_number for version in observed] == [
        5,
        4,
        3,
        2,
        1,
    ]


@pytest.mark.parametrize(
    ("after_version_number", "after_document_version_id"),
    [
        (1, None),
        (
            None,
            UUID("00000000-0000-4000-8000-000000000001"),
        ),
    ],
)
async def test_repository_rejects_partial_version_keyset_position(
    postgresql_session: AsyncSession,
    clean_business_tables: None,
    after_version_number: int | None,
    after_document_version_id: UUID | None,
) -> None:
    repository = SqlAlchemyDocumentVersionRepository(postgresql_session)

    with pytest.raises(
        ValueError,
        match=(
            r"Document version pagination position requires both "
            r"version number and ID\."
        ),
    ):
        await repository.list(
            workspace_id=_WORKSPACE_A_ID,
            document_id=_DOCUMENT_ID,
            limit=20,
            after_version_number=after_version_number,
            after_document_version_id=after_document_version_id,
        )
