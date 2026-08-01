"""Integration tests for the PostgreSQL document-chunk repository."""

from datetime import UTC, datetime
from uuid import UUID

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from supportops.infrastructure.postgresql.transaction import (
    SqlAlchemyTransactionManager,
)
from supportops.modules.knowledge_documents.domain.models import (
    Document,
    DocumentChunk,
    DocumentMediaType,
    DocumentVersion,
    KnowledgeIndexProfile,
)
from supportops.modules.knowledge_documents.domain.repositories import (
    DocumentChunkConflictError,
)
from supportops.modules.knowledge_documents.infrastructure.repository import (
    SqlAlchemyDocumentChunkRepository,
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


async def persist_profiled_version(
    session: AsyncSession,
) -> DocumentVersion:
    """Persist one workspace, document, and profiled pending version."""

    workspace_repository = SqlAlchemyWorkspaceRepository(session)
    document_repository = SqlAlchemyDocumentRepository(session)
    version_repository = SqlAlchemyDocumentVersionRepository(session)
    transaction_manager = SqlAlchemyTransactionManager(session)

    workspace = Workspace(
        id=_WORKSPACE_A_ID,
        name="Platform Support",
        slug="platform-support",
        created_at=_BASE_TIMESTAMP,
        updated_at=_BASE_TIMESTAMP,
    )
    document = Document.create(
        document_id=_DOCUMENT_ID,
        workspace_id=_WORKSPACE_A_ID,
        title="Database Incident Runbook",
        now=_BASE_TIMESTAMP,
    )
    pending = DocumentVersion.create_pending(
        document_version_id=_VERSION_ID,
        workspace_id=_WORKSPACE_A_ID,
        document_id=_DOCUMENT_ID,
        version_number=1,
        media_type=DocumentMediaType.TEXT_MARKDOWN,
        content=(
            "# Connection exhaustion\n\n"
            "Restart the connection pool.\n\n"
            "Escalate if saturation continues.\n"
        ),
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

    async with transaction_manager.transaction():
        await workspace_repository.add(workspace)
        await document_repository.add(document)
        await version_repository.add(profiled)

    return profiled


def create_chunks(
    version: DocumentVersion,
) -> list[DocumentChunk]:
    """Create deterministic chunks in intentionally unsorted input order."""

    second = DocumentChunk.create(
        document_version=version,
        ordinal=1,
        section_path=("Connection exhaustion",),
        content="Escalate if saturation continues.",
        token_count=6,
        now=_BASE_TIMESTAMP,
    )
    first = DocumentChunk.create(
        document_version=version,
        ordinal=0,
        section_path=("Connection exhaustion",),
        content="Restart the connection pool.",
        token_count=6,
        now=_BASE_TIMESTAMP,
    )
    return [second, first]


async def test_repository_persists_counts_and_orders_chunks(
    postgresql_session: AsyncSession,
    clean_business_tables: None,
) -> None:
    version = await persist_profiled_version(postgresql_session)
    repository = SqlAlchemyDocumentChunkRepository(postgresql_session)
    transaction_manager = SqlAlchemyTransactionManager(postgresql_session)
    chunks = create_chunks(version)

    async with transaction_manager.transaction():
        await repository.add_many(chunks)

    persisted = await repository.list_by_version(
        workspace_id=_WORKSPACE_A_ID,
        document_id=_DOCUMENT_ID,
        document_version_id=_VERSION_ID,
    )

    assert [chunk.ordinal for chunk in persisted] == [0, 1]
    assert persisted == [chunks[1], chunks[0]]
    assert (
        await repository.count_by_version(
            workspace_id=_WORKSPACE_A_ID,
            document_id=_DOCUMENT_ID,
            document_version_id=_VERSION_ID,
        )
        == 2
    )


async def test_repository_treats_identical_chunk_rerun_as_idempotent(
    postgresql_session: AsyncSession,
    clean_business_tables: None,
) -> None:
    version = await persist_profiled_version(postgresql_session)
    repository = SqlAlchemyDocumentChunkRepository(postgresql_session)
    transaction_manager = SqlAlchemyTransactionManager(postgresql_session)
    chunks = create_chunks(version)

    async with transaction_manager.transaction():
        await repository.add_many(chunks)
    async with transaction_manager.transaction():
        await repository.add_many(chunks)

    assert (
        await repository.count_by_version(
            workspace_id=_WORKSPACE_A_ID,
            document_id=_DOCUMENT_ID,
            document_version_id=_VERSION_ID,
        )
        == 2
    )


async def test_repository_rejects_conflicting_chunk_at_same_ordinal(
    postgresql_session: AsyncSession,
    clean_business_tables: None,
) -> None:
    version = await persist_profiled_version(postgresql_session)
    repository = SqlAlchemyDocumentChunkRepository(postgresql_session)
    transaction_manager = SqlAlchemyTransactionManager(postgresql_session)
    original = DocumentChunk.create(
        document_version=version,
        ordinal=0,
        section_path=("Connection exhaustion",),
        content="Restart the connection pool.",
        token_count=6,
        now=_BASE_TIMESTAMP,
    )
    conflicting = DocumentChunk.create(
        document_version=version,
        ordinal=0,
        section_path=("Connection exhaustion",),
        content="Increase the connection limit immediately.",
        token_count=7,
        now=_BASE_TIMESTAMP,
    )

    async with transaction_manager.transaction():
        await repository.add_many([original])

    with pytest.raises(
        DocumentChunkConflictError,
        match=(r"Persisted chunk does not match the deterministic rerun\."),
    ):
        async with transaction_manager.transaction():
            await repository.add_many([conflicting])

    assert await repository.list_by_version(
        workspace_id=_WORKSPACE_A_ID,
        document_id=_DOCUMENT_ID,
        document_version_id=_VERSION_ID,
    ) == [original]


async def test_repository_rejects_mixed_version_batch_before_persistence(
    postgresql_session: AsyncSession,
    clean_business_tables: None,
) -> None:
    version = await persist_profiled_version(postgresql_session)
    repository = SqlAlchemyDocumentChunkRepository(postgresql_session)
    transaction_manager = SqlAlchemyTransactionManager(postgresql_session)
    first = DocumentChunk.create(
        document_version=version,
        ordinal=0,
        section_path=(),
        content="Restart the connection pool.",
        token_count=6,
        now=_BASE_TIMESTAMP,
    )
    profile = version.index_profile
    assert profile is not None

    other_version = DocumentVersion.create_pending(
        document_version_id=UUID("8ac21e0c-f869-4d84-a9e1-37d8074c9e54"),
        workspace_id=_WORKSPACE_A_ID,
        document_id=_DOCUMENT_ID,
        version_number=2,
        media_type=DocumentMediaType.TEXT_MARKDOWN,
        content="# Version 2\nEscalate to operations.\n",
        now=_BASE_TIMESTAMP,
    ).bind_index_profile(
        profile,
        now=_BASE_TIMESTAMP,
    )
    second = DocumentChunk.create(
        document_version=other_version,
        ordinal=1,
        section_path=(),
        content="Escalate to operations.",
        token_count=5,
        now=_BASE_TIMESTAMP,
    )

    with pytest.raises(
        ValueError,
        match=r"Chunk batch must belong to one version and profile\.",
    ):
        async with transaction_manager.transaction():
            await repository.add_many([first, second])

    assert (
        await repository.count_by_version(
            workspace_id=_WORKSPACE_A_ID,
            document_id=_DOCUMENT_ID,
            document_version_id=_VERSION_ID,
        )
        == 0
    )


async def test_repository_applies_workspace_and_document_scope_to_reads(
    postgresql_session: AsyncSession,
    clean_business_tables: None,
) -> None:
    version = await persist_profiled_version(postgresql_session)
    repository = SqlAlchemyDocumentChunkRepository(postgresql_session)
    transaction_manager = SqlAlchemyTransactionManager(postgresql_session)
    chunks = create_chunks(version)

    async with transaction_manager.transaction():
        await repository.add_many(chunks)

    assert (
        await repository.list_by_version(
            workspace_id=_WORKSPACE_B_ID,
            document_id=_DOCUMENT_ID,
            document_version_id=_VERSION_ID,
        )
        == []
    )
    assert (
        await repository.list_by_version(
            workspace_id=_WORKSPACE_A_ID,
            document_id=UUID("db00b4aa-4c17-4bf5-a333-226af35069c8"),
            document_version_id=_VERSION_ID,
        )
        == []
    )


async def test_repository_accepts_empty_chunk_batch_without_writes(
    postgresql_session: AsyncSession,
    clean_business_tables: None,
) -> None:
    repository = SqlAlchemyDocumentChunkRepository(postgresql_session)
    transaction_manager = SqlAlchemyTransactionManager(postgresql_session)

    async with transaction_manager.transaction():
        await repository.add_many([])

    assert (
        await repository.count_by_version(
            workspace_id=_WORKSPACE_A_ID,
            document_id=_DOCUMENT_ID,
            document_version_id=_VERSION_ID,
        )
        == 0
    )
