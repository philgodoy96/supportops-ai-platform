"""Integration tests for retry-safe knowledge indexing orchestration."""

from collections.abc import Sequence
from datetime import UTC, datetime
from uuid import UUID, uuid4

import pytest
from qdrant_client import AsyncQdrantClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from supportops.ai.embeddings.contracts import (
    EmbeddingProvider,
)
from supportops.ai.embeddings.mock import (
    MOCK_HASHING_EMBEDDING_MODEL,
    MockEmbeddingProvider,
)
from supportops.ai.embeddings.pricing import (
    DEFAULT_EMBEDDING_PRICING_CATALOG,
)
from supportops.core.settings import Settings
from supportops.infrastructure.postgresql.transaction import (
    SqlAlchemyTransactionManager,
)
from supportops.infrastructure.qdrant import (
    close_qdrant_client,
    create_qdrant_client,
)
from supportops.knowledge_index.chunking.contracts import (
    ChunkingPolicy,
)
from supportops.knowledge_index.chunking.markdown import (
    MarkdownTokenChunker,
)
from supportops.knowledge_index.indexing.results import (
    IndexDocumentVersionResult,
)
from supportops.knowledge_index.indexing.service import (
    IndexDocumentVersion,
)
from supportops.knowledge_index.vector_store.contracts import (
    KnowledgeCollectionProfile,
    KnowledgeVectorPoint,
    KnowledgeVectorStore,
    KnowledgeVectorStoreUnavailableError,
    KnowledgeVersionProjection,
)
from supportops.knowledge_index.vector_store.qdrant import (
    QdrantKnowledgeVectorStore,
)
from supportops.modules.knowledge_documents.domain.models import (
    Document,
    DocumentChunk,
    DocumentMediaType,
    DocumentVersion,
    DocumentVersionStatus,
    KnowledgeIndexProfile,
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

_WORKSPACE_ID = UUID("032c8c87-57cc-4d14-bfbd-04968b4e8cd4")
_DOCUMENT_ID = UUID("276046a2-28ec-4cb1-8bb6-a2ff70f9064b")
_VERSION_ID = UUID("09036916-84cf-4a58-bdf4-09bc52716ec5")

_SOURCE_CONTENT = (
    "# Database recovery\n\n"
    + (
        "Restart the database connection pool and verify "
        "that the primary node accepts new connections. "
    )
    * 18
    + "\n\n"
    + "## Validation\n\n"
    + ("Confirm replication health, application latency, and successful transaction processing. ")
    * 14
    + "\n"
)


class CharacterTokenizer:
    """Deterministic offline tokenizer with one token per character."""

    @property
    def encoding_name(self) -> str:
        """Return the persisted tokenizer identity."""

        return "cl100k_base"

    def encode(self, text: str) -> tuple[int, ...]:
        """Encode every Unicode code point."""

        return tuple(ord(character) for character in text)

    def decode(
        self,
        tokens: Sequence[int],
    ) -> str:
        """Decode Unicode code points."""

        return "".join(chr(token) for token in tokens)


class FailAfterPartialProjectionStore:
    """Persist a partial projection once, then expose a retryable failure."""

    def __init__(
        self,
        *,
        delegate: QdrantKnowledgeVectorStore,
    ) -> None:
        self._delegate = delegate
        self.failed_once = False
        self.partial_point_count = 0

    async def ensure_collection(
        self,
        profile: KnowledgeCollectionProfile,
    ) -> None:
        """Delegate collection creation and compatibility validation."""

        await self._delegate.ensure_collection(profile)

    async def upsert_version_points(
        self,
        *,
        profile: KnowledgeCollectionProfile,
        projection: KnowledgeVersionProjection,
        points: Sequence[KnowledgeVectorPoint],
    ) -> None:
        """Fail after writing only part of the first projection."""

        normalized_points = tuple(points)

        if not self.failed_once:
            self.failed_once = True
            partial_size = max(
                1,
                len(normalized_points) // 2,
            )
            partial_points = normalized_points[:partial_size]
            self.partial_point_count = len(partial_points)

            await self._delegate.upsert_version_points(
                profile=profile,
                projection=projection,
                points=partial_points,
            )

            raise KnowledgeVectorStoreUnavailableError(
                "The knowledge vector store became unavailable after a partial projection."
            )

        await self._delegate.upsert_version_points(
            profile=profile,
            projection=projection,
            points=normalized_points,
        )

    async def count_version_points(
        self,
        *,
        profile: KnowledgeCollectionProfile,
        projection: KnowledgeVersionProjection,
    ) -> int:
        """Delegate exact projection counting."""

        return await self._delegate.count_version_points(
            profile=profile,
            projection=projection,
        )


def create_index_profile(
    *,
    collection_name: str,
) -> KnowledgeIndexProfile:
    """Create one isolated mock embedding profile."""

    return KnowledgeIndexProfile(
        chunking_strategy="markdown-token",
        chunking_version="v1",
        tokenizer_encoding="cl100k_base",
        embedding_provider="mock",
        embedding_model=MOCK_HASHING_EMBEDDING_MODEL,
        embedding_dimensions=64,
        knowledge_collection=collection_name,
        knowledge_vector_name="dense",
    )


def create_collection_profile(
    profile: KnowledgeIndexProfile,
) -> KnowledgeCollectionProfile:
    """Create the Qdrant profile from persisted indexing identity."""

    return KnowledgeCollectionProfile(
        collection_name=profile.knowledge_collection,
        vector_name=profile.knowledge_vector_name,
        dimensions=profile.embedding_dimensions,
    )


def create_projection(
    version: DocumentVersion,
) -> KnowledgeVersionProjection:
    """Create the owned Qdrant projection identity."""

    return KnowledgeVersionProjection(
        workspace_id=version.workspace_id,
        document_id=version.document_id,
        document_version_id=version.id,
    )


def create_indexing_service(
    *,
    session: AsyncSession,
    profile: KnowledgeIndexProfile,
    embedding_provider: EmbeddingProvider,
    vector_store: KnowledgeVectorStore,
) -> IndexDocumentVersion:
    """Compose the service around real persistence adapters."""

    policy = ChunkingPolicy()

    return IndexDocumentVersion(
        version_repository=(SqlAlchemyDocumentVersionRepository(session)),
        chunk_repository=(SqlAlchemyDocumentChunkRepository(session)),
        transaction_manager=(SqlAlchemyTransactionManager(session)),
        chunker=MarkdownTokenChunker(
            policy=policy,
            tokenizer=CharacterTokenizer(),
        ),
        embedding_provider=embedding_provider,
        vector_store=vector_store,
        pricing_catalog=(DEFAULT_EMBEDDING_PRICING_CATALOG),
        index_profile=profile,
        embedding_timeout_seconds=12,
        embedding_batch_size=64,
    )


async def seed_pending_version(
    session_factory: async_sessionmaker[AsyncSession],
) -> DocumentVersion:
    """Persist one workspace, document, and pending source version."""

    created_at = datetime.now(UTC)
    workspace = Workspace.create(
        workspace_id=_WORKSPACE_ID,
        name="Platform Support",
        slug="platform-support",
        now=created_at,
    )
    document = Document.create(
        document_id=_DOCUMENT_ID,
        workspace_id=_WORKSPACE_ID,
        title="Database Recovery Runbook",
        external_reference="database-recovery-runbook",
        now=created_at,
    )
    version = DocumentVersion.create_pending(
        document_version_id=_VERSION_ID,
        workspace_id=_WORKSPACE_ID,
        document_id=_DOCUMENT_ID,
        version_number=1,
        media_type=DocumentMediaType.TEXT_MARKDOWN,
        content=_SOURCE_CONTENT,
        now=created_at,
    )

    async with session_factory() as session:
        transaction_manager = SqlAlchemyTransactionManager(session)

        async with transaction_manager.transaction():
            await SqlAlchemyWorkspaceRepository(session).add(workspace)
            await SqlAlchemyDocumentRepository(session).add(document)
            await SqlAlchemyDocumentVersionRepository(session).add(version)

    return version


async def load_authoritative_state(
    session_factory: async_sessionmaker[AsyncSession],
) -> tuple[
    Document,
    DocumentVersion,
    tuple[DocumentChunk, ...],
]:
    """Load the authoritative document, version, and ordered chunks."""

    async with session_factory() as session:
        document = await SqlAlchemyDocumentRepository(session).get(
            workspace_id=_WORKSPACE_ID,
            document_id=_DOCUMENT_ID,
        )
        version = await SqlAlchemyDocumentVersionRepository(session).get(
            workspace_id=_WORKSPACE_ID,
            document_id=_DOCUMENT_ID,
            document_version_id=_VERSION_ID,
        )
        chunks = await SqlAlchemyDocumentChunkRepository(session).list_by_version(
            workspace_id=_WORKSPACE_ID,
            document_id=_DOCUMENT_ID,
            document_version_id=_VERSION_ID,
        )

    assert document is not None
    assert version is not None

    return (
        document,
        version,
        tuple(chunks),
    )


async def execute_indexing(
    *,
    session_factory: async_sessionmaker[AsyncSession],
    profile: KnowledgeIndexProfile,
    embedding_provider: EmbeddingProvider,
    vector_store: KnowledgeVectorStore,
) -> IndexDocumentVersionResult:
    """Execute indexing through a fresh SQLAlchemy session."""

    async with session_factory() as session:
        service = create_indexing_service(
            session=session,
            profile=profile,
            embedding_provider=embedding_provider,
            vector_store=vector_store,
        )

        return await service.execute(
            workspace_id=_WORKSPACE_ID,
            document_id=_DOCUMENT_ID,
            document_version_id=_VERSION_ID,
        )


async def retrieve_projected_chunk_ids(
    *,
    client: AsyncQdrantClient,
    profile: KnowledgeIndexProfile,
    chunks: tuple[DocumentChunk, ...],
) -> set[str]:
    """Retrieve and return the deterministic point identifiers."""

    records = await client.retrieve(
        collection_name=profile.knowledge_collection,
        ids=[str(chunk.id) for chunk in chunks],
        with_payload=True,
        with_vectors=False,
    )

    return {str(record.id) for record in records}


async def delete_collection_if_present(
    *,
    client: AsyncQdrantClient,
    collection_name: str,
) -> None:
    """Delete one isolated integration collection."""

    if await client.collection_exists(collection_name=collection_name):
        await client.delete_collection(collection_name=collection_name)


async def test_ready_rerun_does_not_duplicate_chunks_or_vectors(
    postgresql_session_factory: async_sessionmaker[AsyncSession],
    integration_settings: Settings,
    clean_business_tables: None,
) -> None:
    """Keep a successfully indexed ready version stable on rerun."""

    del clean_business_tables

    pending = await seed_pending_version(postgresql_session_factory)
    profile = create_index_profile(
        collection_name=(f"supportops-knowledge-idempotency-{uuid4().hex}")
    )
    provider = MockEmbeddingProvider(
        model=profile.embedding_model,
        dimensions=profile.embedding_dimensions,
    )
    client = create_qdrant_client(integration_settings)
    vector_store = QdrantKnowledgeVectorStore(
        client=client,
        batch_size=2,
    )

    try:
        first_result = await execute_indexing(
            session_factory=(postgresql_session_factory),
            profile=profile,
            embedding_provider=provider,
            vector_store=vector_store,
        )

        assert not first_result.already_ready
        assert first_result.version.status is (DocumentVersionStatus.READY)
        assert first_result.chunk_count > 1
        assert first_result.version.id == pending.id

        (
            first_document,
            first_version,
            first_chunks,
        ) = await load_authoritative_state(postgresql_session_factory)

        assert first_document.active_version_id is None
        assert first_version.status is (DocumentVersionStatus.READY)
        assert first_version.chunk_count == len(first_chunks)
        assert len(first_chunks) == (first_result.chunk_count)

        first_chunk_ids = tuple(chunk.id for chunk in first_chunks)
        first_invocation_count = provider.invocation_count

        collection_profile = create_collection_profile(profile)
        projection = create_projection(first_version)

        assert await vector_store.count_version_points(
            profile=collection_profile,
            projection=projection,
        ) == len(first_chunks)

        projected_ids = await retrieve_projected_chunk_ids(
            client=client,
            profile=profile,
            chunks=first_chunks,
        )
        assert projected_ids == {str(chunk_id) for chunk_id in first_chunk_ids}

        second_result = await execute_indexing(
            session_factory=(postgresql_session_factory),
            profile=profile,
            embedding_provider=provider,
            vector_store=vector_store,
        )

        assert second_result.already_ready
        assert second_result.version == first_version
        assert provider.invocation_count == first_invocation_count

        (
            second_document,
            second_version,
            second_chunks,
        ) = await load_authoritative_state(postgresql_session_factory)

        assert second_document.active_version_id is None
        assert second_version == first_version
        assert tuple(chunk.id for chunk in second_chunks) == first_chunk_ids
        assert len(second_chunks) == len(first_chunks)

        assert await vector_store.count_version_points(
            profile=collection_profile,
            projection=projection,
        ) == len(first_chunks)
    finally:
        await provider.close()
        await delete_collection_if_present(
            client=client,
            collection_name=profile.knowledge_collection,
        )
        await close_qdrant_client(client)


async def test_partial_projection_failure_recovers_without_duplicates(
    postgresql_session_factory: async_sessionmaker[AsyncSession],
    integration_settings: Settings,
    clean_business_tables: None,
) -> None:
    """Complete a failed partial projection through a deterministic retry."""

    del clean_business_tables

    await seed_pending_version(postgresql_session_factory)
    profile = create_index_profile(collection_name=(f"supportops-knowledge-recovery-{uuid4().hex}"))
    provider = MockEmbeddingProvider(
        model=profile.embedding_model,
        dimensions=profile.embedding_dimensions,
    )
    client = create_qdrant_client(integration_settings)
    delegate_store = QdrantKnowledgeVectorStore(
        client=client,
        batch_size=2,
    )
    failing_store = FailAfterPartialProjectionStore(delegate=delegate_store)
    collection_profile = create_collection_profile(profile)

    try:
        with pytest.raises(
            KnowledgeVectorStoreUnavailableError,
            match="partial projection",
        ):
            await execute_indexing(
                session_factory=(postgresql_session_factory),
                profile=profile,
                embedding_provider=provider,
                vector_store=failing_store,
            )

        (
            failed_document,
            failed_version,
            failed_chunks,
        ) = await load_authoritative_state(postgresql_session_factory)

        assert failed_document.active_version_id is None
        assert failed_version.status is (DocumentVersionStatus.FAILED)
        assert failed_version.last_error_code == ("knowledge_vector_store_unavailable")
        assert failed_version.embedding_input_tokens is None
        assert failed_version.indexed_at is None
        assert failed_version.chunk_count == len(failed_chunks)
        assert len(failed_chunks) > 1
        assert failing_store.failed_once
        assert 0 < failing_store.partial_point_count < len(failed_chunks)

        projection = create_projection(failed_version)
        partial_projection_count = await delegate_store.count_version_points(
            profile=collection_profile,
            projection=projection,
        )

        assert partial_projection_count == (failing_store.partial_point_count)

        failed_chunk_ids = tuple(chunk.id for chunk in failed_chunks)
        first_invocation_count = provider.invocation_count

        retry_result = await execute_indexing(
            session_factory=(postgresql_session_factory),
            profile=profile,
            embedding_provider=provider,
            vector_store=failing_store,
        )

        assert not retry_result.already_ready
        assert retry_result.version.status is (DocumentVersionStatus.READY)
        assert retry_result.chunk_count == len(failed_chunks)
        assert provider.invocation_count > first_invocation_count

        (
            ready_document,
            ready_version,
            ready_chunks,
        ) = await load_authoritative_state(postgresql_session_factory)

        assert ready_document.active_version_id is None
        assert ready_version.status is (DocumentVersionStatus.READY)
        assert ready_version.last_error_code is None
        assert ready_version.indexed_at is not None
        assert ready_version.chunk_count == len(ready_chunks)
        assert tuple(chunk.id for chunk in ready_chunks) == failed_chunk_ids

        assert await delegate_store.count_version_points(
            profile=collection_profile,
            projection=create_projection(ready_version),
        ) == len(ready_chunks)

        projected_ids = await retrieve_projected_chunk_ids(
            client=client,
            profile=profile,
            chunks=ready_chunks,
        )
        assert projected_ids == {str(chunk_id) for chunk_id in failed_chunk_ids}
    finally:
        await provider.close()
        await delete_collection_if_present(
            client=client,
            collection_name=profile.knowledge_collection,
        )
        await close_qdrant_client(client)
