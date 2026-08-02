"""Integration tests for isolated authoritative semantic retrieval."""

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from hashlib import sha256
from uuid import UUID, uuid4

import pytest
from qdrant_client import AsyncQdrantClient, models
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from supportops.ai.embeddings.contracts import (
    EmbeddingOperation,
    EmbeddingRequest,
)
from supportops.ai.embeddings.mock import (
    MOCK_HASHING_EMBEDDING_MODEL,
    MockEmbeddingProvider,
)
from supportops.ai.embeddings.pricing import (
    EMBEDDING_PRICING_CATALOG_VERSION,
)
from supportops.core.settings import Settings
from supportops.infrastructure.postgresql.transaction import (
    SqlAlchemyTransactionManager,
)
from supportops.infrastructure.qdrant import (
    close_qdrant_client,
    create_qdrant_client,
)
from supportops.knowledge_index.vector_store.contracts import (
    KnowledgeCollectionProfile,
    KnowledgeVectorPoint,
    KnowledgeVersionProjection,
)
from supportops.knowledge_index.vector_store.qdrant import (
    CHUNK_ID_PAYLOAD,
    CHUNK_ORDINAL_PAYLOAD,
    CHUNKING_STRATEGY_PAYLOAD,
    CHUNKING_VERSION_PAYLOAD,
    CONTENT_SHA256_PAYLOAD,
    DOCUMENT_ID_PAYLOAD,
    DOCUMENT_VERSION_ID_PAYLOAD,
    MEDIA_TYPE_PAYLOAD,
    WORKSPACE_ID_PAYLOAD,
    QdrantKnowledgeVectorStore,
)
from supportops.knowledge_retrieval.contracts import (
    KnowledgeSearchRequest,
    KnowledgeSearchResult,
)
from supportops.knowledge_retrieval.postgresql import (
    SqlAlchemyActiveKnowledgeVersionResolver,
    SqlAlchemyKnowledgeChunkHydrator,
)
from supportops.knowledge_retrieval.qdrant import (
    QdrantKnowledgeVectorSearcher,
)
from supportops.knowledge_retrieval.service import (
    SearchKnowledge,
)
from supportops.modules.knowledge_documents.domain.models import (
    Document,
    DocumentChunk,
    DocumentMediaType,
    DocumentVersion,
    KnowledgeIndexProfile,
)
from supportops.modules.knowledge_documents.infrastructure.repository import (
    SqlAlchemyDocumentChunkRepository,
    SqlAlchemyDocumentRepository,
    SqlAlchemyDocumentVersionRepository,
)
from supportops.modules.workspaces.domain.models import (
    Workspace,
)
from supportops.modules.workspaces.infrastructure.repository import (
    SqlAlchemyWorkspaceRepository,
)

pytestmark = pytest.mark.integration

_WORKSPACE_A_ID = UUID("032c8c87-57cc-4d14-bfbd-04968b4e8cd4")
_WORKSPACE_B_ID = UUID("e3d19348-b66f-42ae-9192-feb93060df21")

_DATABASE_DOCUMENT_ID = UUID("276046a2-28ec-4cb1-8bb6-a2ff70f9064b")
_BILLING_DOCUMENT_ID = UUID("10bea98d-dfb1-4c33-884f-a4f36789a2ab")
_OTHER_WORKSPACE_DOCUMENT_ID = UUID("7309be38-3325-47ca-b144-0bedc300d52f")

_DATABASE_ACTIVE_VERSION_ID = UUID("09036916-84cf-4a58-bdf4-09bc52716ec5")
_DATABASE_INACTIVE_VERSION_ID = UUID("ad37db34-91b5-4638-b9a4-0d2230c3133f")
_BILLING_ACTIVE_VERSION_ID = UUID("8ac21e0c-f869-4d84-a9e1-37d8074c9e54")
_OTHER_WORKSPACE_VERSION_ID = UUID("ed80e017-f65d-40c3-980b-a04774923b64")

_CREATED_AT = datetime(
    2026,
    8,
    2,
    4,
    0,
    tzinfo=UTC,
)


@dataclass(frozen=True, slots=True)
class SeededDocumentVersion:
    """One persisted document version and its authoritative chunks."""

    document: Document
    active_document: Document | None
    version: DocumentVersion
    chunks: tuple[DocumentChunk, ...]


def create_profile(
    *,
    collection_name: str,
) -> KnowledgeIndexProfile:
    """Create one isolated mock retrieval profile."""

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


def create_ready_version(
    *,
    document: Document,
    document_version_id: UUID,
    version_number: int,
    source_content: str,
    chunk_contents: tuple[str, ...],
    profile: KnowledgeIndexProfile,
    active: bool,
    created_at: datetime,
) -> SeededDocumentVersion:
    """Create a ready version with deterministic authoritative chunks."""

    pending = DocumentVersion.create_pending(
        document_version_id=document_version_id,
        workspace_id=document.workspace_id,
        document_id=document.id,
        version_number=version_number,
        media_type=DocumentMediaType.TEXT_MARKDOWN,
        content=source_content,
        now=created_at,
    )
    profiled = pending.bind_index_profile(
        profile,
        now=created_at + timedelta(seconds=1),
    )

    chunks = tuple(
        DocumentChunk.create(
            document_version=profiled,
            ordinal=ordinal,
            section_path=("Recovery",),
            content=content,
            token_count=max(
                1,
                len(content.split()),
            ),
            now=created_at + timedelta(seconds=2),
        )
        for ordinal, content in enumerate(chunk_contents)
    )

    ready = profiled.mark_ready(
        chunk_count=len(chunks),
        embedding_input_tokens=sum(chunk.token_count for chunk in chunks),
        embedding_estimated_cost_usd=Decimal("0"),
        embedding_pricing_catalog_version=(EMBEDDING_PRICING_CATALOG_VERSION),
        indexed_at=created_at + timedelta(seconds=3),
    )

    active_document = (
        document.activate_version(
            ready,
            now=created_at + timedelta(seconds=4),
        )
        if active
        else None
    )

    return SeededDocumentVersion(
        document=document,
        active_document=active_document,
        version=ready,
        chunks=chunks,
    )


async def persist_graph(
    *,
    session_factory: async_sessionmaker[AsyncSession],
    workspaces: Sequence[Workspace],
    seeded_versions: Sequence[SeededDocumentVersion],
) -> None:
    """Persist workspaces, documents, versions, chunks, and activation."""

    documents_by_id: dict[UUID, Document] = {}
    active_documents_by_id: dict[UUID, Document] = {}
    chunks_by_version: dict[
        UUID,
        list[DocumentChunk],
    ] = defaultdict(list)

    for seeded in seeded_versions:
        documents_by_id.setdefault(
            seeded.document.id,
            seeded.document,
        )

        if seeded.active_document is not None:
            active_documents_by_id[seeded.document.id] = seeded.active_document

        chunks_by_version[seeded.version.id].extend(seeded.chunks)

    async with session_factory() as session:
        transaction_manager = SqlAlchemyTransactionManager(session)
        workspace_repository = SqlAlchemyWorkspaceRepository(session)
        document_repository = SqlAlchemyDocumentRepository(session)
        version_repository = SqlAlchemyDocumentVersionRepository(session)
        chunk_repository = SqlAlchemyDocumentChunkRepository(session)

        async with transaction_manager.transaction():
            for workspace in workspaces:
                await workspace_repository.add(workspace)

            for document in documents_by_id.values():
                await document_repository.add(document)

            for seeded in seeded_versions:
                await version_repository.add(seeded.version)
                await chunk_repository.add_many(seeded.chunks)

            for active_document in active_documents_by_id.values():
                await document_repository.update(active_document)


async def embed_chunks(
    *,
    provider: MockEmbeddingProvider,
    profile: KnowledgeIndexProfile,
    chunks: tuple[DocumentChunk, ...],
) -> tuple[tuple[float, ...], ...]:
    """Generate deterministic indexing vectors for authoritative chunks."""

    response = await provider.embed(
        EmbeddingRequest(
            operation=(EmbeddingOperation.KNOWLEDGE_INDEXING),
            model=profile.embedding_model,
            inputs=tuple(chunk.content for chunk in chunks),
            dimensions=profile.embedding_dimensions,
            timeout_seconds=12,
            metadata={
                "purpose": "integration-test-indexing",
            },
        )
    )

    return response.embeddings


async def project_version(
    *,
    vector_store: QdrantKnowledgeVectorStore,
    provider: MockEmbeddingProvider,
    profile: KnowledgeIndexProfile,
    seeded: SeededDocumentVersion,
) -> None:
    """Project one persisted version into Qdrant."""

    vectors = await embed_chunks(
        provider=provider,
        profile=profile,
        chunks=seeded.chunks,
    )

    await vector_store.upsert_version_points(
        profile=KnowledgeCollectionProfile(
            collection_name=(profile.knowledge_collection),
            vector_name=(profile.knowledge_vector_name),
            dimensions=(profile.embedding_dimensions),
        ),
        projection=KnowledgeVersionProjection(
            workspace_id=(seeded.version.workspace_id),
            document_id=(seeded.version.document_id),
            document_version_id=(seeded.version.id),
        ),
        points=tuple(
            KnowledgeVectorPoint.from_chunk(
                chunk=chunk,
                media_type=seeded.version.media_type,
                vector=vector,
            )
            for chunk, vector in zip(
                seeded.chunks,
                vectors,
                strict=True,
            )
        ),
    )


async def execute_search(
    *,
    session_factory: async_sessionmaker[AsyncSession],
    client: AsyncQdrantClient,
    provider: MockEmbeddingProvider,
    profile: KnowledgeIndexProfile,
    request: KnowledgeSearchRequest,
) -> KnowledgeSearchResult:
    """Execute semantic retrieval through real persistence adapters."""

    vector_store = QdrantKnowledgeVectorStore(client=client)
    vector_searcher = QdrantKnowledgeVectorSearcher(
        client=client,
        collection_guard=vector_store,
    )

    async with session_factory() as session:
        service = SearchKnowledge(
            active_version_resolver=(SqlAlchemyActiveKnowledgeVersionResolver(session)),
            chunk_hydrator=(SqlAlchemyKnowledgeChunkHydrator(session)),
            embedding_provider=provider,
            vector_searcher=vector_searcher,
            index_profile=profile,
            embedding_timeout_seconds=12,
        )

        return await service.execute(request)


async def delete_collection_if_present(
    *,
    client: AsyncQdrantClient,
    collection_name: str,
) -> None:
    """Delete one isolated integration collection."""

    if await client.collection_exists(collection_name=collection_name):
        await client.delete_collection(collection_name=collection_name)


async def test_retrieval_uses_only_active_versions_in_requested_workspace(
    postgresql_session_factory: async_sessionmaker[AsyncSession],
    integration_settings: Settings,
    clean_business_tables: None,
) -> None:
    """Exclude inactive versions and all cross-workspace knowledge."""

    del clean_business_tables

    collection_name = f"supportops-retrieval-isolation-{uuid4().hex}"
    profile = create_profile(collection_name=collection_name)

    workspace_a = Workspace.create(
        workspace_id=_WORKSPACE_A_ID,
        name="Platform Support",
        slug="platform-support",
        now=_CREATED_AT,
    )
    workspace_b = Workspace.create(
        workspace_id=_WORKSPACE_B_ID,
        name="External Support",
        slug="external-support",
        now=_CREATED_AT,
    )

    database_document = Document.create(
        document_id=_DATABASE_DOCUMENT_ID,
        workspace_id=_WORKSPACE_A_ID,
        title="Database Recovery Runbook",
        external_reference="database-recovery",
        now=_CREATED_AT,
    )
    billing_document = Document.create(
        document_id=_BILLING_DOCUMENT_ID,
        workspace_id=_WORKSPACE_A_ID,
        title="Billing Recovery Runbook",
        external_reference="billing-recovery",
        now=_CREATED_AT,
    )
    other_workspace_document = Document.create(
        document_id=_OTHER_WORKSPACE_DOCUMENT_ID,
        workspace_id=_WORKSPACE_B_ID,
        title="Private Database Runbook",
        external_reference="private-database-recovery",
        now=_CREATED_AT,
    )

    active_database = create_ready_version(
        document=database_document,
        document_version_id=(_DATABASE_ACTIVE_VERSION_ID),
        version_number=1,
        source_content=("# Database recovery\n\nRestart the primary database connection pool."),
        chunk_contents=("Database recovery requires restarting the primary connection pool.",),
        profile=profile,
        active=True,
        created_at=_CREATED_AT,
    )
    inactive_database = create_ready_version(
        document=database_document,
        document_version_id=(_DATABASE_INACTIVE_VERSION_ID),
        version_number=2,
        source_content=("# Database recovery\n\nEmergency database recovery instructions."),
        chunk_contents=(
            "Database recovery database recovery database recovery emergency instructions.",
        ),
        profile=profile,
        active=False,
        created_at=(_CREATED_AT + timedelta(minutes=1)),
    )
    active_billing = create_ready_version(
        document=billing_document,
        document_version_id=(_BILLING_ACTIVE_VERSION_ID),
        version_number=1,
        source_content=("# Billing recovery\n\nReconcile the failed invoice."),
        chunk_contents=("Billing recovery requires reconciling the failed invoice.",),
        profile=profile,
        active=True,
        created_at=(_CREATED_AT + timedelta(minutes=2)),
    )
    other_workspace_active = create_ready_version(
        document=other_workspace_document,
        document_version_id=(_OTHER_WORKSPACE_VERSION_ID),
        version_number=1,
        source_content=("# Database recovery\n\nPrivate customer database recovery instructions."),
        chunk_contents=("Database recovery database recovery private customer instructions.",),
        profile=profile,
        active=True,
        created_at=(_CREATED_AT + timedelta(minutes=3)),
    )

    seeded_versions = (
        active_database,
        inactive_database,
        active_billing,
        other_workspace_active,
    )

    await persist_graph(
        session_factory=postgresql_session_factory,
        workspaces=(
            workspace_a,
            workspace_b,
        ),
        seeded_versions=seeded_versions,
    )

    provider = MockEmbeddingProvider(
        model=profile.embedding_model,
        dimensions=profile.embedding_dimensions,
    )
    client = create_qdrant_client(integration_settings)
    vector_store = QdrantKnowledgeVectorStore(client=client)

    try:
        for seeded in seeded_versions:
            await project_version(
                vector_store=vector_store,
                provider=provider,
                profile=profile,
                seeded=seeded,
            )

        result = await execute_search(
            session_factory=(postgresql_session_factory),
            client=client,
            provider=provider,
            profile=profile,
            request=KnowledgeSearchRequest(
                workspace_id=_WORKSPACE_A_ID,
                query="database recovery",
                top_k=10,
            ),
        )

        assert result.searched_version_count == 2
        assert {evidence.citation.document_id for evidence in result.evidence} == {
            _DATABASE_DOCUMENT_ID,
            _BILLING_DOCUMENT_ID,
        }
        assert {evidence.citation.document_version_id for evidence in result.evidence} == {
            _DATABASE_ACTIVE_VERSION_ID,
            _BILLING_ACTIVE_VERSION_ID,
        }
        assert all(
            evidence.citation.workspace_id == _WORKSPACE_A_ID for evidence in result.evidence
        )
        assert all(
            evidence.citation.document_version_id != _DATABASE_INACTIVE_VERSION_ID
            for evidence in result.evidence
        )
        assert all(
            evidence.citation.document_id != _OTHER_WORKSPACE_DOCUMENT_ID
            for evidence in result.evidence
        )

        filtered_result = await execute_search(
            session_factory=(postgresql_session_factory),
            client=client,
            provider=provider,
            profile=profile,
            request=KnowledgeSearchRequest(
                workspace_id=_WORKSPACE_A_ID,
                query="database recovery",
                top_k=5,
                document_ids=(_DATABASE_DOCUMENT_ID,),
            ),
        )

        assert filtered_result.searched_version_count == 1
        assert len(filtered_result.evidence) == 1
        assert filtered_result.evidence[0].citation.document_id == _DATABASE_DOCUMENT_ID
        assert (
            filtered_result.evidence[0].citation.document_version_id == _DATABASE_ACTIVE_VERSION_ID
        )

        invocation_count_before_empty_scope = provider.invocation_count

        cross_workspace_filter_result = await execute_search(
            session_factory=(postgresql_session_factory),
            client=client,
            provider=provider,
            profile=profile,
            request=KnowledgeSearchRequest(
                workspace_id=_WORKSPACE_A_ID,
                query="private database recovery",
                top_k=5,
                document_ids=(_OTHER_WORKSPACE_DOCUMENT_ID,),
            ),
        )

        assert cross_workspace_filter_result.searched_version_count == 0
        assert cross_workspace_filter_result.evidence == ()
        assert provider.invocation_count == invocation_count_before_empty_scope
    finally:
        await provider.close()
        await delete_collection_if_present(
            client=client,
            collection_name=collection_name,
        )
        await close_qdrant_client(client)


async def test_retrieval_discards_stale_qdrant_provenance(
    postgresql_session_factory: async_sessionmaker[AsyncSession],
    integration_settings: Settings,
    clean_business_tables: None,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Discard a high-scoring point whose hash differs from PostgreSQL."""

    del clean_business_tables

    collection_name = f"supportops-retrieval-consistency-{uuid4().hex}"
    profile = create_profile(collection_name=collection_name)

    workspace = Workspace.create(
        workspace_id=_WORKSPACE_A_ID,
        name="Platform Support",
        slug="platform-support",
        now=_CREATED_AT,
    )
    document = Document.create(
        document_id=_DATABASE_DOCUMENT_ID,
        workspace_id=_WORKSPACE_A_ID,
        title="Database Recovery Runbook",
        external_reference="database-recovery",
        now=_CREATED_AT,
    )
    active_version = create_ready_version(
        document=document,
        document_version_id=(_DATABASE_ACTIVE_VERSION_ID),
        version_number=1,
        source_content=(
            "# Database recovery\n\n"
            "Restart the primary connection pool.\n\n"
            "Verify replication health."
        ),
        chunk_contents=(
            "Restart the primary database connection pool.",
            "Verify database replication health after recovery.",
        ),
        profile=profile,
        active=True,
        created_at=_CREATED_AT,
    )

    await persist_graph(
        session_factory=postgresql_session_factory,
        workspaces=(workspace,),
        seeded_versions=(active_version,),
    )

    provider = MockEmbeddingProvider(
        model=profile.embedding_model,
        dimensions=profile.embedding_dimensions,
    )
    client = create_qdrant_client(integration_settings)
    vector_store = QdrantKnowledgeVectorStore(client=client)

    try:
        await project_version(
            vector_store=vector_store,
            provider=provider,
            profile=profile,
            seeded=active_version,
        )

        query_response = await provider.embed(
            EmbeddingRequest(
                operation=(EmbeddingOperation.KNOWLEDGE_QUERY),
                model=profile.embedding_model,
                inputs=("database recovery",),
                dimensions=(profile.embedding_dimensions),
                timeout_seconds=12,
                metadata={
                    "purpose": ("integration-test-corruption"),
                },
            )
        )
        query_vector = query_response.embeddings[0]
        corrupted_chunk = active_version.chunks[0]

        update_result = await client.upsert(
            collection_name=collection_name,
            points=[
                models.PointStruct(
                    id=str(corrupted_chunk.id),
                    vector={profile.knowledge_vector_name: (list(query_vector))},
                    payload={
                        WORKSPACE_ID_PAYLOAD: str(corrupted_chunk.workspace_id),
                        DOCUMENT_ID_PAYLOAD: str(corrupted_chunk.document_id),
                        DOCUMENT_VERSION_ID_PAYLOAD: str(corrupted_chunk.document_version_id),
                        CHUNK_ID_PAYLOAD: str(corrupted_chunk.id),
                        CHUNK_ORDINAL_PAYLOAD: (corrupted_chunk.ordinal),
                        CONTENT_SHA256_PAYLOAD: sha256(b"stale-derived-projection").hexdigest(),
                        MEDIA_TYPE_PAYLOAD: (DocumentMediaType.TEXT_MARKDOWN.value),
                        CHUNKING_STRATEGY_PAYLOAD: (corrupted_chunk.chunking_strategy),
                        CHUNKING_VERSION_PAYLOAD: (corrupted_chunk.chunking_version),
                    },
                )
            ],
            wait=True,
        )

        assert update_result.status is models.UpdateStatus.COMPLETED

        result = await execute_search(
            session_factory=(postgresql_session_factory),
            client=client,
            provider=provider,
            profile=profile,
            request=KnowledgeSearchRequest(
                workspace_id=_WORKSPACE_A_ID,
                query="database recovery",
                top_k=2,
                document_ids=(_DATABASE_DOCUMENT_ID,),
            ),
        )

        valid_chunk = active_version.chunks[1]

        assert result.searched_version_count == 1
        assert len(result.evidence) == 1
        assert result.evidence[0].rank == 1
        assert result.evidence[0].citation.chunk_id == valid_chunk.id
        assert result.evidence[0].content == valid_chunk.content
        assert result.evidence[0].content_sha256 == valid_chunk.content_sha256
        assert result.evidence[0].citation.chunk_id != corrupted_chunk.id
        assert corrupted_chunk.content not in tuple(
            evidence.content for evidence in result.evidence
        )

        assert "Discarded inconsistent semantic knowledge candidate" in caplog.text
        assert corrupted_chunk.content not in caplog.text
        assert valid_chunk.content not in caplog.text
    finally:
        await provider.close()
        await delete_collection_if_present(
            client=client,
            collection_name=collection_name,
        )
        await close_qdrant_client(client)
