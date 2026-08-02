"""Unit tests for explicit document-version indexing."""

from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from uuid import UUID

import pytest

from supportops.ai.embeddings.contracts import (
    EmbeddingProviderResponse,
    EmbeddingRequest,
    EmbeddingTokenUsage,
)
from supportops.ai.embeddings.errors import (
    EmbeddingInvalidResponseError,
    EmbeddingTimeoutError,
)
from supportops.ai.embeddings.pricing import (
    DEFAULT_EMBEDDING_PRICING_CATALOG,
)
from supportops.knowledge_index.chunking.contracts import (
    ChunkingPolicy,
)
from supportops.knowledge_index.indexing.errors import (
    KnowledgeDocumentVersionNotFoundError,
    KnowledgeIndexProfileMismatchError,
    KnowledgeProjectionCountMismatchError,
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
    KnowledgeVersionProjection,
)
from supportops.modules.knowledge_documents.domain.models import (
    DocumentChunk,
    DocumentMediaType,
    DocumentVersion,
    DocumentVersionStatus,
    KnowledgeIndexProfile,
)
from supportops.modules.knowledge_documents.domain.repositories import (
    DocumentChunkConflictError,
)

_WORKSPACE_ID = UUID("032c8c87-57cc-4d14-bfbd-04968b4e8cd4")
_DOCUMENT_ID = UUID("276046a2-28ec-4cb1-8bb6-a2ff70f9064b")
_VERSION_ID = UUID("09036916-84cf-4a58-bdf4-09bc52716ec5")
_CREATED_AT = datetime(
    2026,
    8,
    2,
    1,
    0,
    tzinfo=UTC,
)


class AdvancingClock:
    """Return monotonically increasing deterministic timestamps."""

    def __init__(self) -> None:
        self._current = _CREATED_AT

    def __call__(self) -> datetime:
        self._current += timedelta(seconds=1)
        return self._current


class FakeTransactionManager:
    """Record transaction completion and rollback."""

    def __init__(self) -> None:
        self.completed = 0
        self.rolled_back = 0

    @asynccontextmanager
    async def transaction(self) -> AsyncIterator[None]:
        try:
            yield
        except Exception:
            self.rolled_back += 1
            raise
        else:
            self.completed += 1


class FakeVersionRepository:
    """In-memory document-version repository fake."""

    def __init__(
        self,
        version: DocumentVersion | None,
    ) -> None:
        self.version = version
        self.updates: list[DocumentVersion] = []

    async def add(
        self,
        version: DocumentVersion,
    ) -> None:
        raise AssertionError("add must not be called")

    async def update(
        self,
        version: DocumentVersion,
    ) -> None:
        self.version = version
        self.updates.append(version)

    async def get(
        self,
        *,
        workspace_id: UUID,
        document_id: UUID,
        document_version_id: UUID,
    ) -> DocumentVersion | None:
        return self._scoped(
            workspace_id=workspace_id,
            document_id=document_id,
            document_version_id=document_version_id,
        )

    async def get_for_update(
        self,
        *,
        workspace_id: UUID,
        document_id: UUID,
        document_version_id: UUID,
    ) -> DocumentVersion | None:
        return self._scoped(
            workspace_id=workspace_id,
            document_id=document_id,
            document_version_id=document_version_id,
        )

    async def get_by_content_hash(
        self,
        *,
        workspace_id: UUID,
        document_id: UUID,
        content_sha256: str,
    ) -> DocumentVersion | None:
        raise AssertionError("get_by_content_hash must not be called")

    async def next_version_number(
        self,
        *,
        workspace_id: UUID,
        document_id: UUID,
    ) -> int:
        raise AssertionError("next_version_number must not be called")

    async def list(
        self,
        *,
        workspace_id: UUID,
        document_id: UUID,
        limit: int,
        after_version_number: int | None = None,
        after_document_version_id: UUID | None = None,
    ) -> Sequence[DocumentVersion]:
        raise AssertionError("list must not be called")

    def _scoped(
        self,
        *,
        workspace_id: UUID,
        document_id: UUID,
        document_version_id: UUID,
    ) -> DocumentVersion | None:
        if self.version is None:
            return None
        if (
            self.version.workspace_id != workspace_id
            or self.version.document_id != document_id
            or self.version.id != document_version_id
        ):
            return None
        return self.version


class FakeChunkRepository:
    """In-memory authoritative chunk repository fake."""

    def __init__(self) -> None:
        self.chunks: dict[UUID, DocumentChunk] = {}
        self.conflict = False

    async def add_many(
        self,
        chunks: Sequence[DocumentChunk],
    ) -> None:
        if self.conflict:
            raise DocumentChunkConflictError("conflicting chunks")

        for chunk in chunks:
            existing = self.chunks.get(chunk.id)
            if existing is not None and existing != chunk:
                raise DocumentChunkConflictError("conflicting chunks")
            self.chunks[chunk.id] = chunk

    async def list_by_version(
        self,
        *,
        workspace_id: UUID,
        document_id: UUID,
        document_version_id: UUID,
    ) -> Sequence[DocumentChunk]:
        return tuple(
            sorted(
                (
                    chunk
                    for chunk in self.chunks.values()
                    if (
                        chunk.workspace_id == workspace_id
                        and chunk.document_id == document_id
                        and chunk.document_version_id == document_version_id
                    )
                ),
                key=lambda chunk: chunk.ordinal,
            )
        )

    async def count_by_version(
        self,
        *,
        workspace_id: UUID,
        document_id: UUID,
        document_version_id: UUID,
    ) -> int:
        return len(
            await self.list_by_version(
                workspace_id=workspace_id,
                document_id=document_id,
                document_version_id=document_version_id,
            )
        )


class FakeChunker:
    """Return deterministic chunks under one policy."""

    def __init__(
        self,
        *,
        chunk_count: int = 3,
    ) -> None:
        self._policy = ChunkingPolicy()
        self.chunk_count = chunk_count
        self.calls = 0

    @property
    def policy(self) -> ChunkingPolicy:
        return self._policy

    def chunk(
        self,
        document_version: DocumentVersion,
    ) -> tuple[DocumentChunk, ...]:
        self.calls += 1
        return tuple(
            DocumentChunk.create(
                document_version=document_version,
                ordinal=index,
                section_path=("Recovery",),
                content=f"Authoritative chunk {index}.",
                token_count=4,
                now=document_version.created_at,
            )
            for index in range(self.chunk_count)
        )


class FakeEmbeddingProvider:
    """Return deterministic vectors or one configured failure."""

    provider_name = "mock"

    def __init__(self) -> None:
        self.requests: list[EmbeddingRequest] = []
        self.error: Exception | None = None
        self.include_usage = True
        self.response_model = "mock-hashing-embedding-v1"
        self.response_provider = "mock"

    async def embed(
        self,
        request: EmbeddingRequest,
    ) -> EmbeddingProviderResponse:
        self.requests.append(request)

        if self.error is not None:
            raise self.error

        usage = (
            EmbeddingTokenUsage(
                input_tokens=len(request.inputs) * 10,
                total_tokens=len(request.inputs) * 10,
            )
            if self.include_usage
            else None
        )

        return EmbeddingProviderResponse(
            embeddings=tuple(
                (
                    float(index + 1),
                    float(index + 2),
                    float(index + 3),
                )
                for index, _ in enumerate(request.inputs)
            ),
            provider=self.response_provider,
            model=self.response_model,
            dimensions=request.dimensions,
            usage=usage,
            provider_request_id=(f"embedding-request-{len(self.requests)}"),
        )

    async def close(self) -> None:
        return None


class FakeVectorStore:
    """Record one complete projected version."""

    def __init__(self) -> None:
        self.ensure_calls: list[KnowledgeCollectionProfile] = []
        self.upsert_calls: list[
            tuple[
                KnowledgeCollectionProfile,
                KnowledgeVersionProjection,
                tuple[KnowledgeVectorPoint, ...],
            ]
        ] = []
        self.count_override: int | None = None
        self.error: Exception | None = None

    async def ensure_collection(
        self,
        profile: KnowledgeCollectionProfile,
    ) -> None:
        self.ensure_calls.append(profile)

    async def upsert_version_points(
        self,
        *,
        profile: KnowledgeCollectionProfile,
        projection: KnowledgeVersionProjection,
        points: Sequence[KnowledgeVectorPoint],
    ) -> None:
        if self.error is not None:
            raise self.error

        self.upsert_calls.append(
            (
                profile,
                projection,
                tuple(points),
            )
        )

    async def count_version_points(
        self,
        *,
        profile: KnowledgeCollectionProfile,
        projection: KnowledgeVersionProjection,
    ) -> int:
        if self.count_override is not None:
            return self.count_override
        if not self.upsert_calls:
            return 0
        return len(self.upsert_calls[-1][2])


def create_profile(
    *,
    chunking_version: str = "v1",
) -> KnowledgeIndexProfile:
    return KnowledgeIndexProfile(
        chunking_strategy="markdown-token",
        chunking_version=chunking_version,
        tokenizer_encoding="cl100k_base",
        embedding_provider="mock",
        embedding_model="mock-hashing-embedding-v1",
        embedding_dimensions=3,
        knowledge_collection=("supportops-knowledge-mock-v1"),
        knowledge_vector_name="dense",
    )


def create_pending_version() -> DocumentVersion:
    return DocumentVersion.create_pending(
        document_version_id=_VERSION_ID,
        workspace_id=_WORKSPACE_ID,
        document_id=_DOCUMENT_ID,
        version_number=1,
        media_type=DocumentMediaType.TEXT_MARKDOWN,
        content=("# Recovery\n\nRestart the connection pool.\n"),
        now=_CREATED_AT,
    )


def create_ready_version() -> DocumentVersion:
    profiled = create_pending_version().bind_index_profile(
        create_profile(),
        now=_CREATED_AT,
    )
    return profiled.mark_ready(
        chunk_count=3,
        embedding_input_tokens=30,
        embedding_estimated_cost_usd=Decimal("0"),
        embedding_pricing_catalog_version=("supportops-embedding-pricing-2026-08-01"),
        indexed_at=_CREATED_AT,
    )


def create_service(
    *,
    version: DocumentVersion | None = None,
) -> tuple[
    IndexDocumentVersion,
    FakeVersionRepository,
    FakeChunkRepository,
    FakeChunker,
    FakeEmbeddingProvider,
    FakeVectorStore,
    FakeTransactionManager,
]:
    version_repository = FakeVersionRepository(version or create_pending_version())
    chunk_repository = FakeChunkRepository()
    chunker = FakeChunker()
    embedding_provider = FakeEmbeddingProvider()
    vector_store = FakeVectorStore()
    transaction_manager = FakeTransactionManager()

    service = IndexDocumentVersion(
        version_repository=version_repository,
        chunk_repository=chunk_repository,
        transaction_manager=transaction_manager,
        chunker=chunker,
        embedding_provider=embedding_provider,
        vector_store=vector_store,
        pricing_catalog=(DEFAULT_EMBEDDING_PRICING_CATALOG),
        index_profile=create_profile(),
        embedding_timeout_seconds=12,
        embedding_batch_size=2,
        clock=AdvancingClock(),
    )

    return (
        service,
        version_repository,
        chunk_repository,
        chunker,
        embedding_provider,
        vector_store,
        transaction_manager,
    )


async def execute(
    service: IndexDocumentVersion,
) -> IndexDocumentVersionResult:
    return await service.execute(
        workspace_id=_WORKSPACE_ID,
        document_id=_DOCUMENT_ID,
        document_version_id=_VERSION_ID,
    )


async def test_indexes_chunks_in_batches_and_marks_version_ready() -> None:
    (
        service,
        version_repository,
        chunk_repository,
        chunker,
        embedding_provider,
        vector_store,
        transaction_manager,
    ) = create_service()

    result = await execute(service)

    assert result.version.status is (DocumentVersionStatus.READY)
    assert not result.already_ready
    assert result.chunk_count == 3
    assert result.embedding_input_tokens == 30
    assert result.estimated_cost_usd == Decimal("0")
    assert result.pricing_catalog_version == ("supportops-embedding-pricing-2026-08-01")

    assert chunker.calls == 1
    assert len(chunk_repository.chunks) == 3
    assert [len(request.inputs) for request in embedding_provider.requests] == [2, 1]

    assert len(vector_store.upsert_calls) == 1
    _, projection, points = vector_store.upsert_calls[0]
    assert projection.document_version_id == _VERSION_ID
    assert len(points) == 3
    assert all(len(point.vector) == 3 for point in points)

    assert version_repository.version == result.version
    assert transaction_manager.completed == 3


async def test_ready_version_returns_without_external_work() -> None:
    (
        service,
        _,
        chunk_repository,
        chunker,
        embedding_provider,
        vector_store,
        transaction_manager,
    ) = create_service(version=create_ready_version())

    result = await execute(service)

    assert result.already_ready
    assert result.version.status is (DocumentVersionStatus.READY)
    assert chunker.calls == 0
    assert chunk_repository.chunks == {}
    assert embedding_provider.requests == []
    assert vector_store.upsert_calls == []
    assert transaction_manager.completed == 1


async def test_failed_version_is_prepared_and_retried() -> None:
    failed = (
        create_pending_version()
        .bind_index_profile(
            create_profile(),
            now=_CREATED_AT,
        )
        .mark_failed(
            error_code="embedding_timeout",
            chunk_count=0,
            now=_CREATED_AT,
        )
    )
    (
        service,
        version_repository,
        _,
        _,
        _,
        _,
        _,
    ) = create_service(version=failed)

    result = await execute(service)

    assert result.version.status is (DocumentVersionStatus.READY)
    assert result.version.last_error_code is None
    assert version_repository.updates[0].status is (DocumentVersionStatus.PENDING)


async def test_embedding_failure_records_failed_state_and_chunk_count() -> None:
    (
        service,
        version_repository,
        chunk_repository,
        _,
        embedding_provider,
        vector_store,
        transaction_manager,
    ) = create_service()
    embedding_provider.error = EmbeddingTimeoutError()

    with pytest.raises(EmbeddingTimeoutError):
        await execute(service)

    assert len(chunk_repository.chunks) == 3
    assert vector_store.upsert_calls == []
    assert version_repository.version is not None
    assert version_repository.version.status is (DocumentVersionStatus.FAILED)
    assert version_repository.version.last_error_code == ("embedding_timeout")
    assert version_repository.version.chunk_count == 3
    assert transaction_manager.completed == 3


async def test_missing_embedding_usage_is_invalid_and_recorded() -> None:
    (
        service,
        version_repository,
        _,
        _,
        embedding_provider,
        _,
        _,
    ) = create_service()
    embedding_provider.include_usage = False

    with pytest.raises(EmbeddingInvalidResponseError):
        await execute(service)

    assert version_repository.version is not None
    assert version_repository.version.status is (DocumentVersionStatus.FAILED)
    assert version_repository.version.last_error_code == ("embedding_invalid_response")


async def test_incomplete_qdrant_projection_is_not_marked_ready() -> None:
    (
        service,
        version_repository,
        _,
        _,
        _,
        vector_store,
        _,
    ) = create_service()
    vector_store.count_override = 2

    with pytest.raises(KnowledgeProjectionCountMismatchError):
        await execute(service)

    assert version_repository.version is not None
    assert version_repository.version.status is (DocumentVersionStatus.FAILED)
    assert version_repository.version.last_error_code == ("knowledge_projection_count_mismatch")
    assert version_repository.version.chunk_count == 3


async def test_persisted_profile_mismatch_does_not_mutate_version() -> None:
    incompatible = create_pending_version().bind_index_profile(
        create_profile(chunking_version="v2"),
        now=_CREATED_AT,
    )
    (
        service,
        version_repository,
        chunk_repository,
        chunker,
        embedding_provider,
        vector_store,
        transaction_manager,
    ) = create_service(version=incompatible)

    with pytest.raises(KnowledgeIndexProfileMismatchError):
        await execute(service)

    assert version_repository.version == incompatible
    assert chunk_repository.chunks == {}
    assert chunker.calls == 0
    assert embedding_provider.requests == []
    assert vector_store.upsert_calls == []
    assert transaction_manager.rolled_back == 1


async def test_missing_scoped_version_fails_before_work() -> None:
    (
        service,
        _,
        _,
        chunker,
        embedding_provider,
        vector_store,
        transaction_manager,
    ) = create_service(version=create_pending_version())

    with pytest.raises(KnowledgeDocumentVersionNotFoundError):
        await service.execute(
            workspace_id=_WORKSPACE_ID,
            document_id=_DOCUMENT_ID,
            document_version_id=UUID(int=999),
        )

    assert chunker.calls == 0
    assert embedding_provider.requests == []
    assert vector_store.upsert_calls == []
    assert transaction_manager.rolled_back == 1


@pytest.mark.parametrize(
    ("embedding_timeout_seconds", "embedding_batch_size"),
    [
        (0, 64),
        (-1, 64),
        (12, 0),
        (12, -1),
    ],
)
def test_service_rejects_invalid_runtime_limits(
    embedding_timeout_seconds: float,
    embedding_batch_size: int,
) -> None:
    version_repository = FakeVersionRepository(create_pending_version())

    with pytest.raises(ValueError):
        IndexDocumentVersion(
            version_repository=version_repository,
            chunk_repository=FakeChunkRepository(),
            transaction_manager=FakeTransactionManager(),
            chunker=FakeChunker(),
            embedding_provider=FakeEmbeddingProvider(),
            vector_store=FakeVectorStore(),
            pricing_catalog=(DEFAULT_EMBEDDING_PRICING_CATALOG),
            index_profile=create_profile(),
            embedding_timeout_seconds=(embedding_timeout_seconds),
            embedding_batch_size=embedding_batch_size,
        )
