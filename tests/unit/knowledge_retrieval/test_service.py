"""Unit tests for active authoritative semantic retrieval."""

from collections.abc import Sequence
from dataclasses import replace
from datetime import UTC, datetime
from hashlib import sha256
from uuid import UUID

import pytest

from supportops.ai.embeddings.contracts import (
    EmbeddingOperation,
    EmbeddingProviderResponse,
    EmbeddingRequest,
    EmbeddingTokenUsage,
)
from supportops.ai.embeddings.errors import (
    EmbeddingInvalidResponseError,
)
from supportops.knowledge_retrieval.contracts import (
    ActiveKnowledgeVersion,
    KnowledgeSearchRequest,
    KnowledgeSearchTarget,
    KnowledgeVectorCandidate,
    KnowledgeVectorSearchRequest,
)
from supportops.knowledge_retrieval.service import (
    SearchKnowledge,
)
from supportops.modules.knowledge_documents.domain.models import (
    DocumentChunk,
    DocumentMediaType,
    KnowledgeIndexProfile,
)

_WORKSPACE_ID = UUID("032c8c87-57cc-4d14-bfbd-04968b4e8cd4")
_DOCUMENT_A_ID = UUID("276046a2-28ec-4cb1-8bb6-a2ff70f9064b")
_DOCUMENT_B_ID = UUID("10bea98d-dfb1-4c33-884f-a4f36789a2ab")
_VERSION_A_ID = UUID("09036916-84cf-4a58-bdf4-09bc52716ec5")
_VERSION_B_ID = UUID("8ac21e0c-f869-4d84-a9e1-37d8074c9e54")
_CREATED_AT = datetime(
    2026,
    8,
    2,
    2,
    30,
    tzinfo=UTC,
)


def create_profile(
    *,
    embedding_model: str = ("mock-hashing-embedding-v1"),
) -> KnowledgeIndexProfile:
    """Create one deterministic retrieval profile."""

    return KnowledgeIndexProfile(
        chunking_strategy="markdown-token",
        chunking_version="v1",
        tokenizer_encoding="cl100k_base",
        embedding_provider="mock",
        embedding_model=embedding_model,
        embedding_dimensions=3,
        knowledge_collection=("supportops-knowledge-mock-v1"),
        knowledge_vector_name="dense",
    )


def create_active_version(
    *,
    document_id: UUID = _DOCUMENT_A_ID,
    document_version_id: UUID = (_VERSION_A_ID),
    title: str = "Database Recovery Runbook",
    profile: KnowledgeIndexProfile | None = None,
) -> ActiveKnowledgeVersion:
    """Create one active ready-version descriptor."""

    return ActiveKnowledgeVersion(
        workspace_id=_WORKSPACE_ID,
        document_id=document_id,
        document_title=title,
        document_external_reference=(f"document-{document_id}"),
        document_version_id=(document_version_id),
        version_number=1,
        media_type=(DocumentMediaType.TEXT_MARKDOWN),
        index_profile=profile or create_profile(),
    )


def create_chunk(
    *,
    chunk_id: UUID,
    document_id: UUID = _DOCUMENT_A_ID,
    document_version_id: UUID = (_VERSION_A_ID),
    ordinal: int = 0,
    content: str = ("Restart the database connection pool."),
) -> DocumentChunk:
    """Create one authoritative PostgreSQL chunk."""

    return DocumentChunk(
        id=chunk_id,
        workspace_id=_WORKSPACE_ID,
        document_id=document_id,
        document_version_id=(document_version_id),
        ordinal=ordinal,
        section_path=("Recovery",),
        content=content,
        content_sha256=sha256(content.encode("utf-8")).hexdigest(),
        token_count=6,
        chunking_strategy="markdown-token",
        chunking_version="v1",
        created_at=_CREATED_AT,
    )


def create_candidate(
    chunk: DocumentChunk,
    *,
    score: float,
) -> KnowledgeVectorCandidate:
    """Create one candidate matching an authoritative chunk."""

    return KnowledgeVectorCandidate(
        chunk_id=chunk.id,
        workspace_id=chunk.workspace_id,
        document_id=chunk.document_id,
        document_version_id=(chunk.document_version_id),
        ordinal=chunk.ordinal,
        content_sha256=(chunk.content_sha256),
        media_type=(DocumentMediaType.TEXT_MARKDOWN),
        chunking_strategy=(chunk.chunking_strategy),
        chunking_version=(chunk.chunking_version),
        score=score,
    )


class FakeActiveVersionResolver:
    """Return configured active versions and record scope."""

    def __init__(
        self,
        versions: Sequence[ActiveKnowledgeVersion] = (),
    ) -> None:
        self.versions = tuple(versions)
        self.calls: list[tuple[UUID, tuple[UUID, ...]]] = []

    async def resolve(
        self,
        *,
        workspace_id: UUID,
        document_ids: tuple[UUID, ...],
    ) -> Sequence[ActiveKnowledgeVersion]:
        self.calls.append(
            (
                workspace_id,
                document_ids,
            )
        )
        return self.versions


class FakeChunkHydrator:
    """Return configured authoritative chunks."""

    def __init__(
        self,
        chunks: Sequence[DocumentChunk] = (),
    ) -> None:
        self.chunks = tuple(chunks)
        self.calls: list[tuple[UUID, tuple[UUID, ...]]] = []

    async def hydrate(
        self,
        *,
        workspace_id: UUID,
        chunk_ids: tuple[UUID, ...],
    ) -> Sequence[DocumentChunk]:
        self.calls.append(
            (
                workspace_id,
                chunk_ids,
            )
        )

        chunks_by_id = {chunk.id: chunk for chunk in self.chunks}

        return tuple(chunks_by_id[chunk_id] for chunk_id in chunk_ids if chunk_id in chunks_by_id)


class FakeEmbeddingProvider:
    """Return one configured query embedding."""

    provider_name = "mock"

    def __init__(self) -> None:
        self.requests: list[EmbeddingRequest] = []
        self.response_provider = "mock"
        self.response_model = "mock-hashing-embedding-v1"
        self.response_dimensions = 3
        self.embeddings: tuple[tuple[float, ...], ...] = (
            (
                1.0,
                0.0,
                0.0,
            ),
        )

    async def embed(
        self,
        request: EmbeddingRequest,
    ) -> EmbeddingProviderResponse:
        self.requests.append(request)

        return EmbeddingProviderResponse(
            embeddings=self.embeddings,
            provider=self.response_provider,
            model=self.response_model,
            dimensions=(self.response_dimensions),
            usage=EmbeddingTokenUsage(
                input_tokens=4,
                total_tokens=4,
            ),
            provider_request_id=("query-embedding-request-1"),
        )

    async def close(self) -> None:
        return None


class FakeVectorSearcher:
    """Return configured non-authoritative candidates."""

    def __init__(
        self,
        candidates: Sequence[KnowledgeVectorCandidate] = (),
    ) -> None:
        self.candidates = tuple(candidates)
        self.requests: list[KnowledgeVectorSearchRequest] = []

    async def search(
        self,
        request: KnowledgeVectorSearchRequest,
    ) -> Sequence[KnowledgeVectorCandidate]:
        self.requests.append(request)
        return self.candidates


def create_service(
    *,
    versions: Sequence[ActiveKnowledgeVersion] = (),
    chunks: Sequence[DocumentChunk] = (),
    candidates: Sequence[KnowledgeVectorCandidate] = (),
    profile: KnowledgeIndexProfile | None = None,
) -> tuple[
    SearchKnowledge,
    FakeActiveVersionResolver,
    FakeChunkHydrator,
    FakeEmbeddingProvider,
    FakeVectorSearcher,
]:
    """Compose the service around test doubles."""

    resolver = FakeActiveVersionResolver(versions)
    hydrator = FakeChunkHydrator(chunks)
    embedding_provider = FakeEmbeddingProvider()
    vector_searcher = FakeVectorSearcher(candidates)
    resolved_profile = profile or create_profile()

    service = SearchKnowledge(
        active_version_resolver=resolver,
        chunk_hydrator=hydrator,
        embedding_provider=(embedding_provider),
        vector_searcher=vector_searcher,
        index_profile=resolved_profile,
        embedding_timeout_seconds=12,
        candidate_multiplier=4,
    )

    return (
        service,
        resolver,
        hydrator,
        embedding_provider,
        vector_searcher,
    )


async def test_empty_active_scope_returns_without_external_work() -> None:
    (
        service,
        resolver,
        hydrator,
        embedding_provider,
        vector_searcher,
    ) = create_service()

    request = KnowledgeSearchRequest(
        workspace_id=_WORKSPACE_ID,
        query="How do I recover the database?",
        document_ids=(_DOCUMENT_A_ID,),
    )

    result = await service.execute(request)

    assert result.request == request
    assert result.searched_version_count == 0
    assert result.evidence == ()
    assert resolver.calls == [
        (
            _WORKSPACE_ID,
            (_DOCUMENT_A_ID,),
        )
    ]
    assert embedding_provider.requests == []
    assert vector_searcher.requests == []
    assert hydrator.calls == []


async def test_incompatible_active_profile_is_safely_omitted(
    caplog: pytest.LogCaptureFixture,
) -> None:
    incompatible = create_active_version(profile=create_profile(embedding_model="other-model"))
    (
        service,
        _,
        hydrator,
        embedding_provider,
        vector_searcher,
    ) = create_service(versions=(incompatible,))

    result = await service.execute(
        KnowledgeSearchRequest(
            workspace_id=_WORKSPACE_ID,
            query="database recovery",
        )
    )

    assert result.searched_version_count == 0
    assert result.evidence == ()
    assert embedding_provider.requests == []
    assert vector_searcher.requests == []
    assert hydrator.calls == []
    assert "Discarded ineligible active" in (caplog.text)


async def test_embeds_query_searches_active_targets_and_hydrates() -> None:
    active_a = create_active_version()
    active_b = create_active_version(
        document_id=_DOCUMENT_B_ID,
        document_version_id=_VERSION_B_ID,
        title="Billing Recovery Runbook",
    )
    chunk_a = create_chunk(
        chunk_id=UUID(int=101),
    )
    chunk_b = create_chunk(
        chunk_id=UUID(int=102),
        document_id=_DOCUMENT_B_ID,
        document_version_id=_VERSION_B_ID,
        content="Reconcile the failed invoice.",
    )
    candidate_a = create_candidate(
        chunk_a,
        score=0.95,
    )
    candidate_b = create_candidate(
        chunk_b,
        score=0.80,
    )

    (
        service,
        resolver,
        hydrator,
        embedding_provider,
        vector_searcher,
    ) = create_service(
        versions=(
            active_a,
            active_b,
        ),
        chunks=(
            chunk_a,
            chunk_b,
        ),
        candidates=(
            candidate_a,
            candidate_b,
        ),
    )

    request = KnowledgeSearchRequest(
        workspace_id=_WORKSPACE_ID,
        query="  recover the database  ",
        top_k=2,
        document_ids=(
            _DOCUMENT_A_ID,
            _DOCUMENT_B_ID,
        ),
    )

    result = await service.execute(request)

    assert resolver.calls == [
        (
            _WORKSPACE_ID,
            (
                _DOCUMENT_A_ID,
                _DOCUMENT_B_ID,
            ),
        )
    ]

    assert len(embedding_provider.requests) == 1
    embedding_request = embedding_provider.requests[0]
    assert embedding_request.operation is (EmbeddingOperation.KNOWLEDGE_QUERY)
    assert embedding_request.inputs == ("recover the database",)
    assert embedding_request.model == ("mock-hashing-embedding-v1")
    assert embedding_request.dimensions == 3
    assert embedding_request.metadata == {"workspace_id": str(_WORKSPACE_ID)}

    assert len(vector_searcher.requests) == 1
    vector_request = vector_searcher.requests[0]
    assert vector_request.workspace_id == (_WORKSPACE_ID)
    assert vector_request.targets == (
        KnowledgeSearchTarget(
            document_id=_DOCUMENT_A_ID,
            document_version_id=_VERSION_A_ID,
        ),
        KnowledgeSearchTarget(
            document_id=_DOCUMENT_B_ID,
            document_version_id=_VERSION_B_ID,
        ),
    )
    assert vector_request.query_vector == (
        1.0,
        0.0,
        0.0,
    )
    assert vector_request.limit == 8

    assert hydrator.calls == [
        (
            _WORKSPACE_ID,
            (
                chunk_a.id,
                chunk_b.id,
            ),
        )
    ]

    assert result.searched_version_count == 2
    assert tuple(item.rank for item in result.evidence) == (
        1,
        2,
    )
    assert tuple(item.score for item in result.evidence) == (
        0.95,
        0.80,
    )
    assert result.evidence[0].content == (chunk_a.content)
    assert result.evidence[0].citation.document_title == "Database Recovery Runbook"
    assert result.evidence[1].content == (chunk_b.content)


async def test_sorts_candidates_and_applies_top_k_after_hydration() -> None:
    active = create_active_version()
    low_chunk = create_chunk(
        chunk_id=UUID(int=201),
        ordinal=0,
        content="Low score.",
    )
    high_chunk = create_chunk(
        chunk_id=UUID(int=202),
        ordinal=1,
        content="High score.",
    )
    middle_chunk = create_chunk(
        chunk_id=UUID(int=203),
        ordinal=2,
        content="Middle score.",
    )

    (
        service,
        _,
        _,
        _,
        _,
    ) = create_service(
        versions=(active,),
        chunks=(
            low_chunk,
            high_chunk,
            middle_chunk,
        ),
        candidates=(
            create_candidate(
                low_chunk,
                score=0.40,
            ),
            create_candidate(
                high_chunk,
                score=0.95,
            ),
            create_candidate(
                middle_chunk,
                score=0.75,
            ),
        ),
    )

    result = await service.execute(
        KnowledgeSearchRequest(
            workspace_id=_WORKSPACE_ID,
            query="recovery",
            top_k=2,
        )
    )

    assert tuple(item.score for item in result.evidence) == (
        0.95,
        0.75,
    )
    assert tuple(item.content for item in result.evidence) == (
        "High score.",
        "Middle score.",
    )


async def test_discards_missing_duplicate_and_inconsistent_candidates(
    caplog: pytest.LogCaptureFixture,
) -> None:
    active = create_active_version()
    valid_chunk = create_chunk(
        chunk_id=UUID(int=301),
        ordinal=0,
        content="Valid evidence.",
    )
    inconsistent_chunk = create_chunk(
        chunk_id=UUID(int=302),
        ordinal=1,
        content="Authoritative content.",
    )
    missing_chunk = create_chunk(
        chunk_id=UUID(int=303),
        ordinal=2,
        content="Missing authoritative row.",
    )

    valid_candidate = create_candidate(
        valid_chunk,
        score=0.70,
    )
    inconsistent_candidate = replace(
        create_candidate(
            inconsistent_chunk,
            score=0.90,
        ),
        content_sha256=sha256(b"different-content").hexdigest(),
    )
    missing_candidate = create_candidate(
        missing_chunk,
        score=0.80,
    )

    (
        service,
        _,
        _,
        _,
        _,
    ) = create_service(
        versions=(active,),
        chunks=(
            valid_chunk,
            inconsistent_chunk,
        ),
        candidates=(
            inconsistent_candidate,
            missing_candidate,
            valid_candidate,
            valid_candidate,
        ),
    )

    result = await service.execute(
        KnowledgeSearchRequest(
            workspace_id=_WORKSPACE_ID,
            query="recovery",
            top_k=3,
        )
    )

    assert len(result.evidence) == 1
    assert result.evidence[0].content == ("Valid evidence.")
    assert "Discarded inconsistent semantic" in (caplog.text)


@pytest.mark.parametrize(
    (
        "response_provider",
        "response_model",
        "response_dimensions",
        "embeddings",
    ),
    [
        (
            "other-provider",
            "mock-hashing-embedding-v1",
            3,
            ((1.0, 0.0, 0.0),),
        ),
        (
            "mock",
            "other-model",
            3,
            ((1.0, 0.0, 0.0),),
        ),
        (
            "mock",
            "mock-hashing-embedding-v1",
            2,
            ((1.0, 0.0),),
        ),
        (
            "mock",
            "mock-hashing-embedding-v1",
            3,
            (
                (1.0, 0.0, 0.0),
                (0.0, 1.0, 0.0),
            ),
        ),
    ],
)
async def test_rejects_invalid_query_embedding_response(
    response_provider: str,
    response_model: str,
    response_dimensions: int,
    embeddings: tuple[tuple[float, ...], ...],
) -> None:
    active = create_active_version()
    (
        service,
        _,
        hydrator,
        embedding_provider,
        vector_searcher,
    ) = create_service(versions=(active,))
    embedding_provider.response_provider = response_provider
    embedding_provider.response_model = response_model
    embedding_provider.response_dimensions = response_dimensions
    embedding_provider.embeddings = embeddings

    with pytest.raises(EmbeddingInvalidResponseError) as captured:
        await service.execute(
            KnowledgeSearchRequest(
                workspace_id=_WORKSPACE_ID,
                query="database recovery",
            )
        )

    assert captured.value.provider_request_id == "query-embedding-request-1"
    assert vector_searcher.requests == []
    assert hydrator.calls == []


@pytest.mark.parametrize(
    (
        "timeout_seconds",
        "candidate_multiplier",
    ),
    [
        (0, 4),
        (-1, 4),
        (12, 0),
        (12, -1),
    ],
)
def test_service_rejects_invalid_runtime_limits(
    timeout_seconds: float,
    candidate_multiplier: int,
) -> None:
    with pytest.raises(ValueError):
        SearchKnowledge(
            active_version_resolver=(FakeActiveVersionResolver()),
            chunk_hydrator=FakeChunkHydrator(),
            embedding_provider=(FakeEmbeddingProvider()),
            vector_searcher=(FakeVectorSearcher()),
            index_profile=create_profile(),
            embedding_timeout_seconds=(timeout_seconds),
            candidate_multiplier=(candidate_multiplier),
        )


def test_service_rejects_provider_profile_mismatch() -> None:
    provider = FakeEmbeddingProvider()
    provider.provider_name = "openai"

    with pytest.raises(
        ValueError,
        match=("Embedding provider must match the retrieval index profile"),
    ):
        SearchKnowledge(
            active_version_resolver=(FakeActiveVersionResolver()),
            chunk_hydrator=FakeChunkHydrator(),
            embedding_provider=provider,
            vector_searcher=(FakeVectorSearcher()),
            index_profile=create_profile(),
            embedding_timeout_seconds=12,
        )
