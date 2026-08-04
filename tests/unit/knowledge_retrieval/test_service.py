"""Unit tests for active authoritative semantic retrieval."""

from collections.abc import Callable, Sequence
from contextlib import AbstractContextManager
from dataclasses import replace
from datetime import UTC, datetime
from hashlib import sha256
from types import TracebackType
from typing import Literal
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
    EmbeddingTimeoutError,
)
from supportops.ai.embeddings.observability import (
    ObservingEmbeddingProvider,
)
from supportops.knowledge_index.vector_store.contracts import (
    KnowledgeVectorStoreUnavailableError,
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
from supportops.observability.context import (
    ActiveObservationContext,
    current_observation_context,
    observation_context_scope,
)
from supportops.observability.contracts import TraceScope
from supportops.observability.models import (
    EventObservation,
    ObservabilityProvider,
    ObservationAttributes,
    ObservationStatus,
    ObservationType,
    ObservationUpdate,
    TraceAttributes,
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
        *,
        error: Exception | None = None,
    ) -> None:
        self.versions = tuple(versions)
        self.error = error
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
        if self.error is not None:
            raise self.error
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
        self.parent_observation_names: list[str | None] = []
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
        self.error: Exception | None = None

    async def embed(
        self,
        request: EmbeddingRequest,
    ) -> EmbeddingProviderResponse:
        self.requests.append(request)
        parent = current_observation_context()
        self.parent_observation_names.append(
            None if parent is None else parent.name,
        )

        if self.error is not None:
            raise self.error

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
        *,
        error: Exception | None = None,
    ) -> None:
        self.candidates = tuple(candidates)
        self.error = error
        self.requests: list[KnowledgeVectorSearchRequest] = []

    async def search(
        self,
        request: KnowledgeVectorSearchRequest,
    ) -> Sequence[KnowledgeVectorCandidate]:
        self.requests.append(request)
        if self.error is not None:
            raise self.error
        return self.candidates


class RecordingObservationScope:
    def __init__(
        self,
        *,
        attributes: ObservationAttributes,
        fail_update: bool = False,
    ) -> None:
        self.attributes = attributes
        self._fail_update = fail_update
        self.updates: list[ObservationUpdate] = []

    @property
    def observation_id(self) -> str | None:
        return "retrieval-observation-1"

    def update(self, update: ObservationUpdate) -> None:
        if self._fail_update:
            raise RuntimeError("synthetic update failure")

        self.updates.append(update)

    def start_observation(
        self,
        attributes: ObservationAttributes,
    ) -> AbstractContextManager["RecordingObservationScope"]:
        del attributes
        raise AssertionError("Nested observations are not expected on the scope.")

    def record_event(self, event: EventObservation) -> None:
        del event
        raise AssertionError("Events are not expected.")


class RecordingObservationManager(AbstractContextManager[RecordingObservationScope]):
    def __init__(
        self,
        *,
        scope: RecordingObservationScope,
        fail_enter: bool = False,
        fail_exit: bool = False,
        on_enter: Callable[[str], None] | None = None,
        on_exit: Callable[[str], None] | None = None,
    ) -> None:
        self._scope = scope
        self._fail_enter = fail_enter
        self._fail_exit = fail_exit
        self._on_enter = on_enter
        self._on_exit = on_exit
        self.exit_calls = 0
        self._context_manager = observation_context_scope(
            ActiveObservationContext(
                name=scope.attributes.name,
                observation_id=scope.observation_id,
            )
        )

    def __enter__(self) -> RecordingObservationScope:
        if self._fail_enter:
            raise RuntimeError("synthetic enter failure")

        self._context_manager.__enter__()
        if self._on_enter is not None:
            self._on_enter(self._scope.attributes.name)
        return self._scope

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> Literal[False]:
        self.exit_calls += 1
        if self._on_exit is not None:
            self._on_exit(self._scope.attributes.name)
        self._context_manager.__exit__(exc_type, exc, traceback)

        if self._fail_exit:
            raise RuntimeError("synthetic exit failure")

        return False


class RecordingObservabilityClient:
    def __init__(
        self,
        *,
        fail_start: bool = False,
        fail_enter: bool = False,
        fail_update: bool = False,
        fail_exit: bool = False,
    ) -> None:
        self._fail_start = fail_start
        self._fail_enter = fail_enter
        self._fail_update = fail_update
        self._fail_exit = fail_exit

        self.attributes: list[ObservationAttributes] = []
        self.scopes: list[RecordingObservationScope] = []
        self.managers: list[RecordingObservationManager] = []
        self.parent_observation_names: list[str | None] = []
        self.lifecycle: list[tuple[str, str]] = []

    @property
    def provider(self) -> ObservabilityProvider:
        return ObservabilityProvider.NOOP

    @property
    def enabled(self) -> bool:
        return True

    def start_trace(
        self,
        attributes: TraceAttributes,
    ) -> AbstractContextManager[TraceScope]:
        del attributes
        raise AssertionError("Retrieval tracing must not create roots.")

    def start_observation(
        self,
        attributes: ObservationAttributes,
    ) -> AbstractContextManager[RecordingObservationScope]:
        if self._fail_start:
            raise RuntimeError("synthetic start failure")

        parent = current_observation_context()
        self.parent_observation_names.append(
            None if parent is None else parent.name,
        )

        scope = RecordingObservationScope(
            attributes=attributes,
            fail_update=self._fail_update,
        )
        manager = RecordingObservationManager(
            scope=scope,
            fail_enter=self._fail_enter,
            fail_exit=self._fail_exit,
            on_enter=lambda name: self.lifecycle.append(("enter", name)),
            on_exit=lambda name: self.lifecycle.append(("exit", name)),
        )

        self.attributes.append(attributes)
        self.scopes.append(scope)
        self.managers.append(manager)

        return manager

    def record_event(self, event: EventObservation) -> None:
        del event
        raise AssertionError("Retrieval tracing must not emit events.")

    def flush(self) -> None:
        return None

    def shutdown(self) -> None:
        return None


def create_service(
    *,
    versions: Sequence[ActiveKnowledgeVersion] = (),
    chunks: Sequence[DocumentChunk] = (),
    candidates: Sequence[KnowledgeVectorCandidate] = (),
    profile: KnowledgeIndexProfile | None = None,
    observability_client: RecordingObservabilityClient | None = None,
    embedding_provider: FakeEmbeddingProvider | ObservingEmbeddingProvider | None = None,
    vector_searcher: FakeVectorSearcher | None = None,
    resolver: FakeActiveVersionResolver | None = None,
) -> tuple[
    SearchKnowledge,
    FakeActiveVersionResolver,
    FakeChunkHydrator,
    FakeEmbeddingProvider | ObservingEmbeddingProvider,
    FakeVectorSearcher,
]:
    """Compose the service around test doubles."""

    resolved_resolver = resolver or FakeActiveVersionResolver(versions)
    hydrator = FakeChunkHydrator(chunks)
    resolved_embedding_provider = embedding_provider or FakeEmbeddingProvider()
    resolved_vector_searcher = vector_searcher or FakeVectorSearcher(candidates)
    resolved_profile = profile or create_profile()

    service = SearchKnowledge(
        active_version_resolver=resolved_resolver,
        chunk_hydrator=hydrator,
        embedding_provider=(resolved_embedding_provider),
        vector_searcher=resolved_vector_searcher,
        index_profile=resolved_profile,
        embedding_timeout_seconds=12,
        candidate_multiplier=4,
        observability_client=observability_client,
    )

    return (
        service,
        resolved_resolver,
        hydrator,
        resolved_embedding_provider,
        resolved_vector_searcher,
    )


def _successful_search_fixture() -> tuple[
    ActiveKnowledgeVersion,
    DocumentChunk,
    KnowledgeVectorCandidate,
]:
    active = create_active_version()
    chunk = create_chunk(chunk_id=UUID(int=401))
    candidate = create_candidate(chunk, score=0.91)
    return active, chunk, candidate


async def test_empty_active_scope_returns_without_external_work() -> None:
    observability = RecordingObservabilityClient()
    (
        service,
        resolver,
        hydrator,
        embedding_provider,
        vector_searcher,
    ) = create_service(observability_client=observability)

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
    assert isinstance(embedding_provider, FakeEmbeddingProvider)
    assert embedding_provider.requests == []
    assert vector_searcher.requests == []
    assert hydrator.calls == []
    assert len(observability.attributes) == 1
    assert observability.attributes[0].name == "knowledge.search"
    assert observability.attributes[0].observation_type is ObservationType.RETRIEVER
    assert observability.scopes[0].updates[0].status is ObservationStatus.OK
    assert observability.scopes[0].updates[0].metadata["evidence_count"] == 0
    assert "embedding.request" not in [attributes.name for attributes in observability.attributes]
    assert "How do I recover the database?" not in str(observability.attributes)


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
    assert isinstance(embedding_provider, FakeEmbeddingProvider)
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

    assert isinstance(embedding_provider, FakeEmbeddingProvider)
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
    assert isinstance(embedding_provider, FakeEmbeddingProvider)
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


async def test_successful_search_creates_one_retriever_observation() -> None:
    active, chunk, candidate = _successful_search_fixture()
    observability = RecordingObservabilityClient()
    (
        service,
        _,
        _,
        embedding_provider,
        _,
    ) = create_service(
        versions=(active,),
        chunks=(chunk,),
        candidates=(candidate,),
        observability_client=observability,
    )

    request = KnowledgeSearchRequest(
        workspace_id=_WORKSPACE_ID,
        query="recover the database",
        top_k=3,
        document_ids=(_DOCUMENT_A_ID,),
    )
    result = await service.execute(request)

    assert len(result.evidence) == 1
    assert len(observability.attributes) == 1
    assert len(observability.scopes) == 1
    assert observability.managers[0].exit_calls == 1

    attributes = observability.attributes[0]
    assert attributes.name == "knowledge.search"
    assert attributes.observation_type is ObservationType.RETRIEVER
    assert attributes.input_data is None
    assert attributes.input_paths == frozenset()
    assert attributes.output_paths == frozenset()
    assert attributes.metadata == {
        "workspace_id": str(_WORKSPACE_ID),
        "top_k": 3,
        "requested_document_count": 1,
        "embedding_provider": "mock",
        "embedding_model": "mock-hashing-embedding-v1",
        "embedding_dimensions": 3,
    }
    assert "recover the database" not in str(attributes.metadata)
    assert request.query not in str(attributes)

    update = observability.scopes[0].updates[0]
    assert update.status is ObservationStatus.OK
    assert update.output_data is None
    assert update.metadata["searched_version_count"] == 1
    assert update.metadata["candidate_count"] == 1
    assert update.metadata["hydrated_candidate_count"] == 1
    assert update.metadata["evidence_count"] == 1
    assert update.metadata["filtered_candidate_count"] == 0
    assert update.metadata["status"] == ObservationStatus.OK.value
    assert isinstance(update.metadata["latency_ms"], int)
    assert chunk.content not in str(update.metadata)
    assert str(chunk.id) not in str(update)
    assert isinstance(embedding_provider, FakeEmbeddingProvider)
    assert len(embedding_provider.requests) == 1


async def test_empty_evidence_remains_ok() -> None:
    active = create_active_version()
    observability = RecordingObservabilityClient()
    (service, _, _, _, _) = create_service(
        versions=(active,),
        candidates=(),
        observability_client=observability,
    )

    result = await service.execute(
        KnowledgeSearchRequest(
            workspace_id=_WORKSPACE_ID,
            query="recover the database",
        )
    )

    assert result.evidence == ()
    update = observability.scopes[0].updates[0]
    assert update.status is ObservationStatus.OK
    assert update.metadata["evidence_count"] == 0
    assert update.metadata["candidate_count"] == 0
    assert update.metadata["status"] == ObservationStatus.OK.value


async def test_normalized_embedding_failure_marks_error_and_preserves_exception() -> None:
    active = create_active_version()
    observability = RecordingObservabilityClient()
    embedding_provider = FakeEmbeddingProvider()
    embedding_provider.error = EmbeddingTimeoutError(
        provider_request_id="timeout-1",
    )
    (service, _, _, _, _) = create_service(
        versions=(active,),
        embedding_provider=embedding_provider,
        observability_client=observability,
    )

    with pytest.raises(EmbeddingTimeoutError) as captured:
        await service.execute(
            KnowledgeSearchRequest(
                workspace_id=_WORKSPACE_ID,
                query="recover the database",
            )
        )

    assert captured.value is embedding_provider.error
    update = observability.scopes[0].updates[0]
    assert update.status is ObservationStatus.ERROR
    assert update.error_code == "embedding_timeout"
    assert update.metadata["error_code"] == "embedding_timeout"


async def test_normalized_vector_store_failure_marks_error() -> None:
    active = create_active_version()
    observability = RecordingObservabilityClient()
    error = KnowledgeVectorStoreUnavailableError(
        "The knowledge vector store is unavailable.",
    )
    (service, _, _, _, _) = create_service(
        versions=(active,),
        vector_searcher=FakeVectorSearcher(error=error),
        observability_client=observability,
    )

    with pytest.raises(KnowledgeVectorStoreUnavailableError) as captured:
        await service.execute(
            KnowledgeSearchRequest(
                workspace_id=_WORKSPACE_ID,
                query="recover the database",
            )
        )

    assert captured.value is error
    update = observability.scopes[0].updates[0]
    assert update.status is ObservationStatus.ERROR
    assert update.error_code == "knowledge_retrieval_unavailable"


async def test_unexpected_failure_uses_safe_normalized_code() -> None:
    observability = RecordingObservabilityClient()
    unexpected = RuntimeError("raw repository failure details")
    (service, _, _, _, _) = create_service(
        resolver=FakeActiveVersionResolver(error=unexpected),
        observability_client=observability,
    )

    with pytest.raises(RuntimeError) as captured:
        await service.execute(
            KnowledgeSearchRequest(
                workspace_id=_WORKSPACE_ID,
                query="recover the database",
            )
        )

    assert captured.value is unexpected
    update = observability.scopes[0].updates[0]
    assert update.status is ObservationStatus.ERROR
    assert update.error_code == "knowledge_retrieval_unexpected_failure"
    assert "raw repository failure details" not in str(update.metadata)
    assert "raw repository failure details" not in str(update)


@pytest.mark.parametrize(
    "client_kwargs",
    [
        {"fail_start": True},
        {"fail_update": True},
        {"fail_exit": True},
    ],
)
async def test_observability_failures_do_not_alter_successful_retrieval(
    client_kwargs: dict[str, bool],
) -> None:
    active, chunk, candidate = _successful_search_fixture()
    observability = RecordingObservabilityClient(**client_kwargs)
    (service, _, _, _, _) = create_service(
        versions=(active,),
        chunks=(chunk,),
        candidates=(candidate,),
        observability_client=observability,
    )

    result = await service.execute(
        KnowledgeSearchRequest(
            workspace_id=_WORKSPACE_ID,
            query="recover the database",
        )
    )

    assert len(result.evidence) == 1
    assert result.evidence[0].content == chunk.content


async def test_result_ordering_and_workspace_filtering_unchanged() -> None:
    active = create_active_version()
    low_chunk = create_chunk(
        chunk_id=UUID(int=501),
        ordinal=0,
        content="Low score.",
    )
    high_chunk = create_chunk(
        chunk_id=UUID(int=502),
        ordinal=1,
        content="High score.",
    )
    observability = RecordingObservabilityClient()
    (service, resolver, _, _, _) = create_service(
        versions=(active,),
        chunks=(low_chunk, high_chunk),
        candidates=(
            create_candidate(low_chunk, score=0.40),
            create_candidate(high_chunk, score=0.95),
        ),
        observability_client=observability,
    )

    result = await service.execute(
        KnowledgeSearchRequest(
            workspace_id=_WORKSPACE_ID,
            query="recovery",
            top_k=2,
            document_ids=(_DOCUMENT_A_ID,),
        )
    )

    assert resolver.calls == [(_WORKSPACE_ID, (_DOCUMENT_A_ID,))]
    assert tuple(item.score for item in result.evidence) == (0.95, 0.40)
    assert all(item.citation.workspace_id == _WORKSPACE_ID for item in result.evidence)
    exported = str(observability.attributes) + str(observability.scopes[0].updates)
    assert "High score." not in exported
    assert "Low score." not in exported
    assert str(high_chunk.id) not in exported
    assert str(low_chunk.id) not in exported


async def test_observing_embedding_provider_nests_without_duplicate_manual_span() -> None:
    active, chunk, candidate = _successful_search_fixture()
    observability = RecordingObservabilityClient()
    inner_provider = FakeEmbeddingProvider()
    observing_provider = ObservingEmbeddingProvider(
        provider=inner_provider,
        observability_client=observability,
    )
    (service, _, _, _, _) = create_service(
        versions=(active,),
        chunks=(chunk,),
        candidates=(candidate,),
        embedding_provider=observing_provider,
        observability_client=observability,
    )

    await service.execute(
        KnowledgeSearchRequest(
            workspace_id=_WORKSPACE_ID,
            query="recover the database",
        )
    )

    assert len(inner_provider.requests) == 1
    names = [attributes.name for attributes in observability.attributes]
    types = [attributes.observation_type for attributes in observability.attributes]
    assert names.count("knowledge.search") == 1
    assert names.count("embedding.request") == 1
    assert types.count(ObservationType.RETRIEVER) == 1
    assert types.count(ObservationType.EMBEDDING) == 1
    assert names[0] == "knowledge.search"
    assert names[1] == "embedding.request"
    assert observability.parent_observation_names == [
        None,
        "knowledge.search",
    ]
    assert inner_provider.parent_observation_names == ["embedding.request"]
    assert observability.lifecycle == [
        ("enter", "knowledge.search"),
        ("enter", "embedding.request"),
        ("exit", "embedding.request"),
        ("exit", "knowledge.search"),
    ]
    assert all(manager.exit_calls == 1 for manager in observability.managers)
    assert current_observation_context() is None
    assert "recover the database" not in str(observability.attributes)
    assert chunk.content not in str(observability.scopes[0].updates)
    assert chunk.content not in str(observability.scopes[1].updates)
