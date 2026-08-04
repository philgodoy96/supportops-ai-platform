"""Unit tests for knowledge indexing runtime composition."""

from typing import Any, cast

import pytest
from qdrant_client import AsyncQdrantClient
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
)

import supportops.knowledge_index.composition as composition
from supportops.ai.embeddings.contracts import (
    EmbeddingProvider,
)
from supportops.ai.embeddings.mock import (
    MockEmbeddingProvider,
)
from supportops.ai.embeddings.observability import (
    ObservingEmbeddingProvider,
)
from supportops.ai.embeddings.openai import (
    OpenAIEmbeddingProvider,
)
from supportops.core.settings import (
    EmbeddingProviderName,
    Settings,
)
from supportops.knowledge_index.chunking.markdown import (
    MarkdownTokenChunker,
)
from supportops.knowledge_index.composition import (
    KNOWLEDGE_VECTOR_NAME,
    MOCK_KNOWLEDGE_COLLECTION,
    OPENAI_KNOWLEDGE_COLLECTION,
    KnowledgeIndexCompositionError,
    KnowledgeIndexRuntime,
    build_knowledge_index_profile,
    create_embedding_provider,
)
from supportops.knowledge_index.vector_store.qdrant import (
    QdrantKnowledgeVectorStore,
)


def create_settings(
    *,
    embedding_provider: EmbeddingProviderName = (EmbeddingProviderName.MOCK),
    embedding_model: str = ("mock-hashing-embedding-v1"),
    embedding_dimensions: int = 64,
    openai_api_key: str | None = None,
) -> Settings:
    """Create valid settings for composition tests."""

    return Settings(
        postgresql_url=("postgresql+asyncpg://supportops:supportops@localhost:5432/supportops"),
        qdrant_url="http://localhost:6333",
        embedding_provider=embedding_provider,
        embedding_model=embedding_model,
        embedding_dimensions=embedding_dimensions,
        openai_api_key=openai_api_key,
    )


def test_builds_mock_index_profile() -> None:
    profile = build_knowledge_index_profile(create_settings())

    assert profile.embedding_provider == "mock"
    assert profile.embedding_model == ("mock-hashing-embedding-v1")
    assert profile.embedding_dimensions == 64
    assert profile.knowledge_collection == (MOCK_KNOWLEDGE_COLLECTION)
    assert profile.knowledge_vector_name == (KNOWLEDGE_VECTOR_NAME)
    assert profile.chunking_strategy == ("markdown-token")
    assert profile.chunking_version == "v1"
    assert profile.tokenizer_encoding == ("cl100k_base")


def test_builds_fixed_openai_index_profile() -> None:
    settings = create_settings(
        embedding_provider=(EmbeddingProviderName.OPENAI),
        embedding_model="text-embedding-3-small",
        embedding_dimensions=1536,
        openai_api_key="test-openai-key",
    )

    profile = build_knowledge_index_profile(settings)

    assert profile.embedding_provider == "openai"
    assert profile.embedding_model == ("text-embedding-3-small")
    assert profile.embedding_dimensions == 1536
    assert profile.knowledge_collection == (OPENAI_KNOWLEDGE_COLLECTION)
    assert profile.knowledge_vector_name == "dense"


@pytest.mark.parametrize(
    ("embedding_model", "embedding_dimensions"),
    [
        ("unsupported-mock-model", 64),
        ("mock-hashing-embedding-v1", 0),
    ],
)
def test_rejects_invalid_mock_profile(
    embedding_model: str,
    embedding_dimensions: int,
) -> None:
    if embedding_dimensions == 0:
        with pytest.raises(ValueError):
            create_settings(
                embedding_model=embedding_model,
                embedding_dimensions=(embedding_dimensions),
            )
        return

    with pytest.raises(
        KnowledgeIndexCompositionError,
        match="mock-hashing-embedding-v1",
    ):
        build_knowledge_index_profile(
            create_settings(
                embedding_model=embedding_model,
                embedding_dimensions=(embedding_dimensions),
            )
        )


@pytest.mark.parametrize(
    ("embedding_model", "embedding_dimensions"),
    [
        ("text-embedding-3-large", 1536),
        ("text-embedding-3-small", 512),
    ],
)
def test_rejects_incompatible_openai_profile(
    embedding_model: str,
    embedding_dimensions: int,
) -> None:
    settings = create_settings(
        embedding_provider=(EmbeddingProviderName.OPENAI),
        embedding_model=embedding_model,
        embedding_dimensions=embedding_dimensions,
        openai_api_key="test-openai-key",
    )

    with pytest.raises(KnowledgeIndexCompositionError):
        build_knowledge_index_profile(settings)


def test_creates_mock_provider() -> None:
    provider = create_embedding_provider(create_settings())

    assert isinstance(provider, ObservingEmbeddingProvider)
    assert isinstance(
        provider.wrapped_provider,
        MockEmbeddingProvider,
    )
    assert provider.provider_name == "mock"
    assert provider.wrapped_provider.model == ("mock-hashing-embedding-v1")
    assert provider.wrapped_provider.dimensions == 64


async def test_creates_and_closes_openai_provider_without_network() -> None:
    settings = create_settings(
        embedding_provider=(EmbeddingProviderName.OPENAI),
        embedding_model="text-embedding-3-small",
        embedding_dimensions=1536,
        openai_api_key="test-openai-key",
    )

    provider = create_embedding_provider(settings)

    assert isinstance(provider, ObservingEmbeddingProvider)
    assert isinstance(
        provider.wrapped_provider,
        OpenAIEmbeddingProvider,
    )
    assert provider.provider_name == "openai"
    assert provider.wrapped_provider.model == ("text-embedding-3-small")
    assert provider.wrapped_provider.dimensions == 1536

    await provider.close()


class FakeProvider:
    """Record provider closure."""

    provider_name = "mock"

    def __init__(self) -> None:
        self.closed = 0

    async def embed(self, request: object) -> object:
        raise AssertionError("embed must not be called")

    async def close(self) -> None:
        self.closed += 1


class FakeObservabilityClient:
    """Record observability shutdown."""

    def __init__(self) -> None:
        self.shutdown_calls = 0
        self.shutdown_error: Exception | None = None

    def shutdown(self) -> None:
        self.shutdown_calls += 1
        if self.shutdown_error is not None:
            raise self.shutdown_error


class FakeQdrantClient:
    """Qdrant placeholder for lifecycle tests."""


class FakeEngine:
    """PostgreSQL engine placeholder for lifecycle tests."""


async def test_runtime_close_releases_every_resource_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = FakeProvider()
    qdrant_close_count = 0
    engine_dispose_count = 0

    async def close_qdrant(
        client: AsyncQdrantClient,
    ) -> None:
        nonlocal qdrant_close_count
        del client
        qdrant_close_count += 1

    async def dispose_engine(
        engine: AsyncEngine,
    ) -> None:
        nonlocal engine_dispose_count
        del engine
        engine_dispose_count += 1

    monkeypatch.setattr(
        composition,
        "close_qdrant_client",
        close_qdrant,
    )
    monkeypatch.setattr(
        composition,
        "dispose_postgresql_engine",
        dispose_engine,
    )

    settings = create_settings()
    profile = build_knowledge_index_profile(settings)
    observability_client = FakeObservabilityClient()

    runtime = KnowledgeIndexRuntime(
        settings=settings,
        engine=cast(
            AsyncEngine,
            FakeEngine(),
        ),
        session_factory=cast(
            async_sessionmaker[AsyncSession],
            object(),
        ),
        qdrant_client=cast(
            AsyncQdrantClient,
            FakeQdrantClient(),
        ),
        embedding_provider=cast(
            EmbeddingProvider,
            provider,
        ),
        index_profile=profile,
        chunker=cast(
            MarkdownTokenChunker,
            object(),
        ),
        vector_store=cast(
            QdrantKnowledgeVectorStore,
            object(),
        ),
        observability_client=cast(
            Any,
            observability_client,
        ),
    )

    await runtime.close()
    await runtime.close()

    assert provider.closed == 1
    assert qdrant_close_count == 1
    assert engine_dispose_count == 1
    assert observability_client.shutdown_calls == 1


async def test_runtime_close_attempts_all_resources_after_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = FakeProvider()
    qdrant_closed = False
    engine_disposed = False

    async def failing_provider_close() -> None:
        provider.closed += 1
        raise RuntimeError("provider close failed")

    monkeypatch.setattr(
        provider,
        "close",
        failing_provider_close,
    )

    async def close_qdrant(
        client: AsyncQdrantClient,
    ) -> None:
        nonlocal qdrant_closed
        del client
        qdrant_closed = True

    async def dispose_engine(
        engine: AsyncEngine,
    ) -> None:
        nonlocal engine_disposed
        del engine
        engine_disposed = True

    monkeypatch.setattr(
        composition,
        "close_qdrant_client",
        close_qdrant,
    )
    monkeypatch.setattr(
        composition,
        "dispose_postgresql_engine",
        dispose_engine,
    )

    settings = create_settings()
    observability_client = FakeObservabilityClient()
    runtime = KnowledgeIndexRuntime(
        settings=settings,
        engine=cast(
            AsyncEngine,
            FakeEngine(),
        ),
        session_factory=cast(
            async_sessionmaker[AsyncSession],
            object(),
        ),
        qdrant_client=cast(
            AsyncQdrantClient,
            FakeQdrantClient(),
        ),
        embedding_provider=cast(
            EmbeddingProvider,
            provider,
        ),
        index_profile=(build_knowledge_index_profile(settings)),
        chunker=cast(
            MarkdownTokenChunker,
            object(),
        ),
        vector_store=cast(
            QdrantKnowledgeVectorStore,
            object(),
        ),
        observability_client=cast(
            Any,
            observability_client,
        ),
    )

    with pytest.raises(
        RuntimeError,
        match="provider close failed",
    ):
        await runtime.close()

    assert qdrant_closed
    assert engine_disposed
    assert observability_client.shutdown_calls == 1


async def test_runtime_observability_shutdown_failure_does_not_raise(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = FakeProvider()
    observability_client = FakeObservabilityClient()
    observability_client.shutdown_error = RuntimeError("shutdown failed")

    async def close_qdrant(
        client: AsyncQdrantClient,
    ) -> None:
        del client

    async def dispose_engine(
        engine: AsyncEngine,
    ) -> None:
        del engine

    monkeypatch.setattr(
        composition,
        "close_qdrant_client",
        close_qdrant,
    )
    monkeypatch.setattr(
        composition,
        "dispose_postgresql_engine",
        dispose_engine,
    )

    settings = create_settings()
    runtime = KnowledgeIndexRuntime(
        settings=settings,
        engine=cast(
            AsyncEngine,
            FakeEngine(),
        ),
        session_factory=cast(
            async_sessionmaker[AsyncSession],
            object(),
        ),
        qdrant_client=cast(
            AsyncQdrantClient,
            FakeQdrantClient(),
        ),
        embedding_provider=cast(
            EmbeddingProvider,
            provider,
        ),
        index_profile=(build_knowledge_index_profile(settings)),
        chunker=cast(
            MarkdownTokenChunker,
            object(),
        ),
        vector_store=cast(
            QdrantKnowledgeVectorStore,
            object(),
        ),
        observability_client=cast(
            Any,
            observability_client,
        ),
    )

    await runtime.close()

    assert provider.closed == 1
    assert observability_client.shutdown_calls == 1


async def test_create_runtime_composes_one_observability_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = create_settings()
    observability_client = FakeObservabilityClient()
    factory_calls: list[Settings] = []

    def fake_observability_factory(
        configured_settings: Settings,
    ) -> FakeObservabilityClient:
        factory_calls.append(configured_settings)
        return observability_client

    monkeypatch.setattr(
        composition,
        "create_observability_client",
        fake_observability_factory,
    )
    monkeypatch.setattr(
        composition,
        "create_postgresql_engine",
        lambda configured_settings: FakeEngine(),
    )
    monkeypatch.setattr(
        composition,
        "create_postgresql_session_factory",
        lambda engine: object(),
    )
    monkeypatch.setattr(
        composition,
        "create_qdrant_client",
        lambda configured_settings: FakeQdrantClient(),
    )

    async def close_qdrant(client: object) -> None:
        del client

    async def dispose_engine(engine: object) -> None:
        del engine

    monkeypatch.setattr(
        composition,
        "close_qdrant_client",
        close_qdrant,
    )
    monkeypatch.setattr(
        composition,
        "dispose_postgresql_engine",
        dispose_engine,
    )

    runtime = await composition.create_knowledge_index_runtime(
        settings=settings,
    )

    try:
        assert runtime.observability_client is cast(Any, observability_client)
        assert factory_calls == [settings]
        assert isinstance(
            runtime.embedding_provider,
            ObservingEmbeddingProvider,
        )
        assert runtime.embedding_provider._observability_client is cast(
            Any,
            observability_client,
        )
        assert isinstance(
            runtime.embedding_provider.wrapped_provider,
            MockEmbeddingProvider,
        )
    finally:
        await runtime.close()

    assert observability_client.shutdown_calls == 1


def test_create_embedding_provider_uses_supplied_observability_client() -> None:
    observability_client = FakeObservabilityClient()

    provider = create_embedding_provider(
        create_settings(),
        observability_client=cast(Any, observability_client),
    )

    assert isinstance(provider, ObservingEmbeddingProvider)
    assert provider._observability_client is cast(
        Any,
        observability_client,
    )


async def test_partial_startup_after_client_shuts_client_down(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = create_settings()
    observability_client = FakeObservabilityClient()
    provider = FakeProvider()

    monkeypatch.setattr(
        composition,
        "create_observability_client",
        lambda configured_settings: observability_client,
    )
    monkeypatch.setattr(
        composition,
        "create_embedding_provider",
        lambda configured_settings, *, observability_client=None: cast(
            EmbeddingProvider,
            provider,
        ),
    )
    monkeypatch.setattr(
        composition,
        "create_postgresql_engine",
        lambda configured_settings: (_ for _ in ()).throw(RuntimeError("postgresql unavailable")),
    )

    with pytest.raises(RuntimeError, match="postgresql unavailable"):
        await composition.create_knowledge_index_runtime(
            settings=settings,
        )

    assert provider.closed == 1
    assert observability_client.shutdown_calls == 1


async def test_partial_startup_before_provider_skips_provider_close(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = create_settings()
    observability_client = FakeObservabilityClient()

    monkeypatch.setattr(
        composition,
        "create_observability_client",
        lambda configured_settings: observability_client,
    )

    def failing_embedding_provider(
        configured_settings: Settings,
        *,
        observability_client: object | None = None,
    ) -> EmbeddingProvider:
        del configured_settings, observability_client
        raise RuntimeError("embedding provider unavailable")

    monkeypatch.setattr(
        composition,
        "create_embedding_provider",
        failing_embedding_provider,
    )

    with pytest.raises(
        RuntimeError,
        match="embedding provider unavailable",
    ):
        await composition.create_knowledge_index_runtime(
            settings=settings,
        )

    assert observability_client.shutdown_calls == 1


async def test_partial_startup_qdrant_failure_closes_provider_and_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = create_settings()
    observability_client = FakeObservabilityClient()
    provider = FakeProvider()

    monkeypatch.setattr(
        composition,
        "create_observability_client",
        lambda configured_settings: observability_client,
    )
    monkeypatch.setattr(
        composition,
        "create_embedding_provider",
        lambda configured_settings, *, observability_client=None: cast(
            EmbeddingProvider,
            provider,
        ),
    )
    monkeypatch.setattr(
        composition,
        "create_postgresql_engine",
        lambda configured_settings: FakeEngine(),
    )
    monkeypatch.setattr(
        composition,
        "create_postgresql_session_factory",
        lambda engine: object(),
    )
    monkeypatch.setattr(
        composition,
        "create_qdrant_client",
        lambda configured_settings: (_ for _ in ()).throw(RuntimeError("qdrant unavailable")),
    )

    async def close_qdrant(client: object) -> None:
        del client

    async def dispose_engine(engine: object) -> None:
        del engine

    monkeypatch.setattr(
        composition,
        "close_qdrant_client",
        close_qdrant,
    )
    monkeypatch.setattr(
        composition,
        "dispose_postgresql_engine",
        dispose_engine,
    )

    with pytest.raises(RuntimeError, match="qdrant unavailable"):
        await composition.create_knowledge_index_runtime(
            settings=settings,
        )

    assert provider.closed == 1
    assert observability_client.shutdown_calls == 1
