"""Unit tests for knowledge indexing runtime composition."""

from typing import cast

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

    assert isinstance(
        provider,
        MockEmbeddingProvider,
    )
    assert provider.provider_name == "mock"
    assert provider.model == ("mock-hashing-embedding-v1")
    assert provider.dimensions == 64


async def test_creates_and_closes_openai_provider_without_network() -> None:
    settings = create_settings(
        embedding_provider=(EmbeddingProviderName.OPENAI),
        embedding_model="text-embedding-3-small",
        embedding_dimensions=1536,
        openai_api_key="test-openai-key",
    )

    provider = create_embedding_provider(settings)

    assert isinstance(
        provider,
        OpenAIEmbeddingProvider,
    )
    assert provider.provider_name == "openai"
    assert provider.model == ("text-embedding-3-small")
    assert provider.dimensions == 1536

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
    )

    await runtime.close()
    await runtime.close()

    assert provider.closed == 1
    assert qdrant_close_count == 1
    assert engine_dispose_count == 1


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
    )

    with pytest.raises(
        RuntimeError,
        match="provider close failed",
    ):
        await runtime.close()

    assert qdrant_closed
    assert engine_disposed
