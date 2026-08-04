"""Unit tests for application resource lifecycle."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from qdrant_client import AsyncQdrantClient
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from supportops.ai.embeddings.contracts import EmbeddingProvider
from supportops.api.lifespan import application_lifespan
from supportops.api.state import ApplicationState
from supportops.core.settings import Settings
from supportops.modules.knowledge_documents.domain.models import (
    KnowledgeIndexProfile,
)
from supportops.observability.contracts import ObservabilityClient
from supportops.observability.models import ObservabilityProvider


def create_settings() -> Settings:
    """Create isolated settings for lifecycle unit tests."""

    return Settings(
        _env_file=None,
        environment="test",
        postgresql_url=("postgresql+asyncpg://supportops:local@localhost:5432/supportops"),
        qdrant_url="http://localhost:6333",
    )


def create_knowledge_index_profile() -> KnowledgeIndexProfile:
    """Create one valid profile matching mock settings defaults."""

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


def create_embedding_provider_mock() -> MagicMock:
    """Create a fake process-scoped embedding provider."""

    provider = MagicMock(spec=EmbeddingProvider)
    provider.close = AsyncMock()
    return provider


def create_observability_client_mock() -> MagicMock:
    """Create a fake process-scoped observability client."""

    client = MagicMock(spec=ObservabilityClient)
    client.provider = ObservabilityProvider.NOOP
    client.enabled = False
    client.shutdown = MagicMock()
    return client


async def test_application_lifespan_creates_and_releases_resources() -> None:
    app = FastAPI()
    settings = create_settings()
    profile = create_knowledge_index_profile()
    embedding_provider = create_embedding_provider_mock()
    observability_client = create_observability_client_mock()
    engine = MagicMock(spec=AsyncEngine)
    session_factory = MagicMock(spec=async_sessionmaker[AsyncSession])
    qdrant_client = MagicMock(spec=AsyncQdrantClient)

    with (
        patch(
            "supportops.api.lifespan.configure_logging",
        ) as configure_logging,
        patch(
            "supportops.api.lifespan.create_observability_client",
            return_value=observability_client,
        ) as create_client,
        patch(
            "supportops.api.lifespan.build_knowledge_index_profile",
            return_value=profile,
        ),
        patch(
            "supportops.api.lifespan.create_embedding_provider",
            return_value=embedding_provider,
        ) as create_provider,
        patch(
            "supportops.api.lifespan.create_postgresql_engine",
            return_value=engine,
        ),
        patch(
            "supportops.api.lifespan.create_postgresql_session_factory",
            return_value=session_factory,
        ),
        patch(
            "supportops.api.lifespan.create_qdrant_client",
            return_value=qdrant_client,
        ),
        patch(
            "supportops.api.lifespan.dispose_postgresql_engine",
            new=AsyncMock(),
        ) as dispose_engine,
        patch(
            "supportops.api.lifespan.close_qdrant_client",
            new=AsyncMock(),
        ) as close_client,
    ):
        async with application_lifespan(app, settings=settings):
            state = app.state.supportops

            assert isinstance(state, ApplicationState)
            assert state.settings is settings
            assert state.embedding_provider is embedding_provider
            assert state.knowledge_index_profile is profile
            assert state.postgresql_engine is engine
            assert state.postgresql_session_factory is session_factory
            assert state.qdrant_client is qdrant_client
            assert state.observability_client is observability_client

        configure_logging.assert_called_once_with(
            environment=settings.environment,
            log_level=settings.log_level,
        )
        create_client.assert_called_once_with(settings)
        create_provider.assert_called_once_with(
            settings,
            observability_client=observability_client,
        )
        embedding_provider.close.assert_awaited_once_with()
        close_client.assert_awaited_once_with(qdrant_client)
        dispose_engine.assert_awaited_once_with(engine)
        observability_client.shutdown.assert_called_once_with()


async def test_application_lifespan_creates_one_observability_client() -> None:
    app = FastAPI()
    settings = create_settings()
    observability_client = create_observability_client_mock()
    embedding_provider = create_embedding_provider_mock()

    with (
        patch("supportops.api.lifespan.configure_logging"),
        patch(
            "supportops.api.lifespan.create_observability_client",
            return_value=observability_client,
        ) as create_client,
        patch(
            "supportops.api.lifespan.build_knowledge_index_profile",
            return_value=create_knowledge_index_profile(),
        ),
        patch(
            "supportops.api.lifespan.create_embedding_provider",
            return_value=embedding_provider,
        ) as create_provider,
        patch(
            "supportops.api.lifespan.create_postgresql_engine",
            return_value=MagicMock(spec=AsyncEngine),
        ),
        patch(
            "supportops.api.lifespan.create_postgresql_session_factory",
            return_value=MagicMock(),
        ),
        patch(
            "supportops.api.lifespan.create_qdrant_client",
            return_value=MagicMock(spec=AsyncQdrantClient),
        ),
        patch(
            "supportops.api.lifespan.dispose_postgresql_engine",
            new=AsyncMock(),
        ),
        patch(
            "supportops.api.lifespan.close_qdrant_client",
            new=AsyncMock(),
        ),
    ):
        async with application_lifespan(app, settings=settings):
            assert app.state.supportops.observability_client is (observability_client)
            assert app.state.supportops.embedding_provider is embedding_provider

        create_client.assert_called_once_with(settings)
        create_provider.assert_called_once_with(
            settings,
            observability_client=observability_client,
        )
        embedding_provider.close.assert_awaited_once_with()
        observability_client.shutdown.assert_called_once_with()


async def test_application_lifespan_closes_provider_before_observability() -> None:
    app = FastAPI()
    settings = create_settings()
    embedding_provider = create_embedding_provider_mock()
    observability_client = create_observability_client_mock()
    call_order: list[str] = []

    async def record_provider_close() -> None:
        call_order.append("provider_close")

    def record_observability_shutdown() -> None:
        call_order.append("observability_shutdown")

    embedding_provider.close = AsyncMock(side_effect=record_provider_close)
    observability_client.shutdown = MagicMock(
        side_effect=record_observability_shutdown,
    )

    with (
        patch("supportops.api.lifespan.configure_logging"),
        patch(
            "supportops.api.lifespan.create_observability_client",
            return_value=observability_client,
        ),
        patch(
            "supportops.api.lifespan.build_knowledge_index_profile",
            return_value=create_knowledge_index_profile(),
        ),
        patch(
            "supportops.api.lifespan.create_embedding_provider",
            return_value=embedding_provider,
        ),
        patch(
            "supportops.api.lifespan.create_postgresql_engine",
            return_value=MagicMock(spec=AsyncEngine),
        ),
        patch(
            "supportops.api.lifespan.create_postgresql_session_factory",
            return_value=MagicMock(),
        ),
        patch(
            "supportops.api.lifespan.create_qdrant_client",
            return_value=MagicMock(spec=AsyncQdrantClient),
        ),
        patch(
            "supportops.api.lifespan.dispose_postgresql_engine",
            new=AsyncMock(),
        ),
        patch(
            "supportops.api.lifespan.close_qdrant_client",
            new=AsyncMock(),
        ),
    ):
        async with application_lifespan(app, settings=settings):
            pass

    assert call_order[0] == "provider_close"
    assert call_order[-1] == "observability_shutdown"


async def test_application_lifespan_attempts_all_cleanup_operations() -> None:
    app = FastAPI()
    settings = create_settings()
    profile = create_knowledge_index_profile()
    embedding_provider = create_embedding_provider_mock()
    embedding_provider.close = AsyncMock(side_effect=RuntimeError("close failed"))
    observability_client = create_observability_client_mock()
    observability_client.shutdown = MagicMock(side_effect=RuntimeError("shutdown failed"))
    engine = MagicMock(spec=AsyncEngine)
    qdrant_client = MagicMock(spec=AsyncQdrantClient)
    close_client = AsyncMock(side_effect=RuntimeError("close failed"))
    dispose_engine = AsyncMock()

    with (
        patch("supportops.api.lifespan.configure_logging"),
        patch(
            "supportops.api.lifespan.create_observability_client",
            return_value=observability_client,
        ),
        patch(
            "supportops.api.lifespan.build_knowledge_index_profile",
            return_value=profile,
        ),
        patch(
            "supportops.api.lifespan.create_embedding_provider",
            return_value=embedding_provider,
        ),
        patch(
            "supportops.api.lifespan.create_postgresql_engine",
            return_value=engine,
        ),
        patch(
            "supportops.api.lifespan.create_postgresql_session_factory",
            return_value=MagicMock(),
        ),
        patch(
            "supportops.api.lifespan.create_qdrant_client",
            return_value=qdrant_client,
        ),
        patch(
            "supportops.api.lifespan.close_qdrant_client",
            new=close_client,
        ),
        patch(
            "supportops.api.lifespan.dispose_postgresql_engine",
            new=dispose_engine,
        ),
    ):
        async with application_lifespan(app, settings=settings):
            pass

    embedding_provider.close.assert_awaited_once_with()
    close_client.assert_awaited_once_with(qdrant_client)
    dispose_engine.assert_awaited_once_with(engine)
    observability_client.shutdown.assert_called_once_with()


async def test_application_lifespan_cleans_up_partial_startup_failure() -> None:
    app = FastAPI()
    settings = create_settings()
    profile = create_knowledge_index_profile()
    embedding_provider = create_embedding_provider_mock()
    observability_client = create_observability_client_mock()
    engine = MagicMock(spec=AsyncEngine)
    close_client = AsyncMock()
    dispose_engine = AsyncMock()

    with (
        patch("supportops.api.lifespan.configure_logging"),
        patch(
            "supportops.api.lifespan.create_observability_client",
            return_value=observability_client,
        ),
        patch(
            "supportops.api.lifespan.build_knowledge_index_profile",
            return_value=profile,
        ),
        patch(
            "supportops.api.lifespan.create_embedding_provider",
            return_value=embedding_provider,
        ) as create_provider,
        patch(
            "supportops.api.lifespan.create_postgresql_engine",
            return_value=engine,
        ),
        patch(
            "supportops.api.lifespan.create_postgresql_session_factory",
            return_value=MagicMock(),
        ),
        patch(
            "supportops.api.lifespan.create_qdrant_client",
            side_effect=RuntimeError("qdrant unavailable"),
        ),
        patch(
            "supportops.api.lifespan.close_qdrant_client",
            new=close_client,
        ),
        patch(
            "supportops.api.lifespan.dispose_postgresql_engine",
            new=dispose_engine,
        ),
        pytest.raises(RuntimeError, match="qdrant unavailable"),
    ):
        async with application_lifespan(app, settings=settings):
            pass

    create_provider.assert_called_once_with(
        settings,
        observability_client=observability_client,
    )
    embedding_provider.close.assert_awaited_once_with()
    close_client.assert_not_awaited()
    dispose_engine.assert_awaited_once_with(engine)
    observability_client.shutdown.assert_called_once_with()


async def test_application_lifespan_default_noop_needs_no_credentials() -> None:
    app = FastAPI()
    settings = create_settings()
    assert settings.ai_observability_provider is ObservabilityProvider.NOOP
    assert settings.langfuse_public_key is None
    assert settings.langfuse_secret_key is None

    with (
        patch("supportops.api.lifespan.configure_logging"),
        patch(
            "supportops.api.lifespan.build_knowledge_index_profile",
            return_value=create_knowledge_index_profile(),
        ),
        patch(
            "supportops.api.lifespan.create_embedding_provider",
            return_value=create_embedding_provider_mock(),
        ),
        patch(
            "supportops.api.lifespan.create_postgresql_engine",
            return_value=MagicMock(spec=AsyncEngine),
        ),
        patch(
            "supportops.api.lifespan.create_postgresql_session_factory",
            return_value=MagicMock(),
        ),
        patch(
            "supportops.api.lifespan.create_qdrant_client",
            return_value=MagicMock(spec=AsyncQdrantClient),
        ),
        patch(
            "supportops.api.lifespan.dispose_postgresql_engine",
            new=AsyncMock(),
        ),
        patch(
            "supportops.api.lifespan.close_qdrant_client",
            new=AsyncMock(),
        ),
    ):
        async with application_lifespan(app, settings=settings):
            client = app.state.supportops.observability_client
            assert client.provider is ObservabilityProvider.NOOP
            assert client.enabled is False


def test_readiness_does_not_depend_on_observability() -> None:
    import inspect

    from supportops.api.health.service import build_readiness_response

    parameter_names = set(inspect.signature(build_readiness_response).parameters)

    assert "observability_client" not in parameter_names
    assert "observability" not in parameter_names


def test_search_knowledge_dependency_shares_process_observability_client() -> None:
    from supportops.knowledge_retrieval.api.dependencies import (
        get_search_knowledge,
    )

    settings = create_settings()
    profile = create_knowledge_index_profile()
    embedding_provider = create_embedding_provider_mock()
    embedding_provider.provider_name = profile.embedding_provider
    observability_client = create_observability_client_mock()
    session = MagicMock(spec=AsyncSession)
    state = ApplicationState(
        settings=settings,
        embedding_provider=embedding_provider,
        knowledge_index_profile=profile,
        postgresql_engine=MagicMock(spec=AsyncEngine),
        postgresql_session_factory=MagicMock(
            spec=async_sessionmaker[AsyncSession],
        ),
        qdrant_client=MagicMock(spec=AsyncQdrantClient),
        observability_client=observability_client,
    )

    with (
        patch(
            "supportops.knowledge_retrieval.api.dependencies.QdrantKnowledgeVectorStore",
        ),
        patch(
            "supportops.knowledge_retrieval.api.dependencies.QdrantKnowledgeVectorSearcher",
        ),
        patch(
            "supportops.knowledge_retrieval.api.dependencies.SqlAlchemyActiveKnowledgeVersionResolver",
        ),
        patch(
            "supportops.knowledge_retrieval.api.dependencies.SqlAlchemyKnowledgeChunkHydrator",
        ),
    ):
        service = get_search_knowledge(session=session, state=state)
        second = get_search_knowledge(session=session, state=state)

    assert service._observability_client is observability_client
    assert state.embedding_provider is embedding_provider
    assert state.observability_client is observability_client
    assert second._observability_client is observability_client
    assert second._observability_client is service._observability_client
