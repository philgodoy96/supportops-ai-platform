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


async def test_application_lifespan_creates_and_releases_resources() -> None:
    app = FastAPI()
    settings = create_settings()
    profile = create_knowledge_index_profile()
    embedding_provider = create_embedding_provider_mock()
    engine = MagicMock(spec=AsyncEngine)
    session_factory = MagicMock(spec=async_sessionmaker[AsyncSession])
    qdrant_client = MagicMock(spec=AsyncQdrantClient)

    with (
        patch(
            "supportops.api.lifespan.configure_logging",
        ) as configure_logging,
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

        configure_logging.assert_called_once_with(
            environment=settings.environment,
            log_level=settings.log_level,
        )
        embedding_provider.close.assert_awaited_once_with()
        close_client.assert_awaited_once_with(qdrant_client)
        dispose_engine.assert_awaited_once_with(engine)


async def test_application_lifespan_attempts_all_cleanup_operations() -> None:
    app = FastAPI()
    settings = create_settings()
    profile = create_knowledge_index_profile()
    embedding_provider = create_embedding_provider_mock()
    embedding_provider.close = AsyncMock(side_effect=RuntimeError("close failed"))
    engine = MagicMock(spec=AsyncEngine)
    qdrant_client = MagicMock(spec=AsyncQdrantClient)
    close_client = AsyncMock(side_effect=RuntimeError("close failed"))
    dispose_engine = AsyncMock()

    with (
        patch("supportops.api.lifespan.configure_logging"),
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


async def test_application_lifespan_cleans_up_partial_startup_failure() -> None:
    app = FastAPI()
    settings = create_settings()
    profile = create_knowledge_index_profile()
    embedding_provider = create_embedding_provider_mock()
    engine = MagicMock(spec=AsyncEngine)
    close_client = AsyncMock()
    dispose_engine = AsyncMock()

    with (
        patch("supportops.api.lifespan.configure_logging"),
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

    embedding_provider.close.assert_awaited_once_with()
    close_client.assert_not_awaited()
    dispose_engine.assert_awaited_once_with(engine)
