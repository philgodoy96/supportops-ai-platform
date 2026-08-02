"""Unit tests for operational health endpoints."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from qdrant_client import AsyncQdrantClient
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from supportops.ai.embeddings.contracts import EmbeddingProvider
from supportops.api.application import create_application
from supportops.api.health.schemas import ApplicationHealthStatus
from supportops.api.state import ApplicationState
from supportops.core.settings import Settings
from supportops.infrastructure.health import (
    DependencyCheckResult,
    DependencyStatus,
)
from supportops.modules.knowledge_documents.domain.models import (
    KnowledgeIndexProfile,
)


def create_settings() -> Settings:
    """Create isolated settings for health endpoint tests."""

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


def create_test_application() -> FastAPI:
    """Create an application with isolated in-memory lifecycle state."""

    settings = create_settings()
    embedding_provider = create_embedding_provider_mock()
    knowledge_index_profile = create_knowledge_index_profile()
    engine = MagicMock(spec=AsyncEngine)
    session_factory = MagicMock(spec=async_sessionmaker[AsyncSession])
    qdrant_client = MagicMock(spec=AsyncQdrantClient)

    @asynccontextmanager
    async def test_lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.supportops = ApplicationState(
            settings=settings,
            embedding_provider=embedding_provider,
            knowledge_index_profile=knowledge_index_profile,
            postgresql_engine=engine,
            postgresql_session_factory=session_factory,
            qdrant_client=qdrant_client,
        )
        yield

    app = create_application(settings)
    app.router.lifespan_context = test_lifespan

    return app


async def test_liveness_does_not_check_dependencies() -> None:
    app = create_test_application()

    with (
        patch(
            "supportops.api.health.router.check_postgresql_health",
            new=AsyncMock(),
        ) as postgresql_check,
        patch(
            "supportops.api.health.router.check_qdrant_health",
            new=AsyncMock(),
        ) as qdrant_check,
    ):
        async with app.router.lifespan_context(app):
            async with AsyncClient(
                transport=ASGITransport(app=app),
                base_url="http://test",
            ) as client:
                response = await client.get("/health/live")

    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}
    postgresql_check.assert_not_awaited()
    qdrant_check.assert_not_awaited()


async def test_readiness_returns_success_for_healthy_dependencies() -> None:
    app = create_test_application()

    with (
        patch(
            "supportops.api.health.router.check_postgresql_health",
            new=AsyncMock(
                return_value=DependencyCheckResult(
                    dependency="postgresql",
                    status=DependencyStatus.HEALTHY,
                )
            ),
        ),
        patch(
            "supportops.api.health.router.check_qdrant_health",
            new=AsyncMock(
                return_value=DependencyCheckResult(
                    dependency="qdrant",
                    status=DependencyStatus.HEALTHY,
                )
            ),
        ),
    ):
        async with app.router.lifespan_context(app):
            async with AsyncClient(
                transport=ASGITransport(app=app),
                base_url="http://test",
            ) as client:
                response = await client.get("/health/ready")

    assert response.status_code == 200
    assert response.json() == {
        "status": "healthy",
        "dependencies": {
            "postgresql": {
                "status": "healthy",
                "detail": None,
            },
            "qdrant": {
                "status": "healthy",
                "detail": None,
            },
        },
    }


async def test_readiness_returns_service_unavailable_for_dependency_failure() -> None:
    app = create_test_application()

    with (
        patch(
            "supportops.api.health.router.check_postgresql_health",
            new=AsyncMock(
                return_value=DependencyCheckResult(
                    dependency="postgresql",
                    status=DependencyStatus.UNHEALTHY,
                    detail="connectivity check failed",
                )
            ),
        ),
        patch(
            "supportops.api.health.router.check_qdrant_health",
            new=AsyncMock(
                return_value=DependencyCheckResult(
                    dependency="qdrant",
                    status=DependencyStatus.HEALTHY,
                )
            ),
        ),
    ):
        async with app.router.lifespan_context(app):
            async with AsyncClient(
                transport=ASGITransport(app=app),
                base_url="http://test",
            ) as client:
                response = await client.get("/health/ready")

    assert response.status_code == 503
    assert response.json()["status"] == ApplicationHealthStatus.UNHEALTHY
    assert response.json()["dependencies"]["postgresql"] == {
        "status": "unhealthy",
        "detail": "connectivity check failed",
    }


async def test_readiness_response_does_not_expose_connection_details() -> None:
    app = create_test_application()

    with (
        patch(
            "supportops.api.health.router.check_postgresql_health",
            new=AsyncMock(
                return_value=DependencyCheckResult(
                    dependency="postgresql",
                    status=DependencyStatus.UNHEALTHY,
                    detail="connectivity check failed",
                )
            ),
        ),
        patch(
            "supportops.api.health.router.check_qdrant_health",
            new=AsyncMock(
                return_value=DependencyCheckResult(
                    dependency="qdrant",
                    status=DependencyStatus.UNHEALTHY,
                    detail="connectivity check timed out",
                )
            ),
        ),
    ):
        async with app.router.lifespan_context(app):
            async with AsyncClient(
                transport=ASGITransport(app=app),
                base_url="http://test",
            ) as client:
                response = await client.get("/health/ready")

    response_body = response.text.lower()

    assert "supportops-local" not in response_body
    assert "localhost" not in response_body
    assert "asyncpg" not in response_body
    assert "6333" not in response_body
