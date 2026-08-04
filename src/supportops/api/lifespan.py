"""FastAPI application lifecycle management."""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from qdrant_client import AsyncQdrantClient
from sqlalchemy.ext.asyncio import AsyncEngine

from supportops.ai.embeddings.contracts import EmbeddingProvider
from supportops.api.state import ApplicationState
from supportops.core.logging import configure_logging
from supportops.core.settings import Settings
from supportops.infrastructure.postgresql import (
    create_postgresql_engine,
    create_postgresql_session_factory,
    dispose_postgresql_engine,
)
from supportops.infrastructure.qdrant import (
    close_qdrant_client,
    create_qdrant_client,
)
from supportops.knowledge_index.composition import (
    build_knowledge_index_profile,
    create_embedding_provider,
)
from supportops.modules.knowledge_documents.domain.models import (
    KnowledgeIndexProfile,
)
from supportops.observability.composition import create_observability_client
from supportops.observability.contracts import ObservabilityClient

logger = logging.getLogger(__name__)


@asynccontextmanager
async def application_lifespan(
    app: FastAPI,
    *,
    settings: Settings,
) -> AsyncIterator[None]:
    """Create and release resources owned by the API process."""

    configure_logging(
        environment=settings.environment,
        log_level=settings.log_level,
    )

    observability_client: ObservabilityClient | None = None
    knowledge_index_profile: KnowledgeIndexProfile | None = None
    embedding_provider: EmbeddingProvider | None = None
    postgresql_engine: AsyncEngine | None = None
    qdrant_client: AsyncQdrantClient | None = None

    try:
        observability_client = create_observability_client(settings)
        knowledge_index_profile = build_knowledge_index_profile(settings)
        embedding_provider = create_embedding_provider(
            settings,
            observability_client=observability_client,
        )
        postgresql_engine = create_postgresql_engine(settings)
        postgresql_session_factory = create_postgresql_session_factory(
            postgresql_engine,
        )
        qdrant_client = create_qdrant_client(settings)

        app.state.supportops = ApplicationState(
            settings=settings,
            embedding_provider=embedding_provider,
            knowledge_index_profile=knowledge_index_profile,
            postgresql_engine=postgresql_engine,
            postgresql_session_factory=postgresql_session_factory,
            qdrant_client=qdrant_client,
            observability_client=observability_client,
        )

        logger.info(
            "application_started",
            extra={
                "application_name": settings.application_name,
                "application_version": settings.application_version,
                "observability_provider": (settings.ai_observability_provider.value),
                "observability_enabled": observability_client.enabled,
                "observability_capture_mode": (settings.langfuse_capture_mode.value),
            },
        )

        yield
    finally:
        if embedding_provider is not None:
            try:
                await embedding_provider.close()
            except Exception:
                provider_name = (
                    knowledge_index_profile.embedding_provider
                    if knowledge_index_profile is not None
                    else "unavailable"
                )
                logger.exception(
                    "embedding_provider_close_failed",
                    extra={
                        "dependency": "embedding_provider",
                        "provider": provider_name,
                    },
                )

        if qdrant_client is not None:
            try:
                await close_qdrant_client(qdrant_client)
            except Exception:
                logger.exception(
                    "qdrant_client_close_failed",
                    extra={"dependency": "qdrant"},
                )

        if postgresql_engine is not None:
            try:
                await dispose_postgresql_engine(postgresql_engine)
            except Exception:
                logger.exception(
                    "postgresql_engine_disposal_failed",
                    extra={"dependency": "postgresql"},
                )

        if observability_client is not None:
            try:
                observability_client.shutdown()
            except Exception:
                logger.exception(
                    "observability_client_shutdown_failed",
                    extra={
                        "dependency": "observability",
                        "observability_provider": (settings.ai_observability_provider.value),
                    },
                )

        logger.info("application_stopped")
