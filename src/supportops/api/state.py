"""Typed resources owned by the FastAPI application process."""

from dataclasses import dataclass, field

from qdrant_client import AsyncQdrantClient
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from supportops.ai.embeddings.contracts import EmbeddingProvider
from supportops.core.settings import Settings
from supportops.modules.knowledge_documents.domain.models import (
    KnowledgeIndexProfile,
)
from supportops.observability.contracts import ObservabilityClient
from supportops.observability.noop import NoOpObservabilityClient


@dataclass(frozen=True, slots=True)
class ApplicationState:
    """Runtime resources created and owned by the application lifecycle."""

    settings: Settings
    embedding_provider: EmbeddingProvider
    knowledge_index_profile: KnowledgeIndexProfile
    postgresql_engine: AsyncEngine
    postgresql_session_factory: async_sessionmaker[AsyncSession]
    qdrant_client: AsyncQdrantClient
    observability_client: ObservabilityClient = field(
        default_factory=NoOpObservabilityClient,
    )
