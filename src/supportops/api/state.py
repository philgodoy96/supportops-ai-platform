"""Typed resources owned by the FastAPI application process."""

from dataclasses import dataclass

from qdrant_client import AsyncQdrantClient
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from supportops.ai.embeddings.contracts import EmbeddingProvider
from supportops.core.settings import Settings
from supportops.modules.knowledge_documents.domain.models import (
    KnowledgeIndexProfile,
)


@dataclass(frozen=True, slots=True)
class ApplicationState:
    """Runtime resources created and owned by the application lifecycle."""

    settings: Settings
    embedding_provider: EmbeddingProvider
    knowledge_index_profile: KnowledgeIndexProfile
    postgresql_engine: AsyncEngine
    postgresql_session_factory: async_sessionmaker[AsyncSession]
    qdrant_client: AsyncQdrantClient
