"""Typed resources owned by the FastAPI application process."""

from dataclasses import dataclass

from qdrant_client import AsyncQdrantClient
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from supportops.core.settings import Settings


@dataclass(frozen=True, slots=True)
class ApplicationState:
    """Runtime resources created and owned by the application lifecycle."""

    settings: Settings
    postgresql_engine: AsyncEngine
    postgresql_session_factory: async_sessionmaker[AsyncSession]
    qdrant_client: AsyncQdrantClient
