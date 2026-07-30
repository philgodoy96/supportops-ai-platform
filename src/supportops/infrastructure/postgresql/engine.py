"""Async SQLAlchemy engine construction and disposal."""

from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from supportops.core.settings import Settings


def create_postgresql_engine(settings: Settings) -> AsyncEngine:
    """Create an async SQLAlchemy engine from validated settings."""

    return create_async_engine(
        str(settings.postgresql_url),
        pool_pre_ping=True,
        pool_size=settings.postgresql_pool_size,
        max_overflow=settings.postgresql_max_overflow,
        pool_timeout=settings.postgresql_pool_timeout_seconds,
    )


async def dispose_postgresql_engine(engine: AsyncEngine) -> None:
    """Release connections owned by an async SQLAlchemy engine."""

    await engine.dispose()
