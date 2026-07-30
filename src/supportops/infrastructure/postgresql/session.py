"""Async SQLAlchemy session factory construction."""

from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker


def create_postgresql_session_factory(
    engine: AsyncEngine,
) -> async_sessionmaker[AsyncSession]:
    """Create the process-owned async session factory."""

    return async_sessionmaker(
        bind=engine,
        class_=AsyncSession,
        autoflush=False,
        expire_on_commit=False,
    )
