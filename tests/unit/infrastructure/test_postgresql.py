"""Unit tests for PostgreSQL infrastructure factories."""

from typing import Any, cast

from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession

from supportops.core.settings import Settings
from supportops.infrastructure.postgresql import (
    create_postgresql_engine,
    create_postgresql_session_factory,
    dispose_postgresql_engine,
)


def create_settings() -> Settings:
    """Create isolated settings for infrastructure unit tests."""

    settings_type = cast(Any, Settings)
    return cast(
        Settings,
        settings_type(
            _env_file=None,
            postgresql_url=("postgresql+asyncpg://supportops:local@localhost:5432/supportops"),
            postgresql_pool_size=7,
            postgresql_max_overflow=3,
            postgresql_pool_timeout_seconds=4,
            qdrant_url="http://localhost:6333",
        ),
    )


async def test_create_postgresql_engine_uses_asyncpg() -> None:
    engine = create_postgresql_engine(create_settings())

    try:
        assert isinstance(engine, AsyncEngine)
        assert engine.url.drivername == "postgresql+asyncpg"
        assert engine.url.database == "supportops"
        assert engine.url.username == "supportops"
        assert engine.url.password == "local"
    finally:
        await dispose_postgresql_engine(engine)


async def test_create_postgresql_session_factory_uses_expected_behavior() -> None:
    engine = create_postgresql_engine(create_settings())
    session_factory = create_postgresql_session_factory(engine)

    try:
        async with session_factory() as session:
            assert isinstance(session, AsyncSession)
            assert session.autoflush is False
            assert session.sync_session.expire_on_commit is False
    finally:
        await dispose_postgresql_engine(engine)
