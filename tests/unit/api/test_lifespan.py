"""Unit tests for application resource lifecycle."""

from unittest.mock import AsyncMock, MagicMock, patch

from fastapi import FastAPI
from qdrant_client import AsyncQdrantClient
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from supportops.api.lifespan import application_lifespan
from supportops.api.state import ApplicationState
from supportops.core.settings import Settings


def create_settings() -> Settings:
    """Create isolated settings for lifecycle unit tests."""

    return Settings(
        _env_file=None,
        environment="test",
        postgresql_url=("postgresql+asyncpg://supportops:local@localhost:5432/supportops"),
        qdrant_url="http://localhost:6333",
    )


async def test_application_lifespan_creates_and_releases_resources() -> None:
    app = FastAPI()
    settings = create_settings()
    engine = MagicMock(spec=AsyncEngine)
    session_factory = MagicMock(spec=async_sessionmaker[AsyncSession])
    qdrant_client = MagicMock(spec=AsyncQdrantClient)

    with (
        patch(
            "supportops.api.lifespan.configure_logging",
        ) as configure_logging,
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
            assert state.postgresql_engine is engine
            assert state.postgresql_session_factory is session_factory
            assert state.qdrant_client is qdrant_client

        configure_logging.assert_called_once_with(
            environment=settings.environment,
            log_level=settings.log_level,
        )
        close_client.assert_awaited_once_with(qdrant_client)
        dispose_engine.assert_awaited_once_with(engine)


async def test_application_lifespan_attempts_all_cleanup_operations() -> None:
    app = FastAPI()
    settings = create_settings()
    engine = MagicMock(spec=AsyncEngine)
    qdrant_client = MagicMock(spec=AsyncQdrantClient)
    close_client = AsyncMock(side_effect=RuntimeError("close failed"))
    dispose_engine = AsyncMock()

    with (
        patch("supportops.api.lifespan.configure_logging"),
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

    close_client.assert_awaited_once_with(qdrant_client)
    dispose_engine.assert_awaited_once_with(engine)
