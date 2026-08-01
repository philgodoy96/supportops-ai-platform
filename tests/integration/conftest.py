"""Shared fixtures for infrastructure integration tests."""

import subprocess
import sys
from collections.abc import AsyncIterator, Iterator

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from supportops.api.application import create_application
from supportops.core.settings import Settings
from supportops.infrastructure.postgresql import (
    create_postgresql_engine,
    create_postgresql_session_factory,
    dispose_postgresql_engine,
)

# Session-level advisory lock shared by every integration test that mutates the
# local PostgreSQL database. Concurrent pytest processes otherwise race on
# cleanup and flake with foreign-key / unique violations.
_INTEGRATION_DATABASE_LOCK_KEY = 742_891_305


@pytest.fixture
def integration_settings() -> Settings:
    """Load validated settings from the integration environment."""

    return Settings()


@pytest.fixture
def integration_application(
    integration_settings: Settings,
) -> FastAPI:
    """Create an application configured for live infrastructure."""

    return create_application(integration_settings)


@pytest.fixture
async def integration_client(
    integration_application: FastAPI,
) -> AsyncIterator[AsyncClient]:
    """Create an HTTP client with the real application lifecycle enabled."""

    async with (
        integration_application.router.lifespan_context(integration_application),
        AsyncClient(
            transport=ASGITransport(app=integration_application),
            base_url="http://test",
        ) as client,
    ):
        yield client


@pytest.fixture(scope="session")
def migrated_database() -> Iterator[None]:
    """Apply Alembic migrations to the shared local integration database."""

    result = subprocess.run(
        [sys.executable, "-m", "alembic", "upgrade", "head"],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        pytest.fail(result.stderr)

    yield None


@pytest.fixture
async def postgresql_engine(
    migrated_database: None,
    integration_settings: Settings,
) -> AsyncIterator[AsyncEngine]:
    """Create a disposable async engine for one integration test."""

    engine = create_postgresql_engine(integration_settings)

    try:
        yield engine
    finally:
        await dispose_postgresql_engine(engine)


@pytest.fixture
def postgresql_session_factory(
    postgresql_engine: AsyncEngine,
) -> async_sessionmaker[AsyncSession]:
    """Create a session factory bound to the test engine."""

    return create_postgresql_session_factory(postgresql_engine)


@pytest.fixture
async def postgresql_session(
    postgresql_session_factory: async_sessionmaker[AsyncSession],
) -> AsyncIterator[AsyncSession]:
    """Open one async session and roll back any leftover transaction."""

    async with postgresql_session_factory() as session:
        try:
            yield session
        finally:
            if session.in_transaction():
                await session.rollback()


@pytest.fixture
async def exclusive_integration_database(
    integration_settings: Settings,
) -> AsyncIterator[None]:
    """Hold the shared integration DB lock for schema-mutating tests."""

    engine = create_postgresql_engine(integration_settings)
    connection = await engine.connect()
    await connection.execute(
        text(f"SELECT pg_advisory_lock({_INTEGRATION_DATABASE_LOCK_KEY})"),
    )

    try:
        yield None
    finally:
        await connection.execute(
            text(f"SELECT pg_advisory_unlock({_INTEGRATION_DATABASE_LOCK_KEY})"),
        )
        await connection.commit()
        await connection.close()
        await dispose_postgresql_engine(engine)


@pytest.fixture
async def clean_business_tables(
    postgresql_engine: AsyncEngine,
) -> AsyncIterator[None]:
    """Serialize and reset business rows around one integration test."""

    lock_connection = await postgresql_engine.connect()
    await lock_connection.execute(
        text(f"SELECT pg_advisory_lock({_INTEGRATION_DATABASE_LOCK_KEY})"),
    )

    async def cleanup() -> None:
        await lock_connection.execute(text("DELETE FROM knowledge_document_chunks"))
        await lock_connection.execute(
            text("UPDATE knowledge_documents SET active_version_id = NULL"),
        )
        await lock_connection.execute(text("DELETE FROM knowledge_document_versions"))
        await lock_connection.execute(text("DELETE FROM knowledge_documents"))
        await lock_connection.execute(text("DELETE FROM ticket_classifications"))
        await lock_connection.execute(text("DELETE FROM llm_invocations"))
        await lock_connection.execute(text("DELETE FROM agent_run_attempts"))
        await lock_connection.execute(text("DELETE FROM agent_runs"))
        await lock_connection.execute(text("DELETE FROM tickets"))
        await lock_connection.execute(text("DELETE FROM workspaces"))
        await lock_connection.commit()

    try:
        await cleanup()
        yield None
        await cleanup()
    finally:
        if lock_connection.in_transaction():
            await lock_connection.rollback()
        await lock_connection.execute(
            text(f"SELECT pg_advisory_unlock({_INTEGRATION_DATABASE_LOCK_KEY})"),
        )
        await lock_connection.commit()
        await lock_connection.close()
