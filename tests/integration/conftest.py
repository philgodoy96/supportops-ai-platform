"""Shared fixtures for infrastructure integration tests."""

from __future__ import annotations

import asyncio
import subprocess
import sys
from collections.abc import AsyncIterator, Callable, Iterator, Mapping

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from sqlalchemy import text
from sqlalchemy.exc import ProgrammingError
from sqlalchemy.ext.asyncio import (
    AsyncConnection,
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
)

from supportops.api.application import create_application
from supportops.core.settings import Settings
from supportops.infrastructure.postgresql import (
    create_postgresql_engine,
    create_postgresql_session_factory,
    dispose_postgresql_engine,
)

if sys.platform == "win32":

    def pytest_asyncio_loop_factories(
        config: object,
        item: object,
    ) -> Mapping[str, Callable[[], asyncio.AbstractEventLoop]]:
        """Force SelectorEventLoop so Psycopg can run under Windows asyncio."""

        del config, item
        return {"selector": asyncio.SelectorEventLoop}


# Session-level advisory lock shared by every integration test that mutates the
# local PostgreSQL database. Concurrent pytest processes otherwise race on
# cleanup and flake with foreign-key / unique violations.
_INTEGRATION_DATABASE_LOCK_KEY = 742_891_305

# Citations RESTRICT-reference knowledge_document_chunks, so clear them first.
BUSINESS_DATA_DELETE_STATEMENTS: tuple[str, ...] = (
    "DELETE FROM support_recommendation_citations",
    "DELETE FROM support_recommendations",
    "DELETE FROM sensitive_execution_grants",
    "DELETE FROM approval_requests",
    "DELETE FROM agent_tool_calls",
    "DELETE FROM knowledge_document_chunks",
    "UPDATE knowledge_documents SET active_version_id = NULL",
    "DELETE FROM knowledge_document_versions",
    "DELETE FROM knowledge_documents",
    "DELETE FROM ticket_classifications",
    "DELETE FROM llm_invocations",
    "DELETE FROM agent_run_attempts",
    "DELETE FROM agent_runs",
    "DELETE FROM tickets",
    "DELETE FROM workspaces",
)


def run_alembic_upgrade_head() -> None:
    """Apply Alembic migrations through head or raise on failure."""

    result = subprocess.run(
        [sys.executable, "-m", "alembic", "upgrade", "head"],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Failed to upgrade integration database to Alembic head:\n{result.stderr}"
        )


async def clear_integration_business_data(
    connection: AsyncConnection,
) -> None:
    """Delete all shared business rows in FK-safe order."""

    for statement in BUSINESS_DATA_DELETE_STATEMENTS:
        await connection.execute(text(statement))


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

    try:
        run_alembic_upgrade_head()
    except RuntimeError as error:
        pytest.fail(str(error))

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
    # Commit so CREATE INDEX CONCURRENTLY (checkpoint setup) is not blocked.
    await connection.commit()

    try:
        yield None
    finally:
        # Schema-mutating tests may leave the shared DB below head on failure
        # or interrupt; restore before releasing the lock so later fixtures
        # (e.g. clean_business_tables) do not hit missing relations.
        run_alembic_upgrade_head()
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
        try:
            await clear_integration_business_data(lock_connection)
            await lock_connection.commit()
        except ProgrammingError:
            # A prior schema-mutating test/interrupt can leave head-shaped code
            # pointed at a DB missing newer relations such as approval_requests.
            await lock_connection.rollback()
            run_alembic_upgrade_head()
            await clear_integration_business_data(lock_connection)
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
