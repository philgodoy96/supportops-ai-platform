"""Unit tests for bounded infrastructure health checks."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

from httpx import Headers
from qdrant_client.http.exceptions import (
    ResponseHandlingException,
    UnexpectedResponse,
)
from sqlalchemy.exc import OperationalError
from sqlalchemy.ext.asyncio import AsyncEngine

from supportops.infrastructure.health import DependencyStatus
from supportops.infrastructure.postgresql.health import (
    check_postgresql_health,
)
from supportops.infrastructure.qdrant.health import check_qdrant_health


def create_postgresql_engine_mock() -> tuple[AsyncEngine, AsyncMock]:
    """Create an async engine mock with a usable connection context."""

    connection = AsyncMock()
    connection_context = AsyncMock()
    connection_context.__aenter__.return_value = connection

    engine = MagicMock(spec=AsyncEngine)
    engine.connect.return_value = connection_context

    return engine, connection


async def test_postgresql_health_returns_healthy() -> None:
    engine, connection = create_postgresql_engine_mock()

    result = await check_postgresql_health(engine, timeout_seconds=1.0)

    connection.execute.assert_awaited_once()
    assert result.dependency == "postgresql"
    assert result.status is DependencyStatus.HEALTHY
    assert result.detail is None
    assert result.is_healthy is True


async def test_postgresql_health_handles_database_failure() -> None:
    engine, connection = create_postgresql_engine_mock()
    connection.execute.side_effect = OperationalError(
        statement="SELECT 1",
        params=None,
        orig=OSError("database unavailable"),
    )

    result = await check_postgresql_health(engine, timeout_seconds=1.0)

    assert result.status is DependencyStatus.UNHEALTHY
    assert result.detail == "connectivity check failed"
    assert result.is_healthy is False


async def test_postgresql_health_handles_timeout() -> None:
    engine, connection = create_postgresql_engine_mock()
    connection.execute.side_effect = asyncio.TimeoutError

    result = await check_postgresql_health(engine, timeout_seconds=1.0)

    assert result.status is DependencyStatus.UNHEALTHY
    assert result.detail == "connectivity check timed out"


async def test_qdrant_health_returns_healthy() -> None:
    client = AsyncMock()
    client.get_collections.return_value = MagicMock(collections=[])

    result = await check_qdrant_health(client, timeout_seconds=1.0)

    client.get_collections.assert_awaited_once_with()
    assert result.dependency == "qdrant"
    assert result.status is DependencyStatus.HEALTHY
    assert result.detail is None


async def test_qdrant_health_handles_provider_failure() -> None:
    client = AsyncMock()
    client.get_collections.side_effect = UnexpectedResponse(
        status_code=503,
        reason_phrase="Service Unavailable",
        content=b"",
        headers=Headers(),
    )

    result = await check_qdrant_health(client, timeout_seconds=1.0)

    assert result.status is DependencyStatus.UNHEALTHY
    assert result.detail == "connectivity check failed"


async def test_qdrant_health_handles_transport_failure() -> None:
    client = AsyncMock()
    client.get_collections.side_effect = ResponseHandlingException(
        OSError("connection refused"),
    )

    result = await check_qdrant_health(client, timeout_seconds=1.0)

    assert result.status is DependencyStatus.UNHEALTHY
    assert result.detail == "connectivity check failed"
    assert result.is_healthy is False


async def test_qdrant_health_handles_timeout() -> None:
    client = AsyncMock()
    client.get_collections.side_effect = asyncio.TimeoutError

    result = await check_qdrant_health(client, timeout_seconds=1.0)

    assert result.status is DependencyStatus.UNHEALTHY
    assert result.detail == "connectivity check timed out"
