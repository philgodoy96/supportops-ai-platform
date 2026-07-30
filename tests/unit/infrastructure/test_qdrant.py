"""Unit tests for Qdrant infrastructure factories."""

from typing import Any, cast
from unittest.mock import AsyncMock

from qdrant_client import AsyncQdrantClient

from supportops.core.settings import Settings
from supportops.infrastructure.qdrant import (
    close_qdrant_client,
    create_qdrant_client,
)


def create_settings(**overrides: object) -> Settings:
    """Create isolated settings for Qdrant unit tests."""

    values: dict[str, object] = {
        "postgresql_url": ("postgresql+asyncpg://supportops:local@localhost:5432/supportops"),
        "qdrant_url": "http://localhost:6333",
    }
    values.update(overrides)

    settings_type = cast(Any, Settings)
    return cast(Settings, settings_type(_env_file=None, **values))


async def test_create_qdrant_client_returns_async_client() -> None:
    client = create_qdrant_client(create_settings())

    try:
        assert isinstance(client, AsyncQdrantClient)
    finally:
        await close_qdrant_client(client)


async def test_close_qdrant_client_closes_owned_resources() -> None:
    client = AsyncMock(spec=AsyncQdrantClient)

    await close_qdrant_client(client)

    client.close.assert_awaited_once_with()
