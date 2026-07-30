"""Integration tests for Qdrant connectivity."""

import pytest

from supportops.core.settings import Settings
from supportops.infrastructure.health import DependencyStatus
from supportops.infrastructure.qdrant import (
    check_qdrant_health,
    close_qdrant_client,
    create_qdrant_client,
)

pytestmark = pytest.mark.integration


async def test_qdrant_connectivity(
    integration_settings: Settings,
) -> None:
    client = create_qdrant_client(integration_settings)

    try:
        result = await check_qdrant_health(
            client,
            integration_settings.dependency_health_timeout_seconds,
        )
    finally:
        await close_qdrant_client(client)

    assert result.dependency == "qdrant"
    assert result.status is DependencyStatus.HEALTHY
    assert result.detail is None
