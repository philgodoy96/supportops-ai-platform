"""Integration tests for PostgreSQL connectivity."""

import pytest

from supportops.core.settings import Settings
from supportops.infrastructure.health import DependencyStatus
from supportops.infrastructure.postgresql import (
    check_postgresql_health,
    create_postgresql_engine,
    dispose_postgresql_engine,
)

pytestmark = pytest.mark.integration


async def test_postgresql_connectivity(
    integration_settings: Settings,
) -> None:
    engine = create_postgresql_engine(integration_settings)

    try:
        result = await check_postgresql_health(
            engine,
            integration_settings.dependency_health_timeout_seconds,
        )
    finally:
        await dispose_postgresql_engine(engine)

    assert result.dependency == "postgresql"
    assert result.status is DependencyStatus.HEALTHY
    assert result.detail is None
