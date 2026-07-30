"""Integration tests for operational health endpoints."""

import pytest
from httpx import AsyncClient

pytestmark = pytest.mark.integration


async def test_liveness_with_live_application(
    integration_client: AsyncClient,
) -> None:
    response = await integration_client.get("/health/live")

    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


async def test_readiness_with_available_dependencies(
    integration_client: AsyncClient,
) -> None:
    response = await integration_client.get("/health/ready")

    assert response.status_code == 200
    assert response.json() == {
        "status": "healthy",
        "dependencies": {
            "postgresql": {
                "status": "healthy",
                "detail": None,
            },
            "qdrant": {
                "status": "healthy",
                "detail": None,
            },
        },
    }
