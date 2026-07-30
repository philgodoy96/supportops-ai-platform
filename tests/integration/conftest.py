"""Shared fixtures for infrastructure integration tests."""

from collections.abc import AsyncIterator

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from supportops.api.application import create_application
from supportops.core.settings import Settings


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
