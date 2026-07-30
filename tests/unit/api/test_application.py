"""Unit tests for FastAPI application construction."""

from uuid import UUID

from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from supportops.api.application import create_application
from supportops.core.settings import Settings


def create_settings() -> Settings:
    """Create isolated settings for application unit tests."""

    return Settings(
        _env_file=None,
        environment="test",
        application_name="SupportOps Test Platform",
        application_version="9.9.9",
        postgresql_url=("postgresql+asyncpg://supportops:local@localhost:5432/supportops"),
        qdrant_url="http://localhost:6333",
    )


def test_create_application_uses_configured_openapi_metadata() -> None:
    app = create_application(create_settings())

    assert isinstance(app, FastAPI)
    assert app.title == "SupportOps Test Platform"
    assert app.version == "9.9.9"
    assert app.routes


async def test_create_application_adds_request_trace_headers() -> None:
    app = create_application(create_settings())
    transport = ASGITransport(app=app)

    async with AsyncClient(
        transport=transport,
        base_url="http://test",
    ) as client:
        response = await client.get("/openapi.json")

    assert response.status_code == 200

    request_id = UUID(response.headers["X-Request-ID"])
    correlation_id = UUID(response.headers["X-Correlation-ID"])

    assert request_id.version == 4
    assert correlation_id == request_id
