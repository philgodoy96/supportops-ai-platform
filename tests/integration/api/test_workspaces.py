"""Integration tests for workspace HTTP endpoints."""

from uuid import UUID

import pytest
from httpx import AsyncClient

pytestmark = pytest.mark.integration


async def test_create_workspace_returns_created_resource(
    integration_client: AsyncClient,
    clean_business_tables: None,
) -> None:
    response = await integration_client.post(
        "/api/v1/workspaces",
        json={
            "name": "  Platform Support  ",
            "slug": "platform-support",
        },
    )

    assert response.status_code == 201

    payload = response.json()

    assert UUID(payload["id"]).version == 4
    assert payload["name"] == "Platform Support"
    assert payload["slug"] == "platform-support"
    assert payload["created_at"] == payload["updated_at"]
    assert UUID(response.headers["X-Request-ID"]).version == 4
    assert response.headers["X-Correlation-ID"] == (response.headers["X-Request-ID"])


async def test_get_workspace_returns_persisted_resource(
    integration_client: AsyncClient,
    clean_business_tables: None,
) -> None:
    create_response = await integration_client.post(
        "/api/v1/workspaces",
        json={
            "name": "Platform Support",
            "slug": "platform-support",
        },
    )
    workspace_id = create_response.json()["id"]

    response = await integration_client.get(
        f"/api/v1/workspaces/{workspace_id}",
    )

    assert response.status_code == 200
    assert response.json() == create_response.json()


async def test_duplicate_workspace_slug_returns_stable_conflict(
    integration_client: AsyncClient,
    clean_business_tables: None,
) -> None:
    payload = {
        "name": "Platform Support",
        "slug": "platform-support",
    }

    first_response = await integration_client.post(
        "/api/v1/workspaces",
        json=payload,
    )
    conflict_response = await integration_client.post(
        "/api/v1/workspaces",
        json={
            "name": "Escalation Support",
            "slug": payload["slug"],
        },
    )

    assert first_response.status_code == 201
    assert conflict_response.status_code == 409
    assert conflict_response.json() == {
        "error": {
            "code": "workspace_slug_conflict",
            "message": "Workspace slug is already in use.",
            "request_id": conflict_response.headers["X-Request-ID"],
        }
    }


async def test_missing_workspace_returns_stable_not_found(
    integration_client: AsyncClient,
    clean_business_tables: None,
) -> None:
    response = await integration_client.get(
        ("/api/v1/workspaces/ee9f2d68-38c5-4b4a-af4b-f6970f7f29fb"),
    )

    assert response.status_code == 404
    assert response.json() == {
        "error": {
            "code": "workspace_not_found",
            "message": "Workspace was not found.",
            "request_id": response.headers["X-Request-ID"],
        }
    }


async def test_malformed_workspace_id_returns_validation_error(
    integration_client: AsyncClient,
    clean_business_tables: None,
) -> None:
    response = await integration_client.get(
        "/api/v1/workspaces/not-a-uuid",
    )

    assert response.status_code == 422


@pytest.mark.parametrize(
    "payload",
    [
        {
            "name": "   ",
            "slug": "platform-support",
        },
        {
            "name": "Platform Support",
            "slug": "Platform-Support",
        },
        {
            "name": "Platform Support",
            "slug": " platform-support ",
        },
        {
            "name": "Platform Support",
            "slug": "platform-support",
            "unexpected": "value",
        },
    ],
)
async def test_create_workspace_rejects_invalid_payload(
    integration_client: AsyncClient,
    clean_business_tables: None,
    payload: dict[str, str],
) -> None:
    response = await integration_client.post(
        "/api/v1/workspaces",
        json=payload,
    )

    assert response.status_code == 422


async def test_health_endpoints_remain_outside_versioned_api(
    integration_client: AsyncClient,
) -> None:
    response = await integration_client.get("/health/live")

    assert response.status_code == 200

    versioned_response = await integration_client.get(
        "/api/v1/health/live",
    )

    assert versioned_response.status_code == 404
