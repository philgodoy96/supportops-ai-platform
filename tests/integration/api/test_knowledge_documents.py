"""Integration tests for workspace-scoped knowledge document endpoints."""

from datetime import UTC, datetime
from typing import cast
from uuid import UUID

import pytest
from httpx import AsyncClient
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

pytestmark = pytest.mark.integration

type JsonObject = dict[str, object]

_MISSING_WORKSPACE_ID = "ee9f2d68-38c5-4b4a-af4b-f6970f7f29fb"


async def create_workspace(
    client: AsyncClient,
    *,
    name: str,
    slug: str,
) -> JsonObject:
    """Create a workspace through the HTTP API."""

    response = await client.post(
        "/api/v1/workspaces",
        json={
            "name": name,
            "slug": slug,
        },
    )
    assert response.status_code == 201
    return cast(JsonObject, response.json())


async def create_document(
    client: AsyncClient,
    *,
    workspace_id: object,
    title: str = "Database Incident Runbook",
    external_reference: str | None = None,
    content: str = ("# Database incidents\nRestart the connection pool.\n"),
) -> JsonObject:
    """Create a document and its first version through the API."""

    response = await client.post(
        f"/api/v1/workspaces/{workspace_id}/documents",
        json={
            "title": title,
            "external_reference": external_reference,
            "media_type": "text/markdown",
            "content": content,
        },
    )
    assert response.status_code == 201
    return cast(JsonObject, response.json())


async def create_version(
    client: AsyncClient,
    *,
    workspace_id: object,
    document_id: object,
    content: str,
) -> JsonObject:
    """Create one new immutable document version."""

    response = await client.post(
        (f"/api/v1/workspaces/{workspace_id}/documents/{document_id}/versions"),
        json={
            "media_type": "text/markdown",
            "content": content,
        },
    )
    assert response.status_code == 201
    return cast(JsonObject, response.json())


async def mark_version_ready(
    engine: AsyncEngine,
    *,
    document_version_id: object,
) -> None:
    """Persist complete ready-state provenance for activation tests."""

    # Must be >= created_at (set by the API at insert time).
    ready_at = datetime.now(UTC)
    async with engine.begin() as connection:
        await connection.execute(
            text(
                """
                UPDATE knowledge_document_versions
                SET
                    status = 'ready',
                    chunking_strategy = 'markdown-token',
                    chunking_version = 'v1',
                    tokenizer_encoding = 'cl100k_base',
                    embedding_provider = 'mock',
                    embedding_model = 'mock-hashing-embedding-v1',
                    embedding_dimensions = 64,
                    knowledge_collection = 'supportops-knowledge-mock-v1',
                    knowledge_vector_name = 'dense',
                    embedding_input_tokens = 20,
                    embedding_estimated_cost_usd = 0,
                    embedding_pricing_catalog_version =
                        'supportops-embedding-pricing-2026-08-01',
                    chunk_count = 1,
                    indexed_at = :ready_at,
                    updated_at = :ready_at
                WHERE id = :document_version_id
                """
            ),
            {
                "ready_at": ready_at,
                "document_version_id": UUID(cast(str, document_version_id)),
            },
        )


async def test_create_document_returns_metadata_and_normalized_detail(
    integration_client: AsyncClient,
    clean_business_tables: None,
) -> None:
    workspace = await create_workspace(
        integration_client,
        name="Platform Support",
        slug="platform-support",
    )

    response = await integration_client.post(
        f"/api/v1/workspaces/{workspace['id']}/documents",
        json={
            "title": "  Database Incident Runbook  ",
            "external_reference": "runbook-database-incidents",
            "media_type": "text/markdown",
            "content": ("\ufeff# Database incidents\r\nRestart the connection pool.\r"),
        },
        headers={"X-Correlation-ID": ("ec54dc08-f223-45f7-8540-e7e5b31500c4")},
    )

    assert response.status_code == 201
    payload = cast(JsonObject, response.json())
    document = cast(JsonObject, payload["document"])
    version = cast(JsonObject, payload["version"])

    assert document["workspace_id"] == workspace["id"]
    assert document["title"] == "Database Incident Runbook"
    assert document["active_version_id"] is None
    assert version["version_number"] == 1
    assert version["status"] == "pending"
    assert version["media_type"] == "text/markdown"
    assert "content" not in version
    assert response.headers["X-Correlation-ID"] == ("ec54dc08-f223-45f7-8540-e7e5b31500c4")
    assert response.headers["X-Request-ID"]

    detail = await integration_client.get(
        f"/api/v1/workspaces/{workspace['id']}/documents/{document['id']}/versions/{version['id']}"
    )

    assert detail.status_code == 200
    detail_payload = cast(JsonObject, detail.json())
    assert detail_payload["content"] == ("# Database incidents\nRestart the connection pool.\n")


async def test_create_document_for_missing_workspace_returns_404(
    integration_client: AsyncClient,
    clean_business_tables: None,
) -> None:
    response = await integration_client.post(
        (f"/api/v1/workspaces/{_MISSING_WORKSPACE_ID}/documents"),
        json={
            "title": "Database Incident Runbook",
            "media_type": "text/markdown",
            "content": "# Database incidents\n",
        },
    )

    assert response.status_code == 404
    assert response.json()["error"]["code"] == ("workspace_not_found")


async def test_external_reference_conflict_is_workspace_scoped(
    integration_client: AsyncClient,
    clean_business_tables: None,
) -> None:
    workspace_a = await create_workspace(
        integration_client,
        name="Platform Support",
        slug="platform-support",
    )
    workspace_b = await create_workspace(
        integration_client,
        name="Customer Success",
        slug="customer-success",
    )
    await create_document(
        integration_client,
        workspace_id=workspace_a["id"],
        external_reference="runbook-database-incidents",
    )

    conflict = await integration_client.post(
        f"/api/v1/workspaces/{workspace_a['id']}/documents",
        json={
            "title": "Duplicate Runbook",
            "external_reference": "runbook-database-incidents",
            "media_type": "text/plain",
            "content": "Different source content.",
        },
    )
    accepted = await integration_client.post(
        f"/api/v1/workspaces/{workspace_b['id']}/documents",
        json={
            "title": "Workspace B Runbook",
            "external_reference": "runbook-database-incidents",
            "media_type": "text/plain",
            "content": "Workspace B source content.",
        },
    )

    assert conflict.status_code == 409
    assert conflict.json()["error"]["code"] == ("document_external_reference_conflict")
    assert accepted.status_code == 201


async def test_document_listing_is_isolated_paginated_and_excludes_content(
    integration_client: AsyncClient,
    clean_business_tables: None,
) -> None:
    workspace_a = await create_workspace(
        integration_client,
        name="Platform Support",
        slug="platform-support",
    )
    workspace_b = await create_workspace(
        integration_client,
        name="Customer Success",
        slug="customer-success",
    )

    for index in range(3):
        await create_document(
            integration_client,
            workspace_id=workspace_a["id"],
            title=f"Workspace A Runbook {index}",
            content=f"Workspace A source {index}.",
        )
    await create_document(
        integration_client,
        workspace_id=workspace_b["id"],
        title="Workspace B Runbook",
        content="Workspace B source.",
    )

    first_page = await integration_client.get(
        f"/api/v1/workspaces/{workspace_a['id']}/documents",
        params={"page_size": 2},
    )
    assert first_page.status_code == 200
    first_payload = cast(JsonObject, first_page.json())
    first_items = cast(list[JsonObject], first_payload["items"])
    assert len(first_items) == 2
    assert first_payload["next_cursor"] is not None

    second_page = await integration_client.get(
        f"/api/v1/workspaces/{workspace_a['id']}/documents",
        params={
            "page_size": 2,
            "cursor": cast(str, first_payload["next_cursor"]),
        },
    )
    assert second_page.status_code == 200
    second_payload = cast(JsonObject, second_page.json())
    second_items = cast(list[JsonObject], second_payload["items"])
    assert len(second_items) == 1
    assert second_payload["next_cursor"] is None

    all_items = [*first_items, *second_items]
    assert len({item["id"] for item in all_items}) == 3
    assert all(item["workspace_id"] == workspace_a["id"] for item in all_items)
    assert all("content" not in item for item in all_items)


async def test_cross_workspace_document_and_version_access_return_404(
    integration_client: AsyncClient,
    clean_business_tables: None,
) -> None:
    workspace_a = await create_workspace(
        integration_client,
        name="Platform Support",
        slug="platform-support",
    )
    workspace_b = await create_workspace(
        integration_client,
        name="Customer Success",
        slug="customer-success",
    )
    created = await create_document(
        integration_client,
        workspace_id=workspace_a["id"],
    )
    document = cast(JsonObject, created["document"])
    version = cast(JsonObject, created["version"])

    document_response = await integration_client.get(
        f"/api/v1/workspaces/{workspace_b['id']}/documents/{document['id']}"
    )
    version_response = await integration_client.get(
        f"/api/v1/workspaces/{workspace_b['id']}"
        f"/documents/{document['id']}"
        f"/versions/{version['id']}"
    )

    assert document_response.status_code == 404
    assert document_response.json()["error"]["code"] == ("document_not_found")
    assert version_response.status_code == 404
    assert version_response.json()["error"]["code"] == ("document_not_found")


async def test_version_creation_listing_detail_and_duplicate_content(
    integration_client: AsyncClient,
    clean_business_tables: None,
) -> None:
    workspace = await create_workspace(
        integration_client,
        name="Platform Support",
        slug="platform-support",
    )
    created = await create_document(
        integration_client,
        workspace_id=workspace["id"],
    )
    document = cast(JsonObject, created["document"])

    second = await create_version(
        integration_client,
        workspace_id=workspace["id"],
        document_id=document["id"],
        content=("\ufeff# Database incidents\r\nEscalate after two failed restarts.\r"),
    )
    third = await create_version(
        integration_client,
        workspace_id=workspace["id"],
        document_id=document["id"],
        content=("# Database incidents\nEscalate immediately for data loss.\n"),
    )

    assert second["version_number"] == 2
    assert third["version_number"] == 3
    assert "content" not in second
    assert "content" not in third

    duplicate = await integration_client.post(
        (f"/api/v1/workspaces/{workspace['id']}/documents/{document['id']}/versions"),
        json={
            "media_type": "text/markdown",
            "content": ("# Database incidents\nEscalate after two failed restarts.\n"),
        },
    )
    assert duplicate.status_code == 409
    assert duplicate.json()["error"]["code"] == ("document_version_content_conflict")

    first_page = await integration_client.get(
        (f"/api/v1/workspaces/{workspace['id']}/documents/{document['id']}/versions"),
        params={"page_size": 2},
    )
    assert first_page.status_code == 200
    first_payload = cast(JsonObject, first_page.json())
    first_items = cast(list[JsonObject], first_payload["items"])
    assert [item["version_number"] for item in first_items] == [3, 2]
    assert first_payload["next_cursor"] is not None
    assert all("content" not in item for item in first_items)

    second_page = await integration_client.get(
        (f"/api/v1/workspaces/{workspace['id']}/documents/{document['id']}/versions"),
        params={
            "page_size": 2,
            "cursor": cast(str, first_payload["next_cursor"]),
        },
    )
    second_payload = cast(JsonObject, second_page.json())
    second_items = cast(list[JsonObject], second_payload["items"])
    assert second_page.status_code == 200
    assert [item["version_number"] for item in second_items] == [1]
    assert second_payload["next_cursor"] is None

    detail = await integration_client.get(
        f"/api/v1/workspaces/{workspace['id']}/documents/{document['id']}/versions/{second['id']}"
    )
    assert detail.status_code == 200
    assert detail.json()["content"] == (
        "# Database incidents\nEscalate after two failed restarts.\n"
    )


async def test_pending_activation_conflicts_and_ready_activation_succeeds(
    integration_client: AsyncClient,
    postgresql_engine: AsyncEngine,
    clean_business_tables: None,
) -> None:
    workspace = await create_workspace(
        integration_client,
        name="Platform Support",
        slug="platform-support",
    )
    created = await create_document(
        integration_client,
        workspace_id=workspace["id"],
    )
    document = cast(JsonObject, created["document"])
    version = cast(JsonObject, created["version"])
    activation_url = (
        f"/api/v1/workspaces/{workspace['id']}"
        f"/documents/{document['id']}"
        f"/versions/{version['id']}/activate"
    )

    pending_response = await integration_client.post(activation_url)
    assert pending_response.status_code == 409
    assert pending_response.json()["error"]["code"] == ("document_version_not_ready")

    await mark_version_ready(
        postgresql_engine,
        document_version_id=version["id"],
    )
    ready_response = await integration_client.post(activation_url)

    assert ready_response.status_code == 200
    assert ready_response.json()["active_version_id"] == (version["id"])

    repeated_response = await integration_client.post(activation_url)
    assert repeated_response.status_code == 200
    assert repeated_response.json()["active_version_id"] == (version["id"])


async def test_invalid_and_cross_collection_cursors_return_stable_error(
    integration_client: AsyncClient,
    clean_business_tables: None,
) -> None:
    workspace = await create_workspace(
        integration_client,
        name="Platform Support",
        slug="platform-support",
    )
    for index in range(2):
        await create_document(
            integration_client,
            workspace_id=workspace["id"],
            title=f"Runbook {index}",
            content=f"Unique content {index}.",
        )

    document_page = await integration_client.get(
        f"/api/v1/workspaces/{workspace['id']}/documents",
        params={"page_size": 1},
    )
    document_cursor = document_page.json()["next_cursor"]
    assert document_cursor is not None

    first_document = document_page.json()["items"][0]
    invalid = await integration_client.get(
        f"/api/v1/workspaces/{workspace['id']}/documents",
        params={"cursor": "not-a-valid-cursor"},
    )
    wrong_collection = await integration_client.get(
        (f"/api/v1/workspaces/{workspace['id']}/documents/{first_document['id']}/versions"),
        params={"cursor": document_cursor},
    )

    for response in (invalid, wrong_collection):
        assert response.status_code == 400
        assert response.json()["error"] == {
            "code": "invalid_pagination_cursor",
            "message": "Pagination cursor is invalid.",
            "request_id": response.headers["X-Request-ID"],
        }


@pytest.mark.parametrize(
    "payload",
    [
        {
            "title": "Unsupported format",
            "media_type": "application/pdf",
            "content": "Binary formats are not supported.",
        },
        {
            "title": "Whitespace-only content",
            "media_type": "text/plain",
            "content": "   \r\n\t",
        },
        {
            "title": "Unexpected field",
            "media_type": "text/plain",
            "content": "Valid source.",
            "automatic_activation": True,
        },
    ],
)
async def test_create_document_rejects_invalid_payloads(
    integration_client: AsyncClient,
    clean_business_tables: None,
    payload: JsonObject,
) -> None:
    workspace = await create_workspace(
        integration_client,
        name="Platform Support",
        slug="platform-support",
    )

    response = await integration_client.post(
        f"/api/v1/workspaces/{workspace['id']}/documents",
        json=payload,
    )

    assert response.status_code == 422


@pytest.mark.parametrize("page_size", [0, 101])
async def test_document_list_rejects_invalid_page_size(
    integration_client: AsyncClient,
    clean_business_tables: None,
    page_size: int,
) -> None:
    workspace = await create_workspace(
        integration_client,
        name="Platform Support",
        slug="platform-support",
    )

    response = await integration_client.get(
        f"/api/v1/workspaces/{workspace['id']}/documents",
        params={"page_size": page_size},
    )

    assert response.status_code == 422
