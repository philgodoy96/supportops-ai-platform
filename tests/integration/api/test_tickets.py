"""Integration tests for workspace-scoped ticket HTTP endpoints."""

from typing import cast
from uuid import uuid4

import pytest
from httpx import AsyncClient

pytestmark = pytest.mark.integration


async def create_workspace(
    client: AsyncClient,
    *,
    name: str,
    slug: str,
) -> dict[str, str]:
    """Create a workspace through the HTTP API."""

    response = await client.post(
        "/api/v1/workspaces",
        json={
            "name": name,
            "slug": slug,
        },
    )

    assert response.status_code == 201

    return cast(dict[str, str], response.json())


async def create_ticket(
    client: AsyncClient,
    *,
    workspace_id: str,
    subject: str = "Unable to access billing",
    external_reference: str | None = None,
    correlation_id: str | None = None,
) -> tuple[dict[str, object], dict[str, str]]:
    """Create a ticket through the HTTP API."""

    headers = {}

    if correlation_id is not None:
        headers["X-Correlation-ID"] = correlation_id

    response = await client.post(
        f"/api/v1/workspaces/{workspace_id}/tickets",
        json={
            "subject": subject,
            "description": ("The dashboard returns an access error."),
            "external_reference": external_reference,
        },
        headers=headers,
    )

    assert response.status_code == 201

    payload = response.json()
    ticket = cast(dict[str, object], payload["ticket"])

    return ticket, dict(response.headers)


async def test_create_ticket_persists_trace_identifiers(
    integration_client: AsyncClient,
    clean_business_tables: None,
) -> None:
    workspace = await create_workspace(
        integration_client,
        name="Platform Support",
        slug="platform-support",
    )
    correlation_id = str(uuid4())

    response = await integration_client.post(
        f"/api/v1/workspaces/{workspace['id']}/tickets",
        json={
            "subject": "Unable to access billing",
            "description": ("The dashboard returns an access error."),
            "external_reference": "SUP-1042",
        },
        headers={"X-Correlation-ID": correlation_id},
    )

    assert response.status_code == 201

    payload = response.json()
    ticket = payload["ticket"]
    processing_run = payload["processing_run"]
    headers = dict(response.headers)

    assert ticket["workspace_id"] == workspace["id"]
    assert ticket["status"] == "open"
    assert ticket["external_reference"] == "SUP-1042"
    assert ticket["ingestion_request_id"] == headers["x-request-id"]
    assert ticket["correlation_id"] == correlation_id
    assert processing_run["status"] == "queued"
    assert processing_run["workflow_name"] == "ticket-processing"
    assert set(processing_run) == {
        "id",
        "status",
        "workflow_name",
        "workflow_version",
    }
    assert headers["x-correlation-id"] == correlation_id


async def test_create_ticket_for_missing_workspace_returns_404(
    integration_client: AsyncClient,
    clean_business_tables: None,
) -> None:
    response = await integration_client.post(
        ("/api/v1/workspaces/ee9f2d68-38c5-4b4a-af4b-f6970f7f29fb/tickets"),
        json={
            "subject": "Unable to access billing",
            "description": "The dashboard returns an error.",
        },
    )

    assert response.status_code == 404
    assert response.json()["error"]["code"] == ("workspace_not_found")


async def test_duplicate_external_reference_is_scoped_to_workspace(
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

    await create_ticket(
        integration_client,
        workspace_id=workspace_a["id"],
        external_reference="SUP-1042",
    )

    conflict = await integration_client.post(
        f"/api/v1/workspaces/{workspace_a['id']}/tickets",
        json={
            "subject": "Duplicate reference",
            "description": "A duplicate upstream identifier.",
            "external_reference": "SUP-1042",
        },
    )

    accepted = await integration_client.post(
        f"/api/v1/workspaces/{workspace_b['id']}/tickets",
        json={
            "subject": "Same reference, different workspace",
            "description": "The identifier is scoped by workspace.",
            "external_reference": "SUP-1042",
        },
    )

    assert conflict.status_code == 409
    assert conflict.json()["error"]["code"] == ("ticket_external_reference_conflict")
    assert accepted.status_code == 201


async def test_cross_workspace_ticket_retrieval_returns_404(
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
    ticket, _ = await create_ticket(
        integration_client,
        workspace_id=workspace_a["id"],
    )

    valid_response = await integration_client.get(
        (f"/api/v1/workspaces/{workspace_a['id']}/tickets/{ticket['id']}"),
    )
    cross_workspace_response = await integration_client.get(
        (f"/api/v1/workspaces/{workspace_b['id']}/tickets/{ticket['id']}"),
    )

    assert valid_response.status_code == 200
    assert cross_workspace_response.status_code == 404
    assert cross_workspace_response.json()["error"]["code"] == ("ticket_not_found")


async def test_ticket_listing_is_isolated_and_paginated(
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
        await create_ticket(
            integration_client,
            workspace_id=workspace_a["id"],
            subject=f"Workspace A ticket {index}",
        )

    await create_ticket(
        integration_client,
        workspace_id=workspace_b["id"],
        subject="Workspace B ticket",
    )

    first_page = await integration_client.get(
        f"/api/v1/workspaces/{workspace_a['id']}/tickets",
        params={"page_size": 2},
    )

    assert first_page.status_code == 200
    first_payload = first_page.json()
    assert len(first_payload["items"]) == 2
    assert first_payload["next_cursor"] is not None

    second_page = await integration_client.get(
        f"/api/v1/workspaces/{workspace_a['id']}/tickets",
        params={
            "page_size": 2,
            "cursor": first_payload["next_cursor"],
        },
    )

    second_payload = second_page.json()

    assert second_page.status_code == 200
    assert len(second_payload["items"]) == 1
    assert second_payload["next_cursor"] is None

    observed_ids = {
        item["id"]
        for item in [
            *first_payload["items"],
            *second_payload["items"],
        ]
    }

    assert len(observed_ids) == 3
    assert all(
        item["workspace_id"] == workspace_a["id"]
        for item in [
            *first_payload["items"],
            *second_payload["items"],
        ]
    )


async def test_empty_ticket_list_and_missing_workspace_are_distinct(
    integration_client: AsyncClient,
    clean_business_tables: None,
) -> None:
    workspace = await create_workspace(
        integration_client,
        name="Platform Support",
        slug="platform-support",
    )

    empty_response = await integration_client.get(
        f"/api/v1/workspaces/{workspace['id']}/tickets",
    )
    missing_response = await integration_client.get(
        ("/api/v1/workspaces/ee9f2d68-38c5-4b4a-af4b-f6970f7f29fb/tickets"),
    )

    assert empty_response.status_code == 200
    assert empty_response.json() == {
        "items": [],
        "next_cursor": None,
    }
    assert missing_response.status_code == 404
    assert missing_response.json()["error"]["code"] == ("workspace_not_found")


async def test_invalid_cursor_returns_stable_client_error(
    integration_client: AsyncClient,
    clean_business_tables: None,
) -> None:
    workspace = await create_workspace(
        integration_client,
        name="Platform Support",
        slug="platform-support",
    )

    response = await integration_client.get(
        f"/api/v1/workspaces/{workspace['id']}/tickets",
        params={"cursor": "not-a-valid-cursor"},
    )

    assert response.status_code == 400
    assert response.json() == {
        "error": {
            "code": "invalid_pagination_cursor",
            "message": "Pagination cursor is invalid.",
            "request_id": response.headers["X-Request-ID"],
        }
    }


@pytest.mark.parametrize(
    "page_size",
    [
        0,
        101,
    ],
)
async def test_ticket_list_rejects_invalid_page_size(
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
        f"/api/v1/workspaces/{workspace['id']}/tickets",
        params={"page_size": page_size},
    )

    assert response.status_code == 422
