"""Integration tests for workspace-scoped AgentRun inspection endpoints."""

from datetime import UTC, datetime, timedelta
from typing import cast
from uuid import UUID

import pytest
from httpx import AsyncClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from supportops.agent_graph.domain.state import (
    CONTROLLED_SUPPORT_WORKFLOW_VERSION,
)
from supportops.modules.agent_runs.domain.models import AgentRunAttempt
from supportops.modules.agent_runs.infrastructure.models import (
    AgentRunAttemptRecord,
    AgentRunRecord,
)

pytestmark = pytest.mark.integration

_NOW = datetime(
    2026,
    7,
    31,
    21,
    0,
    tzinfo=UTC,
)
_ATTEMPT_ONE_ID = UUID(
    "2b39f5b7-b2a4-48d0-b079-fdad286d5315",
)
_ATTEMPT_TWO_ID = UUID(
    "626e0940-cf3b-4b9f-ad49-98bce214469b",
)
_LEASE_TOKEN_ONE = UUID(
    "dd0ae456-3467-41db-93d1-a908f40e8365",
)
_LEASE_TOKEN_TWO = UUID(
    "b36000c4-62d7-4fe1-ad40-96872a245409",
)
_EXECUTION_REQUEST_ONE = UUID(
    "d1fa068f-2278-47a8-b3c9-39ccf91f0a5e",
)
_EXECUTION_REQUEST_TWO = UUID(
    "99988e91-f292-4ada-81b6-58551c96f02b",
)


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
    session: AsyncSession,
    *,
    workspace_id: str,
) -> dict[str, object]:
    """Create a ticket and resolve its initial AgentRun from persistence."""

    response = await client.post(
        f"/api/v1/workspaces/{workspace_id}/tickets",
        json={
            "subject": "Unable to access billing",
            "description": ("The dashboard returns an access error."),
        },
    )

    assert response.status_code == 201

    ticket = cast(dict[str, object], response.json())
    query_result = await session.execute(
        select(AgentRunRecord).where(
            AgentRunRecord.ticket_id == UUID(cast(str, ticket["id"])),
        ),
    )
    record = query_result.scalar_one()
    ticket["processing_run"] = {
        "id": str(record.id),
    }

    return ticket


def processing_run_id(ticket: dict[str, object]) -> str:
    """Extract the scheduled AgentRun identifier from ticket setup state."""

    processing_run = cast(
        dict[str, object],
        ticket["processing_run"],
    )

    return cast(str, processing_run["id"])


def create_attempt(
    *,
    agent_run_id: UUID,
    attempt_id: UUID,
    attempt_number: int,
    worker_id: str,
    lease_token: UUID,
    execution_request_id: UUID,
    started_at: datetime,
) -> AgentRunAttempt:
    """Create one deterministic active AgentRun attempt."""

    return AgentRunAttempt.start(
        attempt_id=attempt_id,
        agent_run_id=agent_run_id,
        attempt_number=attempt_number,
        worker_id=worker_id,
        lease_token=lease_token,
        execution_request_id=execution_request_id,
        now=started_at,
    )


async def test_get_agent_run_returns_scheduled_processing_state(
    integration_client: AsyncClient,
    postgresql_session: AsyncSession,
    clean_business_tables: None,
) -> None:
    workspace = await create_workspace(
        integration_client,
        name="Platform Support",
        slug="platform-support",
    )
    ticket = await create_ticket(
        integration_client,
        postgresql_session,
        workspace_id=workspace["id"],
    )
    agent_run_id = processing_run_id(ticket)

    response = await integration_client.get(
        (f"/api/v1/workspaces/{workspace['id']}/agent-runs/{agent_run_id}"),
    )

    assert response.status_code == 200

    payload = response.json()

    assert payload["id"] == agent_run_id
    assert payload["workspace_id"] == workspace["id"]
    assert payload["ticket_id"] == ticket["id"]
    assert payload["status"] == "queued"
    assert payload["workflow"] == {
        "name": "ticket-processing",
        "version": CONTROLLED_SUPPORT_WORKFLOW_VERSION,
        "trigger_key": "initial-ticket-processing",
    }
    assert payload["attempt_count"] == 0
    assert payload["retryable_failure_count"] == 0
    assert payload["max_retryable_failures"] == 3
    assert "max_attempts" not in payload
    assert payload["first_started_at"] is None
    assert payload["completed_at"] is None
    assert payload["last_error"] is None


async def test_get_agent_run_omits_internal_runtime_fields(
    integration_client: AsyncClient,
    postgresql_session: AsyncSession,
    clean_business_tables: None,
) -> None:
    workspace = await create_workspace(
        integration_client,
        name="Platform Support",
        slug="platform-support",
    )
    ticket = await create_ticket(
        integration_client,
        postgresql_session,
        workspace_id=workspace["id"],
    )
    agent_run_id = processing_run_id(ticket)

    response = await integration_client.get(
        (f"/api/v1/workspaces/{workspace['id']}/agent-runs/{agent_run_id}"),
    )

    assert response.status_code == 200

    payload = response.json()

    assert "lease_owner" not in payload
    assert "lease_token" not in payload
    assert "lease_expires_at" not in payload
    assert "max_attempts" not in payload
    assert "ingestion_request_id" not in payload


async def test_get_agent_run_returns_404_for_missing_run(
    integration_client: AsyncClient,
    clean_business_tables: None,
) -> None:
    workspace = await create_workspace(
        integration_client,
        name="Platform Support",
        slug="platform-support",
    )

    response = await integration_client.get(
        (f"/api/v1/workspaces/{workspace['id']}/agent-runs/69184ef1-4d71-452e-8070-0b784c29368e"),
    )

    assert response.status_code == 404
    assert response.json() == {
        "error": {
            "code": "agent_run_not_found",
            "message": "AgentRun was not found.",
            "request_id": response.headers["X-Request-ID"],
        },
    }


async def test_get_agent_run_hides_cross_workspace_resource(
    integration_client: AsyncClient,
    postgresql_session: AsyncSession,
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
    ticket = await create_ticket(
        integration_client,
        postgresql_session,
        workspace_id=workspace_a["id"],
    )
    agent_run_id = processing_run_id(ticket)

    response = await integration_client.get(
        (f"/api/v1/workspaces/{workspace_b['id']}/agent-runs/{agent_run_id}"),
    )

    assert response.status_code == 404
    assert response.json()["error"]["code"] == ("agent_run_not_found")


async def test_list_agent_run_attempts_returns_empty_history(
    integration_client: AsyncClient,
    postgresql_session: AsyncSession,
    clean_business_tables: None,
) -> None:
    workspace = await create_workspace(
        integration_client,
        name="Platform Support",
        slug="platform-support",
    )
    ticket = await create_ticket(
        integration_client,
        postgresql_session,
        workspace_id=workspace["id"],
    )
    agent_run_id = processing_run_id(ticket)

    response = await integration_client.get(
        (f"/api/v1/workspaces/{workspace['id']}/agent-runs/{agent_run_id}/attempts"),
    )

    assert response.status_code == 200
    assert response.json() == {
        "items": [],
    }


async def test_list_agent_run_attempts_returns_ordered_history(
    integration_client: AsyncClient,
    postgresql_session: AsyncSession,
    clean_business_tables: None,
) -> None:
    workspace = await create_workspace(
        integration_client,
        name="Platform Support",
        slug="platform-support",
    )
    ticket = await create_ticket(
        integration_client,
        postgresql_session,
        workspace_id=workspace["id"],
    )
    agent_run_id = UUID(processing_run_id(ticket))

    first_attempt = create_attempt(
        agent_run_id=agent_run_id,
        attempt_id=_ATTEMPT_ONE_ID,
        attempt_number=1,
        worker_id="worker-a",
        lease_token=_LEASE_TOKEN_ONE,
        execution_request_id=_EXECUTION_REQUEST_ONE,
        started_at=_NOW,
    )
    second_attempt = create_attempt(
        agent_run_id=agent_run_id,
        attempt_id=_ATTEMPT_TWO_ID,
        attempt_number=2,
        worker_id="worker-b",
        lease_token=_LEASE_TOKEN_TWO,
        execution_request_id=_EXECUTION_REQUEST_TWO,
        started_at=_NOW + timedelta(seconds=10),
    )

    postgresql_session.add_all(
        [
            AgentRunAttemptRecord.from_domain(
                second_attempt,
            ),
            AgentRunAttemptRecord.from_domain(
                first_attempt,
            ),
        ],
    )
    await postgresql_session.commit()

    response = await integration_client.get(
        (f"/api/v1/workspaces/{workspace['id']}/agent-runs/{agent_run_id}/attempts"),
    )

    assert response.status_code == 200

    payload = response.json()

    assert [item["attempt_number"] for item in payload["items"]] == [
        1,
        2,
    ]
    assert [item["worker_id"] for item in payload["items"]] == [
        "worker-a",
        "worker-b",
    ]


async def test_attempt_history_omits_fencing_identifiers(
    integration_client: AsyncClient,
    postgresql_session: AsyncSession,
    clean_business_tables: None,
) -> None:
    workspace = await create_workspace(
        integration_client,
        name="Platform Support",
        slug="platform-support",
    )
    ticket = await create_ticket(
        integration_client,
        postgresql_session,
        workspace_id=workspace["id"],
    )
    agent_run_id = UUID(processing_run_id(ticket))

    attempt = create_attempt(
        agent_run_id=agent_run_id,
        attempt_id=_ATTEMPT_ONE_ID,
        attempt_number=1,
        worker_id="worker-a",
        lease_token=_LEASE_TOKEN_ONE,
        execution_request_id=_EXECUTION_REQUEST_ONE,
        started_at=_NOW,
    )

    postgresql_session.add(
        AgentRunAttemptRecord.from_domain(attempt),
    )
    await postgresql_session.commit()

    response = await integration_client.get(
        (f"/api/v1/workspaces/{workspace['id']}/agent-runs/{agent_run_id}/attempts"),
    )

    assert response.status_code == 200

    item = response.json()["items"][0]

    assert item["id"] == str(_ATTEMPT_ONE_ID)
    assert item["attempt_number"] == 1
    assert item["worker_id"] == "worker-a"
    assert item["finished_at"] is None
    assert item["outcome"] is None
    assert item["error"] is None

    assert "agent_run_id" not in item
    assert "lease_token" not in item
    assert "execution_request_id" not in item


async def test_attempt_history_returns_404_for_cross_workspace_run(
    integration_client: AsyncClient,
    postgresql_session: AsyncSession,
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
    ticket = await create_ticket(
        integration_client,
        postgresql_session,
        workspace_id=workspace_a["id"],
    )
    agent_run_id = processing_run_id(ticket)

    response = await integration_client.get(
        (f"/api/v1/workspaces/{workspace_b['id']}/agent-runs/{agent_run_id}/attempts"),
    )

    assert response.status_code == 404
    assert response.json()["error"]["code"] == ("agent_run_not_found")


async def test_agent_run_route_rejects_invalid_uuid(
    integration_client: AsyncClient,
    clean_business_tables: None,
) -> None:
    workspace = await create_workspace(
        integration_client,
        name="Platform Support",
        slug="platform-support",
    )

    response = await integration_client.get(
        (f"/api/v1/workspaces/{workspace['id']}/agent-runs/not-a-valid-uuid"),
    )

    assert response.status_code == 422
