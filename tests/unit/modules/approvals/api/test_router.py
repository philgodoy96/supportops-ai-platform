"""Unit tests for approval inspection routes."""

from datetime import UTC, datetime
from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from supportops.api.application import create_application
from supportops.modules.approvals.api.dependencies import (
    get_approval_request,
    get_list_approval_requests,
)
from supportops.modules.approvals.application.queries import (
    ApprovalRequestListPage,
    ApprovalRequestPageCursor,
)
from supportops.modules.approvals.domain.models import ApprovalRequest


def test_list_approvals_returns_page() -> None:
    app = create_application()
    workspace_id = uuid4()
    approval = SimpleNamespace(
        id=uuid4(),
        workspace_id=workspace_id,
        ticket_id=uuid4(),
        agent_run_id=uuid4(),
        agent_tool_call_id=uuid4(),
        requested_by_llm_invocation_id=uuid4(),
        status="pending",
        tool_name="escalate_ticket",
        tool_version=1,
        input_fingerprint="a" * 64,
        proposed_input={
            "target_queue": "support_operations",
            "reason": "Operational review required.",
        },
        request_reason="Operational review required.",
        expires_at=datetime(2026, 8, 4, 21, 0, tzinfo=UTC),
        decision_actor_reference=None,
        decision_comment=None,
        decision_request_id=None,
        decision_correlation_id=None,
        decided_at=None,
        created_at=datetime(2026, 8, 3, 21, 0, tzinfo=UTC),
        updated_at=datetime(2026, 8, 3, 21, 0, tzinfo=UTC),
    )
    service = SimpleNamespace(
        execute=AsyncMock(
            return_value=ApprovalRequestListPage(
                items=cast(
                    tuple[ApprovalRequest, ...],
                    (approval,),
                ),
                next_cursor=ApprovalRequestPageCursor(
                    created_at=approval.created_at,
                    approval_request_id=approval.id,
                ),
            ),
        ),
    )
    app.dependency_overrides[get_list_approval_requests] = lambda: service

    response = TestClient(app).get(
        f"/api/v1/workspaces/{workspace_id}/approvals",
        params={"status": "pending", "page_size": 1},
    )

    assert response.status_code == 200
    assert response.json()["items"][0]["id"] == str(approval.id)
    assert response.json()["next_cursor"] is not None


def test_get_approval_returns_workspace_detail() -> None:
    app = create_application()
    workspace_id = uuid4()
    approval_id = uuid4()
    approval = SimpleNamespace(
        id=approval_id,
        workspace_id=workspace_id,
        ticket_id=uuid4(),
        agent_run_id=uuid4(),
        agent_tool_call_id=uuid4(),
        requested_by_llm_invocation_id=uuid4(),
        status="approved",
        tool_name="escalate_ticket",
        tool_version=1,
        input_fingerprint="a" * 64,
        proposed_input={
            "target_queue": "support_operations",
            "reason": "Operational review required.",
        },
        request_reason="Operational review required.",
        expires_at=datetime(2026, 8, 4, 21, 0, tzinfo=UTC),
        decision_actor_reference="operator:alice",
        decision_comment=None,
        decision_request_id=uuid4(),
        decision_correlation_id=uuid4(),
        decided_at=datetime(2026, 8, 3, 21, 5, tzinfo=UTC),
        created_at=datetime(2026, 8, 3, 21, 0, tzinfo=UTC),
        updated_at=datetime(2026, 8, 3, 21, 5, tzinfo=UTC),
    )
    service = SimpleNamespace(
        execute=AsyncMock(return_value=approval),
    )
    app.dependency_overrides[get_approval_request] = lambda: service

    response = TestClient(app).get(
        (f"/api/v1/workspaces/{workspace_id}/approvals/{approval_id}"),
    )

    assert response.status_code == 200
    assert response.json()["status"] == "approved"
    payload = response.json()
    assert "grant_id" not in payload
    assert "lease_token" not in payload
    assert "checkpoint" not in payload
    assert "execution_output" not in payload
    assert "raw_prompt" not in payload
    assert "raw_model_output" not in payload


def test_list_approvals_rejects_malformed_cursor() -> None:
    app = create_application()
    workspace_id = uuid4()
    app.dependency_overrides[get_list_approval_requests] = lambda: SimpleNamespace(
        execute=AsyncMock(),
    )

    response = TestClient(app).get(
        f"/api/v1/workspaces/{workspace_id}/approvals",
        params={"cursor": "not-a-valid-cursor"},
    )

    assert response.status_code == 400
    assert response.json()["error"]["code"] == ("invalid_pagination_cursor")
    assert response.json()["error"]["request_id"] == (response.headers["X-Request-ID"])


def test_list_approvals_rejects_cross_endpoint_cursor() -> None:
    from supportops.modules.tickets.api.escalation_pagination import (
        encode_ticket_escalation_cursor,
    )
    from supportops.modules.tickets.application.escalation_queries import (
        TicketEscalationPageCursor,
    )

    app = create_application()
    workspace_id = uuid4()
    app.dependency_overrides[get_list_approval_requests] = lambda: SimpleNamespace(
        execute=AsyncMock(),
    )
    foreign_cursor = encode_ticket_escalation_cursor(
        TicketEscalationPageCursor(
            created_at=datetime(2026, 8, 3, 21, 0, tzinfo=UTC),
            ticket_escalation_id=uuid4(),
        ),
    )

    response = TestClient(app).get(
        f"/api/v1/workspaces/{workspace_id}/approvals",
        params={"cursor": foreign_cursor},
    )

    assert response.status_code == 400
    assert response.json()["error"]["code"] == ("invalid_pagination_cursor")


def test_list_approvals_rejects_unsupported_cursor_version() -> None:
    app = create_application()
    workspace_id = uuid4()
    app.dependency_overrides[get_list_approval_requests] = lambda: SimpleNamespace(
        execute=AsyncMock(),
    )
    encoded = (
        "eyJ2ZXJzaW9uIjoyLCJjcmVhdGVkX2F0Ijoi"
        "MjAyNi0wOC0wM1QyMTowMDowMFoiLCJhcHBy"
        "b3ZhbF9yZXF1ZXN0X2lkIjoiMDAwMDAwMDAt"
        "MDAwMC0wMDAwLTAwMDAtMDAwMDAwMDAwMDAx"
        "In0="
    )

    response = TestClient(app).get(
        f"/api/v1/workspaces/{workspace_id}/approvals",
        params={"cursor": encoded},
    )

    assert response.status_code == 400
    assert response.json()["error"]["code"] == ("invalid_pagination_cursor")


@pytest.mark.parametrize("page_size", [0, 101])
def test_list_approvals_rejects_invalid_page_size(page_size: int) -> None:
    app = create_application()
    workspace_id = uuid4()
    app.dependency_overrides[get_list_approval_requests] = lambda: SimpleNamespace(
        execute=AsyncMock(),
    )

    response = TestClient(app).get(
        f"/api/v1/workspaces/{workspace_id}/approvals",
        params={"page_size": page_size},
    )

    assert response.status_code == 422


def test_approvals_are_registered_in_openapi() -> None:
    app = create_application()
    paths = app.openapi()["paths"]

    assert "/api/v1/workspaces/{workspace_id}/approvals" in paths
    assert ("/api/v1/workspaces/{workspace_id}/approvals/{approval_request_id}") in paths
    assert ("/api/v1/workspaces/{workspace_id}/approvals/{approval_request_id}/approve") in paths
    assert ("/api/v1/workspaces/{workspace_id}/approvals/{approval_request_id}/reject") in paths
