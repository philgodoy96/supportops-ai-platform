"""Unit tests for approval inspection routes."""

from datetime import UTC, datetime
from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock
from uuid import uuid4

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
