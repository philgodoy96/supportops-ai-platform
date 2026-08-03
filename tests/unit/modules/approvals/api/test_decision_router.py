"""Unit tests for approval decision HTTP endpoints."""

from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import UUID, uuid4

from fastapi.testclient import TestClient

from supportops.api.application import create_application
from supportops.core.request_context import (
    RequestContext,
    get_request_context,
)
from supportops.modules.approvals.api.dependencies import (
    get_decide_approval_request,
)
from supportops.modules.approvals.api.schemas import (
    ApprovalDecisionResponse,
)


def _approval(
    *,
    status: str,
    workspace_id: UUID,
    approval_id: UUID,
) -> SimpleNamespace:
    now = datetime(2026, 8, 3, 21, 30, tzinfo=UTC)
    return SimpleNamespace(
        id=approval_id,
        workspace_id=workspace_id,
        ticket_id=uuid4(),
        agent_run_id=uuid4(),
        agent_tool_call_id=uuid4(),
        requested_by_llm_invocation_id=uuid4(),
        status=status,
        tool_name="escalate_ticket",
        tool_version=1,
        input_fingerprint="a" * 64,
        proposed_input={
            "target_queue": "support_operations",
            "reason": "Operational review required.",
        },
        request_reason="Operational review required.",
        expires_at=now,
        decision_actor_reference="operator:alice",
        decision_comment=None,
        decision_request_id=uuid4(),
        decision_correlation_id=uuid4(),
        decided_at=now,
        created_at=now,
        updated_at=now,
    )


def test_approve_endpoint_delegates_to_decision_service() -> None:
    app = create_application()
    workspace_id = uuid4()
    approval_id = uuid4()
    approval = _approval(
        status="approved",
        workspace_id=workspace_id,
        approval_id=approval_id,
    )
    service = SimpleNamespace(
        approve=AsyncMock(
            return_value=SimpleNamespace(
                approval_request=approval,
                idempotent=False,
            ),
        ),
    )
    request_context = RequestContext(
        request_id=uuid4(),
        correlation_id=uuid4(),
    )
    app.dependency_overrides[get_decide_approval_request] = lambda: service
    app.dependency_overrides[get_request_context] = lambda: request_context

    decision_request_id = uuid4()
    response = TestClient(app).post(
        (f"/api/v1/workspaces/{workspace_id}/approvals/{approval_id}/approve"),
        json={
            "actor_reference": "operator:alice",
            "decision_request_id": str(decision_request_id),
        },
    )

    assert response.status_code == 200
    assert response.json()["status"] == "approved"
    assert response.json()["idempotent"] is False
    command = service.approve.await_args.args[0]
    assert command.workspace_id == workspace_id
    assert command.approval_request_id == approval_id
    assert command.actor_reference == "operator:alice"
    assert command.request_id == decision_request_id
    assert command.correlation_id == request_context.correlation_id
    assert command.decided_at.tzinfo is not None
    assert command.decided_at.utcoffset().total_seconds() == 0


def test_reject_endpoint_requires_comment_and_delegates() -> None:
    app = create_application()
    workspace_id = uuid4()
    approval_id = uuid4()
    approval = _approval(
        status="rejected",
        workspace_id=workspace_id,
        approval_id=approval_id,
    )
    service = SimpleNamespace(
        reject=AsyncMock(
            return_value=SimpleNamespace(
                approval_request=approval,
                idempotent=True,
            ),
        ),
    )
    app.dependency_overrides[get_decide_approval_request] = lambda: service
    app.dependency_overrides[get_request_context] = lambda: RequestContext(
        request_id=uuid4(),
        correlation_id=uuid4(),
    )

    response = TestClient(app).post(
        (f"/api/v1/workspaces/{workspace_id}/approvals/{approval_id}/reject"),
        json={
            "actor_reference": "operator:alice",
            "decision_request_id": str(uuid4()),
            "comment": "Do not execute this escalation.",
        },
    )

    assert response.status_code == 200
    assert response.json()["idempotent"] is True
    service.reject.assert_awaited_once()


def test_reject_endpoint_rejects_missing_comment() -> None:
    app = create_application()
    workspace_id = uuid4()
    approval_id = uuid4()
    app.dependency_overrides[get_decide_approval_request] = lambda: SimpleNamespace(
        reject=AsyncMock()
    )

    response = TestClient(app).post(
        (f"/api/v1/workspaces/{workspace_id}/approvals/{approval_id}/reject"),
        json={
            "actor_reference": "operator:alice",
            "decision_request_id": str(uuid4()),
        },
    )

    assert response.status_code == 422


def test_decision_response_contains_no_execution_fields() -> None:
    fields = set(ApprovalDecisionResponse.model_fields)

    assert "grant_id" not in fields
    assert "ticket_escalation_id" not in fields
    assert "execution_output" not in fields
