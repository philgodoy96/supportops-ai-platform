"""Unit tests for deterministic human-approved routing."""

from typing import Any
from uuid import uuid4

from supportops.agent_graph.domain.human_approved_routing import (
    HumanApprovedSupportGraphRoute,
    select_human_approved_support_route,
)
from supportops.agent_graph.domain.human_approved_state import (
    HumanApprovalCheckpointStatus,
    HumanApprovedDecisionKind,
    HumanApprovedSupportGraphStateSnapshot,
)


def _snapshot(
    **updates: Any,
) -> HumanApprovedSupportGraphStateSnapshot:
    values = {
        "state_schema_version": ("human-approved-support-state-v1"),
        "workflow_name": "ticket-processing",
        "workflow_version": "human-approved-support-v1",
        "graph_version": "graph-v1",
        "workspace_id": uuid4(),
        "ticket_id": uuid4(),
        "agent_run_id": uuid4(),
    }
    values.update(updates)
    return HumanApprovedSupportGraphStateSnapshot(
        **values,
    )


def test_initial_state_routes_to_context_loading() -> None:
    decision = select_human_approved_support_route(
        _snapshot(),
    )

    assert decision.route is (HumanApprovedSupportGraphRoute.LOAD_RUN_CONTEXT)


def test_loaded_state_without_classification_routes_to_classification() -> None:
    decision = select_human_approved_support_route(
        _snapshot(run_context_loaded=True),
    )

    assert decision.route is (HumanApprovedSupportGraphRoute.ENSURE_CLASSIFICATION)


def test_sensitive_decision_routes_to_proposal_preparation() -> None:
    decision = select_human_approved_support_route(
        _snapshot(
            run_context_loaded=True,
            classification_id=uuid4(),
            classification_category="product_bug",
            classification_intent="report_problem",
            classification_urgency="high",
            classification_sentiment="negative",
            classification_requires_human_review=True,
            classification_summary="A production issue is reported.",
            decision_kind=(HumanApprovedDecisionKind.SENSITIVE_TOOL),
            decision_invocation_id=uuid4(),
            decision_summary="Escalate for engineering review.",
            proposed_tool_provider_call_id="call-1",
            proposed_tool_name="escalate_ticket",
            proposed_tool_version=1,
            proposed_tool_input={
                "target_queue": "engineering_support",
                "reason": "A product defect requires review.",
            },
            proposed_tool_fingerprint="a" * 64,
            approval_request_reason=("A product defect requires review."),
        ),
    )

    assert decision.route is (HumanApprovedSupportGraphRoute.PREPARE_SENSITIVE_ACTION)


def test_pending_approval_routes_to_interrupt() -> None:
    decision = select_human_approved_support_route(
        _snapshot(
            run_context_loaded=True,
            classification_id=uuid4(),
            classification_category="product_bug",
            classification_intent="report_problem",
            classification_urgency="high",
            classification_sentiment="negative",
            classification_requires_human_review=True,
            classification_summary="A production issue is reported.",
            decision_kind=(HumanApprovedDecisionKind.SENSITIVE_TOOL),
            decision_invocation_id=uuid4(),
            decision_summary="Escalate for engineering review.",
            proposed_tool_provider_call_id="call-1",
            proposed_tool_name="escalate_ticket",
            proposed_tool_version=1,
            proposed_tool_input={
                "target_queue": "engineering_support",
                "reason": "A product defect requires review.",
            },
            proposed_tool_fingerprint="a" * 64,
            approval_request_reason=("A product defect requires review."),
            agent_tool_call_id=uuid4(),
            approval_request_id=uuid4(),
            approval_status=(HumanApprovalCheckpointStatus.PENDING),
            approval_expires_at=("2026-08-04T12:00:00+00:00"),
        ),
    )

    assert decision.route is (HumanApprovedSupportGraphRoute.AWAIT_HUMAN_APPROVAL)


def test_error_routes_fail_closed() -> None:
    decision = select_human_approved_support_route(
        _snapshot(
            current_error_code="workflow_state_invalid",
        ),
    )

    assert decision.route is (HumanApprovedSupportGraphRoute.FAIL_WORKFLOW)
    assert decision.error_code == "workflow_state_invalid"
