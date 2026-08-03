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
    HumanApprovedRecommendationStage,
    HumanApprovedSupportGraphStateSnapshot,
)
from supportops.agent_graph.domain.resume_planning import (
    ApprovalResumeDecisionStatus,
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


def _classified_sensitive(**updates: Any) -> HumanApprovedSupportGraphStateSnapshot:
    values = {
        "run_context_loaded": True,
        "classification_id": uuid4(),
        "classification_category": "product_bug",
        "classification_intent": "report_problem",
        "classification_urgency": "high",
        "classification_sentiment": "negative",
        "classification_requires_human_review": True,
        "classification_summary": "A production issue is reported.",
        "decision_kind": HumanApprovedDecisionKind.SENSITIVE_TOOL,
        "decision_invocation_id": uuid4(),
        "decision_summary": "Escalate for engineering review.",
        "proposed_tool_provider_call_id": "call-1",
        "proposed_tool_name": "escalate_ticket",
        "proposed_tool_version": 1,
        "proposed_tool_input": {
            "target_queue": "engineering_support",
            "reason": "A product defect requires review.",
        },
        "proposed_tool_fingerprint": "a" * 64,
        "approval_request_reason": "A product defect requires review.",
        "agent_tool_call_id": uuid4(),
        "approval_request_id": uuid4(),
        "approval_status": HumanApprovalCheckpointStatus.PENDING,
        "approval_expires_at": "2026-08-04T12:00:00+00:00",
    }
    values.update(updates)
    return _snapshot(**values)


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
        _classified_sensitive(),
    )

    assert decision.route is (HumanApprovedSupportGraphRoute.AWAIT_HUMAN_APPROVAL)


def test_pending_with_resume_payload_routes_to_decision_handling() -> None:
    approval_request_id = uuid4()
    agent_tool_call_id = uuid4()
    decision = select_human_approved_support_route(
        _classified_sensitive(
            approval_request_id=approval_request_id,
            agent_tool_call_id=agent_tool_call_id,
            approval_resume_payload={
                "approval_request_id": approval_request_id,
                "agent_tool_call_id": agent_tool_call_id,
                "decision_status": ApprovalResumeDecisionStatus.APPROVED,
            },
        ),
    )

    assert decision.route is (HumanApprovedSupportGraphRoute.HANDLE_APPROVAL_DECISION)


def test_approved_without_output_routes_to_sensitive_execution() -> None:
    decision = select_human_approved_support_route(
        _classified_sensitive(
            approval_status=HumanApprovalCheckpointStatus.APPROVED,
        ),
    )

    assert decision.route is (HumanApprovedSupportGraphRoute.EXECUTE_SENSITIVE_TOOL)


def test_approved_with_output_routes_to_recommendation() -> None:
    decision = select_human_approved_support_route(
        _classified_sensitive(
            approval_status=HumanApprovalCheckpointStatus.APPROVED,
            sensitive_execution_output={
                "escalation_id": uuid4(),
                "ticket_id": uuid4(),
                "target_queue": "engineering_support",
                "status": "escalated",
            },
        ),
    )

    assert decision.route is (HumanApprovedSupportGraphRoute.DRAFT_GROUNDED_RECOMMENDATION)


def test_rejected_routes_to_recommendation() -> None:
    decision = select_human_approved_support_route(
        _classified_sensitive(
            approval_status=HumanApprovalCheckpointStatus.REJECTED,
        ),
    )

    assert decision.route is (HumanApprovedSupportGraphRoute.DRAFT_GROUNDED_RECOMMENDATION)


def test_expired_routes_to_recommendation() -> None:
    decision = select_human_approved_support_route(
        _classified_sensitive(
            approval_status=HumanApprovalCheckpointStatus.EXPIRED,
        ),
    )

    assert decision.route is (HumanApprovedSupportGraphRoute.DRAFT_GROUNDED_RECOMMENDATION)


def test_drafted_recommendation_routes_to_validation() -> None:
    decision = select_human_approved_support_route(
        _classified_sensitive(
            approval_status=HumanApprovalCheckpointStatus.REJECTED,
            recommendation_invocation_id=uuid4(),
            recommendation_id=uuid4(),
            recommendation_stage=HumanApprovedRecommendationStage.DRAFTED,
        ),
    )

    assert decision.route is (HumanApprovedSupportGraphRoute.VALIDATE_RECOMMENDATION)


def test_validated_recommendation_routes_to_persist() -> None:
    decision = select_human_approved_support_route(
        _classified_sensitive(
            approval_status=HumanApprovalCheckpointStatus.REJECTED,
            recommendation_invocation_id=uuid4(),
            recommendation_id=uuid4(),
            recommendation_stage=HumanApprovedRecommendationStage.VALIDATED,
        ),
    )

    assert decision.route is (HumanApprovedSupportGraphRoute.PERSIST_RECOMMENDATION)


def test_persisted_recommendation_routes_to_complete() -> None:
    decision = select_human_approved_support_route(
        _classified_sensitive(
            approval_status=HumanApprovalCheckpointStatus.REJECTED,
            recommendation_invocation_id=uuid4(),
            recommendation_id=uuid4(),
            recommendation_stage=HumanApprovedRecommendationStage.PERSISTED,
        ),
    )

    assert decision.route is HumanApprovedSupportGraphRoute.COMPLETE_WORKFLOW


def test_error_routes_fail_closed() -> None:
    decision = select_human_approved_support_route(
        _snapshot(
            current_error_code="workflow_state_invalid",
        ),
    )

    assert decision.route is (HumanApprovedSupportGraphRoute.FAIL_WORKFLOW)
    assert decision.error_code == "workflow_state_invalid"
