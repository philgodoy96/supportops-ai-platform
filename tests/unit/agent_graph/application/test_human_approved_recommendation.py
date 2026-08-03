"""Unit tests for human-approved recommendation context."""

from types import SimpleNamespace
from typing import cast

import pytest
from pydantic import JsonValue

from supportops.agent_graph.application.human_approved_recommendation import (
    build_human_approved_recommendation_workflow,
)
from supportops.agent_graph.domain.human_approved_state import (
    HumanApprovalCheckpointStatus,
    HumanApprovedSupportGraphStateSnapshot,
)


def _state(**updates: object) -> HumanApprovedSupportGraphStateSnapshot:
    values: dict[str, object] = {
        "approval_status": HumanApprovalCheckpointStatus.APPROVED,
        "approval_request_reason": "Operational review required.",
        "decision_summary": "Escalate after approval.",
        "analysis_recommended_action": "recommend_escalation",
        "proposed_tool_name": "escalate_ticket",
        "proposed_tool_version": 1,
        "sensitive_execution_output": {
            "escalation_id": "00000000-0000-0000-0000-000000000001",
            "ticket_id": "00000000-0000-0000-0000-000000000002",
            "target_queue": "support_operations",
            "status": "escalated",
        },
    }
    values.update(updates)
    return cast(
        HumanApprovedSupportGraphStateSnapshot,
        SimpleNamespace(**values),
    )


def _as_mapping(value: JsonValue) -> dict[str, JsonValue]:
    assert isinstance(value, dict)
    return value


def test_approved_context_contains_safe_execution_output() -> None:
    workflow = build_human_approved_recommendation_workflow(
        _state(),
    )

    approval = _as_mapping(workflow["approval"])
    sensitive_action = _as_mapping(workflow["sensitive_action"])
    execution_output = _as_mapping(sensitive_action["execution_output"])
    assert approval["status"] == "approved"
    assert execution_output["status"] == "escalated"


def test_rejected_context_cannot_contain_execution_output() -> None:
    with pytest.raises(ValueError, match="cannot expose"):
        build_human_approved_recommendation_workflow(
            _state(
                approval_status=HumanApprovalCheckpointStatus.REJECTED,
            ),
        )


def test_expired_context_cannot_contain_execution_output() -> None:
    with pytest.raises(ValueError, match="cannot expose"):
        build_human_approved_recommendation_workflow(
            _state(
                approval_status=HumanApprovalCheckpointStatus.EXPIRED,
            ),
        )


def test_rejected_context_omits_execution_output() -> None:
    workflow = build_human_approved_recommendation_workflow(
        _state(
            approval_status=HumanApprovalCheckpointStatus.REJECTED,
            sensitive_execution_output=None,
        ),
    )

    approval = _as_mapping(workflow["approval"])
    sensitive_action = _as_mapping(workflow["sensitive_action"])
    assert approval["status"] == "rejected"
    assert sensitive_action["execution_output"] is None


def test_expired_context_omits_execution_output() -> None:
    workflow = build_human_approved_recommendation_workflow(
        _state(
            approval_status=HumanApprovalCheckpointStatus.EXPIRED,
            sensitive_execution_output=None,
        ),
    )

    approval = _as_mapping(workflow["approval"])
    sensitive_action = _as_mapping(workflow["sensitive_action"])
    assert approval["status"] == "expired"
    assert sensitive_action["execution_output"] is None


def test_approved_without_execution_output_cannot_claim_execution() -> None:
    workflow = build_human_approved_recommendation_workflow(
        _state(sensitive_execution_output=None),
    )

    sensitive_action = _as_mapping(workflow["sensitive_action"])
    assert sensitive_action["execution_output"] is None
    assert sensitive_action["tool_name"] == "escalate_ticket"


def test_approved_execution_output_requires_escalated_status() -> None:
    workflow = build_human_approved_recommendation_workflow(
        _state(),
    )

    sensitive_action = _as_mapping(workflow["sensitive_action"])
    execution_output = _as_mapping(sensitive_action["execution_output"])
    assert execution_output == {
        "escalation_id": "00000000-0000-0000-0000-000000000001",
        "ticket_id": "00000000-0000-0000-0000-000000000002",
        "target_queue": "support_operations",
        "status": "escalated",
    }


def test_workflow_includes_approval_and_decision_context() -> None:
    workflow = build_human_approved_recommendation_workflow(
        _state(
            approval_status=HumanApprovalCheckpointStatus.REJECTED,
            sensitive_execution_output=None,
        ),
    )

    assert set(workflow) == {
        "approval",
        "decision",
        "sensitive_action",
    }
    decision = _as_mapping(workflow["decision"])
    assert decision["summary"] == "Escalate after approval."
    assert decision["recommended_action"] == "recommend_escalation"
