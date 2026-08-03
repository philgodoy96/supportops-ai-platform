"""Unit tests for human-approved graph state contracts."""

from uuid import uuid4

import pytest

from supportops.agent_graph.domain.human_approved_state import (
    HUMAN_APPROVED_SUPPORT_GRAPH_VERSION,
    HUMAN_APPROVED_SUPPORT_STATE_SCHEMA_VERSION,
    HUMAN_APPROVED_SUPPORT_WORKFLOW_NAME,
    HUMAN_APPROVED_SUPPORT_WORKFLOW_VERSION,
    HumanApprovedGraphStateIncompatibleError,
    HumanApprovedSupportGraphStateSnapshot,
    create_initial_human_approved_support_state,
    validate_human_approved_support_state,
)


def test_initial_state_uses_new_versioned_identity() -> None:
    workspace_id = uuid4()
    ticket_id = uuid4()
    agent_run_id = uuid4()

    state = create_initial_human_approved_support_state(
        workspace_id=workspace_id,
        ticket_id=ticket_id,
        agent_run_id=agent_run_id,
    )

    assert state["state_schema_version"] == (HUMAN_APPROVED_SUPPORT_STATE_SCHEMA_VERSION)
    assert state["workflow_name"] == (HUMAN_APPROVED_SUPPORT_WORKFLOW_NAME)
    assert state["workflow_version"] == (HUMAN_APPROVED_SUPPORT_WORKFLOW_VERSION)
    assert state["graph_version"] == (HUMAN_APPROVED_SUPPORT_GRAPH_VERSION)
    assert state["workspace_id"] == str(workspace_id)
    assert state["ticket_id"] == str(ticket_id)
    assert state["agent_run_id"] == str(agent_run_id)
    assert state["run_context_loaded"] is False
    assert state["graph_step_count"] == 0
    assert state["decision_turn_count"] == 0
    assert state["tool_call_count"] == 0


def test_initial_state_round_trips_through_strict_snapshot() -> None:
    state = create_initial_human_approved_support_state(
        workspace_id=uuid4(),
        ticket_id=uuid4(),
        agent_run_id=uuid4(),
    )

    snapshot = validate_human_approved_support_state(state)

    assert isinstance(
        snapshot,
        HumanApprovedSupportGraphStateSnapshot,
    )
    assert snapshot.to_graph_state() == state


def test_partial_sensitive_proposal_is_incompatible() -> None:
    state = create_initial_human_approved_support_state(
        workspace_id=uuid4(),
        ticket_id=uuid4(),
        agent_run_id=uuid4(),
    )
    state["decision_kind"] = "sensitive_tool"
    state["decision_invocation_id"] = str(uuid4())
    state["decision_summary"] = "Escalation requires approval."
    state["proposed_tool_name"] = "escalate_ticket"

    with pytest.raises(
        HumanApprovedGraphStateIncompatibleError,
    ):
        validate_human_approved_support_state(state)


def test_partial_approval_checkpoint_is_incompatible() -> None:
    state = create_initial_human_approved_support_state(
        workspace_id=uuid4(),
        ticket_id=uuid4(),
        agent_run_id=uuid4(),
    )
    state.update(
        {
            "decision_kind": "sensitive_tool",
            "decision_invocation_id": str(uuid4()),
            "decision_summary": "Escalation requires approval.",
            "proposed_tool_provider_call_id": "call-1",
            "proposed_tool_name": "escalate_ticket",
            "proposed_tool_version": 1,
            "proposed_tool_input": {
                "target_queue": "support_operations",
                "reason": "Needs operational handling.",
            },
            "proposed_tool_fingerprint": "a" * 64,
            "approval_request_reason": ("Needs operational handling."),
            "agent_tool_call_id": str(uuid4()),
        },
    )

    with pytest.raises(
        HumanApprovedGraphStateIncompatibleError,
    ):
        validate_human_approved_support_state(state)


def test_unknown_checkpoint_field_is_incompatible() -> None:
    state = create_initial_human_approved_support_state(
        workspace_id=uuid4(),
        ticket_id=uuid4(),
        agent_run_id=uuid4(),
    )
    state["unexpected"] = "value"  # type: ignore[typeddict-unknown-key]

    with pytest.raises(
        HumanApprovedGraphStateIncompatibleError,
    ):
        validate_human_approved_support_state(state)
