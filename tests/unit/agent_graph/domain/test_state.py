"""Unit tests for controlled support graph state contracts."""

import json
from typing import cast
from uuid import UUID

import pytest

from supportops.agent_graph.domain.state import (
    CONTROLLED_SUPPORT_GRAPH_VERSION,
    CONTROLLED_SUPPORT_STATE_SCHEMA_VERSION,
    CONTROLLED_SUPPORT_WORKFLOW_NAME,
    CONTROLLED_SUPPORT_WORKFLOW_VERSION,
    GraphStateIncompatibleError,
    create_initial_controlled_support_state,
    validate_controlled_support_state,
)
from supportops.ai.schemas.ticket_classification import (
    TicketCategory,
    TicketIntent,
    TicketSentiment,
    TicketUrgency,
)

WORKSPACE_ID = UUID("11111111-1111-4111-8111-111111111111")
TICKET_ID = UUID("22222222-2222-4222-8222-222222222222")
AGENT_RUN_ID = UUID("33333333-3333-4333-8333-333333333333")
CLASSIFICATION_ID = UUID("44444444-4444-4444-8444-444444444444")
KNOWLEDGE_TOOL_CALL_ID = UUID("55555555-5555-4555-8555-555555555555")
STATUS_TOOL_CALL_ID = UUID("66666666-6666-4666-8666-666666666666")
RETRIEVAL_QUERY_ID = UUID("77777777-7777-4777-8777-777777777777")
FIRST_CHUNK_ID = UUID("88888888-8888-4888-8888-888888888888")
SECOND_CHUNK_ID = UUID("99999999-9999-4999-8999-999999999999")
RECOMMENDATION_INVOCATION_ID = UUID("aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa")
RECOMMENDATION_ID = UUID("bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb")


def _initial_state() -> dict[str, object]:
    return dict(
        create_initial_controlled_support_state(
            workspace_id=WORKSPACE_ID,
            ticket_id=TICKET_ID,
            agent_run_id=AGENT_RUN_ID,
        )
    )


def _add_complete_classification(
    state: dict[str, object],
) -> None:
    state.update(
        {
            "classification_id": str(CLASSIFICATION_ID),
            "classification_category": TicketCategory.HOW_TO.value,
            "classification_intent": TicketIntent.ASK_QUESTION.value,
            "classification_urgency": TicketUrgency.NORMAL.value,
            "classification_sentiment": TicketSentiment.NEUTRAL.value,
            "classification_requires_human_review": False,
            "classification_summary": ("The customer requests an operational procedure."),
        }
    )


def test_initial_state_is_json_compatible_and_minimal() -> None:
    state = _initial_state()

    assert state["state_schema_version"] == CONTROLLED_SUPPORT_STATE_SCHEMA_VERSION
    assert state["workflow_name"] == CONTROLLED_SUPPORT_WORKFLOW_NAME
    assert state["workflow_version"] == CONTROLLED_SUPPORT_WORKFLOW_VERSION
    assert state["graph_version"] == CONTROLLED_SUPPORT_GRAPH_VERSION
    assert state["workspace_id"] == str(WORKSPACE_ID)
    assert state["ticket_id"] == str(TICKET_ID)
    assert state["agent_run_id"] == str(AGENT_RUN_ID)
    assert state["classification_id"] is None
    assert state["graph_step_count"] == 0
    assert state["decision_turn_count"] == 0
    assert state["tool_call_count"] == 0
    assert state["seen_tool_call_fingerprints"] == []
    assert state["tool_call_ids"] == []
    assert state["retrieval_query_ids"] == []
    assert state["retrieved_chunk_ids"] == []
    assert state["service_status_tool_call_ids"] == []
    assert state["analysis_completion"] is None
    assert state["recommendation_invocation_id"] is None
    assert state["recommendation_id"] is None
    assert state["current_error_code"] is None

    assert "agent_run_attempt_id" not in state
    assert "lease_token" not in state
    assert "request_id" not in state
    assert "correlation_id" not in state

    json.dumps(state, sort_keys=True)


def test_state_round_trips_through_json_checkpoint_payload() -> None:
    state = _initial_state()
    _add_complete_classification(state)

    state.update(
        {
            "graph_step_count": 10,
            "decision_turn_count": 3,
            "tool_call_count": 2,
            "seen_tool_call_fingerprints": [
                "a" * 64,
                "b" * 64,
            ],
            "tool_call_ids": [
                str(KNOWLEDGE_TOOL_CALL_ID),
                str(STATUS_TOOL_CALL_ID),
            ],
            "retrieval_query_ids": [
                str(RETRIEVAL_QUERY_ID),
            ],
            "retrieved_chunk_ids": [
                str(FIRST_CHUNK_ID),
                str(SECOND_CHUNK_ID),
            ],
            "service_status_tool_call_ids": [
                str(STATUS_TOOL_CALL_ID),
            ],
            "analysis_completion": {
                "recommended_action": "respond",
                "evidence_sufficient": True,
                "requires_human_review": False,
                "decision_summary": ("Relevant runbook evidence is available."),
            },
            "recommendation_invocation_id": str(RECOMMENDATION_INVOCATION_ID),
            "recommendation_id": str(RECOMMENDATION_ID),
        }
    )

    serialized_state = json.dumps(state, sort_keys=True)
    loaded_state: object = json.loads(serialized_state)

    assert isinstance(loaded_state, dict)

    snapshot = validate_controlled_support_state(cast(dict[str, object], loaded_state))

    assert snapshot.workspace_id == WORKSPACE_ID
    assert snapshot.ticket_id == TICKET_ID
    assert snapshot.agent_run_id == AGENT_RUN_ID
    assert snapshot.classification_id == CLASSIFICATION_ID
    assert snapshot.classification_category is TicketCategory.HOW_TO
    assert snapshot.classification_intent is TicketIntent.ASK_QUESTION
    assert snapshot.classification_urgency is TicketUrgency.NORMAL
    assert snapshot.classification_sentiment is TicketSentiment.NEUTRAL
    assert snapshot.tool_call_ids == (
        KNOWLEDGE_TOOL_CALL_ID,
        STATUS_TOOL_CALL_ID,
    )
    assert snapshot.retrieved_chunk_ids == (
        FIRST_CHUNK_ID,
        SECOND_CHUNK_ID,
    )
    assert snapshot.recommendation_id == RECOMMENDATION_ID

    reconstructed_state = snapshot.to_graph_state()

    assert reconstructed_state == state
    json.dumps(reconstructed_state, sort_keys=True)


@pytest.mark.parametrize(
    ("field_name", "field_value"),
    [
        ("state_schema_version", "unsupported-state"),
        ("workflow_name", "unsupported-workflow"),
        ("workflow_version", "unsupported-version"),
        ("graph_version", "unsupported-graph"),
    ],
)
def test_state_rejects_incompatible_identity(
    field_name: str,
    field_value: object,
) -> None:
    state = _initial_state()
    state[field_name] = field_value

    with pytest.raises(
        GraphStateIncompatibleError,
        match="incompatible with the current schema",
    ):
        validate_controlled_support_state(state)


@pytest.mark.parametrize(
    "unsafe_field_name",
    [
        "agent_run_attempt_id",
        "lease_token",
        "request_id",
        "correlation_id",
        "database_session",
        "provider",
        "raw_exception",
    ],
)
def test_state_rejects_undeclared_runtime_objects(
    unsafe_field_name: str,
) -> None:
    state = _initial_state()
    state[unsafe_field_name] = object()

    with pytest.raises(GraphStateIncompatibleError):
        validate_controlled_support_state(state)


def test_state_rejects_partial_classification() -> None:
    state = _initial_state()
    state["classification_category"] = TicketCategory.HOW_TO.value

    with pytest.raises(GraphStateIncompatibleError):
        validate_controlled_support_state(state)


def test_state_rejects_incomplete_classification_details() -> None:
    state = _initial_state()
    state["classification_id"] = str(CLASSIFICATION_ID)
    state["classification_category"] = TicketCategory.HOW_TO.value

    with pytest.raises(GraphStateIncompatibleError):
        validate_controlled_support_state(state)


def test_state_rejects_coerced_classification_boolean() -> None:
    state = _initial_state()
    _add_complete_classification(state)
    state["classification_requires_human_review"] = "false"

    with pytest.raises(GraphStateIncompatibleError):
        validate_controlled_support_state(state)


def test_state_rejects_tool_count_without_durable_references() -> None:
    state = _initial_state()
    state["decision_turn_count"] = 1
    state["tool_call_count"] = 1

    with pytest.raises(GraphStateIncompatibleError):
        validate_controlled_support_state(state)


def test_state_rejects_tool_calls_exceeding_decision_turns() -> None:
    state = _initial_state()
    state.update(
        {
            "decision_turn_count": 1,
            "tool_call_count": 2,
            "tool_call_ids": [
                str(KNOWLEDGE_TOOL_CALL_ID),
                str(STATUS_TOOL_CALL_ID),
            ],
            "seen_tool_call_fingerprints": [
                "a" * 64,
                "b" * 64,
            ],
        }
    )

    with pytest.raises(GraphStateIncompatibleError):
        validate_controlled_support_state(state)


def test_state_rejects_malformed_tool_fingerprint() -> None:
    state = _initial_state()
    state.update(
        {
            "decision_turn_count": 1,
            "tool_call_count": 1,
            "tool_call_ids": [
                str(KNOWLEDGE_TOOL_CALL_ID),
            ],
            "seen_tool_call_fingerprints": [
                "not-a-sha256-fingerprint",
            ],
        }
    )

    with pytest.raises(GraphStateIncompatibleError):
        validate_controlled_support_state(state)


def test_state_rejects_duplicate_tool_fingerprints() -> None:
    state = _initial_state()
    state.update(
        {
            "decision_turn_count": 2,
            "tool_call_count": 2,
            "tool_call_ids": [
                str(KNOWLEDGE_TOOL_CALL_ID),
                str(STATUS_TOOL_CALL_ID),
            ],
            "seen_tool_call_fingerprints": [
                "a" * 64,
                "a" * 64,
            ],
        }
    )

    with pytest.raises(GraphStateIncompatibleError):
        validate_controlled_support_state(state)


def test_state_rejects_retrieved_chunks_without_query() -> None:
    state = _initial_state()
    state["retrieved_chunk_ids"] = [str(FIRST_CHUNK_ID)]

    with pytest.raises(GraphStateIncompatibleError):
        validate_controlled_support_state(state)


def test_state_rejects_unknown_service_status_tool_reference() -> None:
    state = _initial_state()
    state["service_status_tool_call_ids"] = [str(STATUS_TOOL_CALL_ID)]

    with pytest.raises(GraphStateIncompatibleError):
        validate_controlled_support_state(state)


def test_state_rejects_completion_without_decision_turn() -> None:
    state = _initial_state()
    state["analysis_completion"] = {
        "recommended_action": "respond",
        "evidence_sufficient": True,
        "requires_human_review": False,
        "decision_summary": "Evidence is available.",
    }

    with pytest.raises(GraphStateIncompatibleError):
        validate_controlled_support_state(state)


def test_state_rejects_recommendation_without_invocation() -> None:
    state = _initial_state()
    state["recommendation_id"] = str(RECOMMENDATION_ID)

    with pytest.raises(GraphStateIncompatibleError):
        validate_controlled_support_state(state)


def test_state_rejects_graph_counter_above_schema_bound() -> None:
    state = _initial_state()
    state["graph_step_count"] = 65

    with pytest.raises(GraphStateIncompatibleError):
        validate_controlled_support_state(state)


def test_state_failure_does_not_expose_checkpoint_payload() -> None:
    secret_value = "checkpoint-payload-secret"
    state = _initial_state()
    state["unexpected"] = secret_value

    with pytest.raises(GraphStateIncompatibleError) as exc_info:
        validate_controlled_support_state(state)

    assert secret_value not in str(exc_info.value)
