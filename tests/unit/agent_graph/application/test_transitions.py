"""Unit tests for controlled support graph state transitions."""

import json
from dataclasses import replace
from uuid import UUID

import pytest

from supportops.agent_graph.application.transitions import (
    ControlledSupportStateTransitionError,
    GraphStateTransitionConflictError,
    advance_graph_step,
    attach_analysis_completion,
    attach_classification,
    attach_recommendation,
    attach_recommendation_invocation,
    mark_graph_error,
    reserve_decision_turn,
)
from supportops.agent_graph.domain.completion import (
    CompleteSupportAnalysisInput,
)
from supportops.agent_graph.domain.routing import (
    ControlledSupportRuntimeLimits,
    GraphStepBudgetExhaustedError,
)
from supportops.agent_graph.domain.state import (
    ControlledSupportGraphStateSnapshot,
    create_initial_controlled_support_state,
    validate_controlled_support_state,
)
from supportops.ai.schemas.ticket_classification import (
    TICKET_CLASSIFICATION_SCHEMA_VERSION,
    TicketCategory,
    TicketIntent,
    TicketSentiment,
    TicketUrgency,
)
from supportops.modules.support_recommendations.domain.models import (
    SupportRecommendation,
    SupportRecommendationAction,
)
from supportops.modules.ticket_classifications.domain.models import (
    TicketClassification,
)

_WORKSPACE_ID = UUID("10000000-0000-4000-8000-000000000001")
_TICKET_ID = UUID("20000000-0000-4000-8000-000000000002")
_AGENT_RUN_ID = UUID("30000000-0000-4000-8000-000000000003")
_CLASSIFICATION_ID = UUID("40000000-0000-4000-8000-000000000004")
_CLASSIFICATION_INVOCATION_ID = UUID("50000000-0000-4000-8000-000000000005")
_RECOMMENDATION_INVOCATION_ID = UUID("60000000-0000-4000-8000-000000000006")
_RECOMMENDATION_ID = UUID("70000000-0000-4000-8000-000000000007")

_PROMPT_HASH = "a" * 64


def _initial_state() -> ControlledSupportGraphStateSnapshot:
    return validate_controlled_support_state(
        create_initial_controlled_support_state(
            workspace_id=_WORKSPACE_ID,
            ticket_id=_TICKET_ID,
            agent_run_id=_AGENT_RUN_ID,
        )
    )


def _classification() -> TicketClassification:
    return TicketClassification.create(
        classification_id=_CLASSIFICATION_ID,
        workspace_id=_WORKSPACE_ID,
        ticket_id=_TICKET_ID,
        agent_run_id=_AGENT_RUN_ID,
        accepted_llm_invocation_id=(_CLASSIFICATION_INVOCATION_ID),
        category=TicketCategory.ACCOUNT_ACCESS,
        intent=TicketIntent.REQUEST_ACCESS,
        urgency=TicketUrgency.NORMAL,
        sentiment=TicketSentiment.NEUTRAL,
        requires_human_review=False,
        summary=("The customer needs documented account-access recovery guidance."),
        schema_version=(TICKET_CLASSIFICATION_SCHEMA_VERSION),
        prompt_id="ticket-classification",
        prompt_version=1,
        prompt_content_hash=_PROMPT_HASH,
        provider="mock",
        model="mock-support-model-v1",
    )


def _completion() -> CompleteSupportAnalysisInput:
    return CompleteSupportAnalysisInput(
        recommended_action=(SupportRecommendationAction.RESPOND),
        evidence_sufficient=True,
        requires_human_review=False,
        decision_summary=("Authoritative evidence supports a direct response."),
    )


def _state_ready_for_completion() -> ControlledSupportGraphStateSnapshot:
    state = attach_classification(
        _initial_state(),
        _classification(),
    )

    return reserve_decision_turn(state)


def _state_ready_for_recommendation() -> ControlledSupportGraphStateSnapshot:
    state = attach_analysis_completion(
        _state_ready_for_completion(),
        _completion(),
    )

    return attach_recommendation_invocation(
        state,
        _RECOMMENDATION_INVOCATION_ID,
    )


def _recommendation() -> SupportRecommendation:
    return SupportRecommendation.create(
        recommendation_id=_RECOMMENDATION_ID,
        workspace_id=_WORKSPACE_ID,
        ticket_id=_TICKET_ID,
        agent_run_id=_AGENT_RUN_ID,
        classification_id=_CLASSIFICATION_ID,
        accepted_llm_invocation_id=(_RECOMMENDATION_INVOCATION_ID),
        recommended_action=(SupportRecommendationAction.RESPOND),
        response_text=("Follow the documented account-access recovery procedure."),
        requires_human_review=False,
        decision_summary=("Authoritative evidence supports a direct response."),
        prompt_id="support-recommendation-draft",
        prompt_version=1,
        prompt_content_hash=_PROMPT_HASH,
        provider="mock",
        model="mock-support-model-v1",
    )


def test_advances_graph_step_with_full_revalidation() -> None:
    state = advance_graph_step(_initial_state())

    assert state.graph_step_count == 1
    assert state.workspace_id == _WORKSPACE_ID
    assert state.ticket_id == _TICKET_ID
    assert state.agent_run_id == _AGENT_RUN_ID


def test_graph_step_respects_runtime_limit() -> None:
    limits = ControlledSupportRuntimeLimits(
        max_graph_steps=1,
        max_tool_calls=1,
        max_decision_turns=2,
    )
    state = advance_graph_step(
        _initial_state(),
        limits=limits,
    )

    with pytest.raises(
        GraphStepBudgetExhaustedError,
        match="graph-step budget",
    ):
        advance_graph_step(
            state,
            limits=limits,
        )


def test_projects_persisted_classification() -> None:
    classification = _classification()

    state = attach_classification(
        _initial_state(),
        classification,
    )

    assert state.classification_id == classification.id
    assert state.classification_category is (TicketCategory.ACCOUNT_ACCESS)
    assert state.classification_intent is (TicketIntent.REQUEST_ACCESS)
    assert state.classification_urgency is (TicketUrgency.NORMAL)
    assert state.classification_sentiment is (TicketSentiment.NEUTRAL)
    assert state.classification_requires_human_review is False
    assert state.classification_summary == (classification.summary)


def test_identical_classification_replay_is_idempotent() -> None:
    classification = _classification()
    state = attach_classification(
        _initial_state(),
        classification,
    )

    replayed = attach_classification(
        state,
        classification,
    )

    assert replayed is state


def test_rejects_classification_ownership_mismatch() -> None:
    classification = replace(
        _classification(),
        workspace_id=UUID("80000000-0000-4000-8000-000000000008"),
    )

    with pytest.raises(
        ControlledSupportStateTransitionError,
        match="workspace ownership",
    ):
        attach_classification(
            _initial_state(),
            classification,
        )


def test_rejects_classification_replacement() -> None:
    state = attach_classification(
        _initial_state(),
        _classification(),
    )
    replacement = replace(
        _classification(),
        id=UUID("90000000-0000-4000-8000-000000000009"),
    )

    with pytest.raises(
        GraphStateTransitionConflictError,
        match="different classification",
    ):
        attach_classification(
            state,
            replacement,
        )


def test_reserves_decision_after_classification() -> None:
    state = attach_classification(
        _initial_state(),
        _classification(),
    )

    reserved = reserve_decision_turn(state)

    assert reserved.decision_turn_count == 1
    assert reserved.tool_call_count == 0


def test_terminal_analysis_requires_unresolved_decision() -> None:
    state = attach_classification(
        _initial_state(),
        _classification(),
    )

    with pytest.raises(
        ControlledSupportStateTransitionError,
        match="one unresolved decision",
    ):
        attach_analysis_completion(
            state,
            _completion(),
        )


def test_attaches_terminal_analysis() -> None:
    state = attach_analysis_completion(
        _state_ready_for_completion(),
        _completion(),
    )

    assert state.analysis_completion is not None
    assert state.analysis_completion.recommended_action == "respond"
    assert state.analysis_completion.evidence_sufficient is True


def test_identical_terminal_analysis_replay_is_idempotent() -> None:
    state = attach_analysis_completion(
        _state_ready_for_completion(),
        _completion(),
    )

    replayed = attach_analysis_completion(
        state,
        _completion(),
    )

    assert replayed is state


def test_rejects_terminal_analysis_replacement() -> None:
    state = attach_analysis_completion(
        _state_ready_for_completion(),
        _completion(),
    )
    replacement = CompleteSupportAnalysisInput(
        recommended_action=(SupportRecommendationAction.REQUEST_MORE_INFORMATION),
        evidence_sufficient=False,
        requires_human_review=False,
        decision_summary=("The ticket requires additional diagnostic details."),
    )

    with pytest.raises(
        GraphStateTransitionConflictError,
        match="different terminal",
    ):
        attach_analysis_completion(
            state,
            replacement,
        )


def test_recommendation_invocation_requires_completion() -> None:
    state = _state_ready_for_completion()

    with pytest.raises(
        ControlledSupportStateTransitionError,
        match="requires terminal analysis",
    ):
        attach_recommendation_invocation(
            state,
            _RECOMMENDATION_INVOCATION_ID,
        )


def test_attaches_recommendation_invocation() -> None:
    state = attach_analysis_completion(
        _state_ready_for_completion(),
        _completion(),
    )

    updated = attach_recommendation_invocation(
        state,
        _RECOMMENDATION_INVOCATION_ID,
    )

    assert updated.recommendation_invocation_id == (_RECOMMENDATION_INVOCATION_ID)


def test_attaches_persisted_recommendation() -> None:
    recommendation = _recommendation()

    state = attach_recommendation(
        _state_ready_for_recommendation(),
        recommendation,
    )

    assert state.recommendation_id == recommendation.id


def test_recommendation_replay_is_idempotent() -> None:
    recommendation = _recommendation()
    state = attach_recommendation(
        _state_ready_for_recommendation(),
        recommendation,
    )

    replayed = attach_recommendation(
        state,
        recommendation,
    )

    assert replayed is state


def test_rejects_recommendation_invocation_mismatch() -> None:
    recommendation = replace(
        _recommendation(),
        accepted_llm_invocation_id=UUID("a0000000-0000-4000-8000-000000000010"),
    )

    with pytest.raises(
        ControlledSupportStateTransitionError,
        match="invocation does not match",
    ):
        attach_recommendation(
            _state_ready_for_recommendation(),
            recommendation,
        )


def test_rejects_recommendation_action_mismatch() -> None:
    recommendation = replace(
        _recommendation(),
        recommended_action=(SupportRecommendationAction.REQUEST_MORE_INFORMATION),
    )

    with pytest.raises(
        ControlledSupportStateTransitionError,
        match="action does not match",
    ):
        attach_recommendation(
            _state_ready_for_recommendation(),
            recommendation,
        )


def test_cannot_weaken_human_review_requirement() -> None:
    completion = CompleteSupportAnalysisInput(
        recommended_action=(SupportRecommendationAction.RESPOND),
        evidence_sufficient=True,
        requires_human_review=True,
        decision_summary=("The case requires specialist review."),
    )
    state = attach_analysis_completion(
        _state_ready_for_completion(),
        completion,
    )
    state = attach_recommendation_invocation(
        state,
        _RECOMMENDATION_INVOCATION_ID,
    )
    recommendation = replace(
        _recommendation(),
        requires_human_review=False,
    )

    with pytest.raises(
        ControlledSupportStateTransitionError,
        match="cannot weaken",
    ):
        attach_recommendation(
            state,
            recommendation,
        )


def test_records_first_graph_error() -> None:
    state = mark_graph_error(
        _initial_state(),
        error_code="tool_dependency_unavailable",
    )

    assert state.current_error_code == ("tool_dependency_unavailable")


def test_identical_error_replay_is_idempotent() -> None:
    state = mark_graph_error(
        _initial_state(),
        error_code="tool_dependency_unavailable",
    )

    replayed = mark_graph_error(
        state,
        error_code="tool_dependency_unavailable",
    )

    assert replayed is state


def test_rejects_error_replacement() -> None:
    state = mark_graph_error(
        _initial_state(),
        error_code="tool_dependency_unavailable",
    )

    with pytest.raises(
        GraphStateTransitionConflictError,
        match="different error code",
    ):
        mark_graph_error(
            state,
            error_code="tool_timeout",
        )


def test_successful_transition_stops_after_error() -> None:
    state = mark_graph_error(
        _initial_state(),
        error_code="tool_dependency_unavailable",
    )

    with pytest.raises(
        ControlledSupportStateTransitionError,
        match="after a graph error",
    ):
        advance_graph_step(state)


def test_completed_recommendation_cannot_fail_afterward() -> None:
    state = attach_recommendation(
        _state_ready_for_recommendation(),
        _recommendation(),
    )

    with pytest.raises(
        ControlledSupportStateTransitionError,
        match="cannot transition to workflow failure",
    ):
        mark_graph_error(
            state,
            error_code="unexpected_failure",
        )


def test_transitioned_state_remains_json_compatible() -> None:
    state = attach_recommendation(
        _state_ready_for_recommendation(),
        _recommendation(),
    )
    graph_state = state.to_graph_state()

    serialized = json.dumps(
        graph_state,
        sort_keys=True,
    )

    assert json.loads(serialized) == graph_state
