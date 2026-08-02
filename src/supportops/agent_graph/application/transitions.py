"""Validated state transitions for the controlled support graph."""

from typing import Any
from uuid import UUID

from supportops.agent_graph.domain.completion import (
    CompleteSupportAnalysisInput,
)
from supportops.agent_graph.domain.routing import (
    CONTROLLED_SUPPORT_RUNTIME_LIMITS,
    ControlledSupportRuntimeLimits,
    reserve_next_decision_turn,
    reserve_next_graph_step,
)
from supportops.agent_graph.domain.state import (
    ControlledSupportGraphStateSnapshot,
)
from supportops.modules.support_recommendations.domain.models import (
    SupportRecommendation,
)
from supportops.modules.ticket_classifications.domain.models import (
    TicketClassification,
)


class ControlledSupportStateTransitionError(RuntimeError):
    """Base error for an invalid controlled graph state transition."""

    error_code = "graph_state_transition_invalid"
    retryable = False


class GraphStateTransitionConflictError(ControlledSupportStateTransitionError):
    """Raised when a transition attempts to replace accepted state."""

    error_code = "graph_state_transition_conflict"

    def __init__(self, message: str) -> None:
        super().__init__(message)


def advance_graph_step(
    state: ControlledSupportGraphStateSnapshot,
    *,
    limits: ControlledSupportRuntimeLimits = (CONTROLLED_SUPPORT_RUNTIME_LIMITS),
) -> ControlledSupportGraphStateSnapshot:
    """Reserve one application-owned graph step."""

    _require_active_state(state)

    return _replace_state(
        state,
        graph_step_count=reserve_next_graph_step(
            state,
            limits=limits,
        ),
    )


def attach_classification(
    state: ControlledSupportGraphStateSnapshot,
    classification: TicketClassification,
) -> ControlledSupportGraphStateSnapshot:
    """Project one persisted classification into checkpoint state."""

    _require_active_state(state)
    _validate_classification_ownership(
        state=state,
        classification=classification,
    )

    updates: dict[str, object] = {
        "classification_id": classification.id,
        "classification_category": classification.category,
        "classification_intent": classification.intent,
        "classification_urgency": classification.urgency,
        "classification_sentiment": classification.sentiment,
        "classification_requires_human_review": (classification.requires_human_review),
        "classification_summary": classification.summary,
    }

    if state.classification_id is None:
        return _replace_state(
            state,
            **updates,
        )

    if _classification_projection(state) == updates:
        return state

    raise GraphStateTransitionConflictError(
        "The graph state already contains a different classification."
    )


def reserve_decision_turn(
    state: ControlledSupportGraphStateSnapshot,
    *,
    limits: ControlledSupportRuntimeLimits = (CONTROLLED_SUPPORT_RUNTIME_LIMITS),
) -> ControlledSupportGraphStateSnapshot:
    """Reserve one provider decision turn."""

    _require_active_state(state)

    return _replace_state(
        state,
        decision_turn_count=reserve_next_decision_turn(
            state,
            limits=limits,
        ),
    )


def attach_analysis_completion(
    state: ControlledSupportGraphStateSnapshot,
    completion: CompleteSupportAnalysisInput,
) -> ControlledSupportGraphStateSnapshot:
    """Attach one terminal non-executable analysis decision."""

    _require_active_state(state)

    if state.classification_id is None:
        raise ControlledSupportStateTransitionError(
            "Terminal analysis requires a persisted classification."
        )

    if state.decision_turn_count != state.tool_call_count + 1:
        raise ControlledSupportStateTransitionError(
            "Terminal analysis requires exactly one unresolved decision turn."
        )

    completion_state = completion.to_state()

    if state.analysis_completion is None:
        return _replace_state(
            state,
            analysis_completion=completion_state,
        )

    if state.analysis_completion.model_dump(mode="json") == completion_state:
        return state

    raise GraphStateTransitionConflictError(
        "The graph state already contains a different terminal analysis decision."
    )


def attach_recommendation_invocation(
    state: ControlledSupportGraphStateSnapshot,
    invocation_id: UUID,
) -> ControlledSupportGraphStateSnapshot:
    """Attach the accepted recommendation-drafting invocation."""

    _require_active_state(state)

    if not isinstance(invocation_id, UUID):
        raise TypeError("invocation_id must be a UUID.")

    if state.analysis_completion is None:
        raise ControlledSupportStateTransitionError(
            "A recommendation invocation requires terminal analysis completion."
        )

    if state.recommendation_invocation_id is None:
        return _replace_state(
            state,
            recommendation_invocation_id=invocation_id,
        )

    if state.recommendation_invocation_id == invocation_id:
        return state

    raise GraphStateTransitionConflictError(
        "The graph state already contains a different recommendation invocation."
    )


def attach_recommendation(
    state: ControlledSupportGraphStateSnapshot,
    recommendation: SupportRecommendation,
) -> ControlledSupportGraphStateSnapshot:
    """Attach one transactionally persisted recommendation."""

    if state.current_error_code is not None:
        raise ControlledSupportStateTransitionError(
            "Successful workflow transitions cannot continue after a graph error."
        )

    _validate_recommendation_ownership(
        state=state,
        recommendation=recommendation,
    )

    if state.classification_id is None:
        raise ControlledSupportStateTransitionError(
            "A persisted recommendation requires a projected classification."
        )

    if state.analysis_completion is None:
        raise ControlledSupportStateTransitionError(
            "A persisted recommendation requires terminal analysis completion."
        )

    if state.recommendation_invocation_id is None:
        raise ControlledSupportStateTransitionError(
            "A persisted recommendation requires an accepted recommendation invocation."
        )

    if recommendation.classification_id != state.classification_id:
        raise ControlledSupportStateTransitionError(
            "The recommendation classification does not match graph state."
        )

    if recommendation.accepted_llm_invocation_id != state.recommendation_invocation_id:
        raise ControlledSupportStateTransitionError(
            "The recommendation invocation does not match graph state."
        )

    if recommendation.recommended_action.value != state.analysis_completion.recommended_action:
        raise ControlledSupportStateTransitionError(
            "The persisted recommendation action does not match terminal analysis."
        )

    if state.analysis_completion.requires_human_review and not recommendation.requires_human_review:
        raise ControlledSupportStateTransitionError(
            "The persisted recommendation cannot weaken the terminal human-review requirement."
        )

    if state.recommendation_id is None:
        return _replace_state(
            state,
            recommendation_id=recommendation.id,
        )

    if state.recommendation_id == recommendation.id:
        return state

    raise GraphStateTransitionConflictError(
        "The graph state already contains a different persisted recommendation."
    )


def mark_graph_error(
    state: ControlledSupportGraphStateSnapshot,
    *,
    error_code: str,
) -> ControlledSupportGraphStateSnapshot:
    """Record the first stable application error code."""

    if state.recommendation_id is not None:
        raise ControlledSupportStateTransitionError(
            "A completed recommendation cannot transition to workflow failure."
        )

    if state.current_error_code is None:
        return _replace_state(
            state,
            current_error_code=error_code,
        )

    if state.current_error_code == error_code:
        return state

    raise GraphStateTransitionConflictError(
        "The graph state already contains a different error code."
    )


def _validate_classification_ownership(
    *,
    state: ControlledSupportGraphStateSnapshot,
    classification: TicketClassification,
) -> None:
    ownership_values = (
        (
            classification.workspace_id,
            state.workspace_id,
            "workspace",
        ),
        (
            classification.ticket_id,
            state.ticket_id,
            "ticket",
        ),
        (
            classification.agent_run_id,
            state.agent_run_id,
            "AgentRun",
        ),
    )

    for actual, expected, resource_name in ownership_values:
        if actual != expected:
            raise ControlledSupportStateTransitionError(
                f"Classification {resource_name} ownership does not match graph state."
            )


def _validate_recommendation_ownership(
    *,
    state: ControlledSupportGraphStateSnapshot,
    recommendation: SupportRecommendation,
) -> None:
    ownership_values = (
        (
            recommendation.workspace_id,
            state.workspace_id,
            "workspace",
        ),
        (
            recommendation.ticket_id,
            state.ticket_id,
            "ticket",
        ),
        (
            recommendation.agent_run_id,
            state.agent_run_id,
            "AgentRun",
        ),
    )

    for actual, expected, resource_name in ownership_values:
        if actual != expected:
            raise ControlledSupportStateTransitionError(
                f"Recommendation {resource_name} ownership does not match graph state."
            )


def _classification_projection(
    state: ControlledSupportGraphStateSnapshot,
) -> dict[str, object]:
    return {
        "classification_id": state.classification_id,
        "classification_category": (state.classification_category),
        "classification_intent": state.classification_intent,
        "classification_urgency": state.classification_urgency,
        "classification_sentiment": (state.classification_sentiment),
        "classification_requires_human_review": (state.classification_requires_human_review),
        "classification_summary": (state.classification_summary),
    }


def _require_active_state(
    state: ControlledSupportGraphStateSnapshot,
) -> None:
    if state.current_error_code is not None:
        raise ControlledSupportStateTransitionError(
            "Successful workflow transitions cannot continue after a graph error."
        )

    if state.recommendation_id is not None:
        raise ControlledSupportStateTransitionError(
            "Successful workflow transitions cannot continue after recommendation persistence."
        )


def _replace_state(
    state: ControlledSupportGraphStateSnapshot,
    **updates: object,
) -> ControlledSupportGraphStateSnapshot:
    payload: dict[str, Any] = state.model_dump(mode="python")
    payload.update(updates)

    return ControlledSupportGraphStateSnapshot.model_validate(payload)
