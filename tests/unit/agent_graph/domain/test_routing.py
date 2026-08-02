"""Unit tests for controlled support graph routing policy."""

from dataclasses import FrozenInstanceError
from uuid import UUID

import pytest

from supportops.agent_graph.domain.routing import (
    CONTROLLED_SUPPORT_RUNTIME_LIMITS,
    ControlledSupportGraphRoute,
    ControlledSupportRuntimeLimits,
    DecisionTurnBudgetExhaustedError,
    GraphRoutingStateError,
    GraphRuntimeLimitsIncompatibleError,
    GraphStepBudgetExhaustedError,
    ToolCallBudgetExhaustedError,
    calculate_remaining_workflow_budget,
    reserve_next_decision_turn,
    reserve_next_graph_step,
    reserve_next_tool_call,
    select_controlled_support_route,
)
from supportops.agent_graph.domain.state import (
    CONTROLLED_SUPPORT_GRAPH_VERSION,
    CONTROLLED_SUPPORT_STATE_SCHEMA_VERSION,
    CONTROLLED_SUPPORT_WORKFLOW_NAME,
    CONTROLLED_SUPPORT_WORKFLOW_VERSION,
    ControlledSupportGraphStateSnapshot,
    SupportAnalysisCompletionSnapshot,
)
from supportops.ai.schemas.ticket_classification import (
    TicketCategory,
    TicketIntent,
    TicketSentiment,
    TicketUrgency,
)

_WORKSPACE_ID = UUID("10000000-0000-4000-8000-000000000001")
_TICKET_ID = UUID("20000000-0000-4000-8000-000000000002")
_AGENT_RUN_ID = UUID("30000000-0000-4000-8000-000000000003")
_CLASSIFICATION_ID = UUID("40000000-0000-4000-8000-000000000004")
_RECOMMENDATION_INVOCATION_ID = UUID("50000000-0000-4000-8000-000000000005")
_RECOMMENDATION_ID = UUID("60000000-0000-4000-8000-000000000006")


def _state(
    *,
    classified: bool = False,
    graph_step_count: int = 0,
    decision_turn_count: int = 0,
    tool_call_count: int = 0,
    analysis_completed: bool = False,
    recommendation_invocation_id: UUID | None = None,
    recommendation_id: UUID | None = None,
    current_error_code: str | None = None,
) -> ControlledSupportGraphStateSnapshot:
    tool_call_ids = tuple(
        UUID(f"70000000-0000-4000-8000-{sequence:012d}")
        for sequence in range(
            1,
            tool_call_count + 1,
        )
    )
    fingerprints = tuple(
        f"{sequence:064x}"
        for sequence in range(
            1,
            tool_call_count + 1,
        )
    )

    return ControlledSupportGraphStateSnapshot(
        state_schema_version=(CONTROLLED_SUPPORT_STATE_SCHEMA_VERSION),
        workflow_name=CONTROLLED_SUPPORT_WORKFLOW_NAME,
        workflow_version=(CONTROLLED_SUPPORT_WORKFLOW_VERSION),
        graph_version=CONTROLLED_SUPPORT_GRAPH_VERSION,
        workspace_id=_WORKSPACE_ID,
        ticket_id=_TICKET_ID,
        agent_run_id=_AGENT_RUN_ID,
        classification_id=(_CLASSIFICATION_ID if classified else None),
        classification_category=(TicketCategory.ACCOUNT_ACCESS if classified else None),
        classification_intent=(TicketIntent.REQUEST_ACCESS if classified else None),
        classification_urgency=(TicketUrgency.NORMAL if classified else None),
        classification_sentiment=(TicketSentiment.NEUTRAL if classified else None),
        classification_requires_human_review=(False if classified else None),
        classification_summary=(
            "The customer needs account-access guidance." if classified else None
        ),
        graph_step_count=graph_step_count,
        decision_turn_count=decision_turn_count,
        tool_call_count=tool_call_count,
        seen_tool_call_fingerprints=fingerprints,
        tool_call_ids=tool_call_ids,
        retrieval_query_ids=(),
        retrieved_chunk_ids=(),
        service_status_tool_call_ids=(),
        analysis_completion=(
            SupportAnalysisCompletionSnapshot(
                recommended_action="respond",
                evidence_sufficient=True,
                requires_human_review=False,
                decision_summary=("Available evidence supports a direct response."),
            )
            if analysis_completed
            else None
        ),
        recommendation_invocation_id=(recommendation_invocation_id),
        recommendation_id=recommendation_id,
        current_error_code=current_error_code,
    )


def test_default_runtime_limits_match_approved_policy() -> None:
    assert CONTROLLED_SUPPORT_RUNTIME_LIMITS.max_graph_steps == 16
    assert CONTROLLED_SUPPORT_RUNTIME_LIMITS.max_tool_calls == 3
    assert CONTROLLED_SUPPORT_RUNTIME_LIMITS.max_decision_turns == 4


def test_runtime_limits_reserve_terminal_turn() -> None:
    with pytest.raises(
        ValueError,
        match="reserve at least one terminal turn",
    ):
        ControlledSupportRuntimeLimits(
            max_graph_steps=16,
            max_tool_calls=4,
            max_decision_turns=4,
        )


@pytest.mark.parametrize(
    "field_name",
    [
        "max_graph_steps",
        "max_tool_calls",
        "max_decision_turns",
    ],
)
def test_runtime_limits_require_real_integers(
    field_name: str,
) -> None:
    values: dict[str, object] = {
        "max_graph_steps": 16,
        "max_tool_calls": 3,
        "max_decision_turns": 4,
    }
    values[field_name] = True

    with pytest.raises(
        TypeError,
        match="must be an integer",
    ):
        ControlledSupportRuntimeLimits(
            **values,  # type: ignore[arg-type]
        )


def test_calculates_remaining_budget() -> None:
    budget = calculate_remaining_workflow_budget(
        _state(
            classified=True,
            graph_step_count=5,
            decision_turn_count=2,
            tool_call_count=2,
        )
    )

    assert budget.graph_steps == 11
    assert budget.tool_calls == 1
    assert budget.decision_turns == 2
    assert budget.can_advance_graph is True
    assert budget.can_decide is True
    assert budget.can_execute_tool is True
    assert budget.requires_terminal_decision is False


def test_tool_exhaustion_reserves_terminal_decision() -> None:
    budget = calculate_remaining_workflow_budget(
        _state(
            classified=True,
            decision_turn_count=3,
            tool_call_count=3,
        )
    )

    assert budget.tool_calls == 0
    assert budget.decision_turns == 1
    assert budget.can_decide is True
    assert budget.can_execute_tool is False
    assert budget.requires_terminal_decision is True


def test_rejects_recovered_state_above_runtime_limits() -> None:
    state = _state(
        classified=True,
        decision_turn_count=5,
        tool_call_count=3,
    )

    with pytest.raises(
        GraphRuntimeLimitsIncompatibleError,
        match="exceed the active",
    ):
        calculate_remaining_workflow_budget(state)


def test_routes_unclassified_state_to_classification() -> None:
    decision = select_controlled_support_route(_state())

    assert decision.route is (ControlledSupportGraphRoute.ENSURE_CLASSIFICATION)
    assert decision.error_code is None


def test_routes_classified_state_to_decision() -> None:
    decision = select_controlled_support_route(_state(classified=True))

    assert decision.route is (ControlledSupportGraphRoute.DECIDE_NEXT_ACTION)


def test_routes_tool_exhaustion_to_final_decision() -> None:
    decision = select_controlled_support_route(
        _state(
            classified=True,
            decision_turn_count=3,
            tool_call_count=3,
        )
    )

    assert decision.route is (ControlledSupportGraphRoute.DECIDE_NEXT_ACTION)


def test_routes_terminal_analysis_to_drafting() -> None:
    decision = select_controlled_support_route(
        _state(
            classified=True,
            decision_turn_count=1,
            analysis_completed=True,
        )
    )

    assert decision.route is (ControlledSupportGraphRoute.DRAFT_RECOMMENDATION)


def test_routes_drafted_recommendation_to_persistence() -> None:
    decision = select_controlled_support_route(
        _state(
            classified=True,
            decision_turn_count=1,
            analysis_completed=True,
            recommendation_invocation_id=(_RECOMMENDATION_INVOCATION_ID),
        )
    )

    assert decision.route is (ControlledSupportGraphRoute.PERSIST_RECOMMENDATION)


def test_routes_persisted_recommendation_to_completion() -> None:
    decision = select_controlled_support_route(
        _state(
            classified=True,
            decision_turn_count=1,
            analysis_completed=True,
            recommendation_invocation_id=(_RECOMMENDATION_INVOCATION_ID),
            recommendation_id=_RECOMMENDATION_ID,
        )
    )

    assert decision.route is (ControlledSupportGraphRoute.COMPLETE_WORKFLOW)


def test_current_error_takes_precedence() -> None:
    decision = select_controlled_support_route(
        _state(
            classified=True,
            current_error_code="tool_dependency_unavailable",
        )
    )

    assert decision.route is (ControlledSupportGraphRoute.FAIL_WORKFLOW)
    assert decision.error_code == ("tool_dependency_unavailable")


def test_graph_step_exhaustion_fails_closed() -> None:
    decision = select_controlled_support_route(
        _state(
            classified=True,
            graph_step_count=16,
        )
    )

    assert decision.route is (ControlledSupportGraphRoute.FAIL_WORKFLOW)
    assert decision.error_code == (GraphStepBudgetExhaustedError.error_code)


def test_decision_turn_exhaustion_fails_closed() -> None:
    decision = select_controlled_support_route(
        _state(
            classified=True,
            decision_turn_count=4,
            tool_call_count=3,
        )
    )

    assert decision.route is (ControlledSupportGraphRoute.FAIL_WORKFLOW)
    assert decision.error_code == (DecisionTurnBudgetExhaustedError.error_code)


def test_incompatible_recovery_routes_to_failure() -> None:
    decision = select_controlled_support_route(
        _state(
            classified=True,
            decision_turn_count=5,
            tool_call_count=3,
        )
    )

    assert decision.route is (ControlledSupportGraphRoute.FAIL_WORKFLOW)
    assert decision.error_code == (GraphRuntimeLimitsIncompatibleError.error_code)


def test_reserves_next_graph_step() -> None:
    assert reserve_next_graph_step(_state(graph_step_count=5)) == 6


def test_rejects_graph_step_after_exhaustion() -> None:
    with pytest.raises(
        GraphStepBudgetExhaustedError,
        match="graph-step budget",
    ):
        reserve_next_graph_step(_state(graph_step_count=16))


def test_reserves_next_decision_after_settled_tools() -> None:
    state = _state(
        classified=True,
        decision_turn_count=2,
        tool_call_count=2,
    )

    assert reserve_next_decision_turn(state) == 3


def test_reserves_terminal_turn_after_tool_exhaustion() -> None:
    state = _state(
        classified=True,
        decision_turn_count=3,
        tool_call_count=3,
    )

    assert reserve_next_decision_turn(state) == 4


def test_new_decision_requires_prior_tool_to_be_settled() -> None:
    state = _state(
        classified=True,
        decision_turn_count=2,
        tool_call_count=1,
    )

    with pytest.raises(
        GraphRoutingStateError,
        match="prior tool decision",
    ):
        reserve_next_decision_turn(state)


def test_reserves_tool_sequence_after_decision() -> None:
    state = _state(
        classified=True,
        decision_turn_count=2,
        tool_call_count=1,
    )

    assert reserve_next_tool_call(state) == 2


def test_tool_requires_one_unresolved_decision() -> None:
    state = _state(
        classified=True,
        decision_turn_count=1,
        tool_call_count=1,
    )

    with pytest.raises(
        GraphRoutingStateError,
        match="one unresolved decision",
    ):
        reserve_next_tool_call(state)


def test_tool_call_fails_after_tool_budget_exhaustion() -> None:
    state = _state(
        classified=True,
        decision_turn_count=4,
        tool_call_count=3,
    )

    with pytest.raises(
        ToolCallBudgetExhaustedError,
        match="tool-call budget",
    ):
        reserve_next_tool_call(state)


def test_decision_cannot_begin_without_classification() -> None:
    with pytest.raises(
        GraphRoutingStateError,
        match="persisted classification",
    ):
        reserve_next_decision_turn(_state())


def test_tool_cannot_begin_after_terminal_analysis() -> None:
    state = _state(
        classified=True,
        decision_turn_count=1,
        tool_call_count=0,
        analysis_completed=True,
    )

    with pytest.raises(
        GraphRoutingStateError,
        match="after terminal analysis",
    ):
        reserve_next_tool_call(state)


def test_route_decision_is_immutable() -> None:
    decision = select_controlled_support_route(_state(classified=True))

    with pytest.raises(FrozenInstanceError):
        decision.error_code = "changed"  # type: ignore[misc]


def test_custom_limits_remain_supported() -> None:
    limits = ControlledSupportRuntimeLimits(
        max_graph_steps=8,
        max_tool_calls=1,
        max_decision_turns=2,
    )
    state = _state(
        classified=True,
        decision_turn_count=1,
        tool_call_count=1,
    )

    budget = calculate_remaining_workflow_budget(
        state,
        limits=limits,
    )

    assert budget.graph_steps == 8
    assert budget.tool_calls == 0
    assert budget.decision_turns == 1
    assert budget.requires_terminal_decision is True
