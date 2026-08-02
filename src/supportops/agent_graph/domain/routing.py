"""Deterministic routing and budget policy for the support graph."""

from dataclasses import dataclass
from enum import StrEnum

from supportops.agent_graph.domain.state import (
    CONTROLLED_SUPPORT_GRAPH_STATE_MAX_DECISION_TURNS,
    CONTROLLED_SUPPORT_GRAPH_STATE_MAX_STEPS,
    CONTROLLED_SUPPORT_GRAPH_STATE_MAX_TOOL_CALLS,
    ControlledSupportGraphStateSnapshot,
)

CONTROLLED_SUPPORT_MAX_GRAPH_STEPS = 16
CONTROLLED_SUPPORT_MAX_TOOL_CALLS = 3
CONTROLLED_SUPPORT_MAX_DECISION_TURNS = 4


class ControlledSupportGraphRoute(StrEnum):
    """Deterministic lifecycle routes exposed to graph composition."""

    ENSURE_CLASSIFICATION = "ensure_classification"
    DECIDE_NEXT_ACTION = "decide_next_action"
    DRAFT_RECOMMENDATION = "draft_recommendation"
    PERSIST_RECOMMENDATION = "persist_recommendation"
    COMPLETE_WORKFLOW = "complete_workflow"
    FAIL_WORKFLOW = "fail_workflow"


class ControlledSupportRoutingError(RuntimeError):
    """Base error for non-retryable graph policy violations."""

    error_code = "controlled_support_routing_error"
    retryable = False


class GraphRuntimeLimitsIncompatibleError(ControlledSupportRoutingError):
    """Raised when recovered counters exceed active runtime limits."""

    error_code = "graph_runtime_limits_incompatible"

    def __init__(self) -> None:
        super().__init__(
            "Recovered graph counters exceed the active controlled-workflow runtime limits."
        )


class GraphStepBudgetExhaustedError(ControlledSupportRoutingError):
    """Raised when no application graph step remains."""

    error_code = "graph_step_budget_exhausted"

    def __init__(self) -> None:
        super().__init__("The controlled workflow exhausted its graph-step budget.")


class DecisionTurnBudgetExhaustedError(ControlledSupportRoutingError):
    """Raised when no LLM decision turn remains."""

    error_code = "decision_turn_budget_exhausted"

    def __init__(self) -> None:
        super().__init__("The controlled workflow exhausted its decision-turn budget.")


class ToolCallBudgetExhaustedError(ControlledSupportRoutingError):
    """Raised when no executable tool call remains."""

    error_code = "tool_call_budget_exhausted"

    def __init__(self) -> None:
        super().__init__("The controlled workflow exhausted its tool-call budget.")


class GraphRoutingStateError(ControlledSupportRoutingError):
    """Raised when a transition contradicts graph lifecycle state."""

    error_code = "graph_routing_state_invalid"

    def __init__(self, message: str) -> None:
        super().__init__(message)


def _validate_positive_integer(
    value: int,
    *,
    field_name: str,
) -> None:
    if type(value) is not int:
        raise TypeError(f"{field_name} must be an integer.")

    if value <= 0:
        raise ValueError(f"{field_name} must be positive.")


@dataclass(frozen=True, slots=True)
class ControlledSupportRuntimeLimits:
    """Application runtime limits narrower than state storage bounds."""

    max_graph_steps: int = CONTROLLED_SUPPORT_MAX_GRAPH_STEPS
    max_tool_calls: int = CONTROLLED_SUPPORT_MAX_TOOL_CALLS
    max_decision_turns: int = CONTROLLED_SUPPORT_MAX_DECISION_TURNS

    def __post_init__(self) -> None:
        _validate_positive_integer(
            self.max_graph_steps,
            field_name="max_graph_steps",
        )
        _validate_positive_integer(
            self.max_tool_calls,
            field_name="max_tool_calls",
        )
        _validate_positive_integer(
            self.max_decision_turns,
            field_name="max_decision_turns",
        )

        if self.max_graph_steps > CONTROLLED_SUPPORT_GRAPH_STATE_MAX_STEPS:
            raise ValueError("max_graph_steps exceeds the graph-state structural bound.")

        if self.max_tool_calls > CONTROLLED_SUPPORT_GRAPH_STATE_MAX_TOOL_CALLS:
            raise ValueError("max_tool_calls exceeds the graph-state structural bound.")

        if self.max_decision_turns > CONTROLLED_SUPPORT_GRAPH_STATE_MAX_DECISION_TURNS:
            raise ValueError("max_decision_turns exceeds the graph-state structural bound.")

        if self.max_tool_calls >= self.max_decision_turns:
            raise ValueError(
                "max_decision_turns must reserve at least one "
                "terminal turn after tool-call exhaustion."
            )


CONTROLLED_SUPPORT_RUNTIME_LIMITS = ControlledSupportRuntimeLimits()


@dataclass(frozen=True, slots=True)
class RemainingWorkflowBudget:
    """Remaining operational capacity for one validated graph state."""

    graph_steps: int
    tool_calls: int
    decision_turns: int
    analysis_completed: bool

    def __post_init__(self) -> None:
        values = (
            self.graph_steps,
            self.tool_calls,
            self.decision_turns,
        )

        if any(value < 0 for value in values):
            raise ValueError("Remaining workflow budgets must be non-negative.")

    @property
    def can_advance_graph(self) -> bool:
        """Return whether another application graph step remains."""

        return self.graph_steps > 0

    @property
    def can_decide(self) -> bool:
        """Return whether another model decision may be requested."""

        return not self.analysis_completed and self.decision_turns > 0

    @property
    def can_execute_tool(self) -> bool:
        """Return whether another read-only tool may be executed."""

        return not self.analysis_completed and self.tool_calls > 0 and self.decision_turns > 0

    @property
    def requires_terminal_decision(self) -> bool:
        """Return whether only the terminal decision should remain."""

        return not self.analysis_completed and self.tool_calls == 0 and self.decision_turns > 0


@dataclass(frozen=True, slots=True)
class ControlledSupportRouteDecision:
    """One deterministic route selected from validated graph state."""

    route: ControlledSupportGraphRoute
    error_code: str | None = None

    def __post_init__(self) -> None:
        if self.route is ControlledSupportGraphRoute.FAIL_WORKFLOW:
            if self.error_code is None:
                raise ValueError("Failure routes require an error_code.")

            return

        if self.error_code is not None:
            raise ValueError("Non-failure routes cannot define an error_code.")


def calculate_remaining_workflow_budget(
    state: ControlledSupportGraphStateSnapshot,
    *,
    limits: ControlledSupportRuntimeLimits = (CONTROLLED_SUPPORT_RUNTIME_LIMITS),
) -> RemainingWorkflowBudget:
    """Calculate remaining capacity or reject incompatible recovery."""

    if (
        state.graph_step_count > limits.max_graph_steps
        or state.tool_call_count > limits.max_tool_calls
        or state.decision_turn_count > limits.max_decision_turns
    ):
        raise GraphRuntimeLimitsIncompatibleError()

    return RemainingWorkflowBudget(
        graph_steps=(limits.max_graph_steps - state.graph_step_count),
        tool_calls=(limits.max_tool_calls - state.tool_call_count),
        decision_turns=(limits.max_decision_turns - state.decision_turn_count),
        analysis_completed=(state.analysis_completion is not None),
    )


def select_controlled_support_route(
    state: ControlledSupportGraphStateSnapshot,
    *,
    limits: ControlledSupportRuntimeLimits = (CONTROLLED_SUPPORT_RUNTIME_LIMITS),
) -> ControlledSupportRouteDecision:
    """Select the next lifecycle route without side effects."""

    if state.current_error_code is not None:
        return ControlledSupportRouteDecision(
            route=ControlledSupportGraphRoute.FAIL_WORKFLOW,
            error_code=state.current_error_code,
        )

    if state.recommendation_id is not None:
        return ControlledSupportRouteDecision(route=(ControlledSupportGraphRoute.COMPLETE_WORKFLOW))

    try:
        budget = calculate_remaining_workflow_budget(
            state,
            limits=limits,
        )
    except GraphRuntimeLimitsIncompatibleError as error:
        return ControlledSupportRouteDecision(
            route=ControlledSupportGraphRoute.FAIL_WORKFLOW,
            error_code=error.error_code,
        )

    if not budget.can_advance_graph:
        return ControlledSupportRouteDecision(
            route=ControlledSupportGraphRoute.FAIL_WORKFLOW,
            error_code=(GraphStepBudgetExhaustedError.error_code),
        )

    if state.recommendation_invocation_id is not None:
        return ControlledSupportRouteDecision(
            route=(ControlledSupportGraphRoute.PERSIST_RECOMMENDATION)
        )

    if state.analysis_completion is not None:
        return ControlledSupportRouteDecision(
            route=(ControlledSupportGraphRoute.DRAFT_RECOMMENDATION)
        )

    if state.classification_id is None:
        return ControlledSupportRouteDecision(
            route=(ControlledSupportGraphRoute.ENSURE_CLASSIFICATION)
        )

    if not budget.can_decide:
        return ControlledSupportRouteDecision(
            route=ControlledSupportGraphRoute.FAIL_WORKFLOW,
            error_code=(DecisionTurnBudgetExhaustedError.error_code),
        )

    return ControlledSupportRouteDecision(route=ControlledSupportGraphRoute.DECIDE_NEXT_ACTION)


def reserve_next_graph_step(
    state: ControlledSupportGraphStateSnapshot,
    *,
    limits: ControlledSupportRuntimeLimits = (CONTROLLED_SUPPORT_RUNTIME_LIMITS),
) -> int:
    """Return the next graph-step count or fail closed."""

    budget = calculate_remaining_workflow_budget(
        state,
        limits=limits,
    )

    if not budget.can_advance_graph:
        raise GraphStepBudgetExhaustedError()

    return state.graph_step_count + 1


def reserve_next_decision_turn(
    state: ControlledSupportGraphStateSnapshot,
    *,
    limits: ControlledSupportRuntimeLimits = (CONTROLLED_SUPPORT_RUNTIME_LIMITS),
) -> int:
    """Reserve one decision turn after prior tool work is settled."""

    budget = calculate_remaining_workflow_budget(
        state,
        limits=limits,
    )

    if state.classification_id is None:
        raise GraphRoutingStateError("A decision turn requires a persisted classification.")

    if state.analysis_completion is not None:
        raise GraphRoutingStateError(
            "A decision turn cannot begin after terminal analysis completion."
        )

    if state.decision_turn_count != state.tool_call_count:
        raise GraphRoutingStateError(
            "A new decision turn requires the prior tool decision to be settled."
        )

    if not budget.can_decide:
        raise DecisionTurnBudgetExhaustedError()

    return state.decision_turn_count + 1


def reserve_next_tool_call(
    state: ControlledSupportGraphStateSnapshot,
    *,
    limits: ControlledSupportRuntimeLimits = (CONTROLLED_SUPPORT_RUNTIME_LIMITS),
) -> int:
    """Reserve the next tool-call sequence after a tool decision."""

    budget = calculate_remaining_workflow_budget(
        state,
        limits=limits,
    )

    if state.classification_id is None:
        raise GraphRoutingStateError("A tool call requires a persisted classification.")

    if state.analysis_completion is not None:
        raise GraphRoutingStateError("A tool call cannot begin after terminal analysis completion.")

    if state.decision_turn_count != state.tool_call_count + 1:
        raise GraphRoutingStateError("A tool call requires exactly one unresolved decision turn.")

    if budget.tool_calls == 0:
        raise ToolCallBudgetExhaustedError()

    return state.tool_call_count + 1
