"""Deterministic routing policy for the human-approved graph."""

from dataclasses import dataclass
from enum import StrEnum

from supportops.agent_graph.domain.human_approved_state import (
    HumanApprovalCheckpointStatus,
    HumanApprovedDecisionKind,
    HumanApprovedSupportGraphStateSnapshot,
)


class HumanApprovedSupportGraphRoute(StrEnum):
    """Application-owned graph routes."""

    LOAD_RUN_CONTEXT = "load_run_context"
    ENSURE_CLASSIFICATION = "ensure_classification"
    DECIDE_NEXT_ACTION = "decide_next_action"
    EXECUTE_READ_ONLY_TOOL = "execute_read_only_tool"
    PREPARE_SENSITIVE_ACTION = "prepare_sensitive_action"
    AWAIT_HUMAN_APPROVAL = "await_human_approval"
    HANDLE_APPROVAL_DECISION = "handle_approval_decision"
    EXECUTE_SENSITIVE_TOOL = "execute_sensitive_tool"
    DRAFT_GROUNDED_RECOMMENDATION = "draft_grounded_recommendation"
    VALIDATE_RECOMMENDATION = "validate_recommendation"
    PERSIST_RECOMMENDATION = "persist_recommendation"
    COMPLETE_WORKFLOW = "complete_workflow"
    FAIL_WORKFLOW = "fail_workflow"


@dataclass(frozen=True, slots=True)
class HumanApprovedRouteDecision:
    """One deterministic route selected from validated state."""

    route: HumanApprovedSupportGraphRoute
    error_code: str | None = None

    def __post_init__(self) -> None:
        if self.route is HumanApprovedSupportGraphRoute.FAIL_WORKFLOW:
            if self.error_code is None:
                raise ValueError(
                    "Failure routes require an error_code.",
                )
            return
        if self.error_code is not None:
            raise ValueError(
                "Non-failure routes cannot define an error_code.",
            )


def select_human_approved_support_route(
    state: HumanApprovedSupportGraphStateSnapshot,
) -> HumanApprovedRouteDecision:
    """Select the next route without side effects."""

    if state.current_error_code is not None:
        return HumanApprovedRouteDecision(
            route=HumanApprovedSupportGraphRoute.FAIL_WORKFLOW,
            error_code=state.current_error_code,
        )

    if state.recommendation_id is not None:
        return HumanApprovedRouteDecision(
            route=HumanApprovedSupportGraphRoute.COMPLETE_WORKFLOW,
        )

    if not state.run_context_loaded:
        return HumanApprovedRouteDecision(
            route=HumanApprovedSupportGraphRoute.LOAD_RUN_CONTEXT,
        )

    if state.approval_request_id is not None:
        if state.approval_status is HumanApprovalCheckpointStatus.PENDING:
            return HumanApprovedRouteDecision(
                route=(HumanApprovedSupportGraphRoute.AWAIT_HUMAN_APPROVAL),
            )
        return HumanApprovedRouteDecision(
            route=(HumanApprovedSupportGraphRoute.HANDLE_APPROVAL_DECISION),
        )

    if state.decision_kind is HumanApprovedDecisionKind.SENSITIVE_TOOL:
        return HumanApprovedRouteDecision(
            route=(HumanApprovedSupportGraphRoute.PREPARE_SENSITIVE_ACTION),
        )

    if state.decision_kind is HumanApprovedDecisionKind.READ_ONLY_TOOL:
        return HumanApprovedRouteDecision(
            route=(HumanApprovedSupportGraphRoute.EXECUTE_READ_ONLY_TOOL),
        )

    if state.decision_kind is HumanApprovedDecisionKind.TERMINAL:
        return HumanApprovedRouteDecision(
            route=(HumanApprovedSupportGraphRoute.DRAFT_GROUNDED_RECOMMENDATION),
        )

    if state.classification_id is None:
        return HumanApprovedRouteDecision(
            route=(HumanApprovedSupportGraphRoute.ENSURE_CLASSIFICATION),
        )

    return HumanApprovedRouteDecision(
        route=HumanApprovedSupportGraphRoute.DECIDE_NEXT_ACTION,
    )
