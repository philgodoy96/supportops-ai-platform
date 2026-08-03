"""Grounded recommendation completion for human-approved workflows."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol
from uuid import UUID

from pydantic import BaseModel, JsonValue

from supportops.agent_graph.domain.human_approved_state import (
    HumanApprovalCheckpointStatus,
    HumanApprovedDecisionKind,
    HumanApprovedSupportGraphStateSnapshot,
)
from supportops.modules.agent_runs.application.execution import (
    AgentRunExecutionContext,
)
from supportops.modules.support_recommendations.domain.models import (
    SupportRecommendation,
)


@dataclass(frozen=True, slots=True)
class HumanApprovedRecommendationOutcome:
    """Persisted recommendation provenance."""

    invocation_id: UUID
    recommendation: SupportRecommendation


class HumanApprovedRecommendationExecutor(Protocol):
    """Draft and persist one approval-aware recommendation."""

    async def execute(
        self,
        *,
        context: AgentRunExecutionContext,
        state: HumanApprovedSupportGraphStateSnapshot,
        workflow: Mapping[str, JsonValue],
    ) -> HumanApprovedRecommendationOutcome:
        """Return one durable recommendation."""

        ...


def build_human_approved_recommendation_workflow(
    state: HumanApprovedSupportGraphStateSnapshot,
) -> dict[str, JsonValue]:
    """Build the bounded recommendation context."""

    if state.approval_status is None:
        if state.decision_kind is not HumanApprovedDecisionKind.TERMINAL:
            raise ValueError(
                "Recommendation requires an approval outcome.",
            )
        return {
            "approval": {
                "status": None,
                "request_reason": None,
            },
            "decision": {
                "summary": state.decision_summary,
                "recommended_action": state.analysis_recommended_action,
            },
            "sensitive_action": {
                "tool_name": None,
                "tool_version": None,
                "execution_output": None,
            },
        }

    approval_status = state.approval_status
    status_value = (
        approval_status.value
        if isinstance(approval_status, HumanApprovalCheckpointStatus)
        else str(approval_status)
    )
    execution_output = _project_execution_output(
        state.sensitive_execution_output,
    )

    workflow: dict[str, JsonValue] = {
        "approval": {
            "status": status_value,
            "request_reason": state.approval_request_reason,
        },
        "decision": {
            "summary": state.decision_summary,
            "recommended_action": state.analysis_recommended_action,
        },
        "sensitive_action": {
            "tool_name": state.proposed_tool_name,
            "tool_version": state.proposed_tool_version,
            "execution_output": execution_output,
        },
    }

    if (
        status_value
        in {
            HumanApprovalCheckpointStatus.REJECTED.value,
            HumanApprovalCheckpointStatus.EXPIRED.value,
        }
        and execution_output is not None
    ):
        raise ValueError(
            "Rejected or expired approvals cannot expose execution output.",
        )

    return workflow


def _project_execution_output(
    value: object,
) -> dict[str, JsonValue] | None:
    if value is None:
        return None
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, Mapping):
        return dict(value)
    raise TypeError(
        "sensitive_execution_output must be a mapping or model.",
    )
