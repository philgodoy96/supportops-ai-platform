"""Terminal analysis control for the controlled support graph."""

from typing import Annotated, Self, cast

from pydantic import (
    StrictBool,
    StringConstraints,
    model_validator,
)

from supportops.agent_graph.domain.state import (
    CONTROLLED_SUPPORT_GRAPH_STATE_MAX_DECISION_SUMMARY_LENGTH,
    SupportAnalysisCompletionState,
)
from supportops.agent_tools.domain.contracts import (
    StrictToolSchema,
)
from supportops.ai.gateway.tool_decisions import (
    COMPLETE_SUPPORT_ANALYSIS_CONTROL_NAME,
    LLMTerminalControlDefinition,
)
from supportops.modules.support_recommendations.domain.models import (
    SupportRecommendationAction,
)

COMPLETE_SUPPORT_ANALYSIS_CONTROL_VERSION = 1

AnalysisDecisionSummary = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        min_length=1,
        max_length=(CONTROLLED_SUPPORT_GRAPH_STATE_MAX_DECISION_SUMMARY_LENGTH),
    ),
]


class CompleteSupportAnalysisInput(StrictToolSchema):
    """Validated terminal analysis selected by the decision model."""

    recommended_action: SupportRecommendationAction
    evidence_sufficient: StrictBool
    requires_human_review: StrictBool
    decision_summary: AnalysisDecisionSummary

    @model_validator(mode="after")
    def validate_action_consistency(
        self,
    ) -> Self:
        """Reject contradictory terminal-analysis decisions."""

        if (
            self.recommended_action is SupportRecommendationAction.RESPOND
            and not self.evidence_sufficient
        ):
            raise ValueError("A direct response requires sufficient evidence.")

        if (
            self.recommended_action is SupportRecommendationAction.REQUEST_MORE_INFORMATION
            and self.evidence_sufficient
        ):
            raise ValueError("A request for more information requires insufficient evidence.")

        if (
            self.recommended_action is SupportRecommendationAction.RECOMMEND_ESCALATION
            and not self.requires_human_review
        ):
            raise ValueError("Escalation recommendations require human review.")

        return self

    def to_state(
        self,
    ) -> SupportAnalysisCompletionState:
        """Return a JSON-compatible checkpoint state projection."""

        return cast(
            SupportAnalysisCompletionState,
            self.model_dump(mode="json"),
        )


COMPLETE_SUPPORT_ANALYSIS_CONTROL = LLMTerminalControlDefinition(
    name=COMPLETE_SUPPORT_ANALYSIS_CONTROL_NAME,
    version=COMPLETE_SUPPORT_ANALYSIS_CONTROL_VERSION,
    description=(
        "Complete the support analysis after determining "
        "whether the workflow should respond, request more "
        "information, or recommend human escalation."
    ),
    input_schema=CompleteSupportAnalysisInput,
)


def get_complete_support_analysis_control(
    *,
    version: int,
) -> LLMTerminalControlDefinition:
    """Return the explicitly selected terminal-control version."""

    if version != COMPLETE_SUPPORT_ANALYSIS_CONTROL_VERSION:
        raise ValueError("The requested complete_support_analysis version is not registered.")

    return COMPLETE_SUPPORT_ANALYSIS_CONTROL
