"""Strict terminal-control schema for human-approved support."""

from typing import Annotated, Literal

from pydantic import StrictBool, StringConstraints

from supportops.agent_tools.domain.contracts import StrictToolSchema
from supportops.ai.gateway.tool_decisions import (
    LLMTerminalControlDefinition,
)

HUMAN_APPROVED_SUPPORT_DECISION_OUTPUT_SCHEMA_ID = "human-approved-support-decision-v1"
COMPLETE_HUMAN_APPROVED_SUPPORT_ANALYSIS_CONTROL_NAME = "complete_human_approved_support_analysis"
COMPLETE_HUMAN_APPROVED_SUPPORT_ANALYSIS_CONTROL_VERSION = 1

HumanApprovedRecommendedAction = Literal[
    "respond",
    "request_more_information",
    "recommend_escalation",
]
DecisionSummary = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        min_length=1,
        max_length=500,
    ),
]


class CompleteHumanApprovedSupportAnalysisInput(StrictToolSchema):
    """Terminal graph decision that performs no side effect."""

    schema_version: Literal["human-approved-support-decision-v1"]
    recommended_action: HumanApprovedRecommendedAction
    evidence_sufficient: StrictBool
    requires_human_review: StrictBool
    decision_summary: DecisionSummary


COMPLETE_HUMAN_APPROVED_SUPPORT_ANALYSIS_CONTROL = LLMTerminalControlDefinition(
    name=(COMPLETE_HUMAN_APPROVED_SUPPORT_ANALYSIS_CONTROL_NAME),
    version=(COMPLETE_HUMAN_APPROVED_SUPPORT_ANALYSIS_CONTROL_VERSION),
    description=(
        "Complete the current support analysis without executing "
        "a tool or changing application state."
    ),
    input_schema=CompleteHumanApprovedSupportAnalysisInput,
)
