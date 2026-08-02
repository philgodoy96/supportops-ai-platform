"""Structured output schema for grounded support recommendations."""

from typing import Annotated, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    StrictBool,
    StringConstraints,
    model_validator,
)

from supportops.modules.support_recommendations.domain.models import (
    SUPPORT_RECOMMENDATION_DECISION_SUMMARY_MAX_LENGTH,
    SUPPORT_RECOMMENDATION_RESPONSE_MAX_LENGTH,
    SUPPORT_RECOMMENDATION_SCHEMA_VERSION,
    SupportRecommendationAction,
    SupportRecommendationSchemaVersion,
)

SUPPORT_RECOMMENDATION_OUTPUT_SCHEMA_ID: SupportRecommendationSchemaVersion = (
    SUPPORT_RECOMMENDATION_SCHEMA_VERSION
)

RecommendationResponseText = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        min_length=1,
        max_length=SUPPORT_RECOMMENDATION_RESPONSE_MAX_LENGTH,
    ),
]
RecommendationDecisionSummary = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        min_length=1,
        max_length=SUPPORT_RECOMMENDATION_DECISION_SUMMARY_MAX_LENGTH,
    ),
]


class SupportRecommendationResult(BaseModel):
    """Validated structured recommendation produced by the workflow."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    recommended_action: SupportRecommendationAction
    response_text: RecommendationResponseText
    requires_human_review: StrictBool
    decision_summary: RecommendationDecisionSummary
    schema_version: SupportRecommendationSchemaVersion

    @model_validator(mode="after")
    def require_review_for_escalation(self) -> Self:
        """Require human review whenever escalation is recommended."""

        if (
            self.recommended_action is SupportRecommendationAction.RECOMMEND_ESCALATION
            and not self.requires_human_review
        ):
            raise ValueError("Escalation recommendations require human review.")

        return self
