"""Unit tests for support recommendation structured output."""

import pytest
from pydantic import ValidationError

from supportops.modules.support_recommendations.application.schemas import (
    SUPPORT_RECOMMENDATION_OUTPUT_SCHEMA_ID,
    SupportRecommendationResult,
)
from supportops.modules.support_recommendations.domain.models import (
    SupportRecommendationAction,
)


def test_accepts_immutable_grounded_recommendation() -> None:
    result = SupportRecommendationResult(
        recommended_action=(SupportRecommendationAction.RESPOND),
        response_text=("  Follow the documented account recovery steps.  "),
        requires_human_review=False,
        decision_summary=("  Authoritative runbook evidence is available.  "),
        schema_version="support-recommendation-v1",
    )

    assert result.response_text == ("Follow the documented account recovery steps.")
    assert result.decision_summary == ("Authoritative runbook evidence is available.")
    assert result.schema_version == (SUPPORT_RECOMMENDATION_OUTPUT_SCHEMA_ID)

    with pytest.raises(ValidationError):
        result.response_text = "Changed"  # type: ignore[misc]


def test_requires_review_for_escalation() -> None:
    with pytest.raises(
        ValidationError,
        match="require human review",
    ):
        SupportRecommendationResult(
            recommended_action=(SupportRecommendationAction.RECOMMEND_ESCALATION),
            response_text=("A support specialist should review this case."),
            requires_human_review=False,
            decision_summary=("The case requires a human decision."),
            schema_version="support-recommendation-v1",
        )


def test_accepts_escalation_with_human_review() -> None:
    result = SupportRecommendationResult(
        recommended_action=(SupportRecommendationAction.RECOMMEND_ESCALATION),
        response_text=("A support specialist should review this case."),
        requires_human_review=True,
        decision_summary=("The case requires a human decision."),
        schema_version="support-recommendation-v1",
    )

    assert result.requires_human_review is True


def test_forbids_unknown_fields() -> None:
    with pytest.raises(
        ValidationError,
        match="extra_forbidden",
    ):
        SupportRecommendationResult.model_validate(
            {
                "recommended_action": "respond",
                "response_text": ("Use the documented procedure."),
                "requires_human_review": False,
                "decision_summary": ("Evidence is sufficient."),
                "schema_version": ("support-recommendation-v1"),
                "hidden_reasoning": "not permitted",
            }
        )


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("response_text", ""),
        ("response_text", "x" * 4_001),
        ("decision_summary", ""),
        ("decision_summary", "x" * 501),
    ],
)
def test_enforces_bounded_text(
    field_name: str,
    value: str,
) -> None:
    payload: dict[str, object] = {
        "recommended_action": "respond",
        "response_text": "Use the documented procedure.",
        "requires_human_review": False,
        "decision_summary": "Evidence is sufficient.",
        "schema_version": "support-recommendation-v1",
    }
    payload[field_name] = value

    with pytest.raises(ValidationError):
        SupportRecommendationResult.model_validate(payload)


def test_requires_exact_schema_version() -> None:
    with pytest.raises(ValidationError):
        SupportRecommendationResult(
            recommended_action=(SupportRecommendationAction.RESPOND),
            response_text=("Use the documented procedure."),
            requires_human_review=False,
            decision_summary="Evidence is sufficient.",
            schema_version="support-recommendation-v2",
        )
