"""Unit tests for the terminal support-analysis control."""

import json

import pytest
from pydantic import ValidationError

from supportops.agent_graph.domain.completion import (
    COMPLETE_SUPPORT_ANALYSIS_CONTROL,
    COMPLETE_SUPPORT_ANALYSIS_CONTROL_VERSION,
    CompleteSupportAnalysisInput,
    get_complete_support_analysis_control,
)
from supportops.ai.gateway.tool_decisions import (
    COMPLETE_SUPPORT_ANALYSIS_CONTROL_NAME,
)
from supportops.modules.support_recommendations.domain.models import (
    SupportRecommendationAction,
)


def test_terminal_control_has_stable_identity() -> None:
    assert COMPLETE_SUPPORT_ANALYSIS_CONTROL.name == (COMPLETE_SUPPORT_ANALYSIS_CONTROL_NAME)
    assert COMPLETE_SUPPORT_ANALYSIS_CONTROL.version == (COMPLETE_SUPPORT_ANALYSIS_CONTROL_VERSION)
    assert COMPLETE_SUPPORT_ANALYSIS_CONTROL.input_schema is CompleteSupportAnalysisInput


def test_terminal_control_projects_strict_provider_schema() -> None:
    provider_definition = COMPLETE_SUPPORT_ANALYSIS_CONTROL.to_provider_definition()
    schema = provider_definition.input_schema
    properties = schema["properties"]
    required = schema["required"]

    assert provider_definition.name == ("complete_support_analysis")
    assert provider_definition.version == 1
    assert provider_definition.strict is True

    assert schema["type"] == "object"
    assert schema["additionalProperties"] is False
    assert isinstance(properties, dict)
    assert isinstance(required, list)
    assert set(properties) == {
        "recommended_action",
        "evidence_sufficient",
        "requires_human_review",
        "decision_summary",
    }
    assert set(required) == {
        "recommended_action",
        "evidence_sufficient",
        "requires_human_review",
        "decision_summary",
    }


def test_accepts_direct_response_with_sufficient_evidence() -> None:
    completion = CompleteSupportAnalysisInput(
        recommended_action=(SupportRecommendationAction.RESPOND),
        evidence_sufficient=True,
        requires_human_review=False,
        decision_summary=(
            "  The active runbook contains sufficient guidance for a direct response.  "
        ),
    )

    assert completion.recommended_action is (SupportRecommendationAction.RESPOND)
    assert completion.evidence_sufficient is True
    assert completion.requires_human_review is False
    assert completion.decision_summary == (
        "The active runbook contains sufficient guidance for a direct response."
    )


def test_accepts_request_for_more_information() -> None:
    completion = CompleteSupportAnalysisInput(
        recommended_action=(SupportRecommendationAction.REQUEST_MORE_INFORMATION),
        evidence_sufficient=False,
        requires_human_review=False,
        decision_summary=(
            "The ticket does not identify the affected service or the observed failure."
        ),
    )

    assert completion.recommended_action is (SupportRecommendationAction.REQUEST_MORE_INFORMATION)
    assert completion.evidence_sufficient is False


def test_accepts_grounded_escalation_recommendation() -> None:
    completion = CompleteSupportAnalysisInput(
        recommended_action=(SupportRecommendationAction.RECOMMEND_ESCALATION),
        evidence_sufficient=True,
        requires_human_review=True,
        decision_summary=(
            "The evidence indicates a security-sensitive case requiring specialist review."
        ),
    )

    assert completion.evidence_sufficient is True
    assert completion.requires_human_review is True


def test_accepts_escalation_when_evidence_is_incomplete() -> None:
    completion = CompleteSupportAnalysisInput(
        recommended_action=(SupportRecommendationAction.RECOMMEND_ESCALATION),
        evidence_sufficient=False,
        requires_human_review=True,
        decision_summary=(
            "Available evidence is insufficient, but the reported behavior is security-sensitive."
        ),
    )

    assert completion.evidence_sufficient is False
    assert completion.requires_human_review is True


def test_direct_response_requires_sufficient_evidence() -> None:
    with pytest.raises(
        ValidationError,
        match="direct response requires sufficient evidence",
    ):
        CompleteSupportAnalysisInput(
            recommended_action=(SupportRecommendationAction.RESPOND),
            evidence_sufficient=False,
            requires_human_review=False,
            decision_summary=("Available evidence is incomplete."),
        )


def test_request_more_information_requires_insufficient_evidence() -> None:
    with pytest.raises(
        ValidationError,
        match=("request for more information requires insufficient evidence"),
    ):
        CompleteSupportAnalysisInput(
            recommended_action=(SupportRecommendationAction.REQUEST_MORE_INFORMATION),
            evidence_sufficient=True,
            requires_human_review=False,
            decision_summary=("Evidence is already sufficient."),
        )


def test_escalation_requires_human_review() -> None:
    with pytest.raises(
        ValidationError,
        match="require human review",
    ):
        CompleteSupportAnalysisInput(
            recommended_action=(SupportRecommendationAction.RECOMMEND_ESCALATION),
            evidence_sufficient=True,
            requires_human_review=False,
            decision_summary=("The case requires specialist handling."),
        )


@pytest.mark.parametrize(
    ("field_name", "field_value"),
    [
        (
            "evidence_sufficient",
            "true",
        ),
        (
            "requires_human_review",
            1,
        ),
    ],
)
def test_requires_strict_boolean_values(
    field_name: str,
    field_value: object,
) -> None:
    payload: dict[str, object] = {
        "recommended_action": "respond",
        "evidence_sufficient": True,
        "requires_human_review": False,
        "decision_summary": ("The evidence supports a direct response."),
    }
    payload[field_name] = field_value

    with pytest.raises(ValidationError):
        CompleteSupportAnalysisInput.model_validate(payload)


def test_forbids_unknown_fields() -> None:
    with pytest.raises(
        ValidationError,
        match="extra_forbidden",
    ):
        CompleteSupportAnalysisInput.model_validate(
            {
                "recommended_action": "respond",
                "evidence_sufficient": True,
                "requires_human_review": False,
                "decision_summary": ("The evidence supports a direct response."),
                "hidden_reasoning": ("This field must never be accepted."),
            }
        )


@pytest.mark.parametrize(
    "decision_summary",
    [
        "",
        "x" * 501,
    ],
)
def test_enforces_bounded_decision_summary(
    decision_summary: str,
) -> None:
    with pytest.raises(ValidationError):
        CompleteSupportAnalysisInput(
            recommended_action=(SupportRecommendationAction.RESPOND),
            evidence_sufficient=True,
            requires_human_review=False,
            decision_summary=decision_summary,
        )


def test_projects_json_compatible_checkpoint_state() -> None:
    completion = CompleteSupportAnalysisInput(
        recommended_action=(SupportRecommendationAction.RESPOND),
        evidence_sufficient=True,
        requires_human_review=False,
        decision_summary=("The evidence supports a direct response."),
    )

    state = completion.to_state()

    assert state == {
        "recommended_action": "respond",
        "evidence_sufficient": True,
        "requires_human_review": False,
        "decision_summary": ("The evidence supports a direct response."),
    }

    assert (
        json.loads(
            json.dumps(
                state,
                sort_keys=True,
            )
        )
        == state
    )


def test_schema_is_immutable() -> None:
    completion = CompleteSupportAnalysisInput(
        recommended_action=(SupportRecommendationAction.RESPOND),
        evidence_sufficient=True,
        requires_human_review=False,
        decision_summary=("The evidence supports a direct response."),
    )

    with pytest.raises(ValidationError):
        completion.evidence_sufficient = False  # type: ignore[misc]


def test_control_lookup_requires_explicit_version() -> None:
    assert get_complete_support_analysis_control(version=1) is COMPLETE_SUPPORT_ANALYSIS_CONTROL

    with pytest.raises(
        ValueError,
        match="version is not registered",
    ):
        get_complete_support_analysis_control(version=2)
