from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from supportops.evaluation.grounded_recommendations.human_review import (
    GroundedRecommendationHumanReviewRecord,
    GroundedRecommendationHumanReviewRubricError,
    load_grounded_recommendation_human_review_rubric,
)

PROJECT_ROOT = Path(__file__).resolve().parents[4]

RUBRIC_PATH = (
    PROJECT_ROOT / "evals" / "grounded-recommendations" / "rubrics" / "human-review-rubric-v1.json"
)

RUBRIC_HASH = "4a7c06238b2095412293f70213ca7707408c1be34072646892b42b5f7fa28e06"

EXPECTED_DIMENSIONS = {
    "correctness",
    "grounding",
    "actionability",
    "safety",
    "citation_quality",
    "abstention_quality",
    "human_review_appropriateness",
}


def test_committed_human_review_rubric_is_valid() -> None:
    rubric = load_grounded_recommendation_human_review_rubric(RUBRIC_PATH)

    assert rubric.rubric_id == ("grounded-recommendation-human-review")
    assert rubric.rubric_version == 1
    assert rubric.schema_version == ("grounded-recommendation-human-review-rubric-v1")
    assert rubric.content_hash == RUBRIC_HASH


def test_content_hash_ignores_file_formatting(
    tmp_path: Path,
) -> None:
    payload = json.loads(RUBRIC_PATH.read_text(encoding="utf-8"))

    reformatted_path = tmp_path / "reformatted.json"
    reformatted_path.write_text(
        json.dumps(payload, indent=8, sort_keys=True),
        encoding="utf-8",
    )

    rubric = load_grounded_recommendation_human_review_rubric(reformatted_path)

    assert rubric.content_hash == RUBRIC_HASH


def test_rubric_uses_complete_five_point_scale() -> None:
    rubric = load_grounded_recommendation_human_review_rubric(RUBRIC_PATH)

    assert rubric.scale.minimum == 1
    assert rubric.scale.maximum == 5
    assert set(rubric.scale.labels) == {1, 2, 3, 4, 5}


def test_rubric_contains_required_dimensions() -> None:
    rubric = load_grounded_recommendation_human_review_rubric(RUBRIC_PATH)

    observed = {dimension.dimension for dimension in rubric.dimensions}

    assert len(rubric.dimensions) == 7
    assert len(EXPECTED_DIMENSIONS) == 7
    assert observed == EXPECTED_DIMENSIONS


def test_rubric_requires_review_of_all_initial_cases() -> None:
    rubric = load_grounded_recommendation_human_review_rubric(RUBRIC_PATH)

    assert rubric.review_policy.review_all_cases is True
    assert rubric.review_policy.required_note_for_score_at_or_below == 2
    assert rubric.review_policy.blocking_issue_requires_second_reviewer is True
    assert rubric.review_policy.unresolved_blocking_disagreement_outcome == "inconclusive"


def test_duplicate_dimensions_are_rejected(
    tmp_path: Path,
) -> None:
    payload = json.loads(RUBRIC_PATH.read_text(encoding="utf-8"))
    payload["dimensions"].append(payload["dimensions"][0])

    invalid_path = tmp_path / "duplicate-dimension.json"
    invalid_path.write_text(
        json.dumps(payload),
        encoding="utf-8",
    )

    with pytest.raises(
        GroundedRecommendationHumanReviewRubricError,
        match="review dimensions must be unique",
    ):
        load_grounded_recommendation_human_review_rubric(invalid_path)


def test_incomplete_review_record_fields_are_rejected(
    tmp_path: Path,
) -> None:
    payload = json.loads(RUBRIC_PATH.read_text(encoding="utf-8"))
    payload["review_record_fields"] = ["case_id"]

    invalid_path = tmp_path / "incomplete-fields.json"
    invalid_path.write_text(
        json.dumps(payload),
        encoding="utf-8",
    )

    with pytest.raises(
        GroundedRecommendationHumanReviewRubricError,
        match="review record fields are incomplete",
    ):
        load_grounded_recommendation_human_review_rubric(invalid_path)


def test_invalid_note_threshold_is_rejected(
    tmp_path: Path,
) -> None:
    payload = json.loads(RUBRIC_PATH.read_text(encoding="utf-8"))
    payload["review_policy"]["required_note_for_score_at_or_below"] = 9

    invalid_path = tmp_path / "invalid-threshold.json"
    invalid_path.write_text(
        json.dumps(payload),
        encoding="utf-8",
    )

    with pytest.raises(
        GroundedRecommendationHumanReviewRubricError,
        match="required note threshold",
    ):
        load_grounded_recommendation_human_review_rubric(invalid_path)


def test_review_record_requires_dimension_scores() -> None:
    with pytest.raises(
        ValidationError,
        match="dimension scores must not be empty",
    ):
        GroundedRecommendationHumanReviewRecord(
            reviewer_id="reviewer-1",
            reviewed_at="2026-08-05T12:00:00Z",
            case_id="grounded-recommendation-test-001",
            dimension_scores={},
            evidence_references=("chunk-1",),
            notes="No scores were supplied.",
            blocking_issue=False,
        )


def test_invalid_json_is_rejected(
    tmp_path: Path,
) -> None:
    invalid_path = tmp_path / "invalid.json"
    invalid_path.write_text(
        "not-json",
        encoding="utf-8",
    )

    with pytest.raises(
        GroundedRecommendationHumanReviewRubricError,
        match="invalid human review rubric",
    ):
        load_grounded_recommendation_human_review_rubric(invalid_path)
