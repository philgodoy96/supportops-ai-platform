from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path

import pytest
from pydantic import ValidationError

from supportops.evaluation.grounded_recommendations.ragas_adapter import (
    RagasMetricName,
)
from supportops.evaluation.grounded_recommendations.ragas_scores import (
    GroundedRecommendationRagasCaseScore,
    GroundedRecommendationRagasMetricScore,
    GroundedRecommendationRagasScoreError,
    RagasMetricStatus,
    load_grounded_recommendation_ragas_scores,
)

PROJECT_ROOT = Path(__file__).resolve().parents[4]

SCORES_PATH = (
    PROJECT_ROOT
    / "evals"
    / "grounded-recommendations"
    / "ragas-scores"
    / "grounded-recommendations-eval-v1.static.jsonl"
)

SCORE_ARTIFACT_HASH = "272ed76a351bf66beb2d951d1dc02c7bbb3beb730d09ac12f107b5ec90b091e6"


def test_committed_ragas_score_artifact_loads_and_is_immutable() -> None:
    artifact = load_grounded_recommendation_ragas_scores(SCORES_PATH)

    assert len(artifact.case_scores) == 14
    assert len({case_score.case_id for case_score in artifact.case_scores}) == 14
    assert artifact.content_hash == SCORE_ARTIFACT_HASH


def test_committed_artifact_uses_only_known_metrics() -> None:
    artifact = load_grounded_recommendation_ragas_scores(SCORES_PATH)

    observed = {
        metric_score.metric
        for case_score in artifact.case_scores
        for metric_score in case_score.metrics
    }

    assert observed == set(RagasMetricName)


def test_each_case_contains_four_unique_metric_results() -> None:
    artifact = load_grounded_recommendation_ragas_scores(SCORES_PATH)

    for case_score in artifact.case_scores:
        assert len(case_score.metrics) == 4
        assert len({metric.metric for metric in case_score.metrics}) == 4


def test_duplicate_case_is_rejected(tmp_path: Path) -> None:
    first_line = SCORES_PATH.read_text(encoding="utf-8").splitlines()[0]

    duplicate_path = tmp_path / "duplicate-case.jsonl"
    duplicate_path.write_text(
        f"{first_line}\n{first_line}\n",
        encoding="utf-8",
    )

    with pytest.raises(
        GroundedRecommendationRagasScoreError,
        match="duplicate RAGAS score case_id",
    ):
        load_grounded_recommendation_ragas_scores(duplicate_path)


def test_duplicate_metric_is_rejected() -> None:
    metric = GroundedRecommendationRagasMetricScore(
        metric=RagasMetricName.FAITHFULNESS,
        status=RagasMetricStatus.SUCCEEDED,
        score=Decimal("0.900000"),
    )

    with pytest.raises(
        ValidationError,
        match="duplicate RAGAS metrics",
    ):
        GroundedRecommendationRagasCaseScore(
            case_id="grounded-recommendation-test-001",
            metrics=(metric, metric),
        )


def test_succeeded_metric_requires_score() -> None:
    with pytest.raises(
        ValidationError,
        match="succeeded metric must include a score",
    ):
        GroundedRecommendationRagasMetricScore(
            metric=RagasMetricName.FAITHFULNESS,
            status=RagasMetricStatus.SUCCEEDED,
        )


def test_failed_metric_requires_error_code() -> None:
    with pytest.raises(
        ValidationError,
        match="failed metric must include an error code",
    ):
        GroundedRecommendationRagasMetricScore(
            metric=RagasMetricName.ANSWER_RELEVANCY,
            status=RagasMetricStatus.FAILED,
        )


def test_not_applicable_metric_requires_reason() -> None:
    with pytest.raises(
        ValidationError,
        match="not-applicable metric must include a reason",
    ):
        GroundedRecommendationRagasMetricScore(
            metric=RagasMetricName.CONTEXT_PRECISION,
            status=RagasMetricStatus.NOT_APPLICABLE,
        )


@pytest.mark.parametrize(
    "score",
    [Decimal("-0.000001"), Decimal("1.000001")],
)
def test_score_outside_normalized_range_is_rejected(
    score: Decimal,
) -> None:
    with pytest.raises(ValidationError):
        GroundedRecommendationRagasMetricScore(
            metric=RagasMetricName.CONTEXT_RECALL,
            status=RagasMetricStatus.SUCCEEDED,
            score=score,
        )


def test_empty_artifact_is_rejected(tmp_path: Path) -> None:
    empty_path = tmp_path / "empty.jsonl"
    empty_path.write_text("", encoding="utf-8")

    with pytest.raises(
        GroundedRecommendationRagasScoreError,
        match="must not be empty",
    ):
        load_grounded_recommendation_ragas_scores(empty_path)


def test_invalid_json_reports_line_number(
    tmp_path: Path,
) -> None:
    invalid_path = tmp_path / "invalid.jsonl"
    invalid_path.write_text(
        json.dumps({"case_id": "valid-but-incomplete"}) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(
        GroundedRecommendationRagasScoreError,
        match="invalid RAGAS score line 1",
    ):
        load_grounded_recommendation_ragas_scores(invalid_path)
