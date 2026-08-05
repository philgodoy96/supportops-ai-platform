from __future__ import annotations

import socket
from decimal import Decimal
from pathlib import Path

import pytest

from supportops.evaluation.grounded_recommendations.dataset import (
    load_grounded_recommendation_dataset,
)
from supportops.evaluation.grounded_recommendations.ragas_adapter import (
    RagasMetricName,
)
from supportops.evaluation.grounded_recommendations.ragas_report import (
    build_grounded_recommendation_ragas_report,
)
from supportops.evaluation.grounded_recommendations.ragas_scores import (
    GroundedRecommendationRagasCaseScore,
    GroundedRecommendationRagasMetricScore,
    GroundedRecommendationRagasScoreArtifact,
    RagasMetricStatus,
    load_grounded_recommendation_ragas_scores,
)

PROJECT_ROOT = Path(__file__).resolve().parents[4]

DATASET_PATH = (
    PROJECT_ROOT
    / "evals"
    / "grounded-recommendations"
    / "datasets"
    / "grounded-recommendations-eval-v1.jsonl"
)

SCORES_PATH = (
    PROJECT_ROOT
    / "evals"
    / "grounded-recommendations"
    / "ragas-scores"
    / "grounded-recommendations-eval-v1.static.jsonl"
)


def test_offline_report_requires_no_network(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_connection(*args: object, **kwargs: object) -> object:
        raise AssertionError("network access is forbidden")

    monkeypatch.setattr(socket, "create_connection", fail_connection)

    dataset = load_grounded_recommendation_dataset(DATASET_PATH)
    scores = load_grounded_recommendation_ragas_scores(SCORES_PATH)

    report = build_grounded_recommendation_ragas_report(
        dataset=dataset,
        scores=scores,
    )

    assert report.case_count == 14
    assert report.scored_case_count == 14
    assert report.missing_case_count == 0
    assert report.unknown_case_count == 0


def test_metric_order_is_stable() -> None:
    dataset = load_grounded_recommendation_dataset(DATASET_PATH)
    scores = load_grounded_recommendation_ragas_scores(SCORES_PATH)

    report = build_grounded_recommendation_ragas_report(
        dataset=dataset,
        scores=scores,
    )

    assert tuple(aggregate.metric for aggregate in report.metric_aggregates) == tuple(
        RagasMetricName
    )


def test_metric_averages_and_not_applicable_counts() -> None:
    dataset = load_grounded_recommendation_dataset(DATASET_PATH)
    scores = load_grounded_recommendation_ragas_scores(SCORES_PATH)

    report = build_grounded_recommendation_ragas_report(
        dataset=dataset,
        scores=scores,
    )
    aggregates = {aggregate.metric: aggregate for aggregate in report.metric_aggregates}

    assert aggregates[RagasMetricName.FAITHFULNESS].average_score == Decimal("0.942857")
    assert aggregates[RagasMetricName.ANSWER_RELEVANCY].average_score == Decimal("0.920000")
    assert aggregates[RagasMetricName.CONTEXT_PRECISION].not_applicable_count == 2
    assert aggregates[RagasMetricName.CONTEXT_RECALL].not_applicable_count == 2


def test_failed_metric_is_counted() -> None:
    dataset = load_grounded_recommendation_dataset(DATASET_PATH)
    scores = load_grounded_recommendation_ragas_scores(SCORES_PATH)

    first_case = scores.case_scores[0]
    replacement_metrics = list(first_case.metrics)
    replacement_metrics[0] = GroundedRecommendationRagasMetricScore(
        metric=RagasMetricName.FAITHFULNESS,
        status=RagasMetricStatus.FAILED,
        error_code="evaluator_timeout",
    )

    modified_case = first_case.model_copy(update={"metrics": tuple(replacement_metrics)})
    modified_scores = scores.model_copy(
        update={
            "case_scores": (
                modified_case,
                *scores.case_scores[1:],
            )
        }
    )

    report = build_grounded_recommendation_ragas_report(
        dataset=dataset,
        scores=modified_scores,
    )
    faithfulness = next(
        aggregate
        for aggregate in report.metric_aggregates
        if aggregate.metric is RagasMetricName.FAITHFULNESS
    )

    assert faithfulness.failed_count == 1
    assert faithfulness.succeeded_count == 13


def test_missing_and_unknown_case_counts_are_visible() -> None:
    dataset = load_grounded_recommendation_dataset(DATASET_PATH)
    scores = load_grounded_recommendation_ragas_scores(SCORES_PATH)

    unknown_case = GroundedRecommendationRagasCaseScore(
        case_id="grounded-recommendation-unknown-999",
        metrics=scores.case_scores[0].metrics,
    )

    modified_scores = GroundedRecommendationRagasScoreArtifact(
        case_scores=(
            *scores.case_scores[:-1],
            unknown_case,
        ),
        content_hash=scores.content_hash,
    )

    report = build_grounded_recommendation_ragas_report(
        dataset=dataset,
        scores=modified_scores,
    )

    assert report.scored_case_count == 13
    assert report.missing_case_count == 1
    assert report.unknown_case_count == 1


def test_report_hash_is_deterministic() -> None:
    dataset = load_grounded_recommendation_dataset(DATASET_PATH)
    scores = load_grounded_recommendation_ragas_scores(SCORES_PATH)

    first = build_grounded_recommendation_ragas_report(
        dataset=dataset,
        scores=scores,
    )
    second = build_grounded_recommendation_ragas_report(
        dataset=dataset,
        scores=scores,
    )

    assert first.report_content_hash == second.report_content_hash
    assert first.model_dump(mode="json") == second.model_dump(mode="json")


def test_report_retains_input_hashes() -> None:
    dataset = load_grounded_recommendation_dataset(DATASET_PATH)
    scores = load_grounded_recommendation_ragas_scores(SCORES_PATH)

    report = build_grounded_recommendation_ragas_report(
        dataset=dataset,
        scores=scores,
    )

    assert report.dataset_hash == dataset.content_hash
    assert report.score_artifact_hash == scores.content_hash
