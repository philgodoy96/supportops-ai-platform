from decimal import Decimal
from pathlib import Path

import pytest

from supportops.evaluation.grounded_recommendations.dataset import (
    load_grounded_recommendation_dataset,
)
from supportops.evaluation.grounded_recommendations.evaluator import (
    GroundedRecommendationEvaluationError,
    evaluate_grounded_recommendation_predictions,
)
from supportops.evaluation.grounded_recommendations.predictions import (
    load_grounded_recommendation_predictions,
)

PROJECT_ROOT = Path(__file__).resolve().parents[4]

DATASET_PATH = (
    PROJECT_ROOT
    / "evals"
    / "grounded-recommendations"
    / "datasets"
    / "grounded-recommendations-eval-v1.jsonl"
)

PREDICTIONS_PATH = (
    PROJECT_ROOT
    / "evals"
    / "grounded-recommendations"
    / "predictions"
    / "grounded-recommendations-eval-v1.static.jsonl"
)


def test_static_predictions_produce_expected_metrics() -> None:
    dataset = load_grounded_recommendation_dataset(DATASET_PATH)
    predictions, prediction_hash = load_grounded_recommendation_predictions(PREDICTIONS_PATH)

    report = evaluate_grounded_recommendation_predictions(
        dataset=dataset,
        predictions=predictions,
        prediction_hash=prediction_hash,
    )

    assert report.recommended_action_accuracy.rate == Decimal("1.000000")
    assert report.human_review_accuracy.rate == Decimal("1.000000")
    assert report.evidence_sufficiency_accuracy.rate == Decimal("1.000000")
    assert report.citation_identity_accuracy.rate == Decimal("1.000000")
    assert report.workspace_isolation_rate.rate == Decimal("1.000000")
    assert report.grounded_abstention_accuracy.rate == Decimal("1.000000")
    assert report.prediction_coverage.rate == Decimal("1.000000")


def test_report_hash_is_deterministic() -> None:
    dataset = load_grounded_recommendation_dataset(DATASET_PATH)
    predictions, prediction_hash = load_grounded_recommendation_predictions(PREDICTIONS_PATH)

    first = evaluate_grounded_recommendation_predictions(
        dataset=dataset,
        predictions=predictions,
        prediction_hash=prediction_hash,
    )
    second = evaluate_grounded_recommendation_predictions(
        dataset=dataset,
        predictions=predictions,
        prediction_hash=prediction_hash,
    )

    assert first.report_content_hash == second.report_content_hash


def test_missing_prediction_reduces_coverage() -> None:
    dataset = load_grounded_recommendation_dataset(DATASET_PATH)
    predictions, prediction_hash = load_grounded_recommendation_predictions(PREDICTIONS_PATH)

    report = evaluate_grounded_recommendation_predictions(
        dataset=dataset,
        predictions=predictions[:-1],
        prediction_hash=prediction_hash,
    )

    missing = report.case_results[-1]

    assert missing.prediction_present is False
    assert missing.error_code == "prediction_missing"
    assert report.prediction_coverage.rate is not None
    assert report.prediction_coverage.rate < Decimal("1.000000")


def test_invalid_citation_reduces_citation_accuracy() -> None:
    dataset = load_grounded_recommendation_dataset(DATASET_PATH)
    predictions, prediction_hash = load_grounded_recommendation_predictions(PREDICTIONS_PATH)

    target_index = 5
    target = predictions[target_index]
    assert target.payload is not None

    invalid_payload = target.payload.model_copy(
        update={"citation_chunk_ids": ("ffffffff-ffff-4fff-8fff-ffffffffffff",)}
    )
    invalid_prediction = target.model_copy(update={"payload": invalid_payload})

    modified = list(predictions)
    modified[target_index] = invalid_prediction

    report = evaluate_grounded_recommendation_predictions(
        dataset=dataset,
        predictions=tuple(modified),
        prediction_hash=prediction_hash,
    )

    assert report.citation_identity_accuracy.rate is not None
    assert report.citation_identity_accuracy.rate < Decimal("1.000000")
    assert report.case_results[target_index].citation_identity_correct is False


def test_cross_workspace_evidence_reduces_isolation() -> None:
    dataset = load_grounded_recommendation_dataset(DATASET_PATH)
    predictions, prediction_hash = load_grounded_recommendation_predictions(PREDICTIONS_PATH)

    target_index = 8
    target = predictions[target_index]
    assert target.payload is not None

    unsafe_payload = target.payload.model_copy(update={"foreign_workspace_evidence_count": 1})
    unsafe_prediction = target.model_copy(update={"payload": unsafe_payload})

    modified = list(predictions)
    modified[target_index] = unsafe_prediction

    report = evaluate_grounded_recommendation_predictions(
        dataset=dataset,
        predictions=tuple(modified),
        prediction_hash=prediction_hash,
    )

    assert report.workspace_isolation_rate.rate is not None
    assert report.workspace_isolation_rate.rate < Decimal("1.000000")


def test_unknown_prediction_case_is_rejected() -> None:
    dataset = load_grounded_recommendation_dataset(DATASET_PATH)
    predictions, prediction_hash = load_grounded_recommendation_predictions(PREDICTIONS_PATH)

    unknown = predictions[0].model_copy(update={"case_id": ("grounded-recommendation-unknown-999")})

    with pytest.raises(
        GroundedRecommendationEvaluationError,
        match="unknown prediction case IDs",
    ):
        evaluate_grounded_recommendation_predictions(
            dataset=dataset,
            predictions=(*predictions, unknown),
            prediction_hash=prediction_hash,
        )
