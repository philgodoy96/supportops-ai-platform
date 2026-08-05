from decimal import Decimal
from pathlib import Path

import pytest

from supportops.evaluation.controlled_support.dataset import (
    load_controlled_support_dataset,
)
from supportops.evaluation.controlled_support.evaluator import (
    ControlledSupportEvaluationError,
    evaluate_controlled_support_predictions,
)
from supportops.evaluation.controlled_support.predictions import (
    load_controlled_support_predictions,
)

PROJECT_ROOT = Path(__file__).resolve().parents[4]
DATASET_PATH = (
    PROJECT_ROOT / "evals" / "controlled-support" / "datasets" / "controlled-support-eval-v1.jsonl"
)
PREDICTIONS_PATH = (
    PROJECT_ROOT
    / "evals"
    / "controlled-support"
    / "predictions"
    / "controlled-support-eval-v1.static.jsonl"
)

PREDICTION_HASH = "fae9f36e61a44c7b5bc32b00d48eb4408e1f8e0b4f72239758013f64593cbc74"


def test_static_predictions_produce_expected_metrics() -> None:
    dataset = load_controlled_support_dataset(DATASET_PATH)
    predictions, prediction_hash = load_controlled_support_predictions(PREDICTIONS_PATH)

    report = evaluate_controlled_support_predictions(
        dataset=dataset,
        predictions=predictions,
        prediction_hash=prediction_hash,
    )

    assert prediction_hash == PREDICTION_HASH
    assert report.expected_outcome_accuracy.rate == Decimal("1.000000")
    assert report.required_tool_call_rate.rate == Decimal("1.000000")
    assert report.forbidden_tool_call_rate.rate == Decimal("0.000000")
    assert report.tool_sequence_acceptance_rate.rate == Decimal("1.000000")
    assert report.repeated_tool_call_rate.rate == Decimal("0.000000")
    assert report.step_limit_behavior_accuracy.rate == Decimal("1.000000")
    assert report.recommended_action_accuracy.rate == Decimal("1.000000")
    assert report.human_review_recommendation_accuracy.rate == Decimal("1.000000")
    assert report.citation_validity_rate.rate == Decimal("1.000000")
    assert report.grounded_abstention_accuracy.rate == Decimal("1.000000")
    assert report.workspace_isolation_rate.rate == Decimal("1.000000")
    assert report.successful_completion_rate.rate == Decimal("1.000000")


def test_report_hash_is_deterministic() -> None:
    dataset = load_controlled_support_dataset(DATASET_PATH)
    predictions, prediction_hash = load_controlled_support_predictions(PREDICTIONS_PATH)

    first = evaluate_controlled_support_predictions(
        dataset=dataset,
        predictions=predictions,
        prediction_hash=prediction_hash,
    )
    second = evaluate_controlled_support_predictions(
        dataset=dataset,
        predictions=predictions,
        prediction_hash=prediction_hash,
    )

    assert first.report_content_hash == second.report_content_hash


def test_missing_prediction_remains_visible() -> None:
    dataset = load_controlled_support_dataset(DATASET_PATH)
    predictions, prediction_hash = load_controlled_support_predictions(PREDICTIONS_PATH)

    report = evaluate_controlled_support_predictions(
        dataset=dataset,
        predictions=predictions[:-1],
        prediction_hash=prediction_hash,
    )

    missing = report.case_results[-1]

    assert missing.prediction_present is False
    assert missing.error_code == "prediction_missing"
    assert report.expected_outcome_accuracy.rate is not None
    assert report.expected_outcome_accuracy.rate < Decimal("1.000000")


def test_forbidden_tool_execution_is_counted_as_failure() -> None:
    dataset = load_controlled_support_dataset(DATASET_PATH)
    predictions, prediction_hash = load_controlled_support_predictions(PREDICTIONS_PATH)

    target_index = 2
    target = predictions[target_index]
    assert target.payload is not None

    unsafe_payload = target.payload.model_copy(update={"executed_forbidden_tool_count": 1})
    unsafe_prediction = target.model_copy(update={"payload": unsafe_payload})

    modified = list(predictions)
    modified[target_index] = unsafe_prediction

    report = evaluate_controlled_support_predictions(
        dataset=dataset,
        predictions=tuple(modified),
        prediction_hash=prediction_hash,
    )

    assert report.forbidden_tool_call_rate.numerator_count == 1
    assert report.forbidden_tool_call_rate.rate is not None
    assert report.forbidden_tool_call_rate.rate > Decimal("0.000000")
    assert report.case_results[target_index].expected_outcome_matched is False


def test_unknown_prediction_case_id_is_rejected() -> None:
    dataset = load_controlled_support_dataset(DATASET_PATH)
    predictions, prediction_hash = load_controlled_support_predictions(PREDICTIONS_PATH)

    unknown = predictions[0].model_copy(update={"case_id": "unknown-controlled-support-case-999"})

    with pytest.raises(
        ControlledSupportEvaluationError,
        match="unknown prediction case IDs",
    ):
        evaluate_controlled_support_predictions(
            dataset=dataset,
            predictions=(*predictions, unknown),
            prediction_hash=prediction_hash,
        )
