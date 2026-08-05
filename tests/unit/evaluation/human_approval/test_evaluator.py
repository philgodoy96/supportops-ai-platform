from decimal import Decimal
from pathlib import Path

import pytest

from supportops.evaluation.human_approval.dataset import (
    load_human_approval_dataset,
)
from supportops.evaluation.human_approval.evaluator import (
    HumanApprovalEvaluationError,
    evaluate_human_approval_predictions,
)
from supportops.evaluation.human_approval.predictions import (
    load_human_approval_predictions,
)

PROJECT_ROOT = Path(__file__).resolve().parents[4]
DATASET_PATH = (
    PROJECT_ROOT / "evals" / "human-approval" / "datasets" / "human-approval-eval-v1.jsonl"
)
PREDICTIONS_PATH = (
    PROJECT_ROOT
    / "evals"
    / "human-approval"
    / "predictions"
    / "human-approval-eval-v1.static.jsonl"
)

PREDICTION_HASH = "a7b322cc05847d06492004342d54adb15bed3881fc48a022c2f20281e3f3bd12"


def test_static_predictions_produce_expected_metrics() -> None:
    dataset = load_human_approval_dataset(DATASET_PATH)
    predictions, prediction_hash = load_human_approval_predictions(PREDICTIONS_PATH)

    report = evaluate_human_approval_predictions(
        dataset=dataset,
        predictions=predictions,
        prediction_hash=prediction_hash,
    )

    assert prediction_hash == PREDICTION_HASH
    assert report.expected_outcome_accuracy.rate == Decimal("1.000000")
    assert report.approval_required_accuracy.rate == Decimal("1.000000")
    assert report.unauthorized_sensitive_execution_rate.rate == Decimal("0.000000")
    assert report.approved_execution_success_rate.rate == Decimal("1.000000")
    assert report.rejected_non_execution_rate.rate == Decimal("1.000000")
    assert report.expired_non_execution_rate.rate == Decimal("1.000000")
    assert report.approval_decision_idempotency_rate.rate == Decimal("1.000000")
    assert report.resume_success_rate.rate == Decimal("1.000000")
    assert report.sensitive_action_idempotency_rate.rate == Decimal("1.000000")
    assert report.checkpoint_approval_match_rate.rate == Decimal("1.000000")
    assert report.grant_match_rate.rate == Decimal("1.000000")
    assert report.retry_budget_preservation_rate.rate == Decimal("1.000000")
    assert report.duplicate_escalation_prevention_rate.rate == Decimal("1.000000")
    assert report.successful_finalization_rate.rate == Decimal("1.000000")


def test_report_hash_is_deterministic() -> None:
    dataset = load_human_approval_dataset(DATASET_PATH)
    predictions, prediction_hash = load_human_approval_predictions(PREDICTIONS_PATH)

    first = evaluate_human_approval_predictions(
        dataset=dataset,
        predictions=predictions,
        prediction_hash=prediction_hash,
    )
    second = evaluate_human_approval_predictions(
        dataset=dataset,
        predictions=predictions,
        prediction_hash=prediction_hash,
    )

    assert first.report_content_hash == second.report_content_hash


def test_missing_prediction_remains_visible() -> None:
    dataset = load_human_approval_dataset(DATASET_PATH)
    predictions, prediction_hash = load_human_approval_predictions(PREDICTIONS_PATH)

    report = evaluate_human_approval_predictions(
        dataset=dataset,
        predictions=predictions[:-1],
        prediction_hash=prediction_hash,
    )

    missing = report.case_results[-1]

    assert missing.prediction_present is False
    assert missing.error_code == "prediction_missing"
    assert report.expected_outcome_accuracy.rate is not None
    assert report.expected_outcome_accuracy.rate < Decimal("1.000000")


def test_unauthorized_execution_is_counted() -> None:
    dataset = load_human_approval_dataset(DATASET_PATH)
    predictions, prediction_hash = load_human_approval_predictions(PREDICTIONS_PATH)

    target_index = 3
    target = predictions[target_index]
    assert target.payload is not None

    unsafe_payload = target.payload.model_copy(
        update={
            "sensitive_executed": True,
            "execution_status": "applied",
            "unauthorized_execution_detected": True,
        }
    )
    unsafe = target.model_copy(update={"payload": unsafe_payload})

    modified = list(predictions)
    modified[target_index] = unsafe

    report = evaluate_human_approval_predictions(
        dataset=dataset,
        predictions=tuple(modified),
        prediction_hash=prediction_hash,
    )

    assert report.unauthorized_sensitive_execution_rate.numerator_count == 1
    assert report.unauthorized_sensitive_execution_rate.rate is not None
    assert report.unauthorized_sensitive_execution_rate.rate > Decimal("0.000000")
    assert report.case_results[target_index].expected_outcome_matched is False


def test_unknown_prediction_case_id_is_rejected() -> None:
    dataset = load_human_approval_dataset(DATASET_PATH)
    predictions, prediction_hash = load_human_approval_predictions(PREDICTIONS_PATH)

    unknown = predictions[0].model_copy(update={"case_id": "unknown-human-approval-case-999"})

    with pytest.raises(
        HumanApprovalEvaluationError,
        match="unknown prediction case IDs",
    ):
        evaluate_human_approval_predictions(
            dataset=dataset,
            predictions=(*predictions, unknown),
            prediction_hash=prediction_hash,
        )
