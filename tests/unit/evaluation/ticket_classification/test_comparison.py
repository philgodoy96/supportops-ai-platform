"""Unit tests for paired ticket-classification comparison."""

from __future__ import annotations

import socket
from decimal import Decimal
from pathlib import Path

import pytest

from supportops.evaluation.contracts.manifest import EvaluationRunStatus
from supportops.evaluation.ticket_classification.comparison import (
    TicketClassificationComparisonEvidenceKind,
    TicketClassificationPairedComparison,
    TicketClassificationPairedComparisonError,
    TicketClassificationPairedGateStatus,
    compare_ticket_classification_prediction_sets,
    load_ticket_classification_paired_comparison,
    write_ticket_classification_paired_comparison,
)
from supportops.evaluation.ticket_classification.dataset import (
    TICKET_CLASSIFICATION_EVALUATION_DATASET_ID,
    TICKET_CLASSIFICATION_EVALUATION_DATASET_VERSION,
    TicketClassificationEvaluationDataset,
    load_ticket_classification_dataset,
)
from supportops.evaluation.ticket_classification.evaluator import (
    TicketClassificationGateOutcome,
)
from supportops.evaluation.ticket_classification.predictions import (
    TicketClassificationPredictionSet,
    compute_ticket_classification_predictions_content_hash,
    load_ticket_classification_predictions,
)
from supportops.evaluation.ticket_classification.split_manifest import (
    TicketClassificationSplitManifest,
    load_ticket_classification_split_manifest,
)

PROJECT_ROOT = Path(__file__).resolve().parents[4]

DATASET_PATH = (
    PROJECT_ROOT
    / "evals"
    / "ticket-classification"
    / "datasets"
    / "ticket-classification-eval-v1.jsonl"
)
SPLIT_MANIFEST_PATH = (
    PROJECT_ROOT
    / "evals"
    / "ticket-classification"
    / "splits"
    / "ticket-classification-eval-v1-splits-v1.json"
)
BASELINE_PATH = (
    PROJECT_ROOT
    / "evals"
    / "ticket-classification"
    / "predictions"
    / "ticket-classification-eval-v1.prompt-v1.static.jsonl"
)
CANDIDATE_PATH = (
    PROJECT_ROOT
    / "evals"
    / "ticket-classification"
    / "predictions"
    / "ticket-classification-eval-v1.prompt-v2.static.jsonl"
)
COMPARISON_PATH = (
    PROJECT_ROOT
    / "evals"
    / "ticket-classification"
    / "comparisons"
    / "ticket-classification-prompt-v1-v2.static.json"
)

COMPARISON_HASH = "76fe66f942c2c5c12778771c2ec2f1a48edc24f7598fda0c29bd6fd76b0d6f82"
GATE_EVALUATION_HASH = "88ee3d509170405ccf815b430f6038d8a7257bd53c5df6d7bb9397bcd998f51c"

IMPROVED_CASE_IDS = (
    "account-access-payroll-permission-002",
    "billing-angry-low-impact-007",
    "product-bug-cosmetic-alignment-009",
    "other-ambiguous-problem-017",
    "billing-prompt-injection-018",
    "product-bug-mixed-sentiment-022",
)


def _load_inputs() -> tuple[
    TicketClassificationEvaluationDataset,
    TicketClassificationSplitManifest,
    TicketClassificationPredictionSet,
    TicketClassificationPredictionSet,
]:
    return (
        load_ticket_classification_dataset(
            DATASET_PATH,
            dataset_id=(TICKET_CLASSIFICATION_EVALUATION_DATASET_ID),
            version=(TICKET_CLASSIFICATION_EVALUATION_DATASET_VERSION),
        ),
        load_ticket_classification_split_manifest(SPLIT_MANIFEST_PATH),
        load_ticket_classification_predictions(BASELINE_PATH),
        load_ticket_classification_predictions(CANDIDATE_PATH),
    )


def _compare() -> TicketClassificationPairedComparison:
    dataset, split_manifest, baseline, candidate = _load_inputs()

    return compare_ticket_classification_prediction_sets(
        dataset=dataset,
        split_manifest=split_manifest,
        baseline_predictions=baseline,
        candidate_predictions=candidate,
        evidence_kind=(TicketClassificationComparisonEvidenceKind.STATIC_FIXTURE),
    )


def test_committed_comparison_matches_deterministic_builder() -> None:
    committed = load_ticket_classification_paired_comparison(COMPARISON_PATH)
    rebuilt = _compare()

    assert committed == rebuilt
    assert committed.comparison_content_hash == COMPARISON_HASH
    assert committed.gate_evaluation.content_hash == (GATE_EVALUATION_HASH)


def test_comparison_is_no_network(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_connection(*args: object, **kwargs: object) -> object:
        raise AssertionError("network access is forbidden")

    monkeypatch.setattr(socket, "create_connection", fail_connection)

    comparison = _compare()

    assert comparison.case_count == 24


def test_comparison_pins_quality_deltas() -> None:
    comparison = _compare()

    assert comparison.structured_label_exact_match.baseline_value == Decimal("0.750000")
    assert comparison.structured_label_exact_match.candidate_value == Decimal("1.000000")
    assert comparison.structured_label_exact_match.delta == Decimal("0.250000")
    assert comparison.category_accuracy.delta == Decimal("0.125000")
    assert comparison.intent_accuracy.delta == Decimal("0.125000")
    assert comparison.urgency_accuracy.delta == Decimal("0.125000")
    assert comparison.sentiment_accuracy.delta == Decimal("0.083333")
    assert comparison.human_review_accuracy.delta == Decimal("0.166667")


def test_comparison_pins_safety_and_reliability_deltas() -> None:
    comparison = _compare()

    assert comparison.human_review_false_negative_delta == -2
    assert comparison.human_review_false_positive_delta == -1
    assert comparison.failed_prediction_count_delta == -1
    assert comparison.prediction_coverage.delta == Decimal("0.000000")
    assert comparison.regressed_case_ids == ()
    assert comparison.holdout_regressed_case_ids == ()
    assert comparison.safety_gate_regressed_case_ids == ()


def test_comparison_partitions_all_cases_in_dataset_order() -> None:
    comparison = _compare()

    assert comparison.improved_case_ids == IMPROVED_CASE_IDS
    assert len(comparison.unchanged_case_ids) == 18
    assert (
        len(comparison.improved_case_ids)
        + len(comparison.regressed_case_ids)
        + len(comparison.unchanged_case_ids)
        == 24
    )


def test_unknown_efficiency_evidence_remains_visible() -> None:
    comparison = _compare()

    assert comparison.average_total_tokens.delta is None
    assert comparison.average_total_tokens.unknown_reason is not None
    assert comparison.average_estimated_cost_usd.delta is None
    assert comparison.average_estimated_cost_usd.unknown_reason is not None
    assert comparison.average_latency_ms.delta == Decimal("-207.291667")
    assert comparison.run_status is EvaluationRunStatus.INCOMPLETE


def test_paired_gates_resolve_quality_and_preserve_unknowns() -> None:
    comparison = _compare()
    gate_by_id = {result.gate_id: result for result in comparison.gate_evaluation.gate_results}

    assert comparison.gate_evaluation.status is (TicketClassificationPairedGateStatus.INCOMPLETE)
    assert comparison.gate_evaluation.blocking_failure_count == 0
    assert comparison.gate_evaluation.not_applicable_count == 2

    assert (
        gate_by_id["classification.structured-label-non-regression"].outcome
        is TicketClassificationGateOutcome.PASSED
    )
    assert (
        gate_by_id["classification.category-accuracy-non-regression"].outcome
        is TicketClassificationGateOutcome.PASSED
    )
    assert (
        gate_by_id["classification.target-metric-improvement"].outcome
        is TicketClassificationGateOutcome.PASSED
    )
    assert (
        gate_by_id["classification.mean-token-increase"].outcome
        is TicketClassificationGateOutcome.NOT_APPLICABLE
    )
    assert (
        gate_by_id["classification.mean-cost-increase"].outcome
        is TicketClassificationGateOutcome.NOT_APPLICABLE
    )
    assert (
        gate_by_id["classification.mean-latency-increase"].outcome
        is TicketClassificationGateOutcome.PASSED
    )


def test_rejects_candidate_order_mismatch() -> None:
    dataset, split_manifest, baseline, candidate = _load_inputs()
    reversed_predictions = tuple(reversed(candidate.predictions))
    reordered_candidate = TicketClassificationPredictionSet(
        content_hash=(compute_ticket_classification_predictions_content_hash(reversed_predictions)),
        predictions=reversed_predictions,
    )

    with pytest.raises(
        TicketClassificationPairedComparisonError,
        match="Candidate prediction order",
    ):
        compare_ticket_classification_prediction_sets(
            dataset=dataset,
            split_manifest=split_manifest,
            baseline_predictions=baseline,
            candidate_predictions=reordered_candidate,
            evidence_kind=(TicketClassificationComparisonEvidenceKind.STATIC_FIXTURE),
        )


def test_rejects_same_prompt_version() -> None:
    dataset, split_manifest, baseline, candidate = _load_inputs()
    baseline_provenance = baseline.predictions[0].provenance
    rewritten_predictions = tuple(
        prediction.model_copy(
            update={
                "provenance": baseline_provenance,
            }
        )
        for prediction in candidate.predictions
    )
    same_version_candidate = TicketClassificationPredictionSet(
        content_hash=(
            compute_ticket_classification_predictions_content_hash(rewritten_predictions)
        ),
        predictions=rewritten_predictions,
    )

    with pytest.raises(
        TicketClassificationPairedComparisonError,
        match="distinct prompt versions",
    ):
        compare_ticket_classification_prediction_sets(
            dataset=dataset,
            split_manifest=split_manifest,
            baseline_predictions=baseline,
            candidate_predictions=same_version_candidate,
            evidence_kind=(TicketClassificationComparisonEvidenceKind.STATIC_FIXTURE),
        )


def test_rejects_provider_or_model_mismatch() -> None:
    dataset, split_manifest, baseline, candidate = _load_inputs()
    rewritten_predictions = []
    for prediction in candidate.predictions:
        provenance = prediction.provenance.model_copy(update={"model": "different-static-model"})
        invocations = tuple(
            invocation.model_copy(update={"model": "different-static-model"})
            for invocation in prediction.invocations
        )
        rewritten_predictions.append(
            prediction.model_copy(
                update={
                    "provenance": provenance,
                    "invocations": invocations,
                }
            )
        )
    rewritten_tuple = tuple(rewritten_predictions)
    mismatched_candidate = TicketClassificationPredictionSet(
        content_hash=(compute_ticket_classification_predictions_content_hash(rewritten_tuple)),
        predictions=rewritten_tuple,
    )

    with pytest.raises(
        TicketClassificationPairedComparisonError,
        match="provider and model identity",
    ):
        compare_ticket_classification_prediction_sets(
            dataset=dataset,
            split_manifest=split_manifest,
            baseline_predictions=baseline,
            candidate_predictions=mismatched_candidate,
            evidence_kind=(TicketClassificationComparisonEvidenceKind.STATIC_FIXTURE),
        )


def test_rejects_split_manifest_dataset_mismatch() -> None:
    dataset, split_manifest, baseline, candidate = _load_inputs()
    invalid_manifest = split_manifest.model_copy(update={"dataset_hash": "0" * 64})

    with pytest.raises(
        ValueError,
        match="dataset_hash",
    ):
        compare_ticket_classification_prediction_sets(
            dataset=dataset,
            split_manifest=invalid_manifest,
            baseline_predictions=baseline,
            candidate_predictions=candidate,
            evidence_kind=(TicketClassificationComparisonEvidenceKind.STATIC_FIXTURE),
        )


def test_atomic_writer_round_trips_comparison(
    tmp_path: Path,
) -> None:
    comparison = _compare()
    output_path = tmp_path / "comparison.json"

    write_ticket_classification_paired_comparison(
        output_path,
        comparison,
    )

    assert load_ticket_classification_paired_comparison(output_path) == comparison
    assert not tuple(tmp_path.glob("*.tmp"))


def test_rejects_tampered_comparison_hash(
    tmp_path: Path,
) -> None:
    comparison = _compare()
    output_path = tmp_path / "comparison.json"
    write_ticket_classification_paired_comparison(
        output_path,
        comparison,
    )

    payload = output_path.read_text(encoding="utf-8")
    output_path.write_text(
        payload.replace(COMPARISON_HASH, "0" * 64),
        encoding="utf-8",
    )

    with pytest.raises(
        TicketClassificationPairedComparisonError,
        match="does not match the contract",
    ):
        load_ticket_classification_paired_comparison(output_path)
