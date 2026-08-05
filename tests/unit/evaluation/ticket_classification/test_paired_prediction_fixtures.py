"""Tests for paired static ticket-classification prediction fixtures."""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path

from supportops.evaluation.ticket_classification.dataset import (
    TICKET_CLASSIFICATION_EVALUATION_DATASET_ID,
    TICKET_CLASSIFICATION_EVALUATION_DATASET_VERSION,
    TicketClassificationEvaluationDataset,
    load_ticket_classification_dataset,
)
from supportops.evaluation.ticket_classification.evaluator import (
    evaluate_ticket_classification_predictions,
)
from supportops.evaluation.ticket_classification.predictions import (
    TicketClassificationFailedPrediction,
    TicketClassificationPredictionSet,
    TicketClassificationSuccessfulPrediction,
    load_ticket_classification_predictions,
)

PROJECT_ROOT = Path(__file__).resolve().parents[4]

DATASET_PATH = (
    PROJECT_ROOT
    / "evals"
    / "ticket-classification"
    / "datasets"
    / "ticket-classification-eval-v1.jsonl"
)
PROMPT_V1_PREDICTIONS_PATH = (
    PROJECT_ROOT
    / "evals"
    / "ticket-classification"
    / "predictions"
    / "ticket-classification-eval-v1.prompt-v1.static.jsonl"
)
PROMPT_V2_PREDICTIONS_PATH = (
    PROJECT_ROOT
    / "evals"
    / "ticket-classification"
    / "predictions"
    / "ticket-classification-eval-v1.prompt-v2.static.jsonl"
)

PROMPT_V1_HASH = "3c9107f8685232da86442e63a551cd991d0d8fc174f480a6b0c8ead3afc85da2"
PROMPT_V2_HASH = "af9ebb855fbdfd340b9377f19e3e3bad1a9ff853af5747de9ca66edca884e3f0"
PROMPT_V1_PREDICTIONS_HASH = "e0b2b2b142e91d11119a2527d6bfbca86f75134166ce5c4e100ce39af3ffe984"
PROMPT_V2_PREDICTIONS_HASH = "f0212c9ac778791b57f3743d3c4f13ca82607451f013d2d5c2e150a290520f87"

STATIC_PROVIDER = "static-fixture"
STATIC_MODEL = "ticket-classification-paired-fixture"

DEVELOPMENT_CASE_IDS = {
    "account-access-password-reset-001",
    "account-access-payroll-permission-002",
    "service-incident-latency-004",
    "billing-angry-low-impact-007",
    "product-bug-export-failure-008",
    "product-bug-cosmetic-alignment-009",
    "how-to-audit-log-export-010",
    "security-suspicious-login-013",
    "feature-request-dark-mode-014",
    "other-ambiguous-problem-017",
    "billing-prompt-injection-018",
    "product-bug-mixed-sentiment-022",
}
HOLDOUT_CASE_IDS = {
    "billing-duplicate-charge-005",
    "billing-refund-request-006",
    "how-to-positive-automation-011",
    "feature-request-bulk-update-015",
    "other-sales-inquiry-016",
    "account-access-executive-lockout-020",
    "product-bug-single-user-dashboard-021",
    "billing-positive-feedback-023",
}
SAFETY_GATE_CASE_IDS = {
    "service-incident-global-outage-003",
    "security-exposed-api-key-012",
    "security-prompt-injection-019",
    "security-data-deletion-request-024",
}

V1_INTENTIONAL_LABEL_ERROR_CASE_IDS = {
    "account-access-payroll-permission-002",
    "billing-angry-low-impact-007",
    "other-ambiguous-problem-017",
    "billing-prompt-injection-018",
    "product-bug-mixed-sentiment-022",
}
V1_INTENTIONAL_FAILED_CASE_ID = "product-bug-cosmetic-alignment-009"


def _load_dataset() -> TicketClassificationEvaluationDataset:
    return load_ticket_classification_dataset(
        DATASET_PATH,
        dataset_id=TICKET_CLASSIFICATION_EVALUATION_DATASET_ID,
        version=TICKET_CLASSIFICATION_EVALUATION_DATASET_VERSION,
    )


def _load_pair() -> tuple[
    TicketClassificationPredictionSet,
    TicketClassificationPredictionSet,
]:
    return (
        load_ticket_classification_predictions(PROMPT_V1_PREDICTIONS_PATH),
        load_ticket_classification_predictions(PROMPT_V2_PREDICTIONS_PATH),
    )


def test_static_pair_loads_with_pinned_hashes() -> None:
    baseline, candidate = _load_pair()

    assert len(baseline.predictions) == 24
    assert len(candidate.predictions) == 24
    assert baseline.content_hash == PROMPT_V1_PREDICTIONS_HASH
    assert candidate.content_hash == PROMPT_V2_PREDICTIONS_HASH


def test_static_pair_preserves_dataset_order_and_case_identity() -> None:
    dataset = _load_dataset()
    baseline, candidate = _load_pair()

    dataset_case_ids = tuple(case.case_id for case in dataset.cases)

    assert tuple(prediction.case_id for prediction in baseline.predictions) == dataset_case_ids
    assert tuple(prediction.case_id for prediction in candidate.predictions) == dataset_case_ids


def test_static_pair_has_compatible_runtime_provenance() -> None:
    baseline, candidate = _load_pair()

    baseline_provenance = {
        (
            prediction.provenance.provider,
            prediction.provenance.model,
        )
        for prediction in baseline.predictions
    }
    candidate_provenance = {
        (
            prediction.provenance.provider,
            prediction.provenance.model,
        )
        for prediction in candidate.predictions
    }

    assert baseline_provenance == {(STATIC_PROVIDER, STATIC_MODEL)}
    assert candidate_provenance == baseline_provenance


def test_static_pair_binds_distinct_prompt_versions_and_hashes() -> None:
    baseline, candidate = _load_pair()

    assert {
        prediction.provenance.prompt_id
        for prediction in (*baseline.predictions, *candidate.predictions)
    } == {"ticket-classification"}

    assert {
        (
            prediction.provenance.prompt_version,
            prediction.provenance.prompt_content_hash,
        )
        for prediction in baseline.predictions
    } == {(1, PROMPT_V1_HASH)}
    assert {
        (
            prediction.provenance.prompt_version,
            prediction.provenance.prompt_content_hash,
        )
        for prediction in candidate.predictions
    } == {(2, PROMPT_V2_HASH)}


def test_baseline_fixture_contains_only_declared_development_errors() -> None:
    dataset = _load_dataset()
    baseline, _ = _load_pair()
    baseline_by_case_id = {prediction.case_id: prediction for prediction in baseline.predictions}

    mismatched_case_ids: set[str] = set()
    failed_case_ids: set[str] = set()

    for case in dataset.cases:
        prediction = baseline_by_case_id[case.case_id]

        if isinstance(
            prediction,
            TicketClassificationFailedPrediction,
        ):
            failed_case_ids.add(case.case_id)
            continue

        assert isinstance(
            prediction,
            TicketClassificationSuccessfulPrediction,
        )
        predicted_labels = (
            prediction.output.category,
            prediction.output.intent,
            prediction.output.urgency,
            prediction.output.sentiment,
            prediction.output.requires_human_review,
        )
        expected_labels = (
            case.expected.category,
            case.expected.intent,
            case.expected.urgency,
            case.expected.sentiment,
            case.expected.requires_human_review,
        )
        if predicted_labels != expected_labels:
            mismatched_case_ids.add(case.case_id)

    assert mismatched_case_ids == (V1_INTENTIONAL_LABEL_ERROR_CASE_IDS)
    assert failed_case_ids == {V1_INTENTIONAL_FAILED_CASE_ID}
    assert (mismatched_case_ids | failed_case_ids) <= DEVELOPMENT_CASE_IDS


def test_candidate_fixture_matches_all_expected_labels() -> None:
    dataset = _load_dataset()
    _, candidate = _load_pair()
    candidate_by_case_id = {prediction.case_id: prediction for prediction in candidate.predictions}

    for case in dataset.cases:
        prediction = candidate_by_case_id[case.case_id]

        assert isinstance(
            prediction,
            TicketClassificationSuccessfulPrediction,
        )
        assert prediction.output.category is case.expected.category
        assert prediction.output.intent is case.expected.intent
        assert prediction.output.urgency is case.expected.urgency
        assert prediction.output.sentiment is case.expected.sentiment
        assert prediction.output.requires_human_review is case.expected.requires_human_review
        assert prediction.output.schema_version == case.expected.schema_version


def test_holdout_and_safety_gate_outputs_are_unchanged() -> None:
    baseline, candidate = _load_pair()
    baseline_by_case_id = {prediction.case_id: prediction for prediction in baseline.predictions}
    candidate_by_case_id = {prediction.case_id: prediction for prediction in candidate.predictions}

    protected_case_ids = HOLDOUT_CASE_IDS | SAFETY_GATE_CASE_IDS

    for case_id in protected_case_ids:
        baseline_prediction = baseline_by_case_id[case_id]
        candidate_prediction = candidate_by_case_id[case_id]

        assert isinstance(
            baseline_prediction,
            TicketClassificationSuccessfulPrediction,
        )
        assert isinstance(
            candidate_prediction,
            TicketClassificationSuccessfulPrediction,
        )
        assert baseline_prediction.output == candidate_prediction.output


def test_baseline_report_pins_intentional_fixture_failures() -> None:
    dataset = _load_dataset()
    baseline, _ = _load_pair()

    report = evaluate_ticket_classification_predictions(
        dataset=dataset,
        predictions=baseline,
    )

    assert report.case_count == 24
    assert report.successful_prediction_count == 23
    assert report.failed_prediction_count == 1
    assert report.structured_label_exact_match.match_count == 18
    assert report.structured_label_exact_match.rate == Decimal("0.750000")
    assert report.category_accuracy.match_count == 21
    assert report.intent_accuracy.match_count == 21
    assert report.urgency_accuracy.match_count == 21
    assert report.sentiment_accuracy.match_count == 22
    assert report.human_review_accuracy.match_count == 20
    assert report.human_review.false_negative_count == 2
    assert report.human_review.false_positive_count == 1
    assert report.critical_urgency_recall.recall == Decimal("1.000000")
    assert report.high_risk_human_review_recall.recall == Decimal("0.750000")
    assert report.failure_counts_by_error_code == {"llm_timeout": 1}


def test_candidate_report_is_complete_and_exact() -> None:
    dataset = _load_dataset()
    _, candidate = _load_pair()

    report = evaluate_ticket_classification_predictions(
        dataset=dataset,
        predictions=candidate,
    )

    assert report.case_count == 24
    assert report.successful_prediction_count == 24
    assert report.failed_prediction_count == 0
    assert report.structured_label_exact_match.match_count == 24
    assert report.structured_label_exact_match.rate == Decimal("1.000000")
    assert report.category_accuracy.rate == Decimal("1.000000")
    assert report.intent_accuracy.rate == Decimal("1.000000")
    assert report.urgency_accuracy.rate == Decimal("1.000000")
    assert report.sentiment_accuracy.rate == Decimal("1.000000")
    assert report.human_review_accuracy.rate == Decimal("1.000000")
    assert report.human_review.false_negative_count == 0
    assert report.human_review.false_positive_count == 0
    assert report.high_risk_human_review_recall.recall == Decimal("1.000000")
    assert report.failure_counts_by_error_code == {}


def test_static_pair_preserves_unknown_cost_evidence() -> None:
    dataset = _load_dataset()
    baseline, candidate = _load_pair()

    baseline_report = evaluate_ticket_classification_predictions(
        dataset=dataset,
        predictions=baseline,
    )
    candidate_report = evaluate_ticket_classification_predictions(
        dataset=dataset,
        predictions=candidate,
    )

    assert baseline_report.unknown_pricing_count == 24
    assert candidate_report.unknown_pricing_count == 24
    assert baseline_report.known_estimated_total_cost_usd == Decimal("0")
    assert candidate_report.known_estimated_total_cost_usd == Decimal("0")


def test_static_fixtures_contain_no_provider_response_or_reasoning() -> None:
    fixture_text = (
        PROMPT_V1_PREDICTIONS_PATH.read_text(encoding="utf-8")
        + PROMPT_V2_PREDICTIONS_PATH.read_text(encoding="utf-8")
    ).lower()

    assert "provider_request_id" not in fixture_text
    assert "raw_response" not in fixture_text
    assert "chain_of_thought" not in fixture_text
    assert '"reasoning_tokens":0' in fixture_text
