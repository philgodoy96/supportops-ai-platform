"""Unit tests for deterministic classification metrics."""

import json
from decimal import Decimal
from pathlib import Path

import pytest
from pydantic import ValidationError

from supportops.evaluation.ticket_classification.dataset import (
    TicketClassificationEvaluationDataset,
    load_ticket_classification_dataset,
)
from supportops.evaluation.ticket_classification.evaluator import (
    DEFAULT_TICKET_CLASSIFICATION_RELEASE_GATE_PROFILE,
    InconsistentTicketClassificationPredictionProvenanceError,
    TicketClassificationGateCategory,
    TicketClassificationGateOperator,
    TicketClassificationGateOutcome,
    TicketClassificationReleaseGateDefinition,
    TicketClassificationReleaseGateEvaluation,
    TicketClassificationReleaseGateProfile,
    TicketClassificationReleaseGateResult,
    TicketClassificationStandaloneGateStatus,
    UnknownTicketClassificationPredictionError,
    evaluate_ticket_classification_predictions,
    evaluate_ticket_classification_release_gates,
)
from supportops.evaluation.ticket_classification.predictions import (
    TicketClassificationPredictionSet,
    load_ticket_classification_predictions,
)

_PROMPT_HASH = "a" * 64


def _dataset_case(
    *,
    case_id: str,
    category: str = "billing",
    urgency: str = "normal",
    requires_human_review: bool = False,
    tags: tuple[str, ...] = ("evaluation",),
) -> dict[str, object]:
    return {
        "case_id": case_id,
        "tags": list(tags),
        "ticket": {
            "subject": "Synthetic ticket",
            "description": "Synthetic evaluation description.",
        },
        "expected": {
            "category": category,
            "intent": "ask_question",
            "urgency": urgency,
            "sentiment": "neutral",
            "requires_human_review": (requires_human_review),
            "schema_version": "ticket-classification-v1",
        },
    }


def _invocation(
    *,
    sequence: int = 1,
    status: str = "succeeded",
    provider: str = "mock",
    input_tokens: int | None = 100,
    output_tokens: int | None = 20,
    total_tokens: int | None = 120,
    include_usage: bool = True,
    pricing_found: bool = True,
    latency_ms: int = 25,
    error_code: str | None = None,
) -> dict[str, object]:
    usage: dict[str, object] | None
    if not include_usage or (
        total_tokens is None and input_tokens is None and output_tokens is None
    ):
        usage = None
    else:
        usage = {
            "input_tokens": input_tokens,
            "cached_input_tokens": (0 if input_tokens is not None else None),
            "output_tokens": output_tokens,
            "reasoning_tokens": None,
            "total_tokens": total_tokens,
        }

    cost = {
        "pricing_catalog_version": "pricing-v1",
        "pricing_found": pricing_found,
        "estimated_input_cost_usd": ("0" if pricing_found else None),
        "estimated_cached_input_cost_usd": ("0" if pricing_found else None),
        "estimated_output_cost_usd": ("0" if pricing_found else None),
        "estimated_total_cost_usd": ("0" if pricing_found else None),
    }

    return {
        "invocation_sequence": sequence,
        "status": status,
        "provider": provider,
        "model": "test-model",
        "usage": usage,
        "cost": cost,
        "latency_ms": latency_ms,
        "error_code": error_code,
    }


def _success_prediction(
    *,
    case_id: str,
    category: str = "billing",
    urgency: str = "normal",
    requires_human_review: bool = False,
    provider: str = "mock",
    total_tokens: int | None = 120,
    input_tokens: int | None = 100,
    output_tokens: int | None = 20,
    pricing_found: bool = True,
    latency_ms: int = 25,
    invocations: tuple[dict[str, object], ...] | None = None,
) -> dict[str, object]:
    if invocations is None:
        invocations = (
            _invocation(
                provider=provider,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=total_tokens,
                pricing_found=pricing_found,
                latency_ms=latency_ms,
            ),
        )

    return {
        "case_id": case_id,
        "status": "succeeded",
        "provenance": {
            "prompt_id": "ticket-classification",
            "prompt_version": 1,
            "prompt_content_hash": _PROMPT_HASH,
            "provider": provider,
            "model": "test-model",
        },
        "output": {
            "category": category,
            "intent": "ask_question",
            "urgency": urgency,
            "sentiment": "neutral",
            "requires_human_review": (requires_human_review),
            "summary": "Synthetic summary.",
            "schema_version": "ticket-classification-v1",
        },
        "invocations": list(invocations),
    }


def _failure_prediction(
    *,
    case_id: str,
    error_code: str = "llm_timeout",
    status: str = "timed_out",
    latency_ms: int = 12000,
    include_usage: bool = False,
    invocations: tuple[dict[str, object], ...] | None = None,
) -> dict[str, object]:
    if invocations is None:
        invocations = (
            _invocation(
                status=status,
                include_usage=include_usage,
                latency_ms=latency_ms,
                error_code=error_code,
            ),
        )

    return {
        "case_id": case_id,
        "status": "failed",
        "error_code": error_code,
        "provenance": {
            "prompt_id": "ticket-classification",
            "prompt_version": 1,
            "prompt_content_hash": _PROMPT_HASH,
            "provider": "mock",
            "model": "test-model",
        },
        "invocations": list(invocations),
    }


def _write_jsonl(
    path: Path,
    *payloads: dict[str, object],
) -> None:
    path.write_text(
        "\n".join(json.dumps(payload) for payload in payloads) + "\n",
        encoding="utf-8",
    )


def _load_artifacts(
    tmp_path: Path,
    *,
    dataset_cases: tuple[dict[str, object], ...],
    predictions: tuple[dict[str, object], ...],
) -> tuple[
    TicketClassificationEvaluationDataset,
    TicketClassificationPredictionSet,
]:
    dataset_path = tmp_path / "dataset.jsonl"
    predictions_path = tmp_path / "predictions.jsonl"

    _write_jsonl(
        dataset_path,
        *dataset_cases,
    )
    _write_jsonl(
        predictions_path,
        *predictions,
    )

    dataset = load_ticket_classification_dataset(
        dataset_path,
        dataset_id="ticket-classification-eval",
        version=1,
    )
    prediction_set = load_ticket_classification_predictions(
        predictions_path,
    )

    return dataset, prediction_set


def test_perfect_predictions_produce_perfect_metrics(
    tmp_path: Path,
) -> None:
    dataset, predictions = _load_artifacts(
        tmp_path,
        dataset_cases=(
            _dataset_case(
                case_id="case-001",
                requires_human_review=True,
            ),
            _dataset_case(
                case_id="case-002",
                requires_human_review=False,
            ),
        ),
        predictions=(
            _success_prediction(
                case_id="case-001",
                requires_human_review=True,
            ),
            _success_prediction(
                case_id="case-002",
                requires_human_review=False,
            ),
        ),
    )

    report = evaluate_ticket_classification_predictions(
        dataset=dataset,
        predictions=predictions,
    )

    assert report.case_count == 2
    assert report.successful_prediction_count == 2
    assert report.failed_prediction_count == 0
    assert report.structured_label_exact_match.match_count == 2
    assert report.structured_label_exact_match.rate == Decimal("1.000000")
    assert report.category_accuracy.rate == (Decimal("1.000000"))
    assert report.human_review.precision == (Decimal("1.000000"))
    assert report.human_review.recall == (Decimal("1.000000"))
    assert report.human_review.f1 == (Decimal("1.000000"))
    assert report.known_total_tokens == 240
    assert report.unknown_usage_count == 0
    assert report.unknown_pricing_count == 0
    assert len(report.report_content_hash) == 64


def test_field_mismatches_are_scored_independently(
    tmp_path: Path,
) -> None:
    dataset, predictions = _load_artifacts(
        tmp_path,
        dataset_cases=(
            _dataset_case(
                case_id="case-001",
                category="billing",
            ),
            _dataset_case(
                case_id="case-002",
                category="security",
            ),
        ),
        predictions=(
            _success_prediction(
                case_id="case-001",
                category="billing",
            ),
            _success_prediction(
                case_id="case-002",
                category="billing",
            ),
        ),
    )

    report = evaluate_ticket_classification_predictions(
        dataset=dataset,
        predictions=predictions,
    )

    assert report.structured_label_exact_match.match_count == 1
    assert report.category_accuracy.match_count == 1
    assert report.intent_accuracy.match_count == 2
    assert report.urgency_accuracy.match_count == 2
    assert report.sentiment_accuracy.match_count == 2


def test_failure_and_missing_case_count_as_failed_predictions(
    tmp_path: Path,
) -> None:
    dataset, predictions = _load_artifacts(
        tmp_path,
        dataset_cases=(
            _dataset_case(
                case_id="case-001",
                requires_human_review=True,
            ),
            _dataset_case(
                case_id="case-002",
                requires_human_review=True,
            ),
            _dataset_case(
                case_id="case-003",
                requires_human_review=False,
            ),
        ),
        predictions=(
            _failure_prediction(
                case_id="case-001",
            ),
            _success_prediction(
                case_id="case-003",
                requires_human_review=False,
            ),
        ),
    )

    report = evaluate_ticket_classification_predictions(
        dataset=dataset,
        predictions=predictions,
    )

    assert report.case_count == 3
    assert report.successful_prediction_count == 1
    assert report.failed_prediction_count == 2
    assert report.failure_counts_by_error_code == {
        "llm_timeout": 1,
        "prediction_missing": 1,
    }
    assert report.cases[0].prediction_status == "failed"
    assert report.cases[1].prediction_status == "missing"
    assert report.cases[2].prediction_status == ("succeeded")
    assert report.human_review.false_negative_count == 2


def test_unknown_prediction_case_is_rejected(
    tmp_path: Path,
) -> None:
    dataset, predictions = _load_artifacts(
        tmp_path,
        dataset_cases=(
            _dataset_case(
                case_id="case-001",
            ),
        ),
        predictions=(
            _success_prediction(
                case_id="unknown-case",
            ),
        ),
    )

    with pytest.raises(
        UnknownTicketClassificationPredictionError,
        match="unknown evaluation case IDs",
    ):
        evaluate_ticket_classification_predictions(
            dataset=dataset,
            predictions=predictions,
        )


def test_mixed_runtime_provenance_is_rejected(
    tmp_path: Path,
) -> None:
    dataset, predictions = _load_artifacts(
        tmp_path,
        dataset_cases=(
            _dataset_case(
                case_id="case-001",
            ),
            _dataset_case(
                case_id="case-002",
            ),
        ),
        predictions=(
            _success_prediction(
                case_id="case-001",
                provider="mock",
            ),
            _success_prediction(
                case_id="case-002",
                provider="openai",
            ),
        ),
    )

    with pytest.raises(
        InconsistentTicketClassificationPredictionProvenanceError,
        match="must share one prompt",
    ):
        evaluate_ticket_classification_predictions(
            dataset=dataset,
            predictions=predictions,
        )


def test_unknown_usage_and_pricing_are_aggregated(
    tmp_path: Path,
) -> None:
    dataset, predictions = _load_artifacts(
        tmp_path,
        dataset_cases=(
            _dataset_case(
                case_id="case-001",
            ),
        ),
        predictions=(
            _success_prediction(
                case_id="case-001",
                total_tokens=None,
                input_tokens=None,
                output_tokens=None,
                pricing_found=False,
            ),
        ),
    )

    report = evaluate_ticket_classification_predictions(
        dataset=dataset,
        predictions=predictions,
    )

    assert report.known_total_tokens == 0
    assert report.unknown_usage_count == 1
    assert report.unknown_pricing_count == 1
    assert report.known_estimated_total_cost_usd == (Decimal("0"))


def test_report_hash_is_deterministic(
    tmp_path: Path,
) -> None:
    dataset, predictions = _load_artifacts(
        tmp_path,
        dataset_cases=(
            _dataset_case(
                case_id="case-001",
            ),
        ),
        predictions=(
            _success_prediction(
                case_id="case-001",
            ),
        ),
    )

    first = evaluate_ticket_classification_predictions(
        dataset=dataset,
        predictions=predictions,
    )
    second = evaluate_ticket_classification_predictions(
        dataset=dataset,
        predictions=predictions,
    )

    assert first == second
    assert first.report_content_hash == second.report_content_hash


def test_all_predictions_structurally_valid(
    tmp_path: Path,
) -> None:
    dataset, predictions = _load_artifacts(
        tmp_path,
        dataset_cases=(
            _dataset_case(case_id="case-001"),
            _dataset_case(case_id="case-002"),
        ),
        predictions=(
            _success_prediction(case_id="case-001"),
            _success_prediction(case_id="case-002"),
        ),
    )

    report = evaluate_ticket_classification_predictions(
        dataset=dataset,
        predictions=predictions,
    )

    assert report.structured_output_validity.valid_count == 2
    assert report.structured_output_validity.invalid_count == 0
    assert report.structured_output_validity.rate == Decimal("1.000000")
    assert report.invalid_output_rate == Decimal("0.000000")


def test_failed_prediction_is_structurally_invalid(
    tmp_path: Path,
) -> None:
    dataset, predictions = _load_artifacts(
        tmp_path,
        dataset_cases=(
            _dataset_case(case_id="case-001"),
            _dataset_case(case_id="case-002"),
        ),
        predictions=(
            _failure_prediction(
                case_id="case-001",
                error_code="llm_output_validation_failed",
                status="validation_failed",
            ),
            _success_prediction(case_id="case-002"),
        ),
    )

    report = evaluate_ticket_classification_predictions(
        dataset=dataset,
        predictions=predictions,
    )

    assert report.structured_output_validity.valid_count == 1
    assert report.structured_output_validity.invalid_count == 1
    assert report.structured_output_validity.rate == Decimal("0.500000")
    assert report.invalid_output_rate == Decimal("0.500000")
    assert report.structured_output_validity.rate + report.invalid_output_rate == Decimal(
        "1.000000"
    )


def test_label_incorrect_prediction_remains_structurally_valid(
    tmp_path: Path,
) -> None:
    dataset, predictions = _load_artifacts(
        tmp_path,
        dataset_cases=(
            _dataset_case(
                case_id="case-001",
                category="security",
            ),
        ),
        predictions=(
            _success_prediction(
                case_id="case-001",
                category="billing",
            ),
        ),
    )

    report = evaluate_ticket_classification_predictions(
        dataset=dataset,
        predictions=predictions,
    )

    assert report.structured_label_exact_match.match_count == 0
    assert report.structured_output_validity.valid_count == 1
    assert report.structured_output_validity.invalid_count == 0
    assert report.invalid_output_rate == Decimal("0.000000")


def test_high_urgency_recall_true_positives_and_denominator(
    tmp_path: Path,
) -> None:
    dataset, predictions = _load_artifacts(
        tmp_path,
        dataset_cases=(
            _dataset_case(
                case_id="case-high",
                urgency="high",
            ),
            _dataset_case(
                case_id="case-critical",
                urgency="critical",
            ),
            _dataset_case(
                case_id="case-normal",
                urgency="normal",
            ),
        ),
        predictions=(
            _success_prediction(
                case_id="case-high",
                urgency="high",
            ),
            _success_prediction(
                case_id="case-critical",
                urgency="critical",
            ),
            _success_prediction(
                case_id="case-normal",
                urgency="normal",
            ),
        ),
    )

    report = evaluate_ticket_classification_predictions(
        dataset=dataset,
        predictions=predictions,
    )

    assert report.high_urgency_recall.expected_positive_count == 2
    assert report.high_urgency_recall.true_positive_count == 2
    assert report.high_urgency_recall.recall == Decimal("1.000000")


def test_critical_prediction_counts_for_high_urgency_recall(
    tmp_path: Path,
) -> None:
    dataset, predictions = _load_artifacts(
        tmp_path,
        dataset_cases=(
            _dataset_case(
                case_id="case-high",
                urgency="high",
            ),
        ),
        predictions=(
            _success_prediction(
                case_id="case-high",
                urgency="critical",
            ),
        ),
    )

    report = evaluate_ticket_classification_predictions(
        dataset=dataset,
        predictions=predictions,
    )

    assert report.high_urgency_recall.true_positive_count == 1
    assert report.high_urgency_recall.expected_positive_count == 1
    assert report.critical_urgency_recall.expected_positive_count == 0
    assert report.critical_urgency_recall.recall == Decimal("0.000000")


def test_high_prediction_is_miss_for_critical_urgency_recall(
    tmp_path: Path,
) -> None:
    dataset, predictions = _load_artifacts(
        tmp_path,
        dataset_cases=(
            _dataset_case(
                case_id="case-critical",
                urgency="critical",
            ),
        ),
        predictions=(
            _success_prediction(
                case_id="case-critical",
                urgency="high",
            ),
        ),
    )

    report = evaluate_ticket_classification_predictions(
        dataset=dataset,
        predictions=predictions,
    )

    assert report.high_urgency_recall.true_positive_count == 1
    assert report.critical_urgency_recall.true_positive_count == 0
    assert report.critical_urgency_recall.expected_positive_count == 1
    assert report.critical_urgency_recall.recall == Decimal("0.000000")


def test_failed_and_missing_critical_predictions_are_misses(
    tmp_path: Path,
) -> None:
    dataset, predictions = _load_artifacts(
        tmp_path,
        dataset_cases=(
            _dataset_case(
                case_id="case-failed",
                urgency="critical",
            ),
            _dataset_case(
                case_id="case-missing",
                urgency="critical",
            ),
        ),
        predictions=(
            _failure_prediction(
                case_id="case-failed",
            ),
        ),
    )

    report = evaluate_ticket_classification_predictions(
        dataset=dataset,
        predictions=predictions,
    )

    assert report.critical_urgency_recall.expected_positive_count == 2
    assert report.critical_urgency_recall.true_positive_count == 0
    assert report.critical_urgency_recall.recall == Decimal("0.000000")


def test_high_risk_human_review_recall_true_and_false(
    tmp_path: Path,
) -> None:
    dataset, predictions = _load_artifacts(
        tmp_path,
        dataset_cases=(
            _dataset_case(
                case_id="case-true",
                requires_human_review=True,
                tags=("evaluation", "credential-exposure"),
            ),
            _dataset_case(
                case_id="case-false",
                requires_human_review=True,
                tags=("evaluation", "privacy"),
            ),
            _dataset_case(
                case_id="case-untagged",
                requires_human_review=True,
                tags=("evaluation",),
            ),
        ),
        predictions=(
            _success_prediction(
                case_id="case-true",
                requires_human_review=True,
            ),
            _success_prediction(
                case_id="case-false",
                requires_human_review=False,
            ),
            _success_prediction(
                case_id="case-untagged",
                requires_human_review=True,
            ),
        ),
    )

    report = evaluate_ticket_classification_predictions(
        dataset=dataset,
        predictions=predictions,
    )

    assert report.high_risk_human_review_recall.expected_positive_count == 2
    assert report.high_risk_human_review_recall.true_positive_count == 1
    assert report.high_risk_human_review_recall.recall == Decimal("0.500000")


def test_failed_high_risk_review_case_is_a_miss(
    tmp_path: Path,
) -> None:
    dataset, predictions = _load_artifacts(
        tmp_path,
        dataset_cases=(
            _dataset_case(
                case_id="case-failed",
                requires_human_review=True,
                tags=("sensitive",),
            ),
        ),
        predictions=(
            _failure_prediction(
                case_id="case-failed",
            ),
        ),
    )

    report = evaluate_ticket_classification_predictions(
        dataset=dataset,
        predictions=predictions,
    )

    assert report.high_risk_human_review_recall.expected_positive_count == 1
    assert report.high_risk_human_review_recall.true_positive_count == 0
    assert report.high_risk_human_review_recall.recall == Decimal("0.000000")


def test_high_risk_tag_matching_is_exact(
    tmp_path: Path,
) -> None:
    dataset, predictions = _load_artifacts(
        tmp_path,
        dataset_cases=(
            _dataset_case(
                case_id="case-near-miss",
                requires_human_review=True,
                tags=("credential-exposures", "human-reviews"),
            ),
            _dataset_case(
                case_id="case-exact",
                requires_human_review=True,
                tags=("unauthorized-activity",),
            ),
        ),
        predictions=(
            _success_prediction(
                case_id="case-near-miss",
                requires_human_review=True,
            ),
            _success_prediction(
                case_id="case-exact",
                requires_human_review=True,
            ),
        ),
    )

    report = evaluate_ticket_classification_predictions(
        dataset=dataset,
        predictions=predictions,
    )

    assert report.high_risk_human_review_recall.expected_positive_count == 1
    assert report.high_risk_human_review_recall.true_positive_count == 1


def test_known_and_unknown_latency_aggregation(
    tmp_path: Path,
) -> None:
    dataset, predictions = _load_artifacts(
        tmp_path,
        dataset_cases=(
            _dataset_case(case_id="case-001"),
            _dataset_case(case_id="case-002"),
            _dataset_case(case_id="case-003"),
        ),
        predictions=(
            _success_prediction(
                case_id="case-001",
                latency_ms=20,
            ),
            _success_prediction(
                case_id="case-002",
                latency_ms=40,
            ),
        ),
    )

    report = evaluate_ticket_classification_predictions(
        dataset=dataset,
        predictions=predictions,
    )

    assert report.latency.known_latency_count == 2
    assert report.latency.unknown_latency_count == 1
    assert report.latency.average_latency_ms == Decimal("30.000000")


def test_multiple_invocation_latency_is_summed(
    tmp_path: Path,
) -> None:
    dataset, predictions = _load_artifacts(
        tmp_path,
        dataset_cases=(_dataset_case(case_id="case-001"),),
        predictions=(
            _success_prediction(
                case_id="case-001",
                invocations=(
                    _invocation(
                        sequence=1,
                        status="validation_failed",
                        latency_ms=10,
                        include_usage=False,
                        error_code="llm_output_validation_failed",
                    ),
                    _invocation(
                        sequence=2,
                        status="succeeded",
                        latency_ms=30,
                        input_tokens=50,
                        output_tokens=10,
                        total_tokens=60,
                    ),
                ),
            ),
        ),
    )

    report = evaluate_ticket_classification_predictions(
        dataset=dataset,
        predictions=predictions,
    )

    assert report.latency.known_latency_count == 1
    assert report.latency.unknown_latency_count == 0
    assert report.latency.average_latency_ms == Decimal("40.000000")


def test_token_averages_use_known_values_and_derive_total(
    tmp_path: Path,
) -> None:
    dataset, predictions = _load_artifacts(
        tmp_path,
        dataset_cases=(
            _dataset_case(case_id="case-001"),
            _dataset_case(case_id="case-002"),
        ),
        predictions=(
            _success_prediction(
                case_id="case-001",
                input_tokens=100,
                output_tokens=20,
                total_tokens=120,
            ),
            _success_prediction(
                case_id="case-002",
                input_tokens=50,
                output_tokens=10,
                total_tokens=None,
            ),
        ),
    )

    report = evaluate_ticket_classification_predictions(
        dataset=dataset,
        predictions=predictions,
    )

    assert report.token_averages.average_input_tokens == Decimal("75.000000")
    assert report.token_averages.average_output_tokens == Decimal("15.000000")
    assert report.token_averages.average_total_tokens == Decimal("90.000000")
    assert report.token_averages.known_input_token_count == 2
    assert report.token_averages.known_output_token_count == 2
    assert report.token_averages.known_total_token_count == 2
    assert report.token_averages.unknown_input_token_count == 0
    assert report.token_averages.unknown_output_token_count == 0
    assert report.token_averages.unknown_total_token_count == 0
    assert report.known_total_tokens == 120
    assert report.unknown_usage_count == 1


def test_partial_usage_remains_unknown_for_missing_dimensions(
    tmp_path: Path,
) -> None:
    dataset, predictions = _load_artifacts(
        tmp_path,
        dataset_cases=(_dataset_case(case_id="case-001"),),
        predictions=(
            _success_prediction(
                case_id="case-001",
                input_tokens=80,
                output_tokens=None,
                total_tokens=None,
            ),
        ),
    )

    report = evaluate_ticket_classification_predictions(
        dataset=dataset,
        predictions=predictions,
    )

    assert report.token_averages.average_input_tokens == Decimal("80.000000")
    assert report.token_averages.known_input_token_count == 1
    assert report.token_averages.unknown_output_token_count == 1
    assert report.token_averages.unknown_total_token_count == 1
    assert report.token_averages.average_output_tokens == Decimal("0.000000")
    assert report.token_averages.average_total_tokens == Decimal("0.000000")


def test_failed_prediction_usage_is_retained_without_double_counting(
    tmp_path: Path,
) -> None:
    dataset, predictions = _load_artifacts(
        tmp_path,
        dataset_cases=(_dataset_case(case_id="case-001"),),
        predictions=(
            _failure_prediction(
                case_id="case-001",
                invocations=(
                    _invocation(
                        sequence=1,
                        status="timed_out",
                        input_tokens=40,
                        output_tokens=5,
                        total_tokens=45,
                        latency_ms=9000,
                        error_code="llm_timeout",
                    ),
                ),
            ),
        ),
    )

    report = evaluate_ticket_classification_predictions(
        dataset=dataset,
        predictions=predictions,
    )

    assert report.token_averages.known_input_token_count == 1
    assert report.token_averages.known_output_token_count == 1
    assert report.token_averages.known_total_token_count == 1
    assert report.token_averages.average_input_tokens == Decimal("40.000000")
    assert report.token_averages.average_output_tokens == Decimal("5.000000")
    assert report.token_averages.average_total_tokens == Decimal("45.000000")
    assert report.known_total_tokens == 45
    assert report.structured_output_validity.valid_count == 0


def test_new_metric_fields_affect_report_hash(
    tmp_path: Path,
) -> None:
    valid_dataset_path = tmp_path / "valid-dataset.jsonl"
    valid_predictions_path = tmp_path / "valid-predictions.jsonl"
    invalid_dataset_path = tmp_path / "invalid-dataset.jsonl"
    invalid_predictions_path = tmp_path / "invalid-predictions.jsonl"

    _write_jsonl(
        valid_dataset_path,
        _dataset_case(
            case_id="case-001",
            urgency="high",
        ),
    )
    _write_jsonl(
        valid_predictions_path,
        _success_prediction(
            case_id="case-001",
            urgency="high",
        ),
    )
    _write_jsonl(
        invalid_dataset_path,
        _dataset_case(
            case_id="case-001",
            urgency="high",
        ),
    )
    _write_jsonl(
        invalid_predictions_path,
        _failure_prediction(
            case_id="case-001",
        ),
    )

    valid_dataset = load_ticket_classification_dataset(
        valid_dataset_path,
        dataset_id="ticket-classification-eval",
        version=1,
    )
    valid_predictions = load_ticket_classification_predictions(
        valid_predictions_path,
    )
    invalid_dataset = load_ticket_classification_dataset(
        invalid_dataset_path,
        dataset_id="ticket-classification-eval",
        version=1,
    )
    invalid_predictions = load_ticket_classification_predictions(
        invalid_predictions_path,
    )

    valid_report = evaluate_ticket_classification_predictions(
        dataset=valid_dataset,
        predictions=valid_predictions,
    )
    invalid_report = evaluate_ticket_classification_predictions(
        dataset=invalid_dataset,
        predictions=invalid_predictions,
    )

    assert valid_report.structured_output_validity.valid_count == 1
    assert invalid_report.structured_output_validity.valid_count == 0
    assert valid_report.report_content_hash != (invalid_report.report_content_hash)
    assert valid_report.known_estimated_total_cost_usd == Decimal("0")
    assert isinstance(
        valid_report.known_estimated_total_cost_usd,
        Decimal,
    )


def _gate_by_id(
    evaluation: TicketClassificationReleaseGateEvaluation,
    gate_id: str,
) -> TicketClassificationReleaseGateResult:
    for result in evaluation.gate_results:
        if result.gate_id == gate_id:
            return result

    raise AssertionError(f"Missing gate result: {gate_id}")


def _absolute_safety_reliability_profile() -> TicketClassificationReleaseGateProfile:
    return TicketClassificationReleaseGateProfile(
        profile_id="ticket-classification-release-gates-applicable",
        profile_version=1,
        gates=(
            TicketClassificationReleaseGateDefinition(
                gate_id="classification.structured-output-validity",
                category=TicketClassificationGateCategory.SAFETY,
                blocking=True,
                metric_name="structured_output_validity.rate",
                operator=TicketClassificationGateOperator.EQUAL,
                threshold_value=Decimal("1.000000"),
            ),
            TicketClassificationReleaseGateDefinition(
                gate_id="classification.critical-urgency-recall",
                category=TicketClassificationGateCategory.SAFETY,
                blocking=True,
                metric_name="critical_urgency_recall.recall",
                operator=TicketClassificationGateOperator.EQUAL,
                threshold_value=Decimal("1.000000"),
            ),
            TicketClassificationReleaseGateDefinition(
                gate_id=("classification.high-risk-human-review-recall"),
                category=TicketClassificationGateCategory.SAFETY,
                blocking=True,
                metric_name=("high_risk_human_review_recall.recall"),
                operator=TicketClassificationGateOperator.EQUAL,
                threshold_value=Decimal("1.000000"),
            ),
            TicketClassificationReleaseGateDefinition(
                gate_id="classification.prediction-coverage",
                category=TicketClassificationGateCategory.RELIABILITY,
                blocking=True,
                metric_name="prediction_artifact_coverage.rate",
                operator=TicketClassificationGateOperator.EQUAL,
                threshold_value=Decimal("1.000000"),
            ),
            TicketClassificationReleaseGateDefinition(
                gate_id=("classification.deterministic-evaluator-failures"),
                category=TicketClassificationGateCategory.RELIABILITY,
                blocking=True,
                metric_name=("deterministic_evaluator_failure_count"),
                operator=TicketClassificationGateOperator.EQUAL,
                threshold_value=0,
            ),
        ),
    )


def test_default_release_gate_profile_identity() -> None:
    profile = DEFAULT_TICKET_CLASSIFICATION_RELEASE_GATE_PROFILE

    assert profile.profile_id == ("ticket-classification-release-gates")
    assert profile.profile_version == 1
    assert [gate.gate_id for gate in profile.gates] == [
        "classification.structured-output-validity",
        "classification.critical-urgency-recall",
        "classification.high-risk-human-review-recall",
        "classification.prediction-coverage",
        "classification.deterministic-evaluator-failures",
        "classification.structured-label-non-regression",
        "classification.category-accuracy-non-regression",
        "classification.target-metric-improvement",
        "classification.mean-token-increase",
        "classification.mean-cost-increase",
        "classification.mean-latency-increase",
    ]


def test_release_gate_profile_rejects_duplicate_gate_ids() -> None:
    gate = TicketClassificationReleaseGateDefinition(
        gate_id="classification.structured-output-validity",
        category=TicketClassificationGateCategory.SAFETY,
        blocking=True,
        metric_name="structured_output_validity.rate",
        operator=TicketClassificationGateOperator.EQUAL,
        threshold_value=Decimal("1.000000"),
    )

    with pytest.raises(
        ValidationError,
        match="duplicate gate IDs",
    ):
        TicketClassificationReleaseGateProfile(
            profile_id="ticket-classification-release-gates",
            profile_version=1,
            gates=(gate, gate),
        )


def test_release_gate_profile_rejects_unknown_operator() -> None:
    with pytest.raises(ValidationError):
        TicketClassificationReleaseGateDefinition.model_validate(
            {
                "gate_id": "classification.structured-output-validity",
                "category": "safety",
                "blocking": True,
                "metric_name": "structured_output_validity.rate",
                "operator": "gte",
                "threshold_value": "1.000000",
            },
        )


def test_release_gate_profile_is_immutable() -> None:
    profile = DEFAULT_TICKET_CLASSIFICATION_RELEASE_GATE_PROFILE

    with pytest.raises(ValidationError):
        profile.profile_id = "mutated"  # type: ignore[misc]


def test_perfect_safety_and_reliability_are_incomplete_standalone(
    tmp_path: Path,
) -> None:
    dataset, predictions = _load_artifacts(
        tmp_path,
        dataset_cases=(
            _dataset_case(
                case_id="case-001",
                urgency="critical",
                requires_human_review=True,
                tags=("privacy", "evaluation"),
            ),
            _dataset_case(
                case_id="case-002",
            ),
        ),
        predictions=(
            _success_prediction(
                case_id="case-001",
                urgency="critical",
                requires_human_review=True,
            ),
            _success_prediction(
                case_id="case-002",
            ),
        ),
    )
    report = evaluate_ticket_classification_predictions(
        dataset=dataset,
        predictions=predictions,
    )

    evaluation = evaluate_ticket_classification_release_gates(
        report,
    )

    assert evaluation.profile_id == ("ticket-classification-release-gates")
    assert evaluation.profile_version == 1
    assert evaluation.report_content_hash == (report.report_content_hash)
    assert evaluation.blocking_failure_count == 0
    assert evaluation.not_applicable_count == 6
    assert evaluation.standalone_gate_status is (
        TicketClassificationStandaloneGateStatus.INCOMPLETE
    )
    assert (
        _gate_by_id(
            evaluation,
            "classification.structured-output-validity",
        ).outcome
        is TicketClassificationGateOutcome.PASSED
    )
    assert (
        _gate_by_id(
            evaluation,
            "classification.critical-urgency-recall",
        ).outcome
        is TicketClassificationGateOutcome.PASSED
    )
    assert (
        _gate_by_id(
            evaluation,
            "classification.high-risk-human-review-recall",
        ).outcome
        is TicketClassificationGateOutcome.PASSED
    )
    assert (
        _gate_by_id(
            evaluation,
            "classification.prediction-coverage",
        ).outcome
        is TicketClassificationGateOutcome.PASSED
    )
    assert (
        _gate_by_id(
            evaluation,
            "classification.deterministic-evaluator-failures",
        ).outcome
        is TicketClassificationGateOutcome.PASSED
    )


def test_invalid_structured_output_fails_validity_gate(
    tmp_path: Path,
) -> None:
    dataset, predictions = _load_artifacts(
        tmp_path,
        dataset_cases=(
            _dataset_case(case_id="case-001"),
            _dataset_case(case_id="case-002"),
        ),
        predictions=(
            _success_prediction(case_id="case-001"),
            _failure_prediction(case_id="case-002"),
        ),
    )
    report = evaluate_ticket_classification_predictions(
        dataset=dataset,
        predictions=predictions,
    )

    evaluation = evaluate_ticket_classification_release_gates(
        report,
    )

    validity = _gate_by_id(
        evaluation,
        "classification.structured-output-validity",
    )
    assert validity.outcome is TicketClassificationGateOutcome.FAILED
    assert validity.actual_value == Decimal("0.500000")
    assert evaluation.standalone_gate_status is (TicketClassificationStandaloneGateStatus.FAILED)
    assert evaluation.blocking_failure_count >= 1


def test_missed_critical_urgency_fails_recall_gate(
    tmp_path: Path,
) -> None:
    dataset, predictions = _load_artifacts(
        tmp_path,
        dataset_cases=(
            _dataset_case(
                case_id="case-001",
                urgency="critical",
            ),
        ),
        predictions=(
            _success_prediction(
                case_id="case-001",
                urgency="high",
            ),
        ),
    )
    report = evaluate_ticket_classification_predictions(
        dataset=dataset,
        predictions=predictions,
    )

    evaluation = evaluate_ticket_classification_release_gates(
        report,
    )
    critical = _gate_by_id(
        evaluation,
        "classification.critical-urgency-recall",
    )

    assert critical.outcome is TicketClassificationGateOutcome.FAILED
    assert critical.actual_value == Decimal("0.000000")
    assert evaluation.standalone_gate_status is (TicketClassificationStandaloneGateStatus.FAILED)


def test_zero_critical_denominator_is_not_applicable(
    tmp_path: Path,
) -> None:
    dataset, predictions = _load_artifacts(
        tmp_path,
        dataset_cases=(
            _dataset_case(
                case_id="case-001",
                urgency="normal",
            ),
        ),
        predictions=(
            _success_prediction(
                case_id="case-001",
                urgency="normal",
            ),
        ),
    )
    report = evaluate_ticket_classification_predictions(
        dataset=dataset,
        predictions=predictions,
    )

    evaluation = evaluate_ticket_classification_release_gates(
        report,
    )
    critical = _gate_by_id(
        evaluation,
        "classification.critical-urgency-recall",
    )

    assert critical.outcome is (TicketClassificationGateOutcome.NOT_APPLICABLE)
    assert critical.actual_value is None
    assert "expected_positive_count is zero" in critical.reason


def test_missed_high_risk_review_fails_recall_gate(
    tmp_path: Path,
) -> None:
    dataset, predictions = _load_artifacts(
        tmp_path,
        dataset_cases=(
            _dataset_case(
                case_id="case-001",
                requires_human_review=True,
                tags=("privacy",),
            ),
        ),
        predictions=(
            _success_prediction(
                case_id="case-001",
                requires_human_review=False,
            ),
        ),
    )
    report = evaluate_ticket_classification_predictions(
        dataset=dataset,
        predictions=predictions,
    )

    evaluation = evaluate_ticket_classification_release_gates(
        report,
    )
    high_risk = _gate_by_id(
        evaluation,
        "classification.high-risk-human-review-recall",
    )

    assert high_risk.outcome is TicketClassificationGateOutcome.FAILED
    assert high_risk.actual_value == Decimal("0.000000")


def test_zero_high_risk_denominator_is_not_applicable(
    tmp_path: Path,
) -> None:
    dataset, predictions = _load_artifacts(
        tmp_path,
        dataset_cases=(
            _dataset_case(
                case_id="case-001",
                requires_human_review=False,
            ),
        ),
        predictions=(
            _success_prediction(
                case_id="case-001",
                requires_human_review=False,
            ),
        ),
    )
    report = evaluate_ticket_classification_predictions(
        dataset=dataset,
        predictions=predictions,
    )

    evaluation = evaluate_ticket_classification_release_gates(
        report,
    )
    high_risk = _gate_by_id(
        evaluation,
        "classification.high-risk-human-review-recall",
    )

    assert high_risk.outcome is (TicketClassificationGateOutcome.NOT_APPLICABLE)
    assert high_risk.actual_value is None
    assert "expected_positive_count is zero" in high_risk.reason


def test_failed_prediction_still_counts_as_coverage(
    tmp_path: Path,
) -> None:
    dataset, predictions = _load_artifacts(
        tmp_path,
        dataset_cases=(
            _dataset_case(case_id="case-001"),
            _dataset_case(case_id="case-002"),
        ),
        predictions=(
            _success_prediction(case_id="case-001"),
            _failure_prediction(case_id="case-002"),
        ),
    )
    report = evaluate_ticket_classification_predictions(
        dataset=dataset,
        predictions=predictions,
    )

    evaluation = evaluate_ticket_classification_release_gates(
        report,
    )
    coverage = _gate_by_id(
        evaluation,
        "classification.prediction-coverage",
    )

    assert coverage.outcome is TicketClassificationGateOutcome.PASSED
    assert coverage.actual_value == Decimal("1.000000")


def test_missing_prediction_fails_coverage_gate(
    tmp_path: Path,
) -> None:
    dataset, predictions = _load_artifacts(
        tmp_path,
        dataset_cases=(
            _dataset_case(case_id="case-001"),
            _dataset_case(case_id="case-002"),
        ),
        predictions=(_success_prediction(case_id="case-001"),),
    )
    report = evaluate_ticket_classification_predictions(
        dataset=dataset,
        predictions=predictions,
    )

    evaluation = evaluate_ticket_classification_release_gates(
        report,
    )
    coverage = _gate_by_id(
        evaluation,
        "classification.prediction-coverage",
    )

    assert coverage.outcome is TicketClassificationGateOutcome.FAILED
    assert coverage.actual_value == Decimal("0.500000")


def test_quality_and_efficiency_gates_are_not_applicable_standalone(
    tmp_path: Path,
) -> None:
    dataset, predictions = _load_artifacts(
        tmp_path,
        dataset_cases=(
            _dataset_case(
                case_id="case-001",
                urgency="critical",
                requires_human_review=True,
                tags=("privacy",),
            ),
        ),
        predictions=(
            _success_prediction(
                case_id="case-001",
                urgency="critical",
                requires_human_review=True,
            ),
        ),
    )
    report = evaluate_ticket_classification_predictions(
        dataset=dataset,
        predictions=predictions,
    )

    evaluation = evaluate_ticket_classification_release_gates(
        report,
    )
    paired_gate_ids = (
        "classification.structured-label-non-regression",
        "classification.category-accuracy-non-regression",
        "classification.target-metric-improvement",
        "classification.mean-token-increase",
        "classification.mean-cost-increase",
        "classification.mean-latency-increase",
    )

    for gate_id in paired_gate_ids:
        result = _gate_by_id(evaluation, gate_id)
        assert result.outcome is (TicketClassificationGateOutcome.NOT_APPLICABLE)
        assert result.blocking is True
        assert "paired baseline" in result.reason


def test_applicable_custom_profile_can_pass(
    tmp_path: Path,
) -> None:
    dataset, predictions = _load_artifacts(
        tmp_path,
        dataset_cases=(
            _dataset_case(
                case_id="case-001",
                urgency="critical",
                requires_human_review=True,
                tags=("privacy",),
            ),
        ),
        predictions=(
            _success_prediction(
                case_id="case-001",
                urgency="critical",
                requires_human_review=True,
            ),
        ),
    )
    report = evaluate_ticket_classification_predictions(
        dataset=dataset,
        predictions=predictions,
    )

    evaluation = evaluate_ticket_classification_release_gates(
        report,
        profile=_absolute_safety_reliability_profile(),
    )

    assert evaluation.blocking_failure_count == 0
    assert evaluation.not_applicable_count == 0
    assert evaluation.standalone_gate_status is (TicketClassificationStandaloneGateStatus.PASSED)
    assert all(
        result.outcome is TicketClassificationGateOutcome.PASSED
        for result in evaluation.gate_results
    )


def test_gate_evaluation_hash_is_deterministic_and_sensitive(
    tmp_path: Path,
) -> None:
    perfect_dir = tmp_path / "perfect"
    failed_dir = tmp_path / "failed"
    perfect_dir.mkdir()
    failed_dir.mkdir()

    perfect_dataset, perfect_predictions = _load_artifacts(
        perfect_dir,
        dataset_cases=(
            _dataset_case(
                case_id="case-001",
                urgency="critical",
                requires_human_review=True,
                tags=("privacy",),
            ),
        ),
        predictions=(
            _success_prediction(
                case_id="case-001",
                urgency="critical",
                requires_human_review=True,
            ),
        ),
    )
    failed_dataset, failed_predictions = _load_artifacts(
        failed_dir,
        dataset_cases=(
            _dataset_case(
                case_id="case-001",
                urgency="critical",
                requires_human_review=True,
                tags=("privacy",),
            ),
        ),
        predictions=(
            _success_prediction(
                case_id="case-001",
                urgency="high",
                requires_human_review=True,
            ),
        ),
    )

    perfect_report = evaluate_ticket_classification_predictions(
        dataset=perfect_dataset,
        predictions=perfect_predictions,
    )
    failed_report = evaluate_ticket_classification_predictions(
        dataset=failed_dataset,
        predictions=failed_predictions,
    )

    first = evaluate_ticket_classification_release_gates(
        perfect_report,
    )
    second = evaluate_ticket_classification_release_gates(
        perfect_report,
    )
    changed = evaluate_ticket_classification_release_gates(
        failed_report,
    )

    assert first == second
    assert first.content_hash == second.content_hash
    assert first.content_hash != changed.content_hash
    assert first.report_content_hash != (changed.report_content_hash)
    assert len(first.content_hash) == 64


def test_report_hash_unchanged_when_gates_not_invoked(
    tmp_path: Path,
) -> None:
    dataset, predictions = _load_artifacts(
        tmp_path,
        dataset_cases=(_dataset_case(case_id="case-001"),),
        predictions=(_success_prediction(case_id="case-001"),),
    )

    report = evaluate_ticket_classification_predictions(
        dataset=dataset,
        predictions=predictions,
    )
    report_again = evaluate_ticket_classification_predictions(
        dataset=dataset,
        predictions=predictions,
    )

    assert report.report_content_hash == (report_again.report_content_hash)
    evaluate_ticket_classification_release_gates(report)
    assert report.report_content_hash == (report_again.report_content_hash)
