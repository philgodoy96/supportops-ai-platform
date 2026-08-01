"""Unit tests for deterministic classification metrics."""

import json
from decimal import Decimal
from pathlib import Path

import pytest

from supportops.evaluation.ticket_classification.dataset import (
    TicketClassificationEvaluationDataset,
    load_ticket_classification_dataset,
)
from supportops.evaluation.ticket_classification.evaluator import (
    InconsistentTicketClassificationPredictionProvenanceError,
    UnknownTicketClassificationPredictionError,
    evaluate_ticket_classification_predictions,
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
    requires_human_review: bool = False,
) -> dict[str, object]:
    return {
        "case_id": case_id,
        "tags": ["evaluation"],
        "ticket": {
            "subject": "Synthetic ticket",
            "description": "Synthetic evaluation description.",
        },
        "expected": {
            "category": category,
            "intent": "ask_question",
            "urgency": "normal",
            "sentiment": "neutral",
            "requires_human_review": (requires_human_review),
            "schema_version": "ticket-classification-v1",
        },
    }


def _success_prediction(
    *,
    case_id: str,
    category: str = "billing",
    requires_human_review: bool = False,
    provider: str = "mock",
    total_tokens: int | None = 120,
    pricing_found: bool = True,
) -> dict[str, object]:
    usage = (
        {
            "input_tokens": 100,
            "cached_input_tokens": 0,
            "output_tokens": 20,
            "reasoning_tokens": None,
            "total_tokens": total_tokens,
        }
        if total_tokens is not None
        else None
    )
    cost = {
        "pricing_catalog_version": "pricing-v1",
        "pricing_found": pricing_found,
        "estimated_input_cost_usd": ("0" if pricing_found else None),
        "estimated_cached_input_cost_usd": ("0" if pricing_found else None),
        "estimated_output_cost_usd": ("0" if pricing_found else None),
        "estimated_total_cost_usd": ("0" if pricing_found else None),
    }

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
            "urgency": "normal",
            "sentiment": "neutral",
            "requires_human_review": (requires_human_review),
            "summary": "Synthetic summary.",
            "schema_version": "ticket-classification-v1",
        },
        "invocations": [
            {
                "invocation_sequence": 1,
                "status": "succeeded",
                "provider": provider,
                "model": "test-model",
                "usage": usage,
                "cost": cost,
                "latency_ms": 25,
                "error_code": None,
            },
        ],
    }


def _failure_prediction(
    *,
    case_id: str,
) -> dict[str, object]:
    return {
        "case_id": case_id,
        "status": "failed",
        "error_code": "llm_timeout",
        "provenance": {
            "prompt_id": "ticket-classification",
            "prompt_version": 1,
            "prompt_content_hash": _PROMPT_HASH,
            "provider": "mock",
            "model": "test-model",
        },
        "invocations": [
            {
                "invocation_sequence": 1,
                "status": "timed_out",
                "provider": "mock",
                "model": "test-model",
                "usage": None,
                "cost": {
                    "pricing_catalog_version": "pricing-v1",
                    "pricing_found": True,
                    "estimated_input_cost_usd": None,
                    "estimated_cached_input_cost_usd": None,
                    "estimated_output_cost_usd": None,
                    "estimated_total_cost_usd": None,
                },
                "latency_ms": 12000,
                "error_code": "llm_timeout",
            },
        ],
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
