"""Unit tests for sequential evaluation execution and artifacts."""

import json
from pathlib import Path
from typing import Any

import pytest

from supportops.ai.gateway.contracts import LLMTokenUsage
from supportops.ai.gateway.service import LLMGateway
from supportops.ai.prompts.registry import (
    PromptDefinitionNotFoundError,
)
from supportops.ai.prompts.ticket_classification_v1 import (
    TICKET_CLASSIFICATION_PROMPT_V1,
)
from supportops.ai.providers.mock import (
    MockLLMOutcome,
    MockLLMProvider,
)
from supportops.evaluation.ticket_classification.dataset import (
    load_ticket_classification_dataset,
)
from supportops.evaluation.ticket_classification.predictions import (
    TicketClassificationEvaluationPrediction,
    load_ticket_classification_predictions,
)
from supportops.evaluation.ticket_classification.predictor import (
    TicketClassificationEvaluationPredictor,
)
from supportops.evaluation.ticket_classification.runner import (
    evaluate_ticket_classification_report_release_gates,
    run_ticket_classification_evaluation,
    score_ticket_classification_predictions_with_release_gates,
    write_ticket_classification_evaluation_report,
    write_ticket_classification_predictions,
)

_PINNED_PROMPT_V1_HASH = "3c9107f8685232da86442e63a551cd991d0d8fc174f480a6b0c8ead3afc85da2"


def _dataset_payload(
    *,
    case_id: str,
    category: str,
) -> dict[str, object]:
    return {
        "case_id": case_id,
        "tags": [
            "evaluation",
        ],
        "ticket": {
            "subject": f"Synthetic {case_id}",
            "description": ("Synthetic evaluation ticket description."),
        },
        "expected": {
            "category": category,
            "intent": "ask_question",
            "urgency": "normal",
            "sentiment": "neutral",
            "requires_human_review": False,
            "schema_version": ("ticket-classification-v1"),
        },
    }


def _success_outcome(
    *,
    category: str,
) -> MockLLMOutcome:
    return MockLLMOutcome.success(
        {
            "category": category,
            "intent": "ask_question",
            "urgency": "normal",
            "sentiment": "neutral",
            "requires_human_review": False,
            "summary": "Synthetic evaluation summary.",
            "schema_version": ("ticket-classification-v1"),
        },
        usage=LLMTokenUsage(
            input_tokens=100,
            cached_input_tokens=0,
            output_tokens=20,
            reasoning_tokens=None,
            total_tokens=120,
        ),
    )


def _write_dataset(
    path: Path,
    *payloads: dict[str, object],
) -> None:
    path.write_text(
        "\n".join(json.dumps(payload) for payload in payloads) + "\n",
        encoding="utf-8",
    )


async def test_runner_processes_dataset_sequentially(
    tmp_path: Path,
) -> None:
    dataset_path = tmp_path / "dataset.jsonl"
    _write_dataset(
        dataset_path,
        _dataset_payload(
            case_id="billing-case-001",
            category="billing",
        ),
        _dataset_payload(
            case_id="security-case-002",
            category="security",
        ),
    )
    dataset = load_ticket_classification_dataset(
        dataset_path,
        dataset_id="ticket-classification-eval",
        version=1,
    )
    provider = MockLLMProvider.with_strict_outcomes(
        (
            _success_outcome(
                category="billing",
            ),
            _success_outcome(
                category="security",
            ),
        ),
    )
    predictor = TicketClassificationEvaluationPredictor(
        gateway=LLMGateway(
            provider=provider,
            max_repair_attempts=1,
        ),
        provider_name=provider.provider_name,
        model=provider.model,
        request_timeout_seconds=12,
    )

    try:
        result = await run_ticket_classification_evaluation(
            dataset=dataset,
            predictor=predictor,
            prompt_version=1,
        )
    finally:
        await provider.close()

    assert provider.invocation_count == 2
    assert [prediction.case_id for prediction in result.predictions.predictions] == [
        "billing-case-001",
        "security-case-002",
    ]
    assert result.report.case_count == 2
    assert result.report.successful_prediction_count == 2
    assert result.report.failed_prediction_count == 0
    assert result.report.structured_label_exact_match.match_count == 2
    assert result.report.known_total_tokens == 240
    assert all(
        prediction.provenance.prompt_version == 1
        and prediction.provenance.prompt_content_hash == (_PINNED_PROMPT_V1_HASH)
        for prediction in result.predictions.predictions
    )


async def test_runner_continues_after_case_failure(
    tmp_path: Path,
) -> None:
    dataset_path = tmp_path / "dataset.jsonl"
    _write_dataset(
        dataset_path,
        _dataset_payload(
            case_id="billing-case-001",
            category="billing",
        ),
        _dataset_payload(
            case_id="security-case-002",
            category="security",
        ),
    )
    dataset = load_ticket_classification_dataset(
        dataset_path,
        dataset_id="ticket-classification-eval",
        version=1,
    )
    provider = MockLLMProvider.with_strict_outcomes(
        (
            MockLLMOutcome.timeout(),
            _success_outcome(
                category="security",
            ),
        ),
    )
    predictor = TicketClassificationEvaluationPredictor(
        gateway=LLMGateway(
            provider=provider,
            max_repair_attempts=0,
        ),
        provider_name=provider.provider_name,
        model=provider.model,
        request_timeout_seconds=12,
    )

    try:
        result = await run_ticket_classification_evaluation(
            dataset=dataset,
            predictor=predictor,
            prompt_version=1,
        )
    finally:
        await provider.close()

    assert provider.invocation_count == 2
    assert result.report.successful_prediction_count == 1
    assert result.report.failed_prediction_count == 1
    assert result.report.failure_counts_by_error_code == {
        "llm_timeout": 1,
    }
    assert [case.prediction_status for case in result.report.cases] == [
        "failed",
        "succeeded",
    ]


async def test_artifact_writers_are_reloadable(
    tmp_path: Path,
) -> None:
    dataset_path = tmp_path / "dataset.jsonl"
    predictions_path = tmp_path / "artifacts" / "predictions.jsonl"
    report_path = tmp_path / "artifacts" / "report.json"
    _write_dataset(
        dataset_path,
        _dataset_payload(
            case_id="billing-case-001",
            category="billing",
        ),
    )
    dataset = load_ticket_classification_dataset(
        dataset_path,
        dataset_id="ticket-classification-eval",
        version=1,
    )
    provider = MockLLMProvider.with_strict_outcomes(
        (
            _success_outcome(
                category="billing",
            ),
        ),
    )
    predictor = TicketClassificationEvaluationPredictor(
        gateway=LLMGateway(
            provider=provider,
            max_repair_attempts=1,
        ),
        provider_name=provider.provider_name,
        model=provider.model,
        request_timeout_seconds=12,
    )

    try:
        result = await run_ticket_classification_evaluation(
            dataset=dataset,
            predictor=predictor,
            prompt_version=1,
        )
    finally:
        await provider.close()

    write_ticket_classification_predictions(
        predictions_path,
        result.predictions,
    )
    write_ticket_classification_evaluation_report(
        report_path,
        result.report,
    )

    reloaded_predictions = load_ticket_classification_predictions(
        predictions_path,
    )
    report_payload = json.loads(
        report_path.read_text(
            encoding="utf-8",
        ),
    )

    assert reloaded_predictions.content_hash == result.predictions.content_hash
    assert report_payload["dataset_id"] == ("ticket-classification-eval")
    assert report_payload["case_count"] == 1
    assert report_payload["report_content_hash"] == (result.report.report_content_hash)
    assert predictions_path.read_text(
        encoding="utf-8",
    ).endswith("\n")
    assert report_path.read_text(
        encoding="utf-8",
    ).endswith("\n")


async def test_runner_propagates_selected_prompt_version(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset_path = tmp_path / "dataset.jsonl"
    _write_dataset(
        dataset_path,
        _dataset_payload(
            case_id="billing-case-001",
            category="billing",
        ),
    )
    dataset = load_ticket_classification_dataset(
        dataset_path,
        dataset_id="ticket-classification-eval",
        version=1,
    )
    provider = MockLLMProvider.with_strict_outcomes(
        (
            _success_outcome(
                category="billing",
            ),
        ),
    )
    predictor = TicketClassificationEvaluationPredictor(
        gateway=LLMGateway(
            provider=provider,
            max_repair_attempts=1,
        ),
        provider_name=provider.provider_name,
        model=provider.model,
        request_timeout_seconds=12,
    )
    captured_versions: list[int] = []
    original_predict = predictor.predict

    async def _capture_predict(
        **kwargs: Any,
    ) -> TicketClassificationEvaluationPrediction:
        captured_versions.append(
            kwargs["prompt_version"],
        )
        return await original_predict(**kwargs)

    monkeypatch.setattr(
        predictor,
        "predict",
        _capture_predict,
    )

    try:
        result = await run_ticket_classification_evaluation(
            dataset=dataset,
            predictor=predictor,
            prompt_version=1,
        )
    finally:
        await provider.close()

    assert captured_versions == [1]
    assert result.predictions.predictions[0].provenance.prompt_version == 1
    assert result.predictions.predictions[0].provenance.prompt_content_hash == (
        TICKET_CLASSIFICATION_PROMPT_V1.content_hash
    )


async def test_unsupported_prompt_version_leaves_output_untouched(
    tmp_path: Path,
) -> None:
    dataset_path = tmp_path / "dataset.jsonl"
    predictions_path = tmp_path / "artifacts" / "predictions.jsonl"
    existing_content = '{"preserved":true}\n'
    predictions_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    predictions_path.write_text(
        existing_content,
        encoding="utf-8",
    )
    _write_dataset(
        dataset_path,
        _dataset_payload(
            case_id="billing-case-001",
            category="billing",
        ),
    )
    dataset = load_ticket_classification_dataset(
        dataset_path,
        dataset_id="ticket-classification-eval",
        version=1,
    )
    provider = MockLLMProvider.with_strict_outcomes(
        (
            _success_outcome(
                category="billing",
            ),
        ),
    )
    predictor = TicketClassificationEvaluationPredictor(
        gateway=LLMGateway(
            provider=provider,
            max_repair_attempts=1,
        ),
        provider_name=provider.provider_name,
        model=provider.model,
        request_timeout_seconds=12,
    )

    try:
        with pytest.raises(
            PromptDefinitionNotFoundError,
            match=("Unsupported prompt: ticket-classification version 3"),
        ):
            await run_ticket_classification_evaluation(
                dataset=dataset,
                predictor=predictor,
                prompt_version=3,
            )
    finally:
        await provider.close()

    assert provider.invocation_count == 0
    assert (
        predictions_path.read_text(
            encoding="utf-8",
        )
        == existing_content
    )


def test_score_with_release_gates_preserves_report_and_returns_gates(
    tmp_path: Path,
) -> None:
    from supportops.evaluation.ticket_classification.dataset import (
        load_ticket_classification_dataset,
    )
    from supportops.evaluation.ticket_classification.evaluator import (
        TicketClassificationStandaloneGateStatus,
        evaluate_ticket_classification_predictions,
    )
    from supportops.evaluation.ticket_classification.predictions import (
        load_ticket_classification_predictions,
    )

    dataset_path = tmp_path / "dataset.jsonl"
    predictions_path = tmp_path / "predictions.jsonl"
    dataset_payload = _dataset_payload(
        case_id="billing-case-001",
        category="billing",
    )
    prediction_payload = {
        "case_id": "billing-case-001",
        "status": "succeeded",
        "provenance": {
            "prompt_id": "ticket-classification",
            "prompt_version": 1,
            "prompt_content_hash": _PINNED_PROMPT_V1_HASH,
            "provider": "mock",
            "model": "test-model",
        },
        "output": {
            "category": "billing",
            "intent": "ask_question",
            "urgency": "normal",
            "sentiment": "neutral",
            "requires_human_review": False,
            "summary": "Synthetic evaluation summary.",
            "schema_version": ("ticket-classification-v1"),
        },
        "invocations": [
            {
                "invocation_sequence": 1,
                "status": "succeeded",
                "provider": "mock",
                "model": "test-model",
                "usage": {
                    "input_tokens": 100,
                    "cached_input_tokens": 0,
                    "output_tokens": 20,
                    "reasoning_tokens": None,
                    "total_tokens": 120,
                },
                "cost": {
                    "pricing_catalog_version": "pricing-v1",
                    "pricing_found": True,
                    "estimated_input_cost_usd": "0",
                    "estimated_cached_input_cost_usd": "0",
                    "estimated_output_cost_usd": "0",
                    "estimated_total_cost_usd": "0",
                },
                "latency_ms": 25,
                "error_code": None,
            },
        ],
    }
    _write_dataset(
        dataset_path,
        dataset_payload,
    )
    predictions_path.write_text(
        json.dumps(prediction_payload) + "\n",
        encoding="utf-8",
    )
    dataset = load_ticket_classification_dataset(
        dataset_path,
        dataset_id="ticket-classification-eval",
        version=1,
    )
    predictions = load_ticket_classification_predictions(
        predictions_path,
    )

    baseline_report = evaluate_ticket_classification_predictions(
        dataset=dataset,
        predictions=predictions,
    )
    report, gate_evaluation = score_ticket_classification_predictions_with_release_gates(
        dataset=dataset,
        predictions=predictions,
    )
    gate_from_report = evaluate_ticket_classification_report_release_gates(
        report,
    )

    assert report == baseline_report
    assert report.report_content_hash == (baseline_report.report_content_hash)
    assert gate_evaluation == gate_from_report
    assert gate_evaluation.report_content_hash == (report.report_content_hash)
    assert gate_evaluation.standalone_gate_status is (
        TicketClassificationStandaloneGateStatus.INCOMPLETE
    )
    assert gate_evaluation.blocking_failure_count == 0
    assert gate_evaluation.not_applicable_count == 8
