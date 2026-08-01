"""Unit tests for the classification evaluation CLI."""

import json
from io import StringIO
from pathlib import Path
from typing import NoReturn

from supportops.evaluation.ticket_classification.cli import (
    run_cli,
)
from supportops.evaluation.ticket_classification.settings import (
    TicketClassificationEvaluationSettings,
)

_PROMPT_HASH = "a" * 64


def _dataset_payload() -> dict[str, object]:
    return {
        "case_id": "other-case-001",
        "tags": [
            "other",
            "evaluation",
        ],
        "ticket": {
            "subject": "General sales question",
            "description": ("Who can explain the enterprise pricing options?"),
        },
        "expected": {
            "category": "other",
            "intent": "other",
            "urgency": "normal",
            "sentiment": "neutral",
            "requires_human_review": False,
            "schema_version": ("ticket-classification-v1"),
        },
    }


def _prediction_payload() -> dict[str, object]:
    return {
        "case_id": "other-case-001",
        "status": "succeeded",
        "provenance": {
            "prompt_id": "ticket-classification",
            "prompt_version": 1,
            "prompt_content_hash": _PROMPT_HASH,
            "provider": "mock",
            "model": "mock-ticket-classifier-v1",
        },
        "output": {
            "category": "other",
            "intent": "other",
            "urgency": "normal",
            "sentiment": "neutral",
            "requires_human_review": False,
            "summary": ("The customer has a general question."),
            "schema_version": ("ticket-classification-v1"),
        },
        "invocations": [
            {
                "invocation_sequence": 1,
                "status": "succeeded",
                "provider": "mock",
                "model": "mock-ticket-classifier-v1",
                "usage": {
                    "input_tokens": 120,
                    "cached_input_tokens": None,
                    "output_tokens": 24,
                    "reasoning_tokens": None,
                    "total_tokens": 144,
                },
                "cost": {
                    "pricing_catalog_version": "pricing-v1",
                    "pricing_found": True,
                    "estimated_input_cost_usd": "0",
                    "estimated_cached_input_cost_usd": "0",
                    "estimated_output_cost_usd": "0",
                    "estimated_total_cost_usd": "0",
                },
                "latency_ms": 0,
                "error_code": None,
            },
        ],
    }


def _write_jsonl(
    path: Path,
    payload: dict[str, object],
) -> None:
    path.write_text(
        json.dumps(payload) + "\n",
        encoding="utf-8",
    )


def _settings() -> TicketClassificationEvaluationSettings:
    return TicketClassificationEvaluationSettings(
        _env_file=None,
    )


def _forbidden_settings_factory() -> NoReturn:
    raise AssertionError(
        "Offline score must not load provider settings.",
    )


def _forbidden_runtime_factory(
    **_: object,
) -> NoReturn:
    raise AssertionError(
        "This command must not compose an LLM provider.",
    )


async def test_score_is_offline_and_writes_report(
    tmp_path: Path,
) -> None:
    dataset_path = tmp_path / "dataset.jsonl"
    predictions_path = tmp_path / "predictions.jsonl"
    report_path = tmp_path / "report.json"
    stdout = StringIO()
    stderr = StringIO()

    _write_jsonl(
        dataset_path,
        _dataset_payload(),
    )
    _write_jsonl(
        predictions_path,
        _prediction_payload(),
    )

    exit_code = await run_cli(
        (
            "score",
            "--dataset",
            str(dataset_path),
            "--predictions",
            str(predictions_path),
            "--output",
            str(report_path),
        ),
        stdout=stdout,
        stderr=stderr,
        settings_factory=_forbidden_settings_factory,
        runtime_factory=_forbidden_runtime_factory,
    )

    assert exit_code == 0
    assert stderr.getvalue() == ""
    assert report_path.exists()

    report = json.loads(
        report_path.read_text(
            encoding="utf-8",
        ),
    )
    summary = json.loads(
        stdout.getvalue(),
    )

    assert report["case_count"] == 1
    assert report["structured_label_exact_match"]["rate"] == "1.000000"
    assert summary["command"] == "score"
    assert summary["failed_prediction_count"] == 0
    assert summary["report_path"] == str(report_path)


async def test_openai_requires_explicit_external_permission(
    tmp_path: Path,
) -> None:
    dataset_path = tmp_path / "dataset.jsonl"
    _write_jsonl(
        dataset_path,
        _dataset_payload(),
    )
    stderr = StringIO()

    exit_code = await run_cli(
        (
            "run",
            "--provider",
            "openai",
            "--dataset",
            str(dataset_path),
            "--predictions-output",
            str(tmp_path / "predictions.jsonl"),
            "--output",
            str(tmp_path / "report.json"),
        ),
        stdout=StringIO(),
        stderr=stderr,
        settings_factory=_forbidden_settings_factory,
        runtime_factory=_forbidden_runtime_factory,
    )

    assert exit_code == 2
    assert "--allow-external-provider" in (stderr.getvalue())


async def test_external_permission_is_rejected_for_mock(
    tmp_path: Path,
) -> None:
    dataset_path = tmp_path / "dataset.jsonl"
    _write_jsonl(
        dataset_path,
        _dataset_payload(),
    )
    stderr = StringIO()

    exit_code = await run_cli(
        (
            "run",
            "--provider",
            "mock",
            "--allow-external-provider",
            "--dataset",
            str(dataset_path),
            "--predictions-output",
            str(tmp_path / "predictions.jsonl"),
            "--output",
            str(tmp_path / "report.json"),
        ),
        stdout=StringIO(),
        stderr=stderr,
        settings_factory=_forbidden_settings_factory,
        runtime_factory=_forbidden_runtime_factory,
    )

    assert exit_code == 2
    assert "valid only" in stderr.getvalue()


async def test_mock_run_writes_predictions_and_report(
    tmp_path: Path,
) -> None:
    dataset_path = tmp_path / "dataset.jsonl"
    predictions_path = tmp_path / "artifacts" / "predictions.jsonl"
    report_path = tmp_path / "artifacts" / "report.json"
    stdout = StringIO()
    stderr = StringIO()

    _write_jsonl(
        dataset_path,
        _dataset_payload(),
    )

    exit_code = await run_cli(
        (
            "run",
            "--provider",
            "mock",
            "--dataset",
            str(dataset_path),
            "--predictions-output",
            str(predictions_path),
            "--output",
            str(report_path),
        ),
        stdout=stdout,
        stderr=stderr,
        settings_factory=_settings,
    )

    assert exit_code == 0
    assert stderr.getvalue() == ""
    assert predictions_path.exists()
    assert report_path.exists()

    prediction = json.loads(
        predictions_path.read_text(
            encoding="utf-8",
        ),
    )
    report = json.loads(
        report_path.read_text(
            encoding="utf-8",
        ),
    )
    summary = json.loads(
        stdout.getvalue(),
    )

    assert prediction["case_id"] == "other-case-001"
    assert prediction["status"] == "succeeded"
    assert prediction["provenance"]["provider"] == "mock"
    assert "provider_request_id" not in str(prediction)

    assert report["case_count"] == 1
    assert report["successful_prediction_count"] == 1
    assert summary["command"] == "run"
    assert summary["predictions_path"] == (str(predictions_path))


async def test_allowed_openai_still_requires_configured_key(
    tmp_path: Path,
) -> None:
    dataset_path = tmp_path / "dataset.jsonl"
    _write_jsonl(
        dataset_path,
        _dataset_payload(),
    )
    stderr = StringIO()

    exit_code = await run_cli(
        (
            "run",
            "--provider",
            "openai",
            "--allow-external-provider",
            "--dataset",
            str(dataset_path),
            "--predictions-output",
            str(tmp_path / "predictions.jsonl"),
            "--output",
            str(tmp_path / "report.json"),
        ),
        stdout=StringIO(),
        stderr=stderr,
        settings_factory=_settings,
    )

    assert exit_code == 2
    assert "openai_api_key is required" in (stderr.getvalue())


async def test_invalid_dataset_fails_before_runtime_composition(
    tmp_path: Path,
) -> None:
    dataset_path = tmp_path / "invalid.jsonl"
    dataset_path.write_text(
        '{"invalid":true}\n',
        encoding="utf-8",
    )
    stderr = StringIO()

    exit_code = await run_cli(
        (
            "run",
            "--provider",
            "mock",
            "--dataset",
            str(dataset_path),
            "--predictions-output",
            str(tmp_path / "predictions.jsonl"),
            "--output",
            str(tmp_path / "report.json"),
        ),
        stdout=StringIO(),
        stderr=stderr,
        settings_factory=_forbidden_settings_factory,
        runtime_factory=_forbidden_runtime_factory,
    )

    assert exit_code == 2
    assert "does not match" in stderr.getvalue()
