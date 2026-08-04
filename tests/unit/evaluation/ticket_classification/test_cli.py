"""Unit tests for the classification evaluation CLI."""

import json
from io import StringIO
from pathlib import Path
from typing import Any, NoReturn

import pytest

from supportops.evaluation.ticket_classification.cli import (
    build_parser,
    run_cli,
)
from supportops.evaluation.ticket_classification.settings import (
    TicketClassificationEvaluationSettings,
)

_PROMPT_HASH = "a" * 64
_PINNED_PROMPT_V1_HASH = "3c9107f8685232da86442e63a551cd991d0d8fc174f480a6b0c8ead3afc85da2"


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
    assert prediction["provenance"]["prompt_version"] == 1
    assert prediction["provenance"]["prompt_content_hash"] == (_PINNED_PROMPT_V1_HASH)
    assert "provider_request_id" not in str(prediction)

    assert report["case_count"] == 1
    assert report["successful_prediction_count"] == 1
    assert summary["command"] == "run"
    assert summary["predictions_path"] == (str(predictions_path))


async def test_run_defaults_to_prompt_version_one() -> None:
    parser = build_parser()
    arguments = parser.parse_args(
        (
            "run",
            "--provider",
            "mock",
            "--dataset",
            "dataset.jsonl",
            "--predictions-output",
            "predictions.jsonl",
            "--output",
            "report.json",
        ),
    )

    assert arguments.prompt_version == 1


async def test_run_explicit_prompt_version_one_is_accepted() -> None:
    parser = build_parser()
    arguments = parser.parse_args(
        (
            "run",
            "--provider",
            "mock",
            "--prompt-version",
            "1",
            "--dataset",
            "dataset.jsonl",
            "--predictions-output",
            "predictions.jsonl",
            "--output",
            "report.json",
        ),
    )

    assert arguments.prompt_version == 1


async def test_run_prompt_version_two_reaches_runner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset_path = tmp_path / "dataset.jsonl"
    _write_jsonl(
        dataset_path,
        _dataset_payload(),
    )
    captured_versions: list[int] = []

    async def _capture_run(
        **kwargs: Any,
    ) -> NoReturn:
        captured_versions.append(
            kwargs["prompt_version"],
        )
        raise AssertionError(
            "Stop after capturing the selected prompt version.",
        )

    monkeypatch.setattr(
        ("supportops.evaluation.ticket_classification.cli.run_ticket_classification_evaluation"),
        _capture_run,
    )
    stderr = StringIO()

    exit_code = await run_cli(
        (
            "run",
            "--provider",
            "mock",
            "--prompt-version",
            "2",
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

    assert captured_versions == [2]
    assert exit_code == 1
    assert "unexpectedly" in stderr.getvalue()


@pytest.mark.parametrize(
    "prompt_version",
    (
        "0",
        "-1",
        "abc",
    ),
)
def test_run_rejects_invalid_prompt_versions(
    prompt_version: str,
) -> None:
    parser = build_parser()

    with pytest.raises(SystemExit) as raised:
        parser.parse_args(
            (
                "run",
                "--provider",
                "mock",
                "--prompt-version",
                prompt_version,
                "--dataset",
                "dataset.jsonl",
                "--predictions-output",
                "predictions.jsonl",
                "--output",
                "report.json",
            ),
        )

    assert raised.value.code == 2


async def test_score_does_not_accept_prompt_version(
    tmp_path: Path,
) -> None:
    dataset_path = tmp_path / "dataset.jsonl"
    predictions_path = tmp_path / "predictions.jsonl"
    _write_jsonl(
        dataset_path,
        _dataset_payload(),
    )
    _write_jsonl(
        predictions_path,
        _prediction_payload(),
    )
    parser = build_parser()

    with pytest.raises(SystemExit) as raised:
        parser.parse_args(
            (
                "score",
                "--dataset",
                str(dataset_path),
                "--predictions",
                str(predictions_path),
                "--output",
                str(tmp_path / "report.json"),
                "--prompt-version",
                "1",
            ),
        )

    assert raised.value.code == 2


async def test_unsupported_prompt_version_fails_without_overwrite(
    tmp_path: Path,
) -> None:
    dataset_path = tmp_path / "dataset.jsonl"
    predictions_path = tmp_path / "predictions.jsonl"
    existing_content = '{"preserved":true}\n'
    predictions_path.write_text(
        existing_content,
        encoding="utf-8",
    )
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
            "--prompt-version",
            "2",
            "--dataset",
            str(dataset_path),
            "--predictions-output",
            str(predictions_path),
            "--output",
            str(tmp_path / "report.json"),
        ),
        stdout=StringIO(),
        stderr=stderr,
        settings_factory=_settings,
    )

    assert exit_code == 2
    assert "Unsupported prompt" in stderr.getvalue()
    assert (
        predictions_path.read_text(
            encoding="utf-8",
        )
        == existing_content
    )
    assert not (tmp_path / "report.json").exists()


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
