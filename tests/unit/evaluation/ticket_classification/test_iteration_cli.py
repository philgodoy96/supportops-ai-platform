"""CLI tests for ticket-classification prompt iteration."""

from __future__ import annotations

import json
import socket
from io import StringIO
from pathlib import Path
from typing import NoReturn

import pytest

from supportops.evaluation.ticket_classification.cli import (
    build_parser,
    run_cli,
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
ANALYSIS_PATH = (
    PROJECT_ROOT
    / "evals"
    / "ticket-classification"
    / "analyses"
    / "classification-prompt-v1-failure-analysis.json"
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
DECISION_TEMPLATE_PATH = (
    PROJECT_ROOT
    / "evals"
    / "ticket-classification"
    / "decisions"
    / "ticket-classification-prompt-v2-decision.static.json"
)


def _forbidden_settings_factory() -> NoReturn:
    raise AssertionError("Offline prompt-iteration commands must not load provider settings.")


def _forbidden_runtime_factory(**_: object) -> NoReturn:
    raise AssertionError("Offline prompt-iteration commands must not compose a provider.")


async def test_analyze_validates_committed_artifact_offline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_connection(*args: object, **kwargs: object) -> object:
        raise AssertionError("network access is forbidden")

    monkeypatch.setattr(socket, "create_connection", fail_connection)
    stdout = StringIO()
    stderr = StringIO()

    exit_code = await run_cli(
        (
            "analyze",
            "--dataset",
            str(DATASET_PATH),
            "--split-manifest",
            str(SPLIT_MANIFEST_PATH),
            "--analysis",
            str(ANALYSIS_PATH),
        ),
        stdout=stdout,
        stderr=stderr,
        settings_factory=_forbidden_settings_factory,
        runtime_factory=_forbidden_runtime_factory,
    )

    summary = json.loads(stdout.getvalue())

    assert exit_code == 0
    assert stderr.getvalue() == ""
    assert summary["command"] == "analyze"
    assert summary["analysis_id"] == ("classification-prompt-v1-failure-analysis")
    assert summary["analyzed_split"] == "development"
    assert summary["analyzed_case_count"] == 12
    assert summary["observation_count"] == 8


async def test_compare_writes_artifacts_offline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_connection(*args: object, **kwargs: object) -> object:
        raise AssertionError("network access is forbidden")

    monkeypatch.setattr(socket, "create_connection", fail_connection)
    comparison_output = tmp_path / "comparison.json"
    baseline_manifest_output = tmp_path / "baseline-manifest.json"
    candidate_manifest_output = tmp_path / "candidate-manifest.json"
    pair_manifest_output = tmp_path / "pair-manifest.json"
    stdout = StringIO()
    stderr = StringIO()

    exit_code = await run_cli(
        (
            "compare",
            "--dataset",
            str(DATASET_PATH),
            "--split-manifest",
            str(SPLIT_MANIFEST_PATH),
            "--baseline-predictions",
            str(BASELINE_PATH),
            "--candidate-predictions",
            str(CANDIDATE_PATH),
            "--evidence-kind",
            "static_fixture",
            "--capture-timestamp",
            "2026-08-05T18:00:00Z",
            "--git-commit",
            "8ed9c7a",
            "--output",
            str(comparison_output),
            "--baseline-manifest-output",
            str(baseline_manifest_output),
            "--candidate-manifest-output",
            str(candidate_manifest_output),
            "--pair-manifest-output",
            str(pair_manifest_output),
        ),
        stdout=stdout,
        stderr=stderr,
        settings_factory=_forbidden_settings_factory,
        runtime_factory=_forbidden_runtime_factory,
    )

    summary = json.loads(stdout.getvalue())

    assert exit_code == 0
    assert stderr.getvalue() == ""
    assert comparison_output.exists()
    assert baseline_manifest_output.exists()
    assert candidate_manifest_output.exists()
    assert pair_manifest_output.exists()
    assert summary["command"] == "compare"
    assert summary["comparison_id"] == ("ticket-classification-prompt-v1-v2")
    assert summary["case_count"] == 24
    assert summary["run_status"] == "incomplete"
    assert summary["gate_status"] == "incomplete"
    assert summary["blocking_failure_count"] == 0
    assert summary["not_applicable_count"] == 2


async def test_decide_rebuilds_inconclusive_static_decision(
    tmp_path: Path,
) -> None:
    comparison_output = tmp_path / "comparison.json"
    decision_output = tmp_path / "decision.json"

    compare_exit_code = await run_cli(
        (
            "compare",
            "--dataset",
            str(DATASET_PATH),
            "--split-manifest",
            str(SPLIT_MANIFEST_PATH),
            "--baseline-predictions",
            str(BASELINE_PATH),
            "--candidate-predictions",
            str(CANDIDATE_PATH),
            "--evidence-kind",
            "static_fixture",
            "--capture-timestamp",
            "2026-08-05T18:00:00Z",
            "--git-commit",
            "8ed9c7a",
            "--output",
            str(comparison_output),
            "--baseline-manifest-output",
            str(tmp_path / "baseline-manifest.json"),
            "--candidate-manifest-output",
            str(tmp_path / "candidate-manifest.json"),
            "--pair-manifest-output",
            str(tmp_path / "pair-manifest.json"),
        ),
        stdout=StringIO(),
        stderr=StringIO(),
        settings_factory=_forbidden_settings_factory,
        runtime_factory=_forbidden_runtime_factory,
    )
    assert compare_exit_code == 0

    stdout = StringIO()
    stderr = StringIO()
    decide_exit_code = await run_cli(
        (
            "decide",
            "--comparison",
            str(comparison_output),
            "--decision-template",
            str(DECISION_TEMPLATE_PATH),
            "--output",
            str(decision_output),
        ),
        stdout=stdout,
        stderr=stderr,
        settings_factory=_forbidden_settings_factory,
        runtime_factory=_forbidden_runtime_factory,
    )

    summary = json.loads(stdout.getvalue())

    assert decide_exit_code == 0
    assert stderr.getvalue() == ""
    assert decision_output.exists()
    assert summary["command"] == "decide"
    assert summary["outcome"] == "inconclusive"
    assert summary["run_status"] == "incomplete"
    assert summary["approved_for_runtime_adoption"] is False
    assert summary["separate_runtime_adoption_required"] is True


async def test_analyze_failure_returns_configuration_exit_code(
    tmp_path: Path,
) -> None:
    invalid_analysis = tmp_path / "invalid-analysis.json"
    invalid_analysis.write_text(
        '{"invalid":true}\n',
        encoding="utf-8",
    )
    stderr = StringIO()

    exit_code = await run_cli(
        (
            "analyze",
            "--dataset",
            str(DATASET_PATH),
            "--split-manifest",
            str(SPLIT_MANIFEST_PATH),
            "--analysis",
            str(invalid_analysis),
        ),
        stdout=StringIO(),
        stderr=stderr,
        settings_factory=_forbidden_settings_factory,
        runtime_factory=_forbidden_runtime_factory,
    )

    assert exit_code == 2
    assert "evaluation_error:" in stderr.getvalue()


def test_compare_rejects_external_provider_flags() -> None:
    parser = build_parser()

    with pytest.raises(SystemExit) as raised:
        parser.parse_args(
            (
                "compare",
                "--dataset",
                str(DATASET_PATH),
                "--split-manifest",
                str(SPLIT_MANIFEST_PATH),
                "--baseline-predictions",
                str(BASELINE_PATH),
                "--candidate-predictions",
                str(CANDIDATE_PATH),
                "--evidence-kind",
                "static_fixture",
                "--capture-timestamp",
                "2026-08-05T18:00:00Z",
                "--git-commit",
                "8ed9c7a",
                "--output",
                "comparison.json",
                "--baseline-manifest-output",
                "baseline-manifest.json",
                "--candidate-manifest-output",
                "candidate-manifest.json",
                "--pair-manifest-output",
                "pair-manifest.json",
                "--allow-external-provider",
            )
        )

    assert raised.value.code == 2


@pytest.mark.parametrize(
    "capture_timestamp",
    (
        "2026-08-05T18:00:00",
        "not-a-timestamp",
    ),
)
def test_compare_rejects_invalid_capture_timestamp(
    capture_timestamp: str,
) -> None:
    parser = build_parser()

    with pytest.raises(SystemExit) as raised:
        parser.parse_args(
            (
                "compare",
                "--dataset",
                str(DATASET_PATH),
                "--split-manifest",
                str(SPLIT_MANIFEST_PATH),
                "--baseline-predictions",
                str(BASELINE_PATH),
                "--candidate-predictions",
                str(CANDIDATE_PATH),
                "--evidence-kind",
                "static_fixture",
                "--capture-timestamp",
                capture_timestamp,
                "--git-commit",
                "8ed9c7a",
                "--output",
                "comparison.json",
                "--baseline-manifest-output",
                "baseline-manifest.json",
                "--candidate-manifest-output",
                "candidate-manifest.json",
                "--pair-manifest-output",
                "pair-manifest.json",
            )
        )

    assert raised.value.code == 2
