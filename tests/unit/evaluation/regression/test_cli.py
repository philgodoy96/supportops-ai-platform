"""Unit tests for the repository regression CLI."""

from __future__ import annotations

import json
import socket
from decimal import Decimal
from io import StringIO
from pathlib import Path

import pytest

from supportops.evaluation.regression.cli import build_parser, run_cli
from supportops.evaluation.regression.models import (
    DOMAIN_CONTROLLED_SUPPORT,
    DOMAIN_HUMAN_APPROVAL,
    DOMAIN_SEMANTIC_RETRIEVAL,
    DOMAIN_TICKET_CLASSIFICATION,
    RegressionAggregateStatus,
    RegressionGateCategory,
    RegressionGateOperator,
    RegressionGateOutcome,
    RegressionGateResult,
    RepositoryRegressionResult,
    build_domain_profile_result,
    build_repository_regression_result,
)
from supportops.evaluation.ticket_classification import cli as classification_cli

PROJECT_ROOT = Path(__file__).resolve().parents[4]


def test_default_score_succeeds_and_writes_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(PROJECT_ROOT)
    output_path = tmp_path / "regression.json"
    stdout = StringIO()
    stderr = StringIO()

    exit_code = run_cli(
        ["score", "--output", str(output_path)],
        stdout=stdout,
        stderr=stderr,
    )

    assert exit_code == 0
    assert stderr.getvalue() == ""
    summary = stdout.getvalue()
    assert "status=incomplete" in summary
    assert "blocking_failure_count=0" in summary
    assert DOMAIN_SEMANTIC_RETRIEVAL in summary
    assert DOMAIN_CONTROLLED_SUPPORT in summary
    assert DOMAIN_HUMAN_APPROVAL in summary
    assert DOMAIN_TICKET_CLASSIFICATION in summary
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["status"] == RegressionAggregateStatus.INCOMPLETE.value
    assert payload["blocking_failure_count"] == 0
    assert [item["domain"] for item in payload["domain_results"]] == [
        DOMAIN_SEMANTIC_RETRIEVAL,
        DOMAIN_CONTROLLED_SUPPORT,
        DOMAIN_HUMAN_APPROVAL,
    ]
    assert payload["not_provided_domains"] == [DOMAIN_TICKET_CLASSIFICATION]


def test_default_score_completes_without_network(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(PROJECT_ROOT)

    def _deny_network(*_args: object, **_kwargs: object) -> None:
        raise OSError("network access is forbidden during regression scoring")

    monkeypatch.setattr(socket, "create_connection", _deny_network)

    exit_code = run_cli(["score"], stdout=StringIO(), stderr=StringIO())

    assert exit_code == 0


def test_incomplete_returns_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(PROJECT_ROOT)
    exit_code = run_cli(["score"], stdout=StringIO(), stderr=StringIO())
    assert exit_code == 0


def test_passed_returns_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    passed = build_repository_regression_result(
        domain_results=(
            build_domain_profile_result(
                profile_id="semantic-retrieval-release-gates",
                profile_version=1,
                domain=DOMAIN_SEMANTIC_RETRIEVAL,
                source_report_hash="a" * 64,
                gate_results=(
                    RegressionGateResult(
                        gate_id="retrieval.workspace-isolation",
                        domain=DOMAIN_SEMANTIC_RETRIEVAL,
                        category=RegressionGateCategory.SAFETY,
                        outcome=RegressionGateOutcome.PASSED,
                        blocking=True,
                        metric_name="workspace_isolation_rate.rate",
                        operator=RegressionGateOperator.EQUAL,
                        actual_value=Decimal("1.000000"),
                        threshold_value=Decimal("1.000000"),
                        reason="forced pass",
                    ),
                ),
            ),
        ),
        not_provided_domains=(DOMAIN_TICKET_CLASSIFICATION,),
    )

    def _fake_run(**_kwargs: object) -> RepositoryRegressionResult:
        return passed

    monkeypatch.setattr(
        "supportops.evaluation.regression.cli.run_repository_regression",
        _fake_run,
    )
    exit_code = run_cli(["score"], stdout=StringIO(), stderr=StringIO())
    assert exit_code == 0
    assert passed.status is RegressionAggregateStatus.PASSED


def test_blocking_failure_returns_one(monkeypatch: pytest.MonkeyPatch) -> None:
    failed = build_repository_regression_result(
        domain_results=(
            build_domain_profile_result(
                profile_id="semantic-retrieval-release-gates",
                profile_version=1,
                domain=DOMAIN_SEMANTIC_RETRIEVAL,
                source_report_hash="a" * 64,
                gate_results=(
                    RegressionGateResult(
                        gate_id="retrieval.workspace-isolation",
                        domain=DOMAIN_SEMANTIC_RETRIEVAL,
                        category=RegressionGateCategory.SAFETY,
                        outcome=RegressionGateOutcome.FAILED,
                        blocking=True,
                        metric_name="workspace_isolation_rate.rate",
                        operator=RegressionGateOperator.EQUAL,
                        actual_value=Decimal("0.000000"),
                        threshold_value=Decimal("1.000000"),
                        reason="forced failure",
                    ),
                ),
            ),
        ),
        not_provided_domains=(DOMAIN_TICKET_CLASSIFICATION,),
    )

    def _fake_run(**_kwargs: object) -> RepositoryRegressionResult:
        return failed

    monkeypatch.setattr(
        "supportops.evaluation.regression.cli.run_repository_regression",
        _fake_run,
    )
    exit_code = run_cli(["score"], stdout=StringIO(), stderr=StringIO())
    assert exit_code == 1
    assert failed.status is RegressionAggregateStatus.FAILED


def test_malformed_artifact_returns_artifact_failure_and_preserves_output(
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "bad-dataset.jsonl"
    predictions = tmp_path / "bad-predictions.jsonl"
    output_path = tmp_path / "regression.json"
    preserved = b'{"preserved":true}\n'
    dataset.write_bytes(b"{not-json\n")
    predictions.write_bytes(b"{not-json\n")
    output_path.write_bytes(preserved)
    stderr = StringIO()

    exit_code = run_cli(
        [
            "score",
            "--domain",
            DOMAIN_SEMANTIC_RETRIEVAL,
            "--semantic-retrieval-dataset",
            str(dataset),
            "--semantic-retrieval-predictions",
            str(predictions),
            "--output",
            str(output_path),
        ],
        stdout=StringIO(),
        stderr=stderr,
    )

    assert exit_code == 3
    assert stderr.getvalue().strip() != ""
    assert output_path.read_bytes() == preserved


def test_unsupported_domain_is_usage_error() -> None:
    exit_code = run_cli(
        ["score", "--domain", "not-a-domain"],
        stdout=StringIO(),
        stderr=StringIO(),
    )

    assert exit_code == 2


def test_classification_selection_requires_paths() -> None:
    stderr = StringIO()
    exit_code = run_cli(
        ["score", "--domain", DOMAIN_TICKET_CLASSIFICATION],
        stdout=StringIO(),
        stderr=stderr,
    )

    assert exit_code == 2
    assert "classification-dataset" in stderr.getvalue()


def test_no_provider_or_network_options_exist() -> None:
    help_text = build_parser().format_help()

    assert "--provider" not in help_text
    assert "--allow-external-provider" not in help_text
    assert "langfuse" not in help_text.lower()
    assert "ragas" not in help_text.lower()


def test_existing_classification_cli_remains_unchanged() -> None:
    parser = classification_cli.build_parser()
    assert parser.prog == "supportops-evaluate-classification"
    help_text = parser.format_help()
    assert "score" in help_text
    assert "run" in help_text

    run_help: str | None = None
    for action in parser._actions:
        choices = getattr(action, "choices", None)
        if isinstance(choices, dict) and "run" in choices:
            run_help = choices["run"].format_help()
            break
    assert run_help is not None
    assert "--provider" in run_help
