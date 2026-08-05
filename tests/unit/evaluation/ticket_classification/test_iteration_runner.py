"""Unit tests for offline classification prompt-iteration orchestration."""

from __future__ import annotations

import json
import socket
from datetime import UTC, datetime
from pathlib import Path

import pytest

from supportops.evaluation.contracts.hashing import sha256_hexdigest
from supportops.evaluation.contracts.manifest import EvaluationRunStatus
from supportops.evaluation.ticket_classification.comparison import (
    TicketClassificationComparisonEvidenceKind,
    TicketClassificationPairedGateStatus,
    load_ticket_classification_paired_comparison,
)
from supportops.evaluation.ticket_classification.decision import (
    TicketClassificationPromptDecisionOutcome,
    load_ticket_classification_prompt_decision,
)
from supportops.evaluation.ticket_classification.iteration_runner import (
    TicketClassificationPromptComparisonRunResult,
    run_ticket_classification_prompt_comparison,
    run_ticket_classification_prompt_decision,
    validate_ticket_classification_failure_analysis_artifact,
    write_ticket_classification_prompt_comparison_run,
    write_ticket_classification_prompt_decision_run,
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
COMPARISON_PATH = (
    PROJECT_ROOT
    / "evals"
    / "ticket-classification"
    / "comparisons"
    / "ticket-classification-prompt-v1-v2.static.json"
)
DECISION_PATH = (
    PROJECT_ROOT
    / "evals"
    / "ticket-classification"
    / "decisions"
    / "ticket-classification-prompt-v2-decision.static.json"
)

CAPTURE_TIMESTAMP = datetime(
    2026,
    8,
    5,
    18,
    0,
    tzinfo=UTC,
)
GIT_COMMIT = "8ed9c7a"


def _run_comparison() -> TicketClassificationPromptComparisonRunResult:
    return run_ticket_classification_prompt_comparison(
        dataset_path=DATASET_PATH,
        dataset_id="ticket-classification-eval",
        dataset_version=1,
        split_manifest_path=SPLIT_MANIFEST_PATH,
        baseline_predictions_path=BASELINE_PATH,
        candidate_predictions_path=CANDIDATE_PATH,
        evidence_kind=(TicketClassificationComparisonEvidenceKind.STATIC_FIXTURE),
        capture_timestamp=CAPTURE_TIMESTAMP,
        git_commit=GIT_COMMIT,
    )


def test_validates_committed_failure_analysis() -> None:
    analysis = validate_ticket_classification_failure_analysis_artifact(
        dataset_path=DATASET_PATH,
        dataset_id="ticket-classification-eval",
        dataset_version=1,
        split_manifest_path=SPLIT_MANIFEST_PATH,
        analysis_path=ANALYSIS_PATH,
    )

    assert analysis.analysis_id == ("classification-prompt-v1-failure-analysis")
    assert analysis.analyzed_split == "development"
    assert len(analysis.analyzed_case_ids) == 12


def test_failure_analysis_requires_exact_development_order(
    tmp_path: Path,
) -> None:
    payload = json.loads(ANALYSIS_PATH.read_text(encoding="utf-8"))
    payload["analyzed_case_ids"] = list(reversed(payload["analyzed_case_ids"]))
    content = {key: value for key, value in payload.items() if key != "analysis_content_hash"}
    payload["analysis_content_hash"] = sha256_hexdigest(content)
    invalid_path = tmp_path / "analysis.json"
    invalid_path.write_text(
        json.dumps(payload),
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="development split exactly",
    ):
        validate_ticket_classification_failure_analysis_artifact(
            dataset_path=DATASET_PATH,
            dataset_id="ticket-classification-eval",
            dataset_version=1,
            split_manifest_path=SPLIT_MANIFEST_PATH,
            analysis_path=invalid_path,
        )


def test_comparison_run_is_deterministic_and_no_network(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_connection(*args: object, **kwargs: object) -> object:
        raise AssertionError("network access is forbidden")

    monkeypatch.setattr(socket, "create_connection", fail_connection)

    first = _run_comparison()
    second = _run_comparison()

    assert first == second
    assert first.comparison.case_count == 24
    assert first.comparison.run_status is EvaluationRunStatus.INCOMPLETE
    assert first.comparison.gate_evaluation.status is (
        TicketClassificationPairedGateStatus.INCOMPLETE
    )


def test_comparison_run_emits_distinct_prompt_manifests() -> None:
    result = _run_comparison()

    assert result.baseline_manifest.prompt_version == 1
    assert result.candidate_manifest.prompt_version == 2
    assert result.baseline_manifest.prompt_hash != result.candidate_manifest.prompt_hash
    assert (
        result.baseline_manifest.system_provider
        == result.candidate_manifest.system_provider
        == "static-fixture"
    )
    assert (
        result.baseline_manifest.system_model
        == result.candidate_manifest.system_model
        == "ticket-classification-paired-fixture"
    )
    assert result.baseline_manifest.git_commit == GIT_COMMIT
    assert result.candidate_manifest.git_commit == GIT_COMMIT
    assert result.baseline_manifest.capture_timestamp == CAPTURE_TIMESTAMP
    assert result.candidate_manifest.capture_timestamp == CAPTURE_TIMESTAMP


def test_pair_manifest_binds_reports_and_comparison() -> None:
    result = _run_comparison()

    assert (
        result.pair_manifest.baseline_manifest_content_hash
        == result.baseline_manifest.content_hash()
    )
    assert (
        result.pair_manifest.candidate_manifest_content_hash
        == result.candidate_manifest.content_hash()
    )
    assert (
        result.pair_manifest.baseline_report_content_hash
        == result.comparison.baseline_report_content_hash
    )
    assert (
        result.pair_manifest.candidate_report_content_hash
        == result.comparison.candidate_report_content_hash
    )
    assert result.pair_manifest.comparison_content_hash == result.comparison.comparison_content_hash


def test_comparison_outputs_round_trip_atomically(
    tmp_path: Path,
) -> None:
    result = _run_comparison()
    comparison_output = tmp_path / "comparison.json"
    baseline_manifest_output = tmp_path / "baseline-manifest.json"
    candidate_manifest_output = tmp_path / "candidate-manifest.json"
    pair_manifest_output = tmp_path / "pair-manifest.json"

    write_ticket_classification_prompt_comparison_run(
        comparison_output=comparison_output,
        baseline_manifest_output=baseline_manifest_output,
        candidate_manifest_output=candidate_manifest_output,
        pair_manifest_output=pair_manifest_output,
        result=result,
    )

    assert load_ticket_classification_paired_comparison(comparison_output) == result.comparison
    assert json.loads(baseline_manifest_output.read_text(encoding="utf-8"))["prompt_version"] == 1
    assert json.loads(candidate_manifest_output.read_text(encoding="utf-8"))["prompt_version"] == 2
    assert json.loads(pair_manifest_output.read_text(encoding="utf-8"))[
        "comparison_content_hash"
    ] == (result.comparison.comparison_content_hash)
    assert not tuple(tmp_path.glob("*.tmp"))


def test_decision_run_rebuilds_committed_static_decision() -> None:
    decision = run_ticket_classification_prompt_decision(
        comparison_path=COMPARISON_PATH,
        decision_template_path=DECISION_PATH,
    )

    assert decision.outcome is (TicketClassificationPromptDecisionOutcome.INCONCLUSIVE)
    assert decision.run_status is EvaluationRunStatus.INCOMPLETE
    assert decision.review.approved_for_runtime_adoption is False


def test_decision_rejects_mismatched_template(
    tmp_path: Path,
) -> None:
    payload = json.loads(DECISION_PATH.read_text(encoding="utf-8"))
    payload["comparison_content_hash"] = "0" * 64
    payload["decision_content_hash"] = "0" * 64
    mismatched_path = tmp_path / "decision.json"
    mismatched_path.write_text(
        json.dumps(payload),
        encoding="utf-8",
    )

    with pytest.raises(ValueError):
        run_ticket_classification_prompt_decision(
            comparison_path=COMPARISON_PATH,
            decision_template_path=mismatched_path,
        )


def test_decision_output_round_trips_atomically(
    tmp_path: Path,
) -> None:
    decision = run_ticket_classification_prompt_decision(
        comparison_path=COMPARISON_PATH,
        decision_template_path=DECISION_PATH,
    )
    output_path = tmp_path / "decision.json"

    write_ticket_classification_prompt_decision_run(
        output=output_path,
        decision=decision,
    )

    assert load_ticket_classification_prompt_decision(output_path) == decision
    assert not tuple(tmp_path.glob("*.tmp"))
