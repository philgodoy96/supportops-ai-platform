from __future__ import annotations

import json
import socket
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

import pytest

from supportops.evaluation.contracts.hashing import (
    canonical_json_bytes,
    sha256_bytes,
)
from supportops.evaluation.contracts.manifest import EvaluationRunStatus
from supportops.evaluation.contracts.predictions import EvaluationPredictionStatus
from supportops.evaluation.grounded_recommendations.dataset import (
    load_grounded_recommendation_dataset,
)
from supportops.evaluation.grounded_recommendations.predictions import (
    load_grounded_recommendation_predictions,
)
from supportops.evaluation.grounded_recommendations.ragas_adapter import (
    RagasEvaluationResult,
    RagasEvaluationSample,
    RagasMetricName,
    RagasMetricResult,
)
from supportops.evaluation.grounded_recommendations.ragas_scores import (
    RagasMetricStatus,
    load_grounded_recommendation_ragas_scores,
)
from supportops.evaluation.grounded_recommendations.runner import (
    DEFAULT_GROUNDED_DATASET_PATH,
    DEFAULT_GROUNDED_PREDICTIONS_PATH,
    DEFAULT_GROUNDED_RAGAS_SCORES_PATH,
    ExternalProviderPermissionRequiredError,
    GroundedRecommendationRunnerError,
    run_grounded_recommendation_ragas_evaluation,
    score_grounded_recommendation_artifacts,
    validate_grounded_recommendation_artifacts,
)

PROJECT_ROOT = Path(__file__).resolve().parents[4]
DATASET_PATH = PROJECT_ROOT / DEFAULT_GROUNDED_DATASET_PATH
PREDICTIONS_PATH = PROJECT_ROOT / DEFAULT_GROUNDED_PREDICTIONS_PATH
RAGAS_SCORES_PATH = PROJECT_ROOT / DEFAULT_GROUNDED_RAGAS_SCORES_PATH

_PROMPT_HASH = "a" * 64
_GIT_COMMIT = "b" * 40
_CAPTURE = datetime(2026, 8, 5, 12, 0, tzinfo=UTC)


class _FakeRagasAdapter:
    def __init__(
        self,
        *,
        fail_metrics: frozenset[RagasMetricName] = frozenset(),
        score: Decimal = Decimal("0.875"),
    ) -> None:
        self.fail_metrics = fail_metrics
        self.score = score
        self.calls: list[tuple[tuple[RagasEvaluationSample, ...], tuple[RagasMetricName, ...]]] = []
        self.created = True

    @property
    def runtime_version(self) -> str:
        return "0.4.3"

    def evaluate(
        self,
        *,
        samples: tuple[RagasEvaluationSample, ...],
        metrics: tuple[RagasMetricName, ...],
    ) -> tuple[RagasEvaluationResult, ...]:
        self.calls.append((samples, metrics))
        results: list[RagasEvaluationResult] = []
        for sample in samples:
            metric_results: list[RagasMetricResult] = []
            for metric in metrics:
                if metric in self.fail_metrics:
                    metric_results.append(
                        RagasMetricResult(
                            metric=metric,
                            score=None,
                            error_code=f"ragas_{metric.value}_failed",
                        )
                    )
                else:
                    metric_results.append(
                        RagasMetricResult(
                            metric=metric,
                            score=self.score,
                        )
                    )
            results.append(
                RagasEvaluationResult(
                    case_id=sample.case_id,
                    metrics=tuple(metric_results),
                )
            )
        return tuple(results)


class _RefuseAdapter:
    @property
    def runtime_version(self) -> str:
        raise AssertionError("adapter should not be used")

    def evaluate(
        self,
        *,
        samples: tuple[RagasEvaluationSample, ...],
        metrics: tuple[RagasMetricName, ...],
    ) -> tuple[RagasEvaluationResult, ...]:
        raise AssertionError("adapter should not be used")


def _run_kwargs(tmp_path: Path, **overrides: object) -> dict[str, object]:
    values: dict[str, object] = {
        "dataset_path": DATASET_PATH,
        "predictions_path": PREDICTIONS_PATH,
        "output_dir": tmp_path / "artifacts" / "grounded-run",
        "allow_external_provider": True,
        "system_provider": "openai",
        "system_model": "system-model",
        "evaluator_provider": "openai",
        "evaluator_model": "evaluator-model",
        "evaluator_embedding_model": "text-embedding-3-small",
        "prompt_id": "grounded-recommendation",
        "prompt_version": 1,
        "prompt_hash": _PROMPT_HASH,
        "workflow_name": "ticket-processing",
        "workflow_version": "controlled-support-v1",
        "git_commit": _GIT_COMMIT,
        "pricing_catalog_version": "pricing-v1",
        "ragas_adapter": _FakeRagasAdapter(),
        "capture_timestamp": _CAPTURE,
    }
    values.update(overrides)
    return values


def test_validate_default_committed_dataset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(PROJECT_ROOT)
    result = validate_grounded_recommendation_artifacts(
        dataset_path=DEFAULT_GROUNDED_DATASET_PATH,
    )

    assert result.case_count == 14
    assert result.dataset_id == "grounded-recommendations-eval"
    assert result.prediction_hash is None
    assert len(result.dataset_hash) == 64


def test_validate_optional_predictions_and_scores(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(PROJECT_ROOT)
    result = validate_grounded_recommendation_artifacts(
        dataset_path=DEFAULT_GROUNDED_DATASET_PATH,
        predictions_path=DEFAULT_GROUNDED_PREDICTIONS_PATH,
        ragas_scores_path=DEFAULT_GROUNDED_RAGAS_SCORES_PATH,
    )

    assert result.prediction_count == 14
    assert result.ragas_score_case_count == 14
    assert result.prediction_hash is not None
    assert result.ragas_score_hash is not None


def test_validate_is_no_network(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(PROJECT_ROOT)

    def _deny_network(*_args: object, **_kwargs: object) -> None:
        raise OSError("network access is forbidden during validate")

    monkeypatch.setattr(socket, "create_connection", _deny_network)

    result = validate_grounded_recommendation_artifacts(
        dataset_path=DEFAULT_GROUNDED_DATASET_PATH,
        predictions_path=DEFAULT_GROUNDED_PREDICTIONS_PATH,
    )
    assert result.case_count == 14


def test_score_is_no_network(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(PROJECT_ROOT)

    def _deny_network(*_args: object, **_kwargs: object) -> None:
        raise OSError("network access is forbidden during score")

    monkeypatch.setattr(socket, "create_connection", _deny_network)

    result = score_grounded_recommendation_artifacts(
        dataset_path=DEFAULT_GROUNDED_DATASET_PATH,
        predictions_path=DEFAULT_GROUNDED_PREDICTIONS_PATH,
    )
    assert result.deterministic_report.case_count == 14
    assert result.ragas_report is None


def test_score_writes_deterministic_report_atomically(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(PROJECT_ROOT)
    output_dir = tmp_path / "artifacts" / "score-out"

    result = score_grounded_recommendation_artifacts(
        dataset_path=DEFAULT_GROUNDED_DATASET_PATH,
        predictions_path=DEFAULT_GROUNDED_PREDICTIONS_PATH,
        output_dir=output_dir,
    )

    assert result.deterministic_report_path is not None
    payload = json.loads(result.deterministic_report_path.read_text(encoding="utf-8"))
    assert payload["case_count"] == 14
    assert payload["report_content_hash"] == (result.deterministic_report.report_content_hash)


def test_score_writes_offline_ragas_report_when_supplied(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(PROJECT_ROOT)
    output_dir = tmp_path / "artifacts" / "score-ragas"

    result = score_grounded_recommendation_artifacts(
        dataset_path=DEFAULT_GROUNDED_DATASET_PATH,
        predictions_path=DEFAULT_GROUNDED_PREDICTIONS_PATH,
        ragas_scores_path=DEFAULT_GROUNDED_RAGAS_SCORES_PATH,
        output_dir=output_dir,
    )

    assert result.ragas_report is not None
    assert result.ragas_report_path is not None
    payload = json.loads(result.ragas_report_path.read_text(encoding="utf-8"))
    assert payload["scored_case_count"] == 14
    assert payload["report_content_hash"] == result.ragas_report.report_content_hash


def test_run_refuses_without_acknowledgement_before_adapter_use(
    tmp_path: Path,
) -> None:
    with pytest.raises(ExternalProviderPermissionRequiredError):
        run_grounded_recommendation_ragas_evaluation(
            **_run_kwargs(  # type: ignore[arg-type]
                tmp_path,
                allow_external_provider=False,
                ragas_adapter=_RefuseAdapter(),
            )
        )


def test_run_rejects_committed_evals_output_directory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(PROJECT_ROOT)
    output_dir = PROJECT_ROOT / "evals" / "grounded-recommendations" / "generated"

    with pytest.raises(
        GroundedRecommendationRunnerError,
        match="evals/grounded-recommendations",
    ):
        run_grounded_recommendation_ragas_evaluation(
            **_run_kwargs(  # type: ignore[arg-type]
                PROJECT_ROOT,
                output_dir=output_dir,
            )
        )


def test_run_preserves_dataset_ordering(tmp_path: Path) -> None:
    dataset = load_grounded_recommendation_dataset(DATASET_PATH)
    result = run_grounded_recommendation_ragas_evaluation(
        **_run_kwargs(tmp_path)  # type: ignore[arg-type]
    )

    assert tuple(case_score.case_id for case_score in result.case_scores) == tuple(
        case.case_id for case in dataset.cases
    )


def test_failed_prediction_produces_explicit_metric_failures(
    tmp_path: Path,
) -> None:
    predictions, _ = load_grounded_recommendation_predictions(PREDICTIONS_PATH)
    failed = predictions[0].model_copy(
        update={
            "status": EvaluationPredictionStatus.FAILED,
            "payload": None,
            "error_code": "provider_timeout",
        }
    )
    prediction_path = tmp_path / "predictions.jsonl"
    lines = [failed, *predictions[1:]]
    prediction_path.write_bytes(
        b"".join(
            canonical_json_bytes(item.model_dump(mode="json", exclude_none=False)) + b"\n"
            for item in lines
        )
    )

    result = run_grounded_recommendation_ragas_evaluation(
        **_run_kwargs(  # type: ignore[arg-type]
            tmp_path,
            predictions_path=prediction_path,
        )
    )

    first = result.case_scores[0]
    assert all(metric.status is RagasMetricStatus.FAILED for metric in first.metrics)
    assert all(metric.error_code == "provider_timeout" for metric in first.metrics)
    assert result.manifest.run_status is EvaluationRunStatus.INCOMPLETE
    failures = [
        json.loads(line)
        for line in result.paths.failures_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert failures
    assert failures[0]["error_code"] == "provider_timeout"


def test_no_context_cases_make_context_metrics_not_applicable(
    tmp_path: Path,
) -> None:
    result = run_grounded_recommendation_ragas_evaluation(
        **_run_kwargs(tmp_path)  # type: ignore[arg-type]
    )

    by_id = {case_score.case_id: case_score for case_score in result.case_scores}
    for case_id in (
        "grounded-recommendation-abstention-correct-007",
        "grounded-recommendation-abstention-hallucinated-008",
    ):
        metrics = {metric.metric: metric for metric in by_id[case_id].metrics}
        assert metrics[RagasMetricName.CONTEXT_PRECISION].status is (
            RagasMetricStatus.NOT_APPLICABLE
        )
        assert metrics[RagasMetricName.CONTEXT_RECALL].status is (RagasMetricStatus.NOT_APPLICABLE)
        assert metrics[RagasMetricName.FAITHFULNESS].status is (RagasMetricStatus.SUCCEEDED)
        assert metrics[RagasMetricName.ANSWER_RELEVANCY].status is (RagasMetricStatus.SUCCEEDED)


def test_partial_evaluator_failure_yields_incomplete_manifest(
    tmp_path: Path,
) -> None:
    adapter = _FakeRagasAdapter(
        fail_metrics=frozenset({RagasMetricName.FAITHFULNESS}),
    )
    result = run_grounded_recommendation_ragas_evaluation(
        **_run_kwargs(  # type: ignore[arg-type]
            tmp_path,
            ragas_adapter=adapter,
        )
    )

    assert result.manifest.run_status is EvaluationRunStatus.INCOMPLETE
    assert result.failure_count > 0


def test_complete_fake_evaluator_run_yields_complete_manifest(
    tmp_path: Path,
) -> None:
    result = run_grounded_recommendation_ragas_evaluation(
        **_run_kwargs(tmp_path)  # type: ignore[arg-type]
    )

    assert result.manifest.run_status is EvaluationRunStatus.COMPLETE
    assert result.failure_count == 0
    assert result.paths.failures_path.read_bytes() == b""
    reloaded = load_grounded_recommendation_ragas_scores(result.paths.ragas_scores_path)
    assert reloaded.content_hash == result.score_artifact_hash
    assert result.manifest.evaluation_id == "grounded-recommendations-ragas"
    assert result.manifest.schema_version == "grounded-recommendations-ragas-run-v1"
    assert result.manifest.ragas_version == "0.4.3"
    assert result.manifest.evaluator_embedding_model == "text-embedding-3-small"


def test_same_model_identities_remain_allowed(tmp_path: Path) -> None:
    result = run_grounded_recommendation_ragas_evaluation(
        **_run_kwargs(  # type: ignore[arg-type]
            tmp_path,
            system_model="shared-model",
            evaluator_model="shared-model",
        )
    )
    assert result.manifest.run_status is EvaluationRunStatus.COMPLETE


def test_output_artifact_hashes_are_stable(tmp_path: Path) -> None:
    first = run_grounded_recommendation_ragas_evaluation(
        **_run_kwargs(  # type: ignore[arg-type]
            tmp_path,
            output_dir=tmp_path / "artifacts" / "run-a",
        )
    )
    second = run_grounded_recommendation_ragas_evaluation(
        **_run_kwargs(  # type: ignore[arg-type]
            tmp_path,
            output_dir=tmp_path / "artifacts" / "run-b",
        )
    )

    assert first.score_artifact_hash == second.score_artifact_hash
    assert first.deterministic_report.report_content_hash == (
        second.deterministic_report.report_content_hash
    )
    assert first.ragas_report.report_content_hash == (second.ragas_report.report_content_hash)
    assert first.manifest.content_hash() == second.manifest.content_hash()


def test_failure_before_write_preserves_existing_files(tmp_path: Path) -> None:
    output_dir = tmp_path / "artifacts" / "preserve"
    output_dir.mkdir(parents=True)
    sentinel = output_dir / "deterministic-report.json"
    original = b'{"preserved":true}\n'
    sentinel.write_bytes(original)

    with pytest.raises(ExternalProviderPermissionRequiredError):
        run_grounded_recommendation_ragas_evaluation(
            **_run_kwargs(  # type: ignore[arg-type]
                tmp_path,
                output_dir=output_dir,
                allow_external_provider=False,
                ragas_adapter=_RefuseAdapter(),
            )
        )

    assert sentinel.read_bytes() == original


def test_score_failure_preserves_existing_destination(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(PROJECT_ROOT)
    output_dir = tmp_path / "artifacts" / "score-preserve"
    output_dir.mkdir(parents=True)
    sentinel = output_dir / "deterministic-report.json"
    original = b'{"preserved":true}\n'
    sentinel.write_bytes(original)

    with pytest.raises(FileNotFoundError):
        score_grounded_recommendation_artifacts(
            dataset_path=DEFAULT_GROUNDED_DATASET_PATH,
            predictions_path=tmp_path / "missing-predictions.jsonl",
            output_dir=output_dir,
        )

    assert sentinel.read_bytes() == original


def test_written_score_bytes_match_content_hash(tmp_path: Path) -> None:
    result = run_grounded_recommendation_ragas_evaluation(
        **_run_kwargs(tmp_path)  # type: ignore[arg-type]
    )
    content = result.paths.ragas_scores_path.read_bytes()
    assert sha256_bytes(content) == result.score_artifact_hash
