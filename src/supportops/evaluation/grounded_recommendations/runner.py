"""External grounded recommendation evaluation orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from supportops.evaluation.contracts.artifacts import (
    write_bytes_atomically,
    write_canonical_json_atomically,
)
from supportops.evaluation.contracts.hashing import (
    canonical_json_bytes,
    sha256_bytes,
)
from supportops.evaluation.contracts.manifest import (
    EvaluationManifest,
    EvaluationRunStatus,
)
from supportops.evaluation.contracts.predictions import (
    EvaluationPredictionStatus,
)
from supportops.evaluation.grounded_recommendations.dataset import (
    GroundedRecommendationDatasetError,
    load_grounded_recommendation_dataset,
)
from supportops.evaluation.grounded_recommendations.evaluator import (
    GroundedRecommendationEvaluationError,
    evaluate_grounded_recommendation_predictions,
)
from supportops.evaluation.grounded_recommendations.models import (
    GroundedRecommendationEvaluationCase,
    GroundedRecommendationEvaluationDataset,
    GroundedRecommendationEvaluationReport,
)
from supportops.evaluation.grounded_recommendations.predictions import (
    GroundedRecommendationPrediction,
    GroundedRecommendationPredictionError,
    load_grounded_recommendation_predictions,
)
from supportops.evaluation.grounded_recommendations.ragas_adapter import (
    PINNED_RAGAS_VERSION,
    RagasAdapter,
    RagasEvaluationResult,
    RagasEvaluationSample,
    RagasMetricName,
    normalize_ragas_evaluation_result,
)
from supportops.evaluation.grounded_recommendations.ragas_report import (
    GroundedRecommendationRagasReport,
    GroundedRecommendationRagasReportError,
    build_grounded_recommendation_ragas_report,
)
from supportops.evaluation.grounded_recommendations.ragas_scores import (
    GroundedRecommendationRagasCaseScore,
    GroundedRecommendationRagasMetricScore,
    GroundedRecommendationRagasScoreArtifact,
    GroundedRecommendationRagasScoreError,
    RagasMetricStatus,
    load_grounded_recommendation_ragas_scores,
)

DEFAULT_GROUNDED_DATASET_PATH = Path(
    "evals/grounded-recommendations/datasets/grounded-recommendations-eval-v1.jsonl"
)
DEFAULT_GROUNDED_PREDICTIONS_PATH = Path(
    "evals/grounded-recommendations/predictions/grounded-recommendations-eval-v1.static.jsonl"
)
DEFAULT_GROUNDED_RAGAS_SCORES_PATH = Path(
    "evals/grounded-recommendations/ragas-scores/grounded-recommendations-eval-v1.static.jsonl"
)

_COMMITTED_GROUNDED_ROOT = Path("evals/grounded-recommendations")
_REQUIRED_METRICS: tuple[RagasMetricName, ...] = tuple(RagasMetricName)
_CONTEXT_DEPENDENT_METRICS = frozenset(
    {
        RagasMetricName.CONTEXT_PRECISION,
        RagasMetricName.CONTEXT_RECALL,
    }
)
_NO_CONTEXT_REASON = "case has no retrieved contexts"
_SCHEMA_VERSION = "grounded-recommendations-ragas-run-v1"
_EVALUATION_ID = "grounded-recommendations-ragas"
_EVALUATION_VERSION = 1


class GroundedRecommendationRunnerError(RuntimeError):
    """Raised when grounded recommendation orchestration cannot complete."""


class ExternalProviderPermissionRequiredError(GroundedRecommendationRunnerError):
    """Raised when an external provider was not explicitly acknowledged."""


@dataclass(frozen=True, slots=True)
class GroundedRecommendationArtifactPaths:
    """Resolved artifact locations for one external evaluation run."""

    output_dir: Path
    predictions_path: Path
    ragas_scores_path: Path
    deterministic_report_path: Path
    ragas_report_path: Path
    manifest_path: Path
    failures_path: Path


@dataclass(frozen=True, slots=True)
class GroundedRecommendationValidationResult:
    """Deterministic validation summary for grounded recommendation artifacts."""

    dataset_path: Path
    dataset_id: str
    dataset_version: int
    dataset_hash: str
    case_count: int
    predictions_path: Path | None
    prediction_hash: str | None
    prediction_count: int | None
    ragas_scores_path: Path | None
    ragas_score_hash: str | None
    ragas_score_case_count: int | None


@dataclass(frozen=True, slots=True)
class GroundedRecommendationScoreResult:
    """Offline scoring result for grounded recommendation artifacts."""

    deterministic_report: GroundedRecommendationEvaluationReport
    ragas_report: GroundedRecommendationRagasReport | None
    deterministic_report_path: Path | None
    ragas_report_path: Path | None


@dataclass(frozen=True, slots=True)
class GroundedRecommendationRunResult:
    """External RAGAS evaluation result with written artifacts."""

    paths: GroundedRecommendationArtifactPaths
    manifest: EvaluationManifest
    deterministic_report: GroundedRecommendationEvaluationReport
    ragas_report: GroundedRecommendationRagasReport
    case_scores: tuple[GroundedRecommendationRagasCaseScore, ...]
    score_artifact_hash: str
    failure_count: int


def resolve_grounded_recommendation_artifact_paths(
    *,
    output_dir: Path,
    predictions_path: Path,
) -> GroundedRecommendationArtifactPaths:
    """Resolve the stable output layout under an explicit output directory."""

    return GroundedRecommendationArtifactPaths(
        output_dir=output_dir,
        predictions_path=predictions_path,
        ragas_scores_path=output_dir / "ragas-scores.jsonl",
        deterministic_report_path=output_dir / "deterministic-report.json",
        ragas_report_path=output_dir / "ragas-report.json",
        manifest_path=output_dir / "manifest.json",
        failures_path=output_dir / "failures.jsonl",
    )


def validate_grounded_recommendation_artifacts(
    *,
    dataset_path: Path,
    predictions_path: Path | None = None,
    ragas_scores_path: Path | None = None,
    output_path: Path | None = None,
) -> GroundedRecommendationValidationResult:
    """Validate grounded recommendation artifacts without network access."""

    dataset = load_grounded_recommendation_dataset(dataset_path)
    dataset_case_ids = {case.case_id for case in dataset.cases}

    prediction_hash: str | None = None
    prediction_count: int | None = None
    if predictions_path is not None:
        predictions, prediction_hash = load_grounded_recommendation_predictions(predictions_path)
        prediction_count = len(predictions)
        _require_known_case_ids(
            artifact_name="predictions",
            case_ids=tuple(prediction.case_id for prediction in predictions),
            dataset_case_ids=dataset_case_ids,
        )

    ragas_score_hash: str | None = None
    ragas_score_case_count: int | None = None
    if ragas_scores_path is not None:
        scores = load_grounded_recommendation_ragas_scores(ragas_scores_path)
        ragas_score_hash = scores.content_hash
        ragas_score_case_count = len(scores.case_scores)
        _require_known_case_ids(
            artifact_name="RAGAS scores",
            case_ids=tuple(case_score.case_id for case_score in scores.case_scores),
            dataset_case_ids=dataset_case_ids,
        )

    result = GroundedRecommendationValidationResult(
        dataset_path=dataset_path,
        dataset_id=dataset.dataset_id,
        dataset_version=dataset.dataset_version,
        dataset_hash=dataset.content_hash,
        case_count=len(dataset.cases),
        predictions_path=predictions_path,
        prediction_hash=prediction_hash,
        prediction_count=prediction_count,
        ragas_scores_path=ragas_scores_path,
        ragas_score_hash=ragas_score_hash,
        ragas_score_case_count=ragas_score_case_count,
    )

    if output_path is not None:
        write_canonical_json_atomically(
            output_path,
            {
                "dataset_path": str(dataset_path),
                "dataset_id": result.dataset_id,
                "dataset_version": result.dataset_version,
                "dataset_hash": result.dataset_hash,
                "case_count": result.case_count,
                "predictions_path": (
                    str(predictions_path) if predictions_path is not None else None
                ),
                "prediction_hash": result.prediction_hash,
                "prediction_count": result.prediction_count,
                "ragas_scores_path": (
                    str(ragas_scores_path) if ragas_scores_path is not None else None
                ),
                "ragas_score_hash": result.ragas_score_hash,
                "ragas_score_case_count": result.ragas_score_case_count,
                "valid": True,
            },
        )

    return result


def score_grounded_recommendation_artifacts(
    *,
    dataset_path: Path,
    predictions_path: Path,
    ragas_scores_path: Path | None = None,
    output_dir: Path | None = None,
) -> GroundedRecommendationScoreResult:
    """Build offline grounded recommendation reports without network access."""

    dataset = load_grounded_recommendation_dataset(dataset_path)
    predictions, prediction_hash = load_grounded_recommendation_predictions(predictions_path)
    _require_known_case_ids(
        artifact_name="predictions",
        case_ids=tuple(prediction.case_id for prediction in predictions),
        dataset_case_ids={case.case_id for case in dataset.cases},
    )

    deterministic_report = evaluate_grounded_recommendation_predictions(
        dataset=dataset,
        predictions=predictions,
        prediction_hash=prediction_hash,
    )

    ragas_report: GroundedRecommendationRagasReport | None = None
    if ragas_scores_path is not None:
        scores = load_grounded_recommendation_ragas_scores(ragas_scores_path)
        ragas_report = build_grounded_recommendation_ragas_report(
            dataset=dataset,
            scores=scores,
        )

    deterministic_report_path: Path | None = None
    ragas_report_path: Path | None = None

    if output_dir is not None:
        _reject_committed_output_directory(output_dir)
        deterministic_report_path = output_dir / "deterministic-report.json"
        write_canonical_json_atomically(
            deterministic_report_path,
            deterministic_report.model_dump(mode="json", exclude_none=False),
        )
        if ragas_report is not None:
            ragas_report_path = output_dir / "ragas-report.json"
            write_canonical_json_atomically(
                ragas_report_path,
                ragas_report.model_dump(mode="json", exclude_none=False),
            )

    return GroundedRecommendationScoreResult(
        deterministic_report=deterministic_report,
        ragas_report=ragas_report,
        deterministic_report_path=deterministic_report_path,
        ragas_report_path=ragas_report_path,
    )


def run_grounded_recommendation_ragas_evaluation(
    *,
    dataset_path: Path,
    predictions_path: Path,
    output_dir: Path,
    allow_external_provider: bool,
    system_provider: str,
    system_model: str,
    evaluator_provider: str,
    evaluator_model: str,
    evaluator_embedding_model: str,
    prompt_id: str,
    prompt_version: int,
    prompt_hash: str,
    workflow_name: str,
    workflow_version: str,
    git_commit: str,
    pricing_catalog_version: str,
    ragas_adapter: RagasAdapter,
    capture_timestamp: datetime | None = None,
) -> GroundedRecommendationRunResult:
    """Evaluate existing predictions with RAGAS and write normalized artifacts."""

    if not allow_external_provider:
        raise ExternalProviderPermissionRequiredError(
            "external grounded recommendation evaluation requires "
            "--allow-external-provider acknowledgement"
        )

    if evaluator_provider != "openai":
        raise GroundedRecommendationRunnerError("only evaluator_provider=openai is supported")

    _reject_committed_output_directory(output_dir)
    paths = resolve_grounded_recommendation_artifact_paths(
        output_dir=output_dir,
        predictions_path=predictions_path,
    )
    _reject_committed_artifact_overwrite(paths)

    dataset = load_grounded_recommendation_dataset(dataset_path)
    predictions, prediction_hash = load_grounded_recommendation_predictions(predictions_path)
    dataset_case_ids = {case.case_id for case in dataset.cases}
    _require_known_case_ids(
        artifact_name="predictions",
        case_ids=tuple(prediction.case_id for prediction in predictions),
        dataset_case_ids=dataset_case_ids,
    )

    predictions_by_case = {prediction.case_id: prediction for prediction in predictions}

    case_scores: list[GroundedRecommendationRagasCaseScore] = []
    failure_entries: list[dict[str, object]] = []

    for case in dataset.cases:
        prediction = predictions_by_case.get(case.case_id)
        case_score, failures = _evaluate_case(
            case=case,
            prediction=prediction,
            ragas_adapter=ragas_adapter,
        )
        case_scores.append(case_score)
        failure_entries.extend(failures)

    immutable_case_scores = tuple(case_scores)
    score_bytes = _ragas_scores_jsonl_bytes(immutable_case_scores)
    score_hash = sha256_bytes(score_bytes)
    score_artifact = GroundedRecommendationRagasScoreArtifact(
        case_scores=immutable_case_scores,
        content_hash=score_hash,
    )

    deterministic_report = evaluate_grounded_recommendation_predictions(
        dataset=dataset,
        predictions=predictions,
        prediction_hash=prediction_hash,
    )
    ragas_report = build_grounded_recommendation_ragas_report(
        dataset=dataset,
        scores=score_artifact,
    )

    run_status = _resolve_run_status(
        dataset=dataset,
        predictions_by_case=predictions_by_case,
        case_scores=immutable_case_scores,
    )

    timestamp = capture_timestamp or datetime.now(UTC)
    manifest = EvaluationManifest(
        evaluation_id=_EVALUATION_ID,
        evaluation_version=_EVALUATION_VERSION,
        dataset_id=dataset.dataset_id,
        dataset_version=dataset.dataset_version,
        dataset_hash=dataset.content_hash,
        split_manifest_id=None,
        split_manifest_version=None,
        split_manifest_hash=None,
        split=None,
        system_provider=system_provider,
        system_model=system_model,
        workflow_name=workflow_name,
        workflow_version=workflow_version,
        prompt_id=prompt_id,
        prompt_version=prompt_version,
        prompt_hash=prompt_hash,
        schema_version=_SCHEMA_VERSION,
        embedding_provider=None,
        embedding_model=None,
        embedding_dimensions=None,
        retrieval_profile=None,
        evaluator_provider=evaluator_provider,
        evaluator_model=evaluator_model,
        evaluator_embedding_model=evaluator_embedding_model,
        ragas_version=PINNED_RAGAS_VERSION,
        pricing_catalog_version=pricing_catalog_version,
        capture_timestamp=timestamp,
        git_commit=git_commit,
        prediction_hash=prediction_hash,
        run_status=run_status,
    )

    write_bytes_atomically(paths.ragas_scores_path, score_bytes)
    write_canonical_json_atomically(
        paths.deterministic_report_path,
        deterministic_report.model_dump(mode="json", exclude_none=False),
    )
    write_canonical_json_atomically(
        paths.ragas_report_path,
        ragas_report.model_dump(mode="json", exclude_none=False),
    )
    write_bytes_atomically(
        paths.failures_path,
        _failures_jsonl_bytes(failure_entries),
    )
    write_canonical_json_atomically(
        paths.manifest_path,
        manifest.canonical_payload(),
    )

    return GroundedRecommendationRunResult(
        paths=paths,
        manifest=manifest,
        deterministic_report=deterministic_report,
        ragas_report=ragas_report,
        case_scores=immutable_case_scores,
        score_artifact_hash=score_hash,
        failure_count=len(failure_entries),
    )


def build_grounded_recommendation_question(
    case: GroundedRecommendationEvaluationCase,
) -> str:
    """Build the deterministic RAGAS question for one evaluation case."""

    return f"{case.ticket_subject}\n{case.ticket_description}"


def _evaluate_case(
    *,
    case: GroundedRecommendationEvaluationCase,
    prediction: GroundedRecommendationPrediction | None,
    ragas_adapter: RagasAdapter,
) -> tuple[GroundedRecommendationRagasCaseScore, list[dict[str, object]]]:
    if prediction is None:
        metrics = _failed_metrics_for_prediction(
            error_code="prediction_missing",
        )
        return (
            GroundedRecommendationRagasCaseScore(
                case_id=case.case_id,
                metrics=metrics,
            ),
            _failure_entries_for_metrics(
                case_id=case.case_id,
                kind="prediction_missing",
                metrics=metrics,
            ),
        )

    if prediction.status is EvaluationPredictionStatus.FAILED or prediction.payload is None:
        error_code = prediction.error_code or "prediction_failed"
        metrics = _failed_metrics_for_prediction(error_code=error_code)
        return (
            GroundedRecommendationRagasCaseScore(
                case_id=case.case_id,
                metrics=metrics,
            ),
            _failure_entries_for_metrics(
                case_id=case.case_id,
                kind="prediction_failed",
                metrics=metrics,
            ),
        )

    contexts = tuple(context.content for context in case.retrieved_contexts)
    has_contexts = bool(contexts)
    requested_metrics = tuple(
        metric
        for metric in _REQUIRED_METRICS
        if has_contexts or metric not in _CONTEXT_DEPENDENT_METRICS
    )

    sample = RagasEvaluationSample(
        case_id=case.case_id,
        question=build_grounded_recommendation_question(case),
        response=prediction.payload.response_text,
        contexts=contexts,
        reference_answer=case.reference_answer,
        reference_claims=case.reference_claims,
    )

    adapter_results = ragas_adapter.evaluate(
        samples=(sample,),
        metrics=requested_metrics,
    )
    adapter_result = _require_single_adapter_result(
        case_id=case.case_id,
        results=adapter_results,
    )
    normalized = normalize_ragas_evaluation_result(result=adapter_result)
    metrics_by_name = {metric_score.metric: metric_score for metric_score in normalized.metrics}

    merged_metrics: list[GroundedRecommendationRagasMetricScore] = []
    metric_failures: list[dict[str, object]] = []

    for metric_name in _REQUIRED_METRICS:
        if metric_name in _CONTEXT_DEPENDENT_METRICS and not has_contexts:
            merged_metrics.append(
                GroundedRecommendationRagasMetricScore(
                    metric=metric_name,
                    status=RagasMetricStatus.NOT_APPLICABLE,
                    score=None,
                    error_code=None,
                    reason=_NO_CONTEXT_REASON,
                )
            )
            continue

        existing = metrics_by_name.get(metric_name)
        if existing is None:
            metric_score = GroundedRecommendationRagasMetricScore(
                metric=metric_name,
                status=RagasMetricStatus.FAILED,
                score=None,
                error_code="ragas_metric_missing",
                reason=None,
            )
        else:
            metric_score = existing

        merged_metrics.append(metric_score)
        if metric_score.status is RagasMetricStatus.FAILED:
            metric_failures.append(
                {
                    "case_id": case.case_id,
                    "kind": "metric_failed",
                    "metric": metric_score.metric.value,
                    "error_code": metric_score.error_code or "ragas_metric_failed",
                }
            )

    return (
        GroundedRecommendationRagasCaseScore(
            case_id=case.case_id,
            metrics=tuple(merged_metrics),
        ),
        metric_failures,
    )


def _failure_entries_for_metrics(
    *,
    case_id: str,
    kind: str,
    metrics: tuple[GroundedRecommendationRagasMetricScore, ...],
) -> list[dict[str, object]]:
    return [
        {
            "case_id": case_id,
            "kind": kind,
            "metric": metric.metric.value,
            "error_code": metric.error_code or kind,
        }
        for metric in metrics
    ]


def _failed_metrics_for_prediction(
    *,
    error_code: str,
) -> tuple[GroundedRecommendationRagasMetricScore, ...]:
    return tuple(
        GroundedRecommendationRagasMetricScore(
            metric=metric_name,
            status=RagasMetricStatus.FAILED,
            score=None,
            error_code=error_code,
            reason=None,
        )
        for metric_name in _REQUIRED_METRICS
    )


def _require_single_adapter_result(
    *,
    case_id: str,
    results: tuple[RagasEvaluationResult, ...],
) -> RagasEvaluationResult:
    if len(results) != 1:
        raise GroundedRecommendationRunnerError(
            f"RAGAS adapter returned {len(results)} results for case {case_id}"
        )

    result = results[0]
    if result.case_id != case_id:
        raise GroundedRecommendationRunnerError(
            f"RAGAS adapter returned unexpected case_id: {result.case_id}"
        )
    return result


def _resolve_run_status(
    *,
    dataset: GroundedRecommendationEvaluationDataset,
    predictions_by_case: dict[str, GroundedRecommendationPrediction],
    case_scores: tuple[GroundedRecommendationRagasCaseScore, ...],
) -> EvaluationRunStatus:
    for case in dataset.cases:
        prediction = predictions_by_case.get(case.case_id)
        if prediction is None:
            return EvaluationRunStatus.INCOMPLETE
        if prediction.status is EvaluationPredictionStatus.FAILED:
            return EvaluationRunStatus.INCOMPLETE

    for case_score in case_scores:
        for metric_score in case_score.metrics:
            if metric_score.status is RagasMetricStatus.FAILED:
                return EvaluationRunStatus.INCOMPLETE

    return EvaluationRunStatus.COMPLETE


def _require_known_case_ids(
    *,
    artifact_name: str,
    case_ids: tuple[str, ...],
    dataset_case_ids: set[str],
) -> None:
    unknown = sorted(set(case_ids) - dataset_case_ids)
    if unknown:
        raise GroundedRecommendationRunnerError(
            f"unknown {artifact_name} case IDs: " + ", ".join(unknown)
        )


def _reject_committed_output_directory(output_dir: Path) -> None:
    resolved_output = output_dir.resolve()
    committed_root = _COMMITTED_GROUNDED_ROOT.resolve()
    try:
        resolved_output.relative_to(committed_root)
    except ValueError:
        return
    raise GroundedRecommendationRunnerError(
        "output directories inside evals/grounded-recommendations are not allowed"
    )


def _reject_committed_artifact_overwrite(
    paths: GroundedRecommendationArtifactPaths,
) -> None:
    protected = {
        DEFAULT_GROUNDED_DATASET_PATH.resolve(),
        DEFAULT_GROUNDED_PREDICTIONS_PATH.resolve(),
        DEFAULT_GROUNDED_RAGAS_SCORES_PATH.resolve(),
    }
    candidates = (
        paths.ragas_scores_path,
        paths.deterministic_report_path,
        paths.ragas_report_path,
        paths.manifest_path,
        paths.failures_path,
    )
    for candidate in candidates:
        if candidate.resolve() in protected:
            raise GroundedRecommendationRunnerError(
                "canonical committed grounded recommendation artifacts must not be overwritten"
            )


def _ragas_scores_jsonl_bytes(
    case_scores: tuple[GroundedRecommendationRagasCaseScore, ...],
) -> bytes:
    return b"".join(
        canonical_json_bytes(case_score.model_dump(mode="json", exclude_none=False)) + b"\n"
        for case_score in case_scores
    )


def _failures_jsonl_bytes(entries: list[dict[str, object]]) -> bytes:
    if not entries:
        return b""
    return b"".join(canonical_json_bytes(entry) + b"\n" for entry in entries)


# Re-export exceptions that callers may need to catch from loading boundaries.
__all__ = [
    "DEFAULT_GROUNDED_DATASET_PATH",
    "DEFAULT_GROUNDED_PREDICTIONS_PATH",
    "DEFAULT_GROUNDED_RAGAS_SCORES_PATH",
    "ExternalProviderPermissionRequiredError",
    "GroundedRecommendationArtifactPaths",
    "GroundedRecommendationDatasetError",
    "GroundedRecommendationEvaluationError",
    "GroundedRecommendationPredictionError",
    "GroundedRecommendationRagasReportError",
    "GroundedRecommendationRagasScoreError",
    "GroundedRecommendationRunResult",
    "GroundedRecommendationRunnerError",
    "GroundedRecommendationScoreResult",
    "GroundedRecommendationValidationResult",
    "build_grounded_recommendation_question",
    "resolve_grounded_recommendation_artifact_paths",
    "run_grounded_recommendation_ragas_evaluation",
    "score_grounded_recommendation_artifacts",
    "validate_grounded_recommendation_artifacts",
]
