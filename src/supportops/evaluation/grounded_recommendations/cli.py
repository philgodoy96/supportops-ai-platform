"""Command-line interface for grounded recommendation evaluation."""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import TextIO

from pydantic import ValidationError

from supportops.evaluation.grounded_recommendations.dataset import (
    GroundedRecommendationDatasetError,
)
from supportops.evaluation.grounded_recommendations.evaluator import (
    GroundedRecommendationEvaluationError,
)
from supportops.evaluation.grounded_recommendations.predictions import (
    GroundedRecommendationPredictionError,
)
from supportops.evaluation.grounded_recommendations.ragas_execution import (
    GroundedRecommendationRagasExecutionError,
    OpenAIRagasAdapter,
    OpenAIRagasConfiguration,
)
from supportops.evaluation.grounded_recommendations.ragas_report import (
    GroundedRecommendationRagasReportError,
)
from supportops.evaluation.grounded_recommendations.ragas_scores import (
    GroundedRecommendationRagasScoreError,
)
from supportops.evaluation.grounded_recommendations.runner import (
    DEFAULT_GROUNDED_DATASET_PATH,
    DEFAULT_GROUNDED_PREDICTIONS_PATH,
    ExternalProviderPermissionRequiredError,
    GroundedRecommendationRunnerError,
    GroundedRecommendationRunResult,
    GroundedRecommendationScoreResult,
    GroundedRecommendationValidationResult,
    run_grounded_recommendation_ragas_evaluation,
    score_grounded_recommendation_artifacts,
    validate_grounded_recommendation_artifacts,
)

_EXIT_SUCCESS = 0
_EXIT_USAGE = 2
_EXIT_ARTIFACT_FAILURE = 3
_EXIT_ACKNOWLEDGEMENT_REQUIRED = 4

_EVALUATION_OPENAI_API_KEY_ENV = "SUPPORTOPS_EVALUATION_OPENAI_API_KEY"
_DEFAULT_PRICING_CATALOG_VERSION = "unspecified"
_DEFAULT_WORKFLOW_NAME = "ticket-processing"
_DEFAULT_WORKFLOW_VERSION = "controlled-support-v1"

_ARTIFACT_ERRORS = (
    GroundedRecommendationDatasetError,
    GroundedRecommendationEvaluationError,
    GroundedRecommendationPredictionError,
    GroundedRecommendationRagasExecutionError,
    GroundedRecommendationRagasReportError,
    GroundedRecommendationRagasScoreError,
    GroundedRecommendationRunnerError,
    ValidationError,
    OSError,
    ValueError,
)


def main(argv: Sequence[str] | None = None) -> None:
    """Run the grounded recommendation evaluation CLI and exit."""

    raise SystemExit(run_cli(argv))


def run_cli(
    argv: Sequence[str] | None = None,
    *,
    stdout: TextIO = sys.stdout,
    stderr: TextIO = sys.stderr,
    environ: dict[str, str] | None = None,
) -> int:
    """Execute one grounded recommendation CLI command."""

    parser = build_parser()
    try:
        arguments = parser.parse_args(list(argv) if argv is not None else None)
    except SystemExit as exit_error:
        code = exit_error.code
        if code in (0, None):
            return _EXIT_SUCCESS
        return _EXIT_USAGE

    env = environ if environ is not None else dict(os.environ)

    try:
        if arguments.command == "validate":
            validation_result = validate_grounded_recommendation_artifacts(
                dataset_path=arguments.dataset,
                predictions_path=arguments.predictions,
                ragas_scores_path=arguments.ragas_scores,
                output_path=arguments.output,
            )
            _write_validate_summary(stdout=stdout, result=validation_result)
            return _EXIT_SUCCESS

        if arguments.command == "score":
            score_result = score_grounded_recommendation_artifacts(
                dataset_path=arguments.dataset,
                predictions_path=arguments.predictions,
                ragas_scores_path=arguments.ragas_scores,
                output_dir=arguments.output_dir,
            )
            _write_score_summary(stdout=stdout, result=score_result)
            return _EXIT_SUCCESS

        if arguments.command == "run":
            return _execute_run_command(
                arguments=arguments,
                stdout=stdout,
                stderr=stderr,
                environ=env,
            )

        raise RuntimeError("Grounded recommendation parser produced an unsupported command.")
    except ExternalProviderPermissionRequiredError as error:
        _write_error(stderr=stderr, error=error)
        return _EXIT_ACKNOWLEDGEMENT_REQUIRED
    except _ARTIFACT_ERRORS as error:
        _write_error(stderr=stderr, error=error)
        return _EXIT_ARTIFACT_FAILURE
    except Exception:
        _write_message(
            stderr,
            "Grounded recommendation evaluation failed unexpectedly.",
        )
        return _EXIT_ARTIFACT_FAILURE


def build_parser() -> argparse.ArgumentParser:
    """Build the grounded recommendation evaluation command surface."""

    parser = argparse.ArgumentParser(
        prog="supportops-evaluate-grounded-recommendations",
        description=(
            "Validate and score grounded recommendation evaluation artifacts, "
            "and optionally evaluate existing predictions with RAGAS."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate_parser = subparsers.add_parser(
        "validate",
        help="Validate grounded recommendation artifacts without network access.",
    )
    validate_parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_GROUNDED_DATASET_PATH,
        help="Path to the grounded recommendation dataset JSONL.",
    )
    validate_parser.add_argument(
        "--predictions",
        type=Path,
        default=None,
        help="Optional path to a grounded recommendation prediction JSONL.",
    )
    validate_parser.add_argument(
        "--ragas-scores",
        type=Path,
        default=None,
        help="Optional path to a normalized RAGAS score JSONL artifact.",
    )
    validate_parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path for a canonical validation summary JSON artifact.",
    )

    score_parser = subparsers.add_parser(
        "score",
        help=(
            "Build deterministic complementary reports and optional offline "
            "RAGAS aggregates without network access."
        ),
    )
    score_parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_GROUNDED_DATASET_PATH,
        help="Path to the grounded recommendation dataset JSONL.",
    )
    score_parser.add_argument(
        "--predictions",
        type=Path,
        default=DEFAULT_GROUNDED_PREDICTIONS_PATH,
        help="Path to the grounded recommendation prediction JSONL.",
    )
    score_parser.add_argument(
        "--ragas-scores",
        type=Path,
        default=None,
        help="Optional path to a normalized RAGAS score JSONL artifact.",
    )
    score_parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional directory for deterministic and RAGAS report artifacts.",
    )

    run_parser = subparsers.add_parser(
        "run",
        help=(
            "Evaluate existing grounded recommendation predictions with RAGAS "
            "after explicit external-provider acknowledgement."
        ),
    )
    run_parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="Path to the grounded recommendation dataset JSONL.",
    )
    run_parser.add_argument(
        "--predictions",
        type=Path,
        required=True,
        help="Path to the grounded recommendation prediction JSONL.",
    )
    run_parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for generated evaluation artifacts under artifacts/.",
    )
    run_parser.add_argument(
        "--allow-external-provider",
        action="store_true",
        help="Acknowledge that this run may call an external evaluator provider.",
    )
    run_parser.add_argument(
        "--system-provider",
        required=True,
        help="Provider identity recorded for the system under evaluation.",
    )
    run_parser.add_argument(
        "--system-model",
        required=True,
        help="Model identity recorded for the system under evaluation.",
    )
    run_parser.add_argument(
        "--evaluator-provider",
        required=True,
        help="Evaluator provider identity. Only openai is supported.",
    )
    run_parser.add_argument(
        "--evaluator-model",
        required=True,
        help="Evaluator LLM model identity used by RAGAS.",
    )
    run_parser.add_argument(
        "--evaluator-embedding-model",
        required=True,
        help="Evaluator embedding model identity used by RAGAS.",
    )
    run_parser.add_argument(
        "--prompt-id",
        required=True,
        help="Prompt identity recorded in the evaluation manifest.",
    )
    run_parser.add_argument(
        "--prompt-version",
        type=_positive_integer,
        required=True,
        help="Positive prompt version recorded in the evaluation manifest.",
    )
    run_parser.add_argument(
        "--prompt-hash",
        required=True,
        help="SHA-256 prompt content hash recorded in the evaluation manifest.",
    )
    run_parser.add_argument(
        "--git-commit",
        required=True,
        help="Git commit hash recorded in the evaluation manifest.",
    )
    run_parser.add_argument(
        "--workflow-name",
        default=_DEFAULT_WORKFLOW_NAME,
        help="Workflow name recorded in the evaluation manifest.",
    )
    run_parser.add_argument(
        "--workflow-version",
        default=_DEFAULT_WORKFLOW_VERSION,
        help="Workflow version recorded in the evaluation manifest.",
    )
    run_parser.add_argument(
        "--pricing-catalog-version",
        default=_DEFAULT_PRICING_CATALOG_VERSION,
        help="Pricing catalog version recorded in the evaluation manifest.",
    )
    return parser


def _execute_run_command(
    *,
    arguments: argparse.Namespace,
    stdout: TextIO,
    stderr: TextIO,
    environ: dict[str, str],
) -> int:
    if not arguments.allow_external_provider:
        raise ExternalProviderPermissionRequiredError(
            "external grounded recommendation evaluation requires "
            "--allow-external-provider acknowledgement"
        )

    if arguments.evaluator_provider != "openai":
        raise GroundedRecommendationRunnerError("only evaluator_provider=openai is supported")

    if arguments.system_model == arguments.evaluator_model:
        _write_message(
            stderr,
            "warning: system model and evaluator model identities are equal",
        )

    api_key = environ.get(_EVALUATION_OPENAI_API_KEY_ENV, "").strip()
    if not api_key:
        raise GroundedRecommendationRagasExecutionError(
            f"{_EVALUATION_OPENAI_API_KEY_ENV} is required for external "
            "grounded recommendation evaluation"
        )

    adapter = OpenAIRagasAdapter(
        configuration=OpenAIRagasConfiguration(
            evaluator_model=arguments.evaluator_model,
            evaluator_embedding_model=arguments.evaluator_embedding_model,
        ),
        api_key=api_key,
    )

    result = run_grounded_recommendation_ragas_evaluation(
        dataset_path=arguments.dataset,
        predictions_path=arguments.predictions,
        output_dir=arguments.output_dir,
        allow_external_provider=True,
        system_provider=arguments.system_provider,
        system_model=arguments.system_model,
        evaluator_provider=arguments.evaluator_provider,
        evaluator_model=arguments.evaluator_model,
        evaluator_embedding_model=arguments.evaluator_embedding_model,
        prompt_id=arguments.prompt_id,
        prompt_version=arguments.prompt_version,
        prompt_hash=arguments.prompt_hash,
        workflow_name=arguments.workflow_name,
        workflow_version=arguments.workflow_version,
        git_commit=arguments.git_commit,
        pricing_catalog_version=arguments.pricing_catalog_version,
        ragas_adapter=adapter,
    )
    _write_run_summary(stdout=stdout, result=result)
    return _EXIT_SUCCESS


def _write_validate_summary(
    *,
    stdout: TextIO,
    result: GroundedRecommendationValidationResult,
) -> None:
    _write_message(
        stdout,
        (
            "validate status=valid "
            f"dataset_id={result.dataset_id} "
            f"dataset_version={result.dataset_version} "
            f"case_count={result.case_count} "
            f"dataset_hash={result.dataset_hash}"
        ),
    )
    if result.prediction_hash is not None:
        _write_message(
            stdout,
            (f"predictions hash={result.prediction_hash} count={result.prediction_count}"),
        )
    if result.ragas_score_hash is not None:
        _write_message(
            stdout,
            (f"ragas_scores hash={result.ragas_score_hash} count={result.ragas_score_case_count}"),
        )


def _write_score_summary(
    *,
    stdout: TextIO,
    result: GroundedRecommendationScoreResult,
) -> None:
    report = result.deterministic_report
    _write_message(
        stdout,
        (
            "score status=ok "
            f"dataset_id={report.dataset_id} "
            f"dataset_version={report.dataset_version} "
            f"case_count={report.case_count} "
            f"prediction_hash={report.prediction_hash} "
            f"report_hash={report.report_content_hash}"
        ),
    )
    if result.ragas_report is not None:
        ragas_report = result.ragas_report
        _write_message(
            stdout,
            (
                "ragas_report "
                f"scored_case_count={ragas_report.scored_case_count} "
                f"missing_case_count={ragas_report.missing_case_count} "
                f"report_hash={ragas_report.report_content_hash}"
            ),
        )
    if result.deterministic_report_path is not None:
        _write_message(
            stdout,
            f"deterministic_report={result.deterministic_report_path}",
        )
    if result.ragas_report_path is not None:
        _write_message(
            stdout,
            f"ragas_report_path={result.ragas_report_path}",
        )


def _write_run_summary(
    *,
    stdout: TextIO,
    result: GroundedRecommendationRunResult,
) -> None:
    _write_message(
        stdout,
        (
            "run "
            f"status={result.manifest.run_status.value} "
            f"case_count={len(result.case_scores)} "
            f"failure_count={result.failure_count} "
            f"score_hash={result.score_artifact_hash} "
            f"manifest={result.paths.manifest_path}"
        ),
    )


def _positive_integer(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _write_error(*, stderr: TextIO, error: BaseException) -> None:
    message = str(error).strip() or error.__class__.__name__
    _write_message(stderr, message)


def _write_message(stream: TextIO, message: str) -> None:
    stream.write(f"{message}\n")
    stream.flush()
