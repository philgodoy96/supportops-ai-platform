"""Command-line interface for ticket-classification evaluation."""

import argparse
import asyncio
import json
import sys
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Protocol, TextIO

from pydantic import ValidationError

from supportops.ai.prompts.registry import (
    PromptDefinitionNotFoundError,
)
from supportops.core.settings import LLMProviderName
from supportops.evaluation.ticket_classification.comparison import (
    TicketClassificationComparisonEvidenceKind,
)
from supportops.evaluation.ticket_classification.composition import (
    TicketClassificationEvaluationLLMRuntime,
    create_ticket_classification_evaluation_runtime,
)
from supportops.evaluation.ticket_classification.dataset import (
    TICKET_CLASSIFICATION_EVALUATION_DATASET_ID,
    TICKET_CLASSIFICATION_EVALUATION_DATASET_VERSION,
    TicketClassificationDatasetError,
    load_ticket_classification_dataset,
)
from supportops.evaluation.ticket_classification.evaluator import (
    TicketClassificationEvaluationError,
    TicketClassificationEvaluationReport,
    evaluate_ticket_classification_predictions,
)
from supportops.evaluation.ticket_classification.iteration_runner import (
    run_ticket_classification_prompt_comparison,
    run_ticket_classification_prompt_decision,
    validate_ticket_classification_failure_analysis_artifact,
    write_ticket_classification_prompt_comparison_run,
    write_ticket_classification_prompt_decision_run,
)
from supportops.evaluation.ticket_classification.predictions import (
    TicketClassificationPredictionError,
    load_ticket_classification_predictions,
)
from supportops.evaluation.ticket_classification.predictor import (
    TicketClassificationEvaluationPredictor,
)
from supportops.evaluation.ticket_classification.runner import (
    run_ticket_classification_evaluation,
    write_ticket_classification_evaluation_report,
    write_ticket_classification_predictions,
)
from supportops.evaluation.ticket_classification.settings import (
    TicketClassificationEvaluationSettings,
)

_EXIT_SUCCESS = 0
_EXIT_RUNTIME_FAILURE = 1
_EXIT_USAGE_OR_CONFIGURATION_FAILURE = 2


class TicketClassificationEvaluationCLIError(ValueError):
    """Raised when CLI execution is not explicitly safe."""


class ExternalProviderPermissionRequiredError(
    TicketClassificationEvaluationCLIError,
):
    """Raised when an external provider was not explicitly allowed."""


class EvaluationSettingsFactory(Protocol):
    """Construct evaluation-only environment settings."""

    def __call__(
        self,
    ) -> TicketClassificationEvaluationSettings:
        """Return validated evaluation settings."""

        ...


class EvaluationRuntimeFactory(Protocol):
    """Construct one explicitly selected evaluation runtime."""

    def __call__(
        self,
        *,
        provider_name: LLMProviderName,
        settings: TicketClassificationEvaluationSettings,
    ) -> TicketClassificationEvaluationLLMRuntime:
        """Return one process-scoped evaluation runtime."""

        ...


def main() -> None:
    """Run the evaluation CLI and exit with an operational status."""

    raise SystemExit(
        asyncio.run(
            run_cli(),
        ),
    )


async def run_cli(
    argv: Sequence[str] | None = None,
    *,
    stdout: TextIO = sys.stdout,
    stderr: TextIO = sys.stderr,
    settings_factory: EvaluationSettingsFactory = (TicketClassificationEvaluationSettings),
    runtime_factory: EvaluationRuntimeFactory = (create_ticket_classification_evaluation_runtime),
) -> int:
    """Execute one CLI command with testable process boundaries."""

    parser = build_parser()
    arguments = parser.parse_args(argv)

    try:
        if arguments.command == "score":
            _execute_score_command(
                arguments=arguments,
                stdout=stdout,
            )
        elif arguments.command == "run":
            await _execute_run_command(
                arguments=arguments,
                stdout=stdout,
                settings_factory=settings_factory,
                runtime_factory=runtime_factory,
            )
        elif arguments.command == "analyze":
            _execute_analyze_command(
                arguments=arguments,
                stdout=stdout,
            )
        elif arguments.command == "compare":
            _execute_compare_command(
                arguments=arguments,
                stdout=stdout,
            )
        elif arguments.command == "decide":
            _execute_decide_command(
                arguments=arguments,
                stdout=stdout,
            )
        else:
            raise RuntimeError(
                "Evaluation parser produced an unsupported command.",
            )
    except (
        ExternalProviderPermissionRequiredError,
        PromptDefinitionNotFoundError,
        TicketClassificationDatasetError,
        TicketClassificationPredictionError,
        TicketClassificationEvaluationError,
        ValidationError,
        ValueError,
    ) as error:
        _write_expected_error(
            stderr=stderr,
            error=error,
        )
        return _EXIT_USAGE_OR_CONFIGURATION_FAILURE
    except OSError:
        _write_runtime_error(
            stderr=stderr,
            message=("Classification evaluation failed while reading or writing an artifact."),
        )
        return _EXIT_RUNTIME_FAILURE
    except Exception:
        _write_runtime_error(
            stderr=stderr,
            message=("Classification evaluation failed unexpectedly."),
        )
        return _EXIT_RUNTIME_FAILURE

    return _EXIT_SUCCESS


def build_parser() -> argparse.ArgumentParser:
    """Build the stable classification evaluation command surface."""

    parser = argparse.ArgumentParser(
        prog="supportops-evaluate-classification",
        description=("Evaluate versioned structured ticket classification behavior."),
    )
    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
    )

    score_parser = subparsers.add_parser(
        "score",
        help=("Score an existing prediction JSONL artifact without initializing a provider."),
    )
    _add_dataset_arguments(score_parser)
    score_parser.add_argument(
        "--predictions",
        type=Path,
        required=True,
        help="Path to the prediction JSONL artifact.",
    )
    score_parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path to the deterministic report JSON artifact.",
    )

    run_parser = subparsers.add_parser(
        "run",
        help=("Generate predictions through an explicitly selected provider and score them."),
    )
    _add_dataset_arguments(run_parser)
    run_parser.add_argument(
        "--provider",
        choices=tuple(provider.value for provider in LLMProviderName),
        required=True,
        help="Explicit LLM provider used for this run.",
    )
    run_parser.add_argument(
        "--allow-external-provider",
        action="store_true",
        help=("Required explicit acknowledgement before the OpenAI provider may be initialized."),
    )
    run_parser.add_argument(
        "--prompt-version",
        type=_positive_integer,
        default=1,
        help=(
            "Explicit positive ticket-classification prompt version used "
            "only for prediction generation. Evaluation never selects a "
            "prompt implicitly."
        ),
    )
    run_parser.add_argument(
        "--predictions-output",
        type=Path,
        required=True,
        help="Path to the generated prediction JSONL artifact.",
    )
    run_parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path to the deterministic report JSON artifact.",
    )

    analyze_parser = subparsers.add_parser(
        "analyze",
        help=("Validate a committed failure-analysis artifact without initializing a provider."),
    )
    _add_dataset_arguments(analyze_parser)
    analyze_parser.add_argument(
        "--split-manifest",
        type=Path,
        required=True,
        help="Path to the versioned split-manifest artifact.",
    )
    analyze_parser.add_argument(
        "--analysis",
        type=Path,
        required=True,
        help="Path to the committed failure-analysis artifact.",
    )

    compare_parser = subparsers.add_parser(
        "compare",
        help=(
            "Compare existing baseline and candidate prediction artifacts "
            "without initializing a provider."
        ),
    )
    _add_dataset_arguments(compare_parser)
    compare_parser.add_argument(
        "--split-manifest",
        type=Path,
        required=True,
        help="Path to the versioned split-manifest artifact.",
    )
    compare_parser.add_argument(
        "--baseline-predictions",
        type=Path,
        required=True,
        help="Path to the baseline prediction JSONL artifact.",
    )
    compare_parser.add_argument(
        "--candidate-predictions",
        type=Path,
        required=True,
        help="Path to the candidate prediction JSONL artifact.",
    )
    compare_parser.add_argument(
        "--evidence-kind",
        choices=tuple(kind.value for kind in TicketClassificationComparisonEvidenceKind),
        required=True,
        help="Authority represented by the paired prediction evidence.",
    )
    compare_parser.add_argument(
        "--capture-timestamp",
        type=_aware_datetime,
        required=True,
        help="Aware ISO-8601 capture timestamp recorded in evaluation manifests.",
    )
    compare_parser.add_argument(
        "--git-commit",
        required=True,
        help="Git commit recorded in evaluation manifests.",
    )
    compare_parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path to the paired comparison JSON artifact.",
    )
    compare_parser.add_argument(
        "--baseline-manifest-output",
        type=Path,
        required=True,
        help="Path to the baseline evaluation manifest JSON artifact.",
    )
    compare_parser.add_argument(
        "--candidate-manifest-output",
        type=Path,
        required=True,
        help="Path to the candidate evaluation manifest JSON artifact.",
    )
    compare_parser.add_argument(
        "--pair-manifest-output",
        type=Path,
        required=True,
        help="Path to the paired provenance manifest JSON artifact.",
    )

    decide_parser = subparsers.add_parser(
        "decide",
        help=(
            "Rebuild a governed prompt decision from comparison and review "
            "evidence without initializing a provider."
        ),
    )
    decide_parser.add_argument(
        "--comparison",
        type=Path,
        required=True,
        help="Path to the paired comparison JSON artifact.",
    )
    decide_parser.add_argument(
        "--decision-template",
        type=Path,
        required=True,
        help=("Path to the decision template supplying immutable review and governance fields."),
    )
    decide_parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path to the rebuilt prompt decision JSON artifact.",
    )

    return parser


def _add_dataset_arguments(
    parser: argparse.ArgumentParser,
) -> None:
    parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="Path to the versioned evaluation JSONL dataset.",
    )
    parser.add_argument(
        "--dataset-id",
        default=(TICKET_CLASSIFICATION_EVALUATION_DATASET_ID),
        help="Stable dataset identity recorded in the report.",
    )
    parser.add_argument(
        "--dataset-version",
        type=_positive_integer,
        default=(TICKET_CLASSIFICATION_EVALUATION_DATASET_VERSION),
        help="Positive dataset version recorded in the report.",
    )


def _execute_score_command(
    *,
    arguments: argparse.Namespace,
    stdout: TextIO,
) -> None:
    dataset = load_ticket_classification_dataset(
        arguments.dataset,
        dataset_id=arguments.dataset_id,
        version=arguments.dataset_version,
    )
    predictions = load_ticket_classification_predictions(
        arguments.predictions,
    )
    report = evaluate_ticket_classification_predictions(
        dataset=dataset,
        predictions=predictions,
    )

    write_ticket_classification_evaluation_report(
        arguments.output,
        report,
    )
    _write_summary(
        stdout=stdout,
        command="score",
        predictions_path=arguments.predictions,
        report_path=arguments.output,
        report=report,
    )


async def _execute_run_command(
    *,
    arguments: argparse.Namespace,
    stdout: TextIO,
    settings_factory: EvaluationSettingsFactory,
    runtime_factory: EvaluationRuntimeFactory,
) -> None:
    provider_name = LLMProviderName(
        arguments.provider,
    )
    _validate_external_provider_permission(
        provider_name=provider_name,
        allow_external_provider=(arguments.allow_external_provider),
    )

    dataset = load_ticket_classification_dataset(
        arguments.dataset,
        dataset_id=arguments.dataset_id,
        version=arguments.dataset_version,
    )
    settings = settings_factory()
    runtime = runtime_factory(
        provider_name=provider_name,
        settings=settings,
    )

    try:
        predictor = TicketClassificationEvaluationPredictor(
            gateway=runtime.gateway,
            provider_name=runtime.provider.provider_name,
            model=runtime.model,
            request_timeout_seconds=(settings.llm_request_timeout_seconds),
        )
        result = await run_ticket_classification_evaluation(
            dataset=dataset,
            predictor=predictor,
            prompt_version=arguments.prompt_version,
        )
    finally:
        await runtime.close()

    write_ticket_classification_predictions(
        arguments.predictions_output,
        result.predictions,
    )
    write_ticket_classification_evaluation_report(
        arguments.output,
        result.report,
    )
    _write_summary(
        stdout=stdout,
        command="run",
        predictions_path=arguments.predictions_output,
        report_path=arguments.output,
        report=result.report,
    )


def _execute_analyze_command(
    *,
    arguments: argparse.Namespace,
    stdout: TextIO,
) -> None:
    analysis = validate_ticket_classification_failure_analysis_artifact(
        dataset_path=arguments.dataset,
        dataset_id=arguments.dataset_id,
        dataset_version=arguments.dataset_version,
        split_manifest_path=arguments.split_manifest,
        analysis_path=arguments.analysis,
    )
    _write_json_line(
        stdout=stdout,
        payload={
            "command": "analyze",
            "analysis_id": analysis.analysis_id,
            "analysis_version": analysis.analysis_version,
            "analyzed_split": analysis.analyzed_split,
            "analyzed_case_count": len(analysis.analyzed_case_ids),
            "observation_count": len(analysis.observations),
            "analysis_content_hash": analysis.analysis_content_hash,
            "analysis_path": str(arguments.analysis),
        },
    )


def _execute_compare_command(
    *,
    arguments: argparse.Namespace,
    stdout: TextIO,
) -> None:
    result = run_ticket_classification_prompt_comparison(
        dataset_path=arguments.dataset,
        dataset_id=arguments.dataset_id,
        dataset_version=arguments.dataset_version,
        split_manifest_path=arguments.split_manifest,
        baseline_predictions_path=arguments.baseline_predictions,
        candidate_predictions_path=arguments.candidate_predictions,
        evidence_kind=TicketClassificationComparisonEvidenceKind(
            arguments.evidence_kind,
        ),
        capture_timestamp=arguments.capture_timestamp,
        git_commit=arguments.git_commit,
    )
    write_ticket_classification_prompt_comparison_run(
        comparison_output=arguments.output,
        baseline_manifest_output=arguments.baseline_manifest_output,
        candidate_manifest_output=arguments.candidate_manifest_output,
        pair_manifest_output=arguments.pair_manifest_output,
        result=result,
    )
    comparison = result.comparison
    _write_json_line(
        stdout=stdout,
        payload={
            "command": "compare",
            "comparison_id": comparison.comparison_id,
            "comparison_version": comparison.comparison_version,
            "evidence_kind": comparison.evidence_kind.value,
            "case_count": comparison.case_count,
            "run_status": comparison.run_status.value,
            "gate_status": comparison.gate_evaluation.status.value,
            "blocking_failure_count": (comparison.gate_evaluation.blocking_failure_count),
            "not_applicable_count": (comparison.gate_evaluation.not_applicable_count),
            "improved_case_count": len(comparison.improved_case_ids),
            "regressed_case_count": len(comparison.regressed_case_ids),
            "comparison_content_hash": comparison.comparison_content_hash,
            "baseline_manifest_content_hash": (result.pair_manifest.baseline_manifest_content_hash),
            "candidate_manifest_content_hash": (
                result.pair_manifest.candidate_manifest_content_hash
            ),
            "pair_manifest_content_hash": result.pair_manifest.content_hash,
            "comparison_path": str(arguments.output),
            "baseline_manifest_path": str(arguments.baseline_manifest_output),
            "candidate_manifest_path": str(arguments.candidate_manifest_output),
            "pair_manifest_path": str(arguments.pair_manifest_output),
        },
    )


def _execute_decide_command(
    *,
    arguments: argparse.Namespace,
    stdout: TextIO,
) -> None:
    decision = run_ticket_classification_prompt_decision(
        comparison_path=arguments.comparison,
        decision_template_path=arguments.decision_template,
    )
    write_ticket_classification_prompt_decision_run(
        output=arguments.output,
        decision=decision,
    )
    _write_json_line(
        stdout=stdout,
        payload={
            "command": "decide",
            "decision_id": decision.decision_id,
            "decision_version": decision.decision_version,
            "outcome": decision.outcome.value,
            "run_status": decision.run_status.value,
            "approved_for_runtime_adoption": (decision.review.approved_for_runtime_adoption),
            "separate_runtime_adoption_required": (decision.separate_runtime_adoption_required),
            "blocking_reason_count": len(decision.blocking_reasons),
            "decision_content_hash": decision.decision_content_hash,
            "comparison_content_hash": decision.comparison_content_hash,
            "decision_path": str(arguments.output),
        },
    )


def _validate_external_provider_permission(
    *,
    provider_name: LLMProviderName,
    allow_external_provider: bool,
) -> None:
    if provider_name is LLMProviderName.OPENAI and not allow_external_provider:
        raise ExternalProviderPermissionRequiredError(
            "OpenAI evaluation requires --allow-external-provider."
        )

    if provider_name is LLMProviderName.MOCK and allow_external_provider:
        raise TicketClassificationEvaluationCLIError(
            "--allow-external-provider is valid only when --provider openai is selected."
        )


def _write_summary(
    *,
    stdout: TextIO,
    command: str,
    predictions_path: Path,
    report_path: Path,
    report: TicketClassificationEvaluationReport,
) -> None:
    _write_json_line(
        stdout=stdout,
        payload={
            "command": command,
            "dataset_id": report.dataset_id,
            "dataset_version": report.dataset_version,
            "case_count": report.case_count,
            "successful_prediction_count": (report.successful_prediction_count),
            "failed_prediction_count": (report.failed_prediction_count),
            "structured_label_exact_match_rate": str(
                report.structured_label_exact_match.rate,
            ),
            "known_total_tokens": report.known_total_tokens,
            "known_estimated_total_cost_usd": str(
                report.known_estimated_total_cost_usd,
            ),
            "predictions_path": str(predictions_path),
            "report_path": str(report_path),
            "report_content_hash": (report.report_content_hash),
        },
    )


def _write_json_line(
    *,
    stdout: TextIO,
    payload: Mapping[str, object],
) -> None:
    stdout.write(
        json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n",
    )


def _write_expected_error(
    *,
    stderr: TextIO,
    error: Exception,
) -> None:
    if isinstance(error, ValidationError):
        message = "Classification evaluation configuration is invalid."
    else:
        message = str(error)

    stderr.write(
        f"evaluation_error: {message}\n",
    )


def _write_runtime_error(
    *,
    stderr: TextIO,
    message: str,
) -> None:
    stderr.write(
        f"evaluation_runtime_error: {message}\n",
    )


def _positive_integer(
    value: str,
) -> int:
    parsed_value = int(value)

    if parsed_value <= 0:
        raise argparse.ArgumentTypeError(
            "value must be a positive integer",
        )

    return parsed_value


def _aware_datetime(value: str) -> datetime:
    normalized = value[:-1] + "+00:00" if value.endswith("Z") else value

    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "value must be an aware ISO-8601 datetime",
        ) from error

    if parsed.tzinfo is None:
        raise argparse.ArgumentTypeError(
            "value must be an aware ISO-8601 datetime",
        )

    return parsed


if __name__ == "__main__":
    main()
