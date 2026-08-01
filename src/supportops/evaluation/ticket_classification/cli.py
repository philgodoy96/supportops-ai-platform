"""Command-line interface for ticket-classification evaluation."""

import argparse
import asyncio
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol, TextIO

from pydantic import ValidationError

from supportops.core.settings import LLMProviderName
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
        else:
            raise RuntimeError(
                "Evaluation parser produced an unsupported command.",
            )
    except (
        ExternalProviderPermissionRequiredError,
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
    payload = {
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
    }

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


if __name__ == "__main__":
    main()
