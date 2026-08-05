"""Command-line interface for repository evaluation regression."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import TextIO

from pydantic import ValidationError

from supportops.evaluation.controlled_support import (
    ControlledSupportDatasetError,
    ControlledSupportEvaluationError,
    ControlledSupportPredictionError,
)
from supportops.evaluation.human_approval import (
    HumanApprovalDatasetError,
    HumanApprovalEvaluationError,
    HumanApprovalPredictionError,
)
from supportops.evaluation.regression.models import (
    DOMAIN_CONTROLLED_SUPPORT,
    DOMAIN_HUMAN_APPROVAL,
    DOMAIN_SEMANTIC_RETRIEVAL,
    DOMAIN_TICKET_CLASSIFICATION,
    STABLE_DOMAIN_ORDER,
    RegressionAggregateStatus,
    RepositoryRegressionResult,
)
from supportops.evaluation.regression.runner import (
    MissingClassificationArtifactsError,
    RegressionRunnerError,
    UnknownRegressionDomainError,
    run_repository_regression,
)
from supportops.evaluation.semantic_retrieval import (
    SemanticRetrievalDatasetError,
    SemanticRetrievalEvaluationError,
    SemanticRetrievalPredictionError,
)
from supportops.evaluation.ticket_classification.dataset import (
    TICKET_CLASSIFICATION_EVALUATION_DATASET_ID,
    TICKET_CLASSIFICATION_EVALUATION_DATASET_VERSION,
    TicketClassificationDatasetError,
)
from supportops.evaluation.ticket_classification.evaluator import (
    TicketClassificationEvaluationError,
)
from supportops.evaluation.ticket_classification.predictions import (
    TicketClassificationPredictionError,
)

_EXIT_SUCCESS = 0
_EXIT_BLOCKING_FAILURE = 1
_EXIT_USAGE = 2
_EXIT_ARTIFACT_FAILURE = 3

_ARTIFACT_ERRORS = (
    ControlledSupportDatasetError,
    ControlledSupportEvaluationError,
    ControlledSupportPredictionError,
    HumanApprovalDatasetError,
    HumanApprovalEvaluationError,
    HumanApprovalPredictionError,
    SemanticRetrievalDatasetError,
    SemanticRetrievalEvaluationError,
    SemanticRetrievalPredictionError,
    TicketClassificationDatasetError,
    TicketClassificationEvaluationError,
    TicketClassificationPredictionError,
    ValidationError,
    OSError,
)


class RegressionCLIError(ValueError):
    """Raised for regression CLI usage or configuration failures."""


def main(argv: Sequence[str] | None = None) -> None:
    """Run the repository regression CLI and exit with a status code."""

    raise SystemExit(run_cli(argv))


def run_cli(
    argv: Sequence[str] | None = None,
    *,
    stdout: TextIO = sys.stdout,
    stderr: TextIO = sys.stderr,
) -> int:
    """Execute one regression CLI command with testable process boundaries."""

    parser = build_parser()
    try:
        arguments = parser.parse_args(list(argv) if argv is not None else None)
    except SystemExit as exit_error:
        code = exit_error.code
        if code in (0, None):
            return _EXIT_SUCCESS
        return _EXIT_USAGE

    try:
        if arguments.command != "score":
            raise RuntimeError("Regression parser produced an unsupported command.")

        selected_domains = _resolve_selected_domains(arguments)
        result = run_repository_regression(
            domains=selected_domains,
            semantic_retrieval_dataset=arguments.semantic_retrieval_dataset,
            semantic_retrieval_predictions=arguments.semantic_retrieval_predictions,
            controlled_support_dataset=arguments.controlled_support_dataset,
            controlled_support_predictions=arguments.controlled_support_predictions,
            human_approval_dataset=arguments.human_approval_dataset,
            human_approval_predictions=arguments.human_approval_predictions,
            classification_dataset=arguments.classification_dataset,
            classification_predictions=arguments.classification_predictions,
            classification_dataset_id=arguments.classification_dataset_id,
            classification_dataset_version=arguments.classification_dataset_version,
            output_path=arguments.output,
        )
    except (
        RegressionCLIError,
        UnknownRegressionDomainError,
        MissingClassificationArtifactsError,
    ) as error:
        _write_error(stderr=stderr, error=error)
        return _EXIT_USAGE
    except _ARTIFACT_ERRORS as error:
        _write_error(stderr=stderr, error=error)
        return _EXIT_ARTIFACT_FAILURE
    except RegressionRunnerError as error:
        _write_error(stderr=stderr, error=error)
        return _EXIT_ARTIFACT_FAILURE
    except Exception:
        _write_message(
            stderr,
            "Repository regression evaluation failed unexpectedly.",
        )
        return _EXIT_ARTIFACT_FAILURE

    _write_summary(stdout=stdout, result=result, output_path=arguments.output)

    if result.status is RegressionAggregateStatus.FAILED:
        return _EXIT_BLOCKING_FAILURE
    return _EXIT_SUCCESS


def build_parser() -> argparse.ArgumentParser:
    """Build the stable repository regression command surface."""

    parser = argparse.ArgumentParser(
        prog="supportops-evaluate-regression",
        description=(
            "Score committed static evaluation evidence and evaluate "
            "deterministic repository release gates without network calls."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    score_parser = subparsers.add_parser(
        "score",
        help=(
            "Score selected domain dataset/prediction pairs and evaluate "
            "release gates without provider execution."
        ),
    )
    score_parser.add_argument(
        "--domain",
        action="append",
        choices=tuple(STABLE_DOMAIN_ORDER),
        dest="domains",
        help=(
            "Domain to score. May be repeated. Defaults to semantic-retrieval, "
            "controlled-support, and human-approval."
        ),
    )
    score_parser.add_argument(
        "--semantic-retrieval-dataset",
        type=Path,
        default=None,
        help="Path to the semantic-retrieval evaluation dataset JSONL.",
    )
    score_parser.add_argument(
        "--semantic-retrieval-predictions",
        type=Path,
        default=None,
        help="Path to the semantic-retrieval static prediction JSONL.",
    )
    score_parser.add_argument(
        "--controlled-support-dataset",
        type=Path,
        default=None,
        help="Path to the controlled-support evaluation dataset JSONL.",
    )
    score_parser.add_argument(
        "--controlled-support-predictions",
        type=Path,
        default=None,
        help="Path to the controlled-support static prediction JSONL.",
    )
    score_parser.add_argument(
        "--human-approval-dataset",
        type=Path,
        default=None,
        help="Path to the human-approval evaluation dataset JSONL.",
    )
    score_parser.add_argument(
        "--human-approval-predictions",
        type=Path,
        default=None,
        help="Path to the human-approval static prediction JSONL.",
    )
    score_parser.add_argument(
        "--classification-dataset",
        type=Path,
        default=None,
        help="Path to the ticket-classification evaluation dataset JSONL.",
    )
    score_parser.add_argument(
        "--classification-predictions",
        type=Path,
        default=None,
        help="Path to the ticket-classification prediction JSONL.",
    )
    score_parser.add_argument(
        "--classification-dataset-id",
        default=TICKET_CLASSIFICATION_EVALUATION_DATASET_ID,
        help="Stable classification dataset identity recorded while scoring.",
    )
    score_parser.add_argument(
        "--classification-dataset-version",
        type=_positive_integer,
        default=TICKET_CLASSIFICATION_EVALUATION_DATASET_VERSION,
        help="Positive classification dataset version recorded while scoring.",
    )
    score_parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path for the canonical repository regression JSON artifact.",
    )
    return parser


def _resolve_selected_domains(arguments: argparse.Namespace) -> tuple[str, ...]:
    selected = tuple(arguments.domains) if arguments.domains else None
    if selected is not None and not selected:
        raise RegressionCLIError("at least one --domain must be selected")

    if (
        selected is not None
        and DOMAIN_TICKET_CLASSIFICATION in selected
        and (
            arguments.classification_dataset is None or arguments.classification_predictions is None
        )
    ):
        raise RegressionCLIError(
            "selecting ticket-classification requires "
            "--classification-dataset and --classification-predictions",
        )

    if selected is None:
        return (
            DOMAIN_SEMANTIC_RETRIEVAL,
            DOMAIN_CONTROLLED_SUPPORT,
            DOMAIN_HUMAN_APPROVAL,
        )
    return selected


def _write_summary(
    *,
    stdout: TextIO,
    result: RepositoryRegressionResult,
    output_path: Path | None,
) -> None:
    domain_statuses = ", ".join(
        f"{domain_result.domain}={domain_result.status.value}"
        for domain_result in result.domain_results
    )
    not_provided = ",".join(result.not_provided_domains) if result.not_provided_domains else "none"
    lines = [
        f"status={result.status.value}",
        f"blocking_failure_count={result.blocking_failure_count}",
        f"incomplete_domain_count={result.incomplete_domain_count}",
        f"domains={domain_statuses}",
        f"not_provided_domains={not_provided}",
        f"content_hash={result.content_hash}",
    ]
    if output_path is not None:
        lines.append(f"output={output_path.as_posix()}")
    _write_message(stdout, "\n".join(lines))


def _write_error(*, stderr: TextIO, error: BaseException) -> None:
    message = str(error).strip() or error.__class__.__name__
    _write_message(stderr, message)


def _write_message(stream: TextIO, message: str) -> None:
    stream.write(f"{message}\n")
    stream.flush()


def _positive_integer(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed
