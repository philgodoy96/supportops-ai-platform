"""Deterministic score-only repository regression runner."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from pydantic import ValidationError

from supportops.evaluation.contracts.artifacts import write_canonical_json_atomically
from supportops.evaluation.controlled_support import (
    ControlledSupportDatasetError,
    ControlledSupportEvaluationError,
    ControlledSupportPredictionError,
    evaluate_controlled_support_predictions,
    load_controlled_support_dataset,
    load_controlled_support_predictions,
)
from supportops.evaluation.human_approval import (
    HumanApprovalDatasetError,
    HumanApprovalEvaluationError,
    HumanApprovalPredictionError,
    evaluate_human_approval_predictions,
    load_human_approval_dataset,
    load_human_approval_predictions,
)
from supportops.evaluation.regression.gates import (
    adapt_classification_release_gate_evaluation,
    evaluate_controlled_support_release_gates,
    evaluate_human_approval_release_gates,
    evaluate_semantic_retrieval_release_gates,
)
from supportops.evaluation.regression.models import (
    DOMAIN_CONTROLLED_SUPPORT,
    DOMAIN_HUMAN_APPROVAL,
    DOMAIN_SEMANTIC_RETRIEVAL,
    DOMAIN_TICKET_CLASSIFICATION,
    STABLE_DOMAIN_ORDER,
    SUPPORTED_REGRESSION_DOMAINS,
    RegressionDomainProfileResult,
    RepositoryRegressionResult,
    build_repository_regression_result,
)
from supportops.evaluation.semantic_retrieval import (
    SemanticRetrievalDatasetError,
    SemanticRetrievalEvaluationError,
    SemanticRetrievalPredictionError,
    evaluate_semantic_retrieval_predictions,
    load_semantic_retrieval_dataset,
    load_semantic_retrieval_predictions,
)
from supportops.evaluation.ticket_classification.dataset import (
    TICKET_CLASSIFICATION_EVALUATION_DATASET_ID,
    TICKET_CLASSIFICATION_EVALUATION_DATASET_VERSION,
    TicketClassificationDatasetError,
    load_ticket_classification_dataset,
)
from supportops.evaluation.ticket_classification.evaluator import (
    TicketClassificationEvaluationError,
)
from supportops.evaluation.ticket_classification.predictions import (
    TicketClassificationPredictionError,
    load_ticket_classification_predictions,
)
from supportops.evaluation.ticket_classification.runner import (
    score_ticket_classification_predictions_with_release_gates,
)

DEFAULT_SEMANTIC_RETRIEVAL_DATASET_PATH = Path(
    "evals/semantic-retrieval/datasets/semantic-retrieval-eval-v1.jsonl"
)
DEFAULT_SEMANTIC_RETRIEVAL_PREDICTIONS_PATH = Path(
    "evals/semantic-retrieval/predictions/semantic-retrieval-eval-v1.static.jsonl"
)
DEFAULT_CONTROLLED_SUPPORT_DATASET_PATH = Path(
    "evals/controlled-support/datasets/controlled-support-eval-v1.jsonl"
)
DEFAULT_CONTROLLED_SUPPORT_PREDICTIONS_PATH = Path(
    "evals/controlled-support/predictions/controlled-support-eval-v1.static.jsonl"
)
DEFAULT_HUMAN_APPROVAL_DATASET_PATH = Path(
    "evals/human-approval/datasets/human-approval-eval-v1.jsonl"
)
DEFAULT_HUMAN_APPROVAL_PREDICTIONS_PATH = Path(
    "evals/human-approval/predictions/human-approval-eval-v1.static.jsonl"
)

DEFAULT_REGRESSION_DOMAINS: tuple[str, ...] = (
    DOMAIN_SEMANTIC_RETRIEVAL,
    DOMAIN_CONTROLLED_SUPPORT,
    DOMAIN_HUMAN_APPROVAL,
)


class RegressionRunnerError(ValueError):
    """Raised when repository regression cannot be completed."""


class UnknownRegressionDomainError(RegressionRunnerError):
    """Raised when an unsupported domain is selected."""


class MissingClassificationArtifactsError(RegressionRunnerError):
    """Raised when classification is selected without artifact paths."""


def run_repository_regression(
    *,
    domains: Sequence[str] | None = None,
    semantic_retrieval_dataset: Path | None = None,
    semantic_retrieval_predictions: Path | None = None,
    controlled_support_dataset: Path | None = None,
    controlled_support_predictions: Path | None = None,
    human_approval_dataset: Path | None = None,
    human_approval_predictions: Path | None = None,
    classification_dataset: Path | None = None,
    classification_predictions: Path | None = None,
    classification_dataset_id: str = TICKET_CLASSIFICATION_EVALUATION_DATASET_ID,
    classification_dataset_version: int = TICKET_CLASSIFICATION_EVALUATION_DATASET_VERSION,
    output_path: Path | None = None,
) -> RepositoryRegressionResult:
    """Score selected domains from static artifacts and aggregate release gates.

    Performs no network calls, provider execution, database access, or RAGAS.
    Existing output artifacts are never replaced when scoring fails.
    """

    selected_domains = _normalize_selected_domains(domains)
    domain_results: list[RegressionDomainProfileResult] = []

    try:
        for domain in selected_domains:
            if domain == DOMAIN_SEMANTIC_RETRIEVAL:
                domain_results.append(
                    _score_semantic_retrieval(
                        dataset_path=(
                            semantic_retrieval_dataset or DEFAULT_SEMANTIC_RETRIEVAL_DATASET_PATH
                        ),
                        predictions_path=(
                            semantic_retrieval_predictions
                            or DEFAULT_SEMANTIC_RETRIEVAL_PREDICTIONS_PATH
                        ),
                    )
                )
            elif domain == DOMAIN_CONTROLLED_SUPPORT:
                domain_results.append(
                    _score_controlled_support(
                        dataset_path=(
                            controlled_support_dataset or DEFAULT_CONTROLLED_SUPPORT_DATASET_PATH
                        ),
                        predictions_path=(
                            controlled_support_predictions
                            or DEFAULT_CONTROLLED_SUPPORT_PREDICTIONS_PATH
                        ),
                    )
                )
            elif domain == DOMAIN_HUMAN_APPROVAL:
                domain_results.append(
                    _score_human_approval(
                        dataset_path=(
                            human_approval_dataset or DEFAULT_HUMAN_APPROVAL_DATASET_PATH
                        ),
                        predictions_path=(
                            human_approval_predictions or DEFAULT_HUMAN_APPROVAL_PREDICTIONS_PATH
                        ),
                    )
                )
            elif domain == DOMAIN_TICKET_CLASSIFICATION:
                domain_results.append(
                    _score_ticket_classification(
                        dataset_path=classification_dataset,
                        predictions_path=classification_predictions,
                        dataset_id=classification_dataset_id,
                        dataset_version=classification_dataset_version,
                    )
                )
            else:
                raise UnknownRegressionDomainError(f"unsupported regression domain: {domain}")
    except (
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
        RegressionRunnerError,
    ):
        raise
    except Exception as error:
        raise RegressionRunnerError(
            f"repository regression scoring failed: {error}",
        ) from error

    not_provided_domains: tuple[str, ...] = ()
    if DOMAIN_TICKET_CLASSIFICATION not in selected_domains:
        not_provided_domains = (DOMAIN_TICKET_CLASSIFICATION,)

    result = build_repository_regression_result(
        domain_results=tuple(domain_results),
        not_provided_domains=not_provided_domains,
    )

    if output_path is not None:
        write_canonical_json_atomically(output_path, result)

    return result


def _normalize_selected_domains(domains: Sequence[str] | None) -> tuple[str, ...]:
    selected = tuple(domains) if domains is not None else DEFAULT_REGRESSION_DOMAINS
    if not selected:
        raise RegressionRunnerError("at least one regression domain must be selected")

    unknown = sorted({domain for domain in selected if domain not in SUPPORTED_REGRESSION_DOMAINS})
    if unknown:
        raise UnknownRegressionDomainError(
            "unsupported regression domain(s): " + ", ".join(unknown),
        )

    order = {domain: index for index, domain in enumerate(STABLE_DOMAIN_ORDER)}
    unique: list[str] = []
    seen: set[str] = set()
    for domain in selected:
        if domain not in seen:
            unique.append(domain)
            seen.add(domain)
    return tuple(sorted(unique, key=lambda domain: order[domain]))


def _score_semantic_retrieval(
    *,
    dataset_path: Path,
    predictions_path: Path,
) -> RegressionDomainProfileResult:
    dataset = load_semantic_retrieval_dataset(dataset_path)
    predictions, prediction_hash = load_semantic_retrieval_predictions(predictions_path)
    report = evaluate_semantic_retrieval_predictions(
        dataset=dataset,
        predictions=predictions,
        prediction_hash=prediction_hash,
    )
    return evaluate_semantic_retrieval_release_gates(report)


def _score_controlled_support(
    *,
    dataset_path: Path,
    predictions_path: Path,
) -> RegressionDomainProfileResult:
    dataset = load_controlled_support_dataset(dataset_path)
    predictions, prediction_hash = load_controlled_support_predictions(predictions_path)
    report = evaluate_controlled_support_predictions(
        dataset=dataset,
        predictions=predictions,
        prediction_hash=prediction_hash,
    )
    return evaluate_controlled_support_release_gates(report)


def _score_human_approval(
    *,
    dataset_path: Path,
    predictions_path: Path,
) -> RegressionDomainProfileResult:
    dataset = load_human_approval_dataset(dataset_path)
    predictions, prediction_hash = load_human_approval_predictions(predictions_path)
    report = evaluate_human_approval_predictions(
        dataset=dataset,
        predictions=predictions,
        prediction_hash=prediction_hash,
    )
    return evaluate_human_approval_release_gates(report)


def _score_ticket_classification(
    *,
    dataset_path: Path | None,
    predictions_path: Path | None,
    dataset_id: str,
    dataset_version: int,
) -> RegressionDomainProfileResult:
    if dataset_path is None or predictions_path is None:
        raise MissingClassificationArtifactsError(
            "ticket-classification requires --classification-dataset and "
            "--classification-predictions because no committed static "
            "classification prediction fixture exists",
        )

    dataset = load_ticket_classification_dataset(
        dataset_path,
        dataset_id=dataset_id,
        version=dataset_version,
    )
    predictions = load_ticket_classification_predictions(predictions_path)
    _report, gate_evaluation = score_ticket_classification_predictions_with_release_gates(
        dataset=dataset,
        predictions=predictions,
    )
    return adapt_classification_release_gate_evaluation(gate_evaluation)
