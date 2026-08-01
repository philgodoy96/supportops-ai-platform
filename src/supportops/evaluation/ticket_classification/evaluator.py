"""Deterministic metrics for structured ticket classification."""

import hashlib
import json
from collections import Counter
from decimal import ROUND_HALF_UP, Decimal
from typing import Literal

from pydantic import BaseModel, ConfigDict

from supportops.evaluation.ticket_classification.dataset import (
    TicketClassificationEvaluationDataset,
)
from supportops.evaluation.ticket_classification.models import (
    TicketClassificationEvaluationCase,
    TicketClassificationExpectedLabels,
)
from supportops.evaluation.ticket_classification.predictions import (
    TicketClassificationEvaluationPrediction,
    TicketClassificationFailedPrediction,
    TicketClassificationPredictionProvenance,
    TicketClassificationPredictionSet,
    TicketClassificationSuccessfulPrediction,
)

_RATE_QUANTUM = Decimal("0.000001")
_MISSING_PREDICTION_ERROR_CODE = "prediction_missing"


class TicketClassificationEvaluationError(ValueError):
    """Raised when predictions cannot be evaluated safely."""


class UnknownTicketClassificationPredictionError(
    TicketClassificationEvaluationError,
):
    """Raised when predictions reference cases outside the dataset."""


class InconsistentTicketClassificationPredictionProvenanceError(
    TicketClassificationEvaluationError,
):
    """Raised when one artifact mixes prompt or runtime identity."""


class TicketClassificationMetric(BaseModel):
    """One count and deterministic rate over the full dataset."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    match_count: int
    rate: Decimal


class TicketClassificationHumanReviewMetrics(BaseModel):
    """Binary metrics for the human-review recommendation."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    true_positive_count: int
    false_positive_count: int
    false_negative_count: int
    true_negative_count: int
    accuracy: Decimal
    precision: Decimal
    recall: Decimal
    f1: Decimal


class TicketClassificationCaseEvaluationResult(BaseModel):
    """Deterministic result for one dataset case."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    case_id: str
    tags: tuple[str, ...]
    prediction_status: Literal[
        "succeeded",
        "failed",
        "missing",
    ]
    expected: TicketClassificationExpectedLabels
    predicted: TicketClassificationExpectedLabels | None
    structured_label_exact_match: bool
    category_match: bool
    intent_match: bool
    urgency_match: bool
    sentiment_match: bool
    human_review_match: bool
    error_code: str | None


class TicketClassificationEvaluationReportContent(BaseModel):
    """Reproducible report content before report hashing."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    dataset_id: str
    dataset_version: int
    dataset_content_hash: str
    predictions_content_hash: str

    prompt_id: str
    prompt_version: int
    prompt_content_hash: str
    provider: str
    model: str

    case_count: int
    successful_prediction_count: int
    failed_prediction_count: int

    structured_label_exact_match: TicketClassificationMetric
    category_accuracy: TicketClassificationMetric
    intent_accuracy: TicketClassificationMetric
    urgency_accuracy: TicketClassificationMetric
    sentiment_accuracy: TicketClassificationMetric
    human_review_accuracy: TicketClassificationMetric
    human_review: TicketClassificationHumanReviewMetrics

    failure_counts_by_error_code: dict[str, int]

    known_total_tokens: int
    unknown_usage_count: int
    known_estimated_total_cost_usd: Decimal
    unknown_pricing_count: int
    pricing_catalog_versions: tuple[str, ...]

    cases: tuple[
        TicketClassificationCaseEvaluationResult,
        ...,
    ]


class TicketClassificationEvaluationReport(
    TicketClassificationEvaluationReportContent,
):
    """Complete deterministic evaluation report."""

    report_content_hash: str


def evaluate_ticket_classification_predictions(
    *,
    dataset: TicketClassificationEvaluationDataset,
    predictions: TicketClassificationPredictionSet,
) -> TicketClassificationEvaluationReport:
    """Evaluate aligned predictions against one versioned dataset."""

    prediction_by_case_id = {
        prediction.case_id: prediction for prediction in predictions.predictions
    }
    dataset_case_ids = {case.case_id for case in dataset.cases}
    unknown_case_ids = set(prediction_by_case_id) - dataset_case_ids

    if unknown_case_ids:
        formatted_case_ids = ", ".join(
            sorted(unknown_case_ids),
        )
        raise UnknownTicketClassificationPredictionError(
            f"Predictions contain unknown evaluation case IDs: {formatted_case_ids}.",
        )

    provenance = _require_consistent_provenance(
        predictions.predictions,
    )

    case_results: list[TicketClassificationCaseEvaluationResult] = []
    failure_counts: Counter[str] = Counter()

    exact_match_count = 0
    category_match_count = 0
    intent_match_count = 0
    urgency_match_count = 0
    sentiment_match_count = 0
    human_review_match_count = 0
    successful_prediction_count = 0

    true_positive_count = 0
    false_positive_count = 0
    false_negative_count = 0
    true_negative_count = 0

    for case in dataset.cases:
        prediction = prediction_by_case_id.get(
            case.case_id,
        )

        if prediction is None:
            result = _missing_case_result(case)
            failure_counts[_MISSING_PREDICTION_ERROR_CODE] += 1

            if case.expected.requires_human_review:
                false_negative_count += 1
            else:
                true_negative_count += 1

            case_results.append(result)
            continue

        if isinstance(
            prediction,
            TicketClassificationFailedPrediction,
        ):
            result = _failed_case_result(
                case=case,
                prediction=prediction,
            )
            failure_counts[prediction.error_code.value] += 1

            if case.expected.requires_human_review:
                false_negative_count += 1
            else:
                true_negative_count += 1

            case_results.append(result)
            continue

        successful_prediction_count += 1
        result = _successful_case_result(
            case=case,
            prediction=prediction,
        )
        case_results.append(result)

        exact_match_count += int(
            result.structured_label_exact_match,
        )
        category_match_count += int(
            result.category_match,
        )
        intent_match_count += int(
            result.intent_match,
        )
        urgency_match_count += int(
            result.urgency_match,
        )
        sentiment_match_count += int(
            result.sentiment_match,
        )
        human_review_match_count += int(
            result.human_review_match,
        )

        expected_review = case.expected.requires_human_review
        predicted_review = prediction.output.requires_human_review

        if expected_review and predicted_review:
            true_positive_count += 1
        elif not expected_review and predicted_review:
            false_positive_count += 1
        elif expected_review and not predicted_review:
            false_negative_count += 1
        else:
            true_negative_count += 1

    (
        known_total_tokens,
        unknown_usage_count,
        known_estimated_total_cost_usd,
        unknown_pricing_count,
        pricing_catalog_versions,
    ) = _aggregate_operational_metadata(
        predictions.predictions,
    )

    case_count = dataset.case_count
    failed_prediction_count = case_count - successful_prediction_count

    content = TicketClassificationEvaluationReportContent(
        dataset_id=dataset.dataset_id,
        dataset_version=dataset.version,
        dataset_content_hash=dataset.content_hash,
        predictions_content_hash=predictions.content_hash,
        prompt_id=provenance.prompt_id,
        prompt_version=provenance.prompt_version,
        prompt_content_hash=(provenance.prompt_content_hash),
        provider=provenance.provider,
        model=provenance.model,
        case_count=case_count,
        successful_prediction_count=(successful_prediction_count),
        failed_prediction_count=failed_prediction_count,
        structured_label_exact_match=_metric(
            match_count=exact_match_count,
            case_count=case_count,
        ),
        category_accuracy=_metric(
            match_count=category_match_count,
            case_count=case_count,
        ),
        intent_accuracy=_metric(
            match_count=intent_match_count,
            case_count=case_count,
        ),
        urgency_accuracy=_metric(
            match_count=urgency_match_count,
            case_count=case_count,
        ),
        sentiment_accuracy=_metric(
            match_count=sentiment_match_count,
            case_count=case_count,
        ),
        human_review_accuracy=_metric(
            match_count=human_review_match_count,
            case_count=case_count,
        ),
        human_review=(
            TicketClassificationHumanReviewMetrics(
                true_positive_count=true_positive_count,
                false_positive_count=false_positive_count,
                false_negative_count=false_negative_count,
                true_negative_count=true_negative_count,
                accuracy=_rate(
                    true_positive_count + true_negative_count,
                    case_count,
                ),
                precision=_rate(
                    true_positive_count,
                    true_positive_count + false_positive_count,
                ),
                recall=_rate(
                    true_positive_count,
                    true_positive_count + false_negative_count,
                ),
                f1=_f1(
                    true_positive_count=(true_positive_count),
                    false_positive_count=(false_positive_count),
                    false_negative_count=(false_negative_count),
                ),
            )
        ),
        failure_counts_by_error_code=dict(
            sorted(
                failure_counts.items(),
            ),
        ),
        known_total_tokens=known_total_tokens,
        unknown_usage_count=unknown_usage_count,
        known_estimated_total_cost_usd=(known_estimated_total_cost_usd),
        unknown_pricing_count=unknown_pricing_count,
        pricing_catalog_versions=(pricing_catalog_versions),
        cases=tuple(case_results),
    )

    return TicketClassificationEvaluationReport(
        **content.model_dump(),
        report_content_hash=_compute_report_content_hash(
            content,
        ),
    )


def _successful_case_result(
    *,
    case: TicketClassificationEvaluationCase,
    prediction: TicketClassificationSuccessfulPrediction,
) -> TicketClassificationCaseEvaluationResult:
    from supportops.evaluation.ticket_classification.models import (
        TicketClassificationEvaluationCase,
    )

    if not isinstance(
        case,
        TicketClassificationEvaluationCase,
    ):
        raise TypeError(
            "case must be a TicketClassificationEvaluationCase.",
        )

    predicted = TicketClassificationExpectedLabels(
        category=prediction.output.category,
        intent=prediction.output.intent,
        urgency=prediction.output.urgency,
        sentiment=prediction.output.sentiment,
        requires_human_review=(prediction.output.requires_human_review),
        schema_version=prediction.output.schema_version,
    )

    category_match = predicted.category is case.expected.category
    intent_match = predicted.intent is case.expected.intent
    urgency_match = predicted.urgency is case.expected.urgency
    sentiment_match = predicted.sentiment is case.expected.sentiment
    human_review_match = predicted.requires_human_review is case.expected.requires_human_review

    exact_match = all(
        (
            category_match,
            intent_match,
            urgency_match,
            sentiment_match,
            human_review_match,
        ),
    )

    return TicketClassificationCaseEvaluationResult(
        case_id=case.case_id,
        tags=case.tags,
        prediction_status="succeeded",
        expected=case.expected,
        predicted=predicted,
        structured_label_exact_match=exact_match,
        category_match=category_match,
        intent_match=intent_match,
        urgency_match=urgency_match,
        sentiment_match=sentiment_match,
        human_review_match=human_review_match,
        error_code=None,
    )


def _failed_case_result(
    *,
    case: TicketClassificationEvaluationCase,
    prediction: TicketClassificationFailedPrediction,
) -> TicketClassificationCaseEvaluationResult:
    from supportops.evaluation.ticket_classification.models import (
        TicketClassificationEvaluationCase,
    )

    if not isinstance(
        case,
        TicketClassificationEvaluationCase,
    ):
        raise TypeError(
            "case must be a TicketClassificationEvaluationCase.",
        )

    return TicketClassificationCaseEvaluationResult(
        case_id=case.case_id,
        tags=case.tags,
        prediction_status="failed",
        expected=case.expected,
        predicted=None,
        structured_label_exact_match=False,
        category_match=False,
        intent_match=False,
        urgency_match=False,
        sentiment_match=False,
        human_review_match=False,
        error_code=prediction.error_code.value,
    )


def _missing_case_result(
    case: TicketClassificationEvaluationCase,
) -> TicketClassificationCaseEvaluationResult:
    from supportops.evaluation.ticket_classification.models import (
        TicketClassificationEvaluationCase,
    )

    if not isinstance(
        case,
        TicketClassificationEvaluationCase,
    ):
        raise TypeError(
            "case must be a TicketClassificationEvaluationCase.",
        )

    return TicketClassificationCaseEvaluationResult(
        case_id=case.case_id,
        tags=case.tags,
        prediction_status="missing",
        expected=case.expected,
        predicted=None,
        structured_label_exact_match=False,
        category_match=False,
        intent_match=False,
        urgency_match=False,
        sentiment_match=False,
        human_review_match=False,
        error_code=_MISSING_PREDICTION_ERROR_CODE,
    )


def _require_consistent_provenance(
    predictions: tuple[
        TicketClassificationEvaluationPrediction,
        ...,
    ],
) -> TicketClassificationPredictionProvenance:
    provenance_by_identity = {
        (
            prediction.provenance.prompt_id,
            prediction.provenance.prompt_version,
            prediction.provenance.prompt_content_hash,
            prediction.provenance.provider,
            prediction.provenance.model,
        ): prediction.provenance
        for prediction in predictions
    }

    if len(provenance_by_identity) != 1:
        raise (
            InconsistentTicketClassificationPredictionProvenanceError(
                "Predictions must share one prompt, provider, and model provenance identity.",
            )
        )

    return next(
        iter(
            provenance_by_identity.values(),
        ),
    )


def _aggregate_operational_metadata(
    predictions: tuple[
        TicketClassificationEvaluationPrediction,
        ...,
    ],
) -> tuple[
    int,
    int,
    Decimal,
    int,
    tuple[str, ...],
]:
    known_total_tokens = 0
    unknown_usage_count = 0
    known_estimated_total_cost_usd = Decimal("0")
    unknown_pricing_count = 0
    pricing_catalog_versions: set[str] = set()

    for prediction in predictions:
        for invocation in prediction.invocations:
            usage = invocation.usage
            if usage is None or usage.total_tokens is None:
                unknown_usage_count += 1
            else:
                known_total_tokens += usage.total_tokens

            cost = invocation.cost
            pricing_catalog_versions.add(
                cost.pricing_catalog_version,
            )

            if not cost.pricing_found:
                unknown_pricing_count += 1

            if cost.estimated_total_cost_usd is not None:
                known_estimated_total_cost_usd += cost.estimated_total_cost_usd

    return (
        known_total_tokens,
        unknown_usage_count,
        known_estimated_total_cost_usd,
        unknown_pricing_count,
        tuple(
            sorted(
                pricing_catalog_versions,
            ),
        ),
    )


def _metric(
    *,
    match_count: int,
    case_count: int,
) -> TicketClassificationMetric:
    return TicketClassificationMetric(
        match_count=match_count,
        rate=_rate(
            match_count,
            case_count,
        ),
    )


def _rate(
    numerator: int,
    denominator: int,
) -> Decimal:
    if denominator == 0:
        return Decimal("0.000000")

    return (Decimal(numerator) / Decimal(denominator)).quantize(
        _RATE_QUANTUM,
        rounding=ROUND_HALF_UP,
    )


def _f1(
    *,
    true_positive_count: int,
    false_positive_count: int,
    false_negative_count: int,
) -> Decimal:
    denominator = 2 * true_positive_count + false_positive_count + false_negative_count

    if denominator == 0:
        return Decimal("0.000000")

    return (Decimal(2 * true_positive_count) / Decimal(denominator)).quantize(
        _RATE_QUANTUM,
        rounding=ROUND_HALF_UP,
    )


def _compute_report_content_hash(
    content: TicketClassificationEvaluationReportContent,
) -> str:
    canonical_content = json.dumps(
        content.model_dump(
            mode="json",
        ),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )

    return hashlib.sha256(
        canonical_content.encode("utf-8"),
    ).hexdigest()
