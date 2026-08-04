"""Deterministic metrics for structured ticket classification."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from decimal import ROUND_HALF_UP, Decimal
from enum import StrEnum
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from supportops.ai.schemas.ticket_classification import TicketUrgency
from supportops.evaluation.contracts.hashing import sha256_hexdigest
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

# Application-owned high-risk tags for specialized human-review recall.
_HIGH_RISK_HUMAN_REVIEW_TAGS: frozenset[str] = frozenset(
    {
        "credential-exposure",
        "privacy",
        "prompt-injection",
        "sensitive",
        "unauthorized-activity",
        "critical",
        "human-review",
    },
)

_HIGH_OR_CRITICAL_URGENCIES: frozenset[TicketUrgency] = frozenset(
    {
        TicketUrgency.HIGH,
        TicketUrgency.CRITICAL,
    },
)


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


class TicketClassificationValidityMetrics(BaseModel):
    """Structured-output validity over the full dataset case count."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    valid_count: int
    invalid_count: int
    rate: Decimal


class TicketClassificationRecallMetrics(BaseModel):
    """Recall over an expected-positive subset of dataset cases."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    true_positive_count: int
    expected_positive_count: int
    recall: Decimal


class TicketClassificationLatencyMetrics(BaseModel):
    """Average complete-system latency across known prediction traces."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    average_latency_ms: Decimal
    known_latency_count: int
    unknown_latency_count: int


class TicketClassificationTokenAverageMetrics(BaseModel):
    """Invocation token averages that preserve unknown usage dimensions."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    average_input_tokens: Decimal
    average_output_tokens: Decimal
    average_total_tokens: Decimal
    known_input_token_count: int
    known_output_token_count: int
    known_total_token_count: int
    unknown_input_token_count: int
    unknown_output_token_count: int
    unknown_total_token_count: int


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

    structured_output_validity: TicketClassificationValidityMetrics
    invalid_output_rate: Decimal
    high_urgency_recall: TicketClassificationRecallMetrics
    critical_urgency_recall: TicketClassificationRecallMetrics
    high_risk_human_review_recall: TicketClassificationRecallMetrics
    latency: TicketClassificationLatencyMetrics
    token_averages: TicketClassificationTokenAverageMetrics

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


class TicketClassificationGateCategory(StrEnum):
    """Release-gate category for ticket-classification reports."""

    SAFETY = "safety"
    QUALITY = "quality"
    RELIABILITY = "reliability"
    EFFICIENCY = "efficiency"


class TicketClassificationGateOutcome(StrEnum):
    """Explicit outcome for one release gate."""

    PASSED = "passed"
    FAILED = "failed"
    NOT_APPLICABLE = "not_applicable"


class TicketClassificationGateOperator(StrEnum):
    """Supported comparison operators for absolute release gates."""

    EQUAL = "equal"


class TicketClassificationStandaloneGateStatus(StrEnum):
    """Aggregate status for a standalone classification gate profile."""

    PASSED = "passed"
    FAILED = "failed"
    INCOMPLETE = "incomplete"


class TicketClassificationReleaseGateProfileError(ValueError):
    """Raised when a release-gate profile fails validation."""


class TicketClassificationReleaseGateDefinition(BaseModel):
    """One immutable gate inside a classification release profile."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    gate_id: str = Field(min_length=1)
    category: TicketClassificationGateCategory
    blocking: bool
    metric_name: str = Field(min_length=1)
    operator: TicketClassificationGateOperator
    threshold_value: Decimal | int


class TicketClassificationReleaseGateProfile(BaseModel):
    """Frozen profile of release gates for ticket classification."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    profile_id: str = Field(min_length=1)
    profile_version: int = Field(ge=1)
    gates: tuple[TicketClassificationReleaseGateDefinition, ...] = Field(
        min_length=1,
    )

    @model_validator(mode="after")
    def validate_unique_gate_ids(self) -> Self:
        gate_ids = [gate.gate_id for gate in self.gates]
        if len(gate_ids) != len(set(gate_ids)):
            raise TicketClassificationReleaseGateProfileError(
                "Release-gate profile contains duplicate gate IDs.",
            )

        return self


class TicketClassificationReleaseGateResult(BaseModel):
    """Deterministic result for one classification release gate."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    gate_id: str
    category: TicketClassificationGateCategory
    outcome: TicketClassificationGateOutcome
    blocking: bool
    actual_value: Decimal | int | None
    operator: TicketClassificationGateOperator
    threshold_value: Decimal | int
    metric_name: str
    reason: str


class TicketClassificationReleaseGateEvaluationContent(BaseModel):
    """Reproducible gate-evaluation content before hashing."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    profile_id: str
    profile_version: int
    report_content_hash: str
    gate_results: tuple[TicketClassificationReleaseGateResult, ...]
    blocking_failure_count: int
    not_applicable_count: int
    standalone_gate_status: TicketClassificationStandaloneGateStatus


class TicketClassificationReleaseGateEvaluation(
    TicketClassificationReleaseGateEvaluationContent,
):
    """Complete deterministic release-gate evaluation for one report."""

    content_hash: str


_PERFECT_RATE = Decimal("1.000000")
_PAIRED_BASELINE_REASON = "paired baseline comparison is required for standalone report evaluation"

DEFAULT_TICKET_CLASSIFICATION_RELEASE_GATE_PROFILE = TicketClassificationReleaseGateProfile(
    profile_id="ticket-classification-release-gates",
    profile_version=1,
    gates=(
        TicketClassificationReleaseGateDefinition(
            gate_id="classification.structured-output-validity",
            category=TicketClassificationGateCategory.SAFETY,
            blocking=True,
            metric_name="structured_output_validity.rate",
            operator=TicketClassificationGateOperator.EQUAL,
            threshold_value=_PERFECT_RATE,
        ),
        TicketClassificationReleaseGateDefinition(
            gate_id="classification.critical-urgency-recall",
            category=TicketClassificationGateCategory.SAFETY,
            blocking=True,
            metric_name="critical_urgency_recall.recall",
            operator=TicketClassificationGateOperator.EQUAL,
            threshold_value=_PERFECT_RATE,
        ),
        TicketClassificationReleaseGateDefinition(
            gate_id=("classification.high-risk-human-review-recall"),
            category=TicketClassificationGateCategory.SAFETY,
            blocking=True,
            metric_name=("high_risk_human_review_recall.recall"),
            operator=TicketClassificationGateOperator.EQUAL,
            threshold_value=_PERFECT_RATE,
        ),
        TicketClassificationReleaseGateDefinition(
            gate_id="classification.prediction-coverage",
            category=TicketClassificationGateCategory.RELIABILITY,
            blocking=True,
            metric_name="prediction_artifact_coverage.rate",
            operator=TicketClassificationGateOperator.EQUAL,
            threshold_value=_PERFECT_RATE,
        ),
        TicketClassificationReleaseGateDefinition(
            gate_id=("classification.deterministic-evaluator-failures"),
            category=TicketClassificationGateCategory.RELIABILITY,
            blocking=True,
            metric_name=("deterministic_evaluator_failure_count"),
            operator=TicketClassificationGateOperator.EQUAL,
            threshold_value=0,
        ),
        TicketClassificationReleaseGateDefinition(
            gate_id=("classification.structured-label-non-regression"),
            category=TicketClassificationGateCategory.QUALITY,
            blocking=True,
            metric_name=("structured_label_exact_match.non_regression"),
            operator=TicketClassificationGateOperator.EQUAL,
            threshold_value=_PERFECT_RATE,
        ),
        TicketClassificationReleaseGateDefinition(
            gate_id=("classification.category-accuracy-non-regression"),
            category=TicketClassificationGateCategory.QUALITY,
            blocking=True,
            metric_name="category_accuracy.non_regression",
            operator=TicketClassificationGateOperator.EQUAL,
            threshold_value=_PERFECT_RATE,
        ),
        TicketClassificationReleaseGateDefinition(
            gate_id="classification.target-metric-improvement",
            category=TicketClassificationGateCategory.QUALITY,
            blocking=True,
            metric_name="target_metric.improvement",
            operator=TicketClassificationGateOperator.EQUAL,
            threshold_value=_PERFECT_RATE,
        ),
        TicketClassificationReleaseGateDefinition(
            gate_id="classification.mean-token-increase",
            category=TicketClassificationGateCategory.EFFICIENCY,
            blocking=True,
            metric_name="mean_total_tokens.increase",
            operator=TicketClassificationGateOperator.EQUAL,
            threshold_value=0,
        ),
        TicketClassificationReleaseGateDefinition(
            gate_id="classification.mean-cost-increase",
            category=TicketClassificationGateCategory.EFFICIENCY,
            blocking=True,
            metric_name="mean_estimated_cost_usd.increase",
            operator=TicketClassificationGateOperator.EQUAL,
            threshold_value=0,
        ),
        TicketClassificationReleaseGateDefinition(
            gate_id="classification.mean-latency-increase",
            category=TicketClassificationGateCategory.EFFICIENCY,
            blocking=True,
            metric_name="mean_latency_ms.increase",
            operator=TicketClassificationGateOperator.EQUAL,
            threshold_value=0,
        ),
    ),
)


def evaluate_ticket_classification_release_gates(
    report: TicketClassificationEvaluationReport,
    *,
    profile: TicketClassificationReleaseGateProfile = (
        DEFAULT_TICKET_CLASSIFICATION_RELEASE_GATE_PROFILE
    ),
) -> TicketClassificationReleaseGateEvaluation:
    """Evaluate one classification report against a frozen release-gate profile.

    Gate evaluation runs only after a valid report exists. Exceptions raised
    while generating that report prevent gate evaluation entirely and are not
    represented as a failure count inside the report.
    """

    if not isinstance(
        profile,
        TicketClassificationReleaseGateProfile,
    ):
        raise TicketClassificationReleaseGateProfileError(
            "profile must be a TicketClassificationReleaseGateProfile.",
        )

    gate_results = tuple(
        _evaluate_release_gate(
            report=report,
            gate=gate,
        )
        for gate in profile.gates
    )

    blocking_failure_count = sum(
        1
        for result in gate_results
        if result.blocking and result.outcome is TicketClassificationGateOutcome.FAILED
    )
    not_applicable_count = sum(
        1
        for result in gate_results
        if result.outcome is TicketClassificationGateOutcome.NOT_APPLICABLE
    )
    standalone_gate_status = _aggregate_standalone_gate_status(
        gate_results,
    )

    content = TicketClassificationReleaseGateEvaluationContent(
        profile_id=profile.profile_id,
        profile_version=profile.profile_version,
        report_content_hash=report.report_content_hash,
        gate_results=gate_results,
        blocking_failure_count=blocking_failure_count,
        not_applicable_count=not_applicable_count,
        standalone_gate_status=standalone_gate_status,
    )

    return TicketClassificationReleaseGateEvaluation(
        **content.model_dump(),
        content_hash=_compute_gate_evaluation_content_hash(
            content,
        ),
    )


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
    structurally_valid_count = 0

    true_positive_count = 0
    false_positive_count = 0
    false_negative_count = 0
    true_negative_count = 0

    high_urgency_true_positive_count = 0
    high_urgency_expected_positive_count = 0
    critical_urgency_true_positive_count = 0
    critical_urgency_expected_positive_count = 0
    high_risk_review_true_positive_count = 0
    high_risk_review_expected_positive_count = 0

    for case in dataset.cases:
        prediction = prediction_by_case_id.get(
            case.case_id,
        )
        structurally_valid = _is_structurally_valid(
            prediction,
        )

        if structurally_valid:
            structurally_valid_count += 1

        if _is_high_urgency_expected(case):
            high_urgency_expected_positive_count += 1
            if structurally_valid and _is_high_urgency_predicted(
                prediction,
            ):
                high_urgency_true_positive_count += 1

        if _is_critical_urgency_expected(case):
            critical_urgency_expected_positive_count += 1
            if structurally_valid and _is_critical_urgency_predicted(
                prediction,
            ):
                critical_urgency_true_positive_count += 1

        if _is_high_risk_human_review_expected(case):
            high_risk_review_expected_positive_count += 1
            if structurally_valid and _predicted_requires_human_review(
                prediction,
            ):
                high_risk_review_true_positive_count += 1

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
    latency_metrics = _aggregate_latency_metrics(
        dataset=dataset,
        prediction_by_case_id=prediction_by_case_id,
    )
    token_averages = _aggregate_token_average_metrics(
        predictions.predictions,
    )

    case_count = dataset.case_count
    failed_prediction_count = case_count - successful_prediction_count
    invalid_count = case_count - structurally_valid_count
    structured_output_validity = TicketClassificationValidityMetrics(
        valid_count=structurally_valid_count,
        invalid_count=invalid_count,
        rate=_rate(
            structurally_valid_count,
            case_count,
        ),
    )
    invalid_output_rate = _rate(
        invalid_count,
        case_count,
    )

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
        structured_output_validity=(structured_output_validity),
        invalid_output_rate=invalid_output_rate,
        high_urgency_recall=TicketClassificationRecallMetrics(
            true_positive_count=(high_urgency_true_positive_count),
            expected_positive_count=(high_urgency_expected_positive_count),
            recall=_rate(
                high_urgency_true_positive_count,
                high_urgency_expected_positive_count,
            ),
        ),
        critical_urgency_recall=TicketClassificationRecallMetrics(
            true_positive_count=(critical_urgency_true_positive_count),
            expected_positive_count=(critical_urgency_expected_positive_count),
            recall=_rate(
                critical_urgency_true_positive_count,
                critical_urgency_expected_positive_count,
            ),
        ),
        high_risk_human_review_recall=(
            TicketClassificationRecallMetrics(
                true_positive_count=(high_risk_review_true_positive_count),
                expected_positive_count=(high_risk_review_expected_positive_count),
                recall=_rate(
                    high_risk_review_true_positive_count,
                    high_risk_review_expected_positive_count,
                ),
            )
        ),
        latency=latency_metrics,
        token_averages=token_averages,
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


def _aggregate_latency_metrics(
    *,
    dataset: TicketClassificationEvaluationDataset,
    prediction_by_case_id: dict[
        str,
        TicketClassificationEvaluationPrediction,
    ],
) -> TicketClassificationLatencyMetrics:
    """Aggregate one complete-system latency sample per prediction.

    Predictions have no separate total latency field, so each known sample is
    the sum of that prediction's invocation ``latency_ms`` values. Missing
    predictions contribute only to ``unknown_latency_count`` and never become
    zero-valued latency samples.
    """

    known_latency_sum = 0
    known_latency_count = 0
    unknown_latency_count = 0

    for case in dataset.cases:
        prediction = prediction_by_case_id.get(
            case.case_id,
        )
        if prediction is None:
            unknown_latency_count += 1
            continue

        known_latency_sum += sum(invocation.latency_ms for invocation in prediction.invocations)
        known_latency_count += 1

    return TicketClassificationLatencyMetrics(
        average_latency_ms=_average(
            known_latency_sum,
            known_latency_count,
        ),
        known_latency_count=known_latency_count,
        unknown_latency_count=unknown_latency_count,
    )


def _aggregate_token_average_metrics(
    predictions: tuple[
        TicketClassificationEvaluationPrediction,
        ...,
    ],
) -> TicketClassificationTokenAverageMetrics:
    known_input_token_sum = 0
    known_output_token_sum = 0
    known_total_token_sum = 0
    known_input_token_count = 0
    known_output_token_count = 0
    known_total_token_count = 0
    unknown_input_token_count = 0
    unknown_output_token_count = 0
    unknown_total_token_count = 0

    for prediction in predictions:
        for invocation in prediction.invocations:
            usage = invocation.usage

            if usage is None:
                unknown_input_token_count += 1
                unknown_output_token_count += 1
                unknown_total_token_count += 1
                continue

            if usage.input_tokens is None:
                unknown_input_token_count += 1
            else:
                known_input_token_sum += usage.input_tokens
                known_input_token_count += 1

            if usage.output_tokens is None:
                unknown_output_token_count += 1
            else:
                known_output_token_sum += usage.output_tokens
                known_output_token_count += 1

            if usage.total_tokens is not None:
                known_total_token_sum += usage.total_tokens
                known_total_token_count += 1
            elif usage.input_tokens is not None and usage.output_tokens is not None:
                known_total_token_sum += usage.input_tokens + usage.output_tokens
                known_total_token_count += 1
            else:
                unknown_total_token_count += 1

    return TicketClassificationTokenAverageMetrics(
        average_input_tokens=_average(
            known_input_token_sum,
            known_input_token_count,
        ),
        average_output_tokens=_average(
            known_output_token_sum,
            known_output_token_count,
        ),
        average_total_tokens=_average(
            known_total_token_sum,
            known_total_token_count,
        ),
        known_input_token_count=known_input_token_count,
        known_output_token_count=known_output_token_count,
        known_total_token_count=known_total_token_count,
        unknown_input_token_count=unknown_input_token_count,
        unknown_output_token_count=unknown_output_token_count,
        unknown_total_token_count=unknown_total_token_count,
    )


def _is_structurally_valid(
    prediction: TicketClassificationEvaluationPrediction | None,
) -> bool:
    """Return whether a prediction is a schema-valid succeeded output.

    Validity requires a succeeded prediction with parsed
    ``TicketClassificationResult`` output. Label correctness is irrelevant.
    Failed, missing, and schema-invalid outcomes are structurally invalid.
    """

    return isinstance(
        prediction,
        TicketClassificationSuccessfulPrediction,
    )


def _is_high_urgency_expected(
    case: TicketClassificationEvaluationCase,
) -> bool:
    return case.expected.urgency in _HIGH_OR_CRITICAL_URGENCIES


def _is_critical_urgency_expected(
    case: TicketClassificationEvaluationCase,
) -> bool:
    return case.expected.urgency is TicketUrgency.CRITICAL


def _is_high_urgency_predicted(
    prediction: TicketClassificationEvaluationPrediction | None,
) -> bool:
    if not isinstance(
        prediction,
        TicketClassificationSuccessfulPrediction,
    ):
        return False

    return prediction.output.urgency in _HIGH_OR_CRITICAL_URGENCIES


def _is_critical_urgency_predicted(
    prediction: TicketClassificationEvaluationPrediction | None,
) -> bool:
    if not isinstance(
        prediction,
        TicketClassificationSuccessfulPrediction,
    ):
        return False

    return prediction.output.urgency is TicketUrgency.CRITICAL


def _is_high_risk_human_review_expected(
    case: TicketClassificationEvaluationCase,
) -> bool:
    if not case.expected.requires_human_review:
        return False

    return any(tag in _HIGH_RISK_HUMAN_REVIEW_TAGS for tag in case.tags)


def _predicted_requires_human_review(
    prediction: TicketClassificationEvaluationPrediction | None,
) -> bool:
    if not isinstance(
        prediction,
        TicketClassificationSuccessfulPrediction,
    ):
        return False

    return prediction.output.requires_human_review


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


def _average(
    total: int,
    count: int,
) -> Decimal:
    if count == 0:
        return Decimal("0.000000")

    return (Decimal(total) / Decimal(count)).quantize(
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


_PAIRED_COMPARISON_GATE_IDS: frozenset[str] = frozenset(
    {
        "classification.structured-label-non-regression",
        "classification.category-accuracy-non-regression",
        "classification.target-metric-improvement",
        "classification.mean-token-increase",
        "classification.mean-cost-increase",
        "classification.mean-latency-increase",
    },
)


def _evaluate_release_gate(
    *,
    report: TicketClassificationEvaluationReport,
    gate: TicketClassificationReleaseGateDefinition,
) -> TicketClassificationReleaseGateResult:
    if gate.gate_id in _PAIRED_COMPARISON_GATE_IDS:
        return _not_applicable_gate_result(
            gate=gate,
            reason=_PAIRED_BASELINE_REASON,
        )

    if gate.gate_id == "classification.structured-output-validity":
        return _compare_absolute_gate(
            gate=gate,
            actual_value=report.structured_output_validity.rate,
        )

    if gate.gate_id == "classification.critical-urgency-recall":
        return _evaluate_recall_gate(
            gate=gate,
            expected_positive_count=(report.critical_urgency_recall.expected_positive_count),
            actual_value=report.critical_urgency_recall.recall,
            zero_denominator_reason=("critical_urgency_recall.expected_positive_count is zero"),
        )

    if gate.gate_id == ("classification.high-risk-human-review-recall"):
        return _evaluate_recall_gate(
            gate=gate,
            expected_positive_count=(report.high_risk_human_review_recall.expected_positive_count),
            actual_value=report.high_risk_human_review_recall.recall,
            zero_denominator_reason=(
                "high_risk_human_review_recall.expected_positive_count is zero"
            ),
        )

    if gate.gate_id == "classification.prediction-coverage":
        return _compare_absolute_gate(
            gate=gate,
            actual_value=_prediction_artifact_coverage_rate(
                report,
            ),
        )

    if gate.gate_id == ("classification.deterministic-evaluator-failures"):
        return _compare_absolute_gate(
            gate=gate,
            actual_value=0,
            passed_reason=("deterministic evaluator failure count is zero for a valid report"),
        )

    raise TicketClassificationReleaseGateProfileError(
        f"Unknown release gate ID: {gate.gate_id}.",
    )


def _evaluate_recall_gate(
    *,
    gate: TicketClassificationReleaseGateDefinition,
    expected_positive_count: int,
    actual_value: Decimal,
    zero_denominator_reason: str,
) -> TicketClassificationReleaseGateResult:
    if expected_positive_count == 0:
        return _not_applicable_gate_result(
            gate=gate,
            reason=zero_denominator_reason,
        )

    return _compare_absolute_gate(
        gate=gate,
        actual_value=actual_value,
    )


def _prediction_artifact_coverage_rate(
    report: TicketClassificationEvaluationReport,
) -> Decimal:
    covered_count = sum(
        1 for case in report.cases if case.prediction_status in {"succeeded", "failed"}
    )

    return _rate(
        covered_count,
        report.case_count,
    )


def _compare_absolute_gate(
    *,
    gate: TicketClassificationReleaseGateDefinition,
    actual_value: Decimal | int,
    passed_reason: str | None = None,
) -> TicketClassificationReleaseGateResult:
    if gate.operator is not TicketClassificationGateOperator.EQUAL:
        raise TicketClassificationReleaseGateProfileError(
            f"Unknown release-gate operator: {gate.operator!r}.",
        )

    passed = actual_value == gate.threshold_value
    if passed:
        reason = passed_reason or (f"{gate.metric_name} equals {gate.threshold_value}")
        outcome = TicketClassificationGateOutcome.PASSED
    else:
        reason = f"{gate.metric_name} {actual_value} does not equal {gate.threshold_value}"
        outcome = TicketClassificationGateOutcome.FAILED

    return TicketClassificationReleaseGateResult(
        gate_id=gate.gate_id,
        category=gate.category,
        outcome=outcome,
        blocking=gate.blocking,
        actual_value=actual_value,
        operator=gate.operator,
        threshold_value=gate.threshold_value,
        metric_name=gate.metric_name,
        reason=reason,
    )


def _not_applicable_gate_result(
    *,
    gate: TicketClassificationReleaseGateDefinition,
    reason: str,
) -> TicketClassificationReleaseGateResult:
    return TicketClassificationReleaseGateResult(
        gate_id=gate.gate_id,
        category=gate.category,
        outcome=TicketClassificationGateOutcome.NOT_APPLICABLE,
        blocking=gate.blocking,
        actual_value=None,
        operator=gate.operator,
        threshold_value=gate.threshold_value,
        metric_name=gate.metric_name,
        reason=reason,
    )


def _aggregate_standalone_gate_status(
    gate_results: tuple[TicketClassificationReleaseGateResult, ...],
) -> TicketClassificationStandaloneGateStatus:
    blocking_results = tuple(result for result in gate_results if result.blocking)

    if any(result.outcome is TicketClassificationGateOutcome.FAILED for result in blocking_results):
        return TicketClassificationStandaloneGateStatus.FAILED

    if any(
        result.outcome is TicketClassificationGateOutcome.NOT_APPLICABLE
        for result in blocking_results
    ):
        return TicketClassificationStandaloneGateStatus.INCOMPLETE

    return TicketClassificationStandaloneGateStatus.PASSED


def _compute_gate_evaluation_content_hash(
    content: TicketClassificationReleaseGateEvaluationContent,
) -> str:
    return sha256_hexdigest(content)
