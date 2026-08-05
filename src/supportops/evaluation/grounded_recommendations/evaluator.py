from __future__ import annotations

from decimal import ROUND_HALF_UP, Decimal

from supportops.evaluation.contracts.hashing import sha256_hexdigest
from supportops.evaluation.contracts.predictions import (
    EvaluationPredictionStatus,
)
from supportops.evaluation.grounded_recommendations.models import (
    CountRateMetric,
    GroundedRecommendationAction,
    GroundedRecommendationCaseResult,
    GroundedRecommendationEvaluationCase,
    GroundedRecommendationEvaluationDataset,
    GroundedRecommendationEvaluationReport,
    GroundedRecommendationPredictionPayload,
    MeanMetric,
)
from supportops.evaluation.grounded_recommendations.predictions import (
    GroundedRecommendationPrediction,
)

_METRIC_QUANTUM = Decimal("0.000001")


class GroundedRecommendationEvaluationError(ValueError):
    """Raised when grounded recommendation scoring cannot complete."""


def evaluate_grounded_recommendation_predictions(
    *,
    dataset: GroundedRecommendationEvaluationDataset,
    predictions: tuple[GroundedRecommendationPrediction, ...],
    prediction_hash: str,
) -> GroundedRecommendationEvaluationReport:
    """Score grounded recommendation predictions deterministically."""

    predictions_by_case = {prediction.case_id: prediction for prediction in predictions}
    dataset_case_ids = {case.case_id for case in dataset.cases}

    unknown_case_ids = sorted(set(predictions_by_case) - dataset_case_ids)
    if unknown_case_ids:
        raise GroundedRecommendationEvaluationError(
            "unknown prediction case IDs: " + ", ".join(unknown_case_ids)
        )

    case_results: list[GroundedRecommendationCaseResult] = []

    action_results: list[bool] = []
    review_results: list[bool] = []
    sufficiency_results: list[bool] = []
    citation_results: list[bool] = []
    workspace_results: list[bool] = []
    abstention_results: list[bool] = []
    coverage_results: list[bool] = []

    latency_values: list[Decimal] = []
    token_values: list[Decimal] = []
    cost_values: list[Decimal] = []

    unknown_latency_count = 0
    unknown_token_count = 0
    unknown_cost_count = 0

    for case in dataset.cases:
        prediction = predictions_by_case.get(case.case_id)

        if prediction is None:
            result = _missing_case_result(case)

            unknown_latency_count += 1
            unknown_token_count += 1
            unknown_cost_count += 1
        else:
            _collect_usage(
                prediction=prediction,
                latency_values=latency_values,
                token_values=token_values,
                cost_values=cost_values,
            )

            if prediction.latency_ms is None:
                unknown_latency_count += 1

            if _total_tokens(prediction) is None:
                unknown_token_count += 1

            if prediction.estimated_cost_usd is None:
                unknown_cost_count += 1

            if prediction.status is EvaluationPredictionStatus.FAILED or prediction.payload is None:
                result = _failed_case_result(
                    case=case,
                    error_code=prediction.error_code,
                )
            else:
                result = _score_case(
                    case=case,
                    payload=prediction.payload,
                )

        case_results.append(result)
        coverage_results.append(result.prediction_present)

        if result.recommended_action_correct is not None:
            action_results.append(result.recommended_action_correct)

        if result.human_review_correct is not None:
            review_results.append(result.human_review_correct)

        if result.evidence_sufficiency_correct is not None:
            sufficiency_results.append(result.evidence_sufficiency_correct)

        if result.citation_identity_correct is not None:
            citation_results.append(result.citation_identity_correct)

        workspace_results.append(result.workspace_isolated)

        if result.grounded_abstention_correct is not None:
            abstention_results.append(result.grounded_abstention_correct)

    report_without_hash = {
        "dataset_id": dataset.dataset_id,
        "dataset_version": dataset.dataset_version,
        "schema_version": dataset.schema_version,
        "workflow_name": dataset.workflow_name,
        "workflow_version": dataset.workflow_version,
        "dataset_hash": dataset.content_hash,
        "prediction_hash": prediction_hash,
        "case_count": len(dataset.cases),
        "recommended_action_accuracy": _count_rate(action_results),
        "human_review_accuracy": _count_rate(review_results),
        "evidence_sufficiency_accuracy": _count_rate(sufficiency_results),
        "citation_identity_accuracy": _count_rate(citation_results),
        "workspace_isolation_rate": _count_rate(workspace_results),
        "grounded_abstention_accuracy": _count_rate(abstention_results),
        "prediction_coverage": _count_rate(coverage_results),
        "average_latency_ms": _mean_metric(
            latency_values,
            unknown_count=unknown_latency_count,
        ),
        "average_total_tokens": _mean_metric(
            token_values,
            unknown_count=unknown_token_count,
        ),
        "estimated_cost_usd": _mean_metric(
            cost_values,
            unknown_count=unknown_cost_count,
        ),
        "case_results": tuple(case_results),
    }

    return GroundedRecommendationEvaluationReport(
        **report_without_hash,
        report_content_hash=sha256_hexdigest(report_without_hash),
    )


def _score_case(
    *,
    case: GroundedRecommendationEvaluationCase,
    payload: GroundedRecommendationPredictionPayload,
) -> GroundedRecommendationCaseResult:
    action_correct = payload.recommended_action is case.expected_action
    review_correct = payload.requires_human_review == case.expected_requires_human_review
    sufficiency_correct = payload.evidence_sufficient == case.expected_evidence_sufficient

    citation_correct = _citation_identity_correct(
        case=case,
        payload=payload,
    )

    workspace_isolated = payload.foreign_workspace_evidence_count == 0

    abstention_correct = (
        payload.evidence_sufficient is False
        and payload.recommended_action is case.expected_action
        and payload.recommended_action
        in {
            GroundedRecommendationAction.REQUEST_MORE_INFORMATION,
            GroundedRecommendationAction.RECOMMEND_ESCALATION,
        }
        if case.expected_evidence_sufficient is False
        else None
    )

    return GroundedRecommendationCaseResult(
        case_id=case.case_id,
        prediction_present=True,
        prediction_succeeded=True,
        recommended_action_correct=action_correct,
        human_review_correct=review_correct,
        evidence_sufficiency_correct=sufficiency_correct,
        citation_identity_correct=citation_correct,
        workspace_isolated=workspace_isolated,
        grounded_abstention_correct=abstention_correct,
    )


def _citation_identity_correct(
    *,
    case: GroundedRecommendationEvaluationCase,
    payload: GroundedRecommendationPredictionPayload,
) -> bool | None:
    expected = set(case.expected_citation_chunk_ids)
    predicted = set(payload.citation_chunk_ids)
    retrieved = set(payload.retrieved_chunk_ids)

    if not expected and not predicted:
        return None

    return expected.issubset(predicted) and predicted.issubset(retrieved)


def _missing_case_result(
    case: GroundedRecommendationEvaluationCase,
) -> GroundedRecommendationCaseResult:
    return GroundedRecommendationCaseResult(
        case_id=case.case_id,
        prediction_present=False,
        prediction_succeeded=False,
        recommended_action_correct=False,
        human_review_correct=False,
        evidence_sufficiency_correct=False,
        citation_identity_correct=(False if case.expected_citation_chunk_ids else None),
        workspace_isolated=False,
        grounded_abstention_correct=(False if case.expected_evidence_sufficient is False else None),
        error_code="prediction_missing",
    )


def _failed_case_result(
    *,
    case: GroundedRecommendationEvaluationCase,
    error_code: str | None,
) -> GroundedRecommendationCaseResult:
    return GroundedRecommendationCaseResult(
        case_id=case.case_id,
        prediction_present=True,
        prediction_succeeded=False,
        recommended_action_correct=False,
        human_review_correct=False,
        evidence_sufficiency_correct=False,
        citation_identity_correct=(False if case.expected_citation_chunk_ids else None),
        workspace_isolated=False,
        grounded_abstention_correct=(False if case.expected_evidence_sufficient is False else None),
        error_code=error_code,
    )


def _collect_usage(
    *,
    prediction: GroundedRecommendationPrediction,
    latency_values: list[Decimal],
    token_values: list[Decimal],
    cost_values: list[Decimal],
) -> None:
    if prediction.latency_ms is not None:
        latency_values.append(Decimal(prediction.latency_ms))

    total_tokens = _total_tokens(prediction)
    if total_tokens is not None:
        token_values.append(Decimal(total_tokens))

    if prediction.estimated_cost_usd is not None:
        cost_values.append(prediction.estimated_cost_usd)


def _total_tokens(
    prediction: GroundedRecommendationPrediction,
) -> int | None:
    if prediction.input_tokens is None or prediction.output_tokens is None:
        return None

    return prediction.input_tokens + prediction.output_tokens


def _count_rate(values: list[bool]) -> CountRateMetric:
    numerator = sum(values)
    denominator = len(values)

    return CountRateMetric(
        numerator_count=numerator,
        denominator_count=denominator,
        rate=(_quantize(Decimal(numerator) / Decimal(denominator)) if denominator else None),
    )


def _mean_metric(
    values: list[Decimal],
    *,
    unknown_count: int,
) -> MeanMetric:
    total = sum(values, start=Decimal("0"))

    return MeanMetric(
        total=total,
        known_count=len(values),
        unknown_count=unknown_count,
        average=(_quantize(total / Decimal(len(values))) if values else None),
    )


def _quantize(value: Decimal) -> Decimal:
    return value.quantize(
        _METRIC_QUANTUM,
        rounding=ROUND_HALF_UP,
    )
