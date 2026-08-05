from __future__ import annotations

from decimal import ROUND_HALF_UP, Decimal

from supportops.evaluation.contracts.hashing import sha256_hexdigest
from supportops.evaluation.contracts.predictions import (
    EvaluationPredictionStatus,
)
from supportops.evaluation.controlled_support.models import (
    ControlledSupportCaseResult,
    ControlledSupportEvaluationCase,
    ControlledSupportEvaluationDataset,
    ControlledSupportEvaluationReport,
    ControlledSupportPredictionPayload,
    CountRateMetric,
    MeanMetric,
)
from supportops.evaluation.controlled_support.predictions import (
    ControlledSupportPrediction,
)

_METRIC_QUANTUM = Decimal("0.000001")


class ControlledSupportEvaluationError(ValueError):
    """Raised when deterministic controlled-support scoring fails."""


def evaluate_controlled_support_predictions(
    *,
    dataset: ControlledSupportEvaluationDataset,
    predictions: tuple[ControlledSupportPrediction, ...],
    prediction_hash: str,
) -> ControlledSupportEvaluationReport:
    """Score static controlled-support traces deterministically."""

    predictions_by_case = {prediction.case_id: prediction for prediction in predictions}
    dataset_case_ids = {case.case_id for case in dataset.cases}

    unknown_case_ids = sorted(set(predictions_by_case) - dataset_case_ids)
    if unknown_case_ids:
        raise ControlledSupportEvaluationError(
            "unknown prediction case IDs: " + ", ".join(unknown_case_ids)
        )

    case_results: list[ControlledSupportCaseResult] = []

    expected_outcomes: list[bool] = []
    required_tool_results: list[bool] = []
    forbidden_execution_results: list[bool] = []
    sequence_results: list[bool] = []
    repeated_acceptance_results: list[bool] = []
    step_results: list[bool] = []
    action_results: list[bool] = []
    review_results: list[bool] = []
    citation_results: list[bool] = []
    abstention_results: list[bool] = []
    workspace_results: list[bool] = []
    completion_results: list[bool] = []

    tool_call_values: list[Decimal] = []
    llm_invocation_values: list[Decimal] = []
    latency_values: list[Decimal] = []
    token_values: list[Decimal] = []
    cost_values: list[Decimal] = []

    unknown_tool_count = 0
    unknown_llm_count = 0
    unknown_latency_count = 0
    unknown_token_count = 0
    unknown_cost_count = 0

    for case in dataset.cases:
        prediction = predictions_by_case.get(case.case_id)

        if prediction is None:
            result = _missing_result(case)
            case_results.append(result)
            _append_case_metrics(
                case=case,
                result=result,
                required_tool_results=required_tool_results,
                forbidden_execution_results=forbidden_execution_results,
                sequence_results=sequence_results,
                repeated_acceptance_results=repeated_acceptance_results,
                step_results=step_results,
                action_results=action_results,
                review_results=review_results,
                citation_results=citation_results,
                abstention_results=abstention_results,
                workspace_results=workspace_results,
                completion_results=completion_results,
            )
            expected_outcomes.append(False)
            unknown_tool_count += 1
            unknown_llm_count += 1
            unknown_latency_count += 1
            unknown_token_count += 1
            unknown_cost_count += 1
            continue

        _collect_envelope_usage(
            prediction=prediction,
            latency_values=latency_values,
            token_values=token_values,
            cost_values=cost_values,
        )

        if prediction.latency_ms is None:
            unknown_latency_count += 1
        if _prediction_total_tokens(prediction) is None:
            unknown_token_count += 1
        if prediction.estimated_cost_usd is None:
            unknown_cost_count += 1

        if prediction.status is EvaluationPredictionStatus.FAILED or prediction.payload is None:
            result = _score_failed_prediction(
                case=case,
                error_code=prediction.error_code,
            )
            unknown_tool_count += 1
            unknown_llm_count += 1
        else:
            result = _score_payload(case, prediction.payload)
            tool_call_values.append(Decimal(prediction.payload.tool_call_count))
            llm_invocation_values.append(Decimal(prediction.payload.llm_invocation_count))

        case_results.append(result)
        expected_outcomes.append(result.expected_outcome_matched)
        _append_case_metrics(
            case=case,
            result=result,
            required_tool_results=required_tool_results,
            forbidden_execution_results=forbidden_execution_results,
            sequence_results=sequence_results,
            repeated_acceptance_results=repeated_acceptance_results,
            step_results=step_results,
            action_results=action_results,
            review_results=review_results,
            citation_results=citation_results,
            abstention_results=abstention_results,
            workspace_results=workspace_results,
            completion_results=completion_results,
        )

    report_without_hash = {
        "dataset_id": dataset.dataset_id,
        "dataset_version": dataset.dataset_version,
        "schema_version": dataset.schema_version,
        "workflow_name": dataset.workflow_name,
        "workflow_version": dataset.workflow_version,
        "dataset_hash": dataset.content_hash,
        "prediction_hash": prediction_hash,
        "case_count": len(dataset.cases),
        "expected_outcome_accuracy": _count_rate(expected_outcomes),
        "required_tool_call_rate": _count_rate(required_tool_results),
        "forbidden_tool_call_rate": _event_rate(forbidden_execution_results),
        "tool_sequence_acceptance_rate": _count_rate(sequence_results),
        "repeated_tool_call_rate": _event_rate(repeated_acceptance_results),
        "step_limit_behavior_accuracy": _count_rate(step_results),
        "recommended_action_accuracy": _count_rate(action_results),
        "human_review_recommendation_accuracy": _count_rate(review_results),
        "citation_validity_rate": _count_rate(citation_results),
        "grounded_abstention_accuracy": _count_rate(abstention_results),
        "workspace_isolation_rate": _count_rate(workspace_results),
        "successful_completion_rate": _count_rate(completion_results),
        "average_tool_calls": _mean_metric(
            tool_call_values,
            unknown_count=unknown_tool_count,
        ),
        "average_llm_invocations": _mean_metric(
            llm_invocation_values,
            unknown_count=unknown_llm_count,
        ),
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

    return ControlledSupportEvaluationReport(
        **report_without_hash,
        report_content_hash=sha256_hexdigest(report_without_hash),
    )


def _score_payload(
    case: ControlledSupportEvaluationCase,
    payload: ControlledSupportPredictionPayload,
) -> ControlledSupportCaseResult:
    accepted_tools = {tool_call.tool_name for tool_call in payload.tool_calls if tool_call.accepted}

    required_tools_satisfied = (
        set(case.required_tool_calls).issubset(accepted_tools) if case.required_tool_calls else None
    )

    forbidden_detected = payload.executed_forbidden_tool_count > 0 or bool(
        set(case.forbidden_tool_calls) & accepted_tools
    )

    sequence_accepted = (
        payload.accepted_tool_sequence == case.expected_tool_sequence
        if case.expected_tool_sequence
        else None
    )

    repeated_accepted = payload.accepted_repeated_tool_count > 0

    step_correct = payload.step_limit_violated == case.expected_step_limit_violation

    action_correct = (
        payload.recommended_action == case.expected_recommended_action
        if case.expected_recommended_action is not None
        else None
    )

    review_correct = (
        payload.requires_human_review == case.expected_requires_human_review
        if case.expected_requires_human_review is not None
        else None
    )

    citation_valid = _citation_validity(case, payload)

    abstention_correct = (
        payload.evidence_sufficient == case.expected_evidence_sufficient
        and payload.recommended_action == case.expected_recommended_action
        if case.expected_evidence_sufficient is False
        else None
    )

    workspace_isolated = payload.foreign_workspace_evidence_count == 0

    completion_correct = payload.completed == case.expected_completion

    expected_outcome_matched = (
        case.expected_error_code is None
        and completion_correct
        and not forbidden_detected
        and not repeated_accepted
        and step_correct
        and required_tools_satisfied is not False
        and sequence_accepted is not False
        and action_correct is not False
        and review_correct is not False
        and citation_valid is not False
        and abstention_correct is not False
        and workspace_isolated
    )

    return ControlledSupportCaseResult(
        case_id=case.case_id,
        prediction_present=True,
        expected_outcome_matched=expected_outcome_matched,
        required_tools_satisfied=required_tools_satisfied,
        forbidden_tool_execution_detected=(
            forbidden_detected if case.forbidden_tool_calls else None
        ),
        tool_sequence_accepted=sequence_accepted,
        repeated_tool_accepted=(repeated_accepted if not case.allow_repeated_tool_calls else None),
        step_limit_behavior_correct=step_correct,
        recommended_action_correct=action_correct,
        human_review_correct=review_correct,
        citation_valid=citation_valid,
        grounded_abstention_correct=abstention_correct,
        workspace_evidence_isolated=workspace_isolated,
        completion_correct=completion_correct,
    )


def _score_failed_prediction(
    *,
    case: ControlledSupportEvaluationCase,
    error_code: str | None,
) -> ControlledSupportCaseResult:
    expected_failure = (
        case.expected_error_code is not None
        and error_code == case.expected_error_code
        and not case.expected_completion
    )

    return ControlledSupportCaseResult(
        case_id=case.case_id,
        prediction_present=True,
        expected_outcome_matched=expected_failure,
        required_tools_satisfied=None,
        forbidden_tool_execution_detected=(False if case.forbidden_tool_calls else None),
        tool_sequence_accepted=None,
        repeated_tool_accepted=(
            False if case.expected_error_code == "tool_repeated_call" else None
        ),
        step_limit_behavior_correct=(
            expected_failure
            if case.expected_step_limit_violation
            else not case.expected_step_limit_violation
        ),
        recommended_action_correct=None,
        human_review_correct=None,
        citation_valid=(False if case.expected_error_code == "invalid_citation" else None),
        grounded_abstention_correct=None,
        workspace_evidence_isolated=True,
        completion_correct=expected_failure,
        error_code=error_code,
    )


def _missing_result(
    case: ControlledSupportEvaluationCase,
) -> ControlledSupportCaseResult:
    return ControlledSupportCaseResult(
        case_id=case.case_id,
        prediction_present=False,
        expected_outcome_matched=False,
        required_tools_satisfied=(False if case.required_tool_calls else None),
        forbidden_tool_execution_detected=(False if case.forbidden_tool_calls else None),
        tool_sequence_accepted=(False if case.expected_tool_sequence else None),
        repeated_tool_accepted=(False if not case.allow_repeated_tool_calls else None),
        step_limit_behavior_correct=False,
        recommended_action_correct=(
            False if case.expected_recommended_action is not None else None
        ),
        human_review_correct=(False if case.expected_requires_human_review is not None else None),
        citation_valid=(False if case.expected_citation_chunk_ids else None),
        grounded_abstention_correct=(False if case.expected_evidence_sufficient is False else None),
        workspace_evidence_isolated=False,
        completion_correct=False,
        error_code="prediction_missing",
    )


def _citation_validity(
    case: ControlledSupportEvaluationCase,
    payload: ControlledSupportPredictionPayload,
) -> bool | None:
    expected = set(case.expected_citation_chunk_ids)
    citations = set(payload.citation_chunk_ids)
    retrieved = set(payload.retrieved_chunk_ids)

    if not expected and not citations:
        return None

    return expected.issubset(citations) and citations.issubset(retrieved)


def _append_case_metrics(
    *,
    case: ControlledSupportEvaluationCase,
    result: ControlledSupportCaseResult,
    required_tool_results: list[bool],
    forbidden_execution_results: list[bool],
    sequence_results: list[bool],
    repeated_acceptance_results: list[bool],
    step_results: list[bool],
    action_results: list[bool],
    review_results: list[bool],
    citation_results: list[bool],
    abstention_results: list[bool],
    workspace_results: list[bool],
    completion_results: list[bool],
) -> None:
    if result.required_tools_satisfied is not None:
        required_tool_results.append(result.required_tools_satisfied)

    if result.forbidden_tool_execution_detected is not None:
        forbidden_execution_results.append(result.forbidden_tool_execution_detected)

    if result.tool_sequence_accepted is not None:
        sequence_results.append(result.tool_sequence_accepted)

    if result.repeated_tool_accepted is not None:
        repeated_acceptance_results.append(result.repeated_tool_accepted)

    step_results.append(result.step_limit_behavior_correct)

    if result.recommended_action_correct is not None:
        action_results.append(result.recommended_action_correct)

    if result.human_review_correct is not None:
        review_results.append(result.human_review_correct)

    if result.citation_valid is not None:
        expected_invalid_citation = case.expected_error_code == "invalid_citation"
        citation_results.append(
            not result.citation_valid if expected_invalid_citation else result.citation_valid
        )

    if result.grounded_abstention_correct is not None:
        abstention_results.append(result.grounded_abstention_correct)

    if result.workspace_evidence_isolated is not None:
        workspace_results.append(result.workspace_evidence_isolated)

    completion_results.append(result.completion_correct)


def _collect_envelope_usage(
    *,
    prediction: ControlledSupportPrediction,
    latency_values: list[Decimal],
    token_values: list[Decimal],
    cost_values: list[Decimal],
) -> None:
    if prediction.latency_ms is not None:
        latency_values.append(Decimal(prediction.latency_ms))

    total_tokens = _prediction_total_tokens(prediction)
    if total_tokens is not None:
        token_values.append(Decimal(total_tokens))

    if prediction.estimated_cost_usd is not None:
        cost_values.append(prediction.estimated_cost_usd)


def _prediction_total_tokens(
    prediction: ControlledSupportPrediction,
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


def _event_rate(values: list[bool]) -> CountRateMetric:
    event_count = sum(values)
    denominator = len(values)

    return CountRateMetric(
        numerator_count=event_count,
        denominator_count=denominator,
        rate=(_quantize(Decimal(event_count) / Decimal(denominator)) if denominator else None),
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
