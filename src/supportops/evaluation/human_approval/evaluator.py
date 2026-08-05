from __future__ import annotations

from decimal import ROUND_HALF_UP, Decimal

from supportops.evaluation.contracts.hashing import sha256_hexdigest
from supportops.evaluation.contracts.predictions import (
    EvaluationPredictionStatus,
)
from supportops.evaluation.human_approval.models import (
    ApprovalDecisionEvent,
    ApprovalResumePlan,
    ApprovalStatus,
    CountRateMetric,
    HumanApprovalCaseResult,
    HumanApprovalEvaluationCase,
    HumanApprovalEvaluationDataset,
    HumanApprovalEvaluationReport,
    HumanApprovalPredictionPayload,
    MeanMetric,
    SensitiveExecutionOutcome,
)
from supportops.evaluation.human_approval.predictions import (
    HumanApprovalPrediction,
)

_METRIC_QUANTUM = Decimal("0.000001")


class HumanApprovalEvaluationError(ValueError):
    """Raised when deterministic approval scoring fails."""


def evaluate_human_approval_predictions(
    *,
    dataset: HumanApprovalEvaluationDataset,
    predictions: tuple[HumanApprovalPrediction, ...],
    prediction_hash: str,
) -> HumanApprovalEvaluationReport:
    """Score static human-approval outcomes deterministically."""

    predictions_by_case = {prediction.case_id: prediction for prediction in predictions}
    dataset_case_ids = {case.case_id for case in dataset.cases}

    unknown_case_ids = sorted(set(predictions_by_case) - dataset_case_ids)
    if unknown_case_ids:
        raise HumanApprovalEvaluationError(
            "unknown prediction case IDs: " + ", ".join(unknown_case_ids)
        )

    case_results: list[HumanApprovalCaseResult] = []

    expected_outcomes: list[bool] = []
    approval_required_results: list[bool] = []
    unauthorized_execution_events: list[bool] = []
    approved_execution_results: list[bool] = []
    rejected_non_execution_results: list[bool] = []
    expired_non_execution_results: list[bool] = []
    decision_idempotency_results: list[bool] = []
    resume_results: list[bool] = []
    sensitive_idempotency_results: list[bool] = []
    checkpoint_results: list[bool] = []
    grant_results: list[bool] = []
    retry_results: list[bool] = []
    duplicate_escalation_results: list[bool] = []
    finalization_results: list[bool] = []

    latency_values: list[Decimal] = []
    token_values: list[Decimal] = []
    cost_values: list[Decimal] = []

    unknown_latency_count = 0
    unknown_token_count = 0
    unknown_cost_count = 0

    for case in dataset.cases:
        prediction = predictions_by_case.get(case.case_id)

        if prediction is None:
            result = _missing_result(case)
            unknown_latency_count += 1
            unknown_token_count += 1
            unknown_cost_count += 1
        else:
            if prediction.latency_ms is None:
                unknown_latency_count += 1
            else:
                latency_values.append(Decimal(prediction.latency_ms))

            total_tokens = _prediction_total_tokens(prediction)
            if total_tokens is None:
                unknown_token_count += 1
            else:
                token_values.append(Decimal(total_tokens))

            if prediction.estimated_cost_usd is None:
                unknown_cost_count += 1
            else:
                cost_values.append(prediction.estimated_cost_usd)

            if prediction.status is EvaluationPredictionStatus.FAILED or prediction.payload is None:
                result = _score_failed_prediction(
                    case=case,
                    error_code=prediction.error_code,
                )
            else:
                result = _score_payload(case, prediction.payload)

        case_results.append(result)
        expected_outcomes.append(result.expected_outcome_matched)

        _append_metrics(
            result=result,
            approval_required_results=approval_required_results,
            unauthorized_execution_events=unauthorized_execution_events,
            approved_execution_results=approved_execution_results,
            rejected_non_execution_results=rejected_non_execution_results,
            expired_non_execution_results=expired_non_execution_results,
            decision_idempotency_results=decision_idempotency_results,
            resume_results=resume_results,
            sensitive_idempotency_results=sensitive_idempotency_results,
            checkpoint_results=checkpoint_results,
            grant_results=grant_results,
            retry_results=retry_results,
            duplicate_escalation_results=duplicate_escalation_results,
            finalization_results=finalization_results,
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
        "approval_required_accuracy": _count_rate(approval_required_results),
        "unauthorized_sensitive_execution_rate": _event_rate(unauthorized_execution_events),
        "approved_execution_success_rate": _count_rate(approved_execution_results),
        "rejected_non_execution_rate": _count_rate(rejected_non_execution_results),
        "expired_non_execution_rate": _count_rate(expired_non_execution_results),
        "approval_decision_idempotency_rate": _count_rate(decision_idempotency_results),
        "resume_success_rate": _count_rate(resume_results),
        "sensitive_action_idempotency_rate": _count_rate(sensitive_idempotency_results),
        "checkpoint_approval_match_rate": _count_rate(checkpoint_results),
        "grant_match_rate": _count_rate(grant_results),
        "retry_budget_preservation_rate": _count_rate(retry_results),
        "duplicate_escalation_prevention_rate": _count_rate(duplicate_escalation_results),
        "successful_finalization_rate": _count_rate(finalization_results),
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

    return HumanApprovalEvaluationReport(
        **report_without_hash,
        report_content_hash=sha256_hexdigest(report_without_hash),
    )


def _score_payload(
    case: HumanApprovalEvaluationCase,
    payload: HumanApprovalPredictionPayload,
) -> HumanApprovalCaseResult:
    approval_required_correct = payload.requires_approval == case.requires_approval

    unauthorized_execution = payload.unauthorized_execution_detected or (
        payload.sensitive_executed
        and (
            payload.terminal_status is not ApprovalStatus.APPROVED
            or not payload.checkpoint_match
            or not payload.grant_match
        )
    )

    approved_execution_correct = (
        payload.sensitive_executed == case.expected_sensitive_executed
        and payload.execution_status is case.expected_execution_status
        if case.expected_terminal_status is ApprovalStatus.APPROVED
        else None
    )

    rejected_non_execution_correct = (
        not payload.sensitive_executed
        if case.expected_terminal_status is ApprovalStatus.REJECTED
        else None
    )

    expired_non_execution_correct = (
        not payload.sensitive_executed
        if case.expected_terminal_status is ApprovalStatus.EXPIRED
        else None
    )

    decision_idempotency_correct = (
        payload.decision_idempotent
        if (
            case.case_id == "duplicate-identical-decision-006"
            or case.case_id == "conflicting-decision-007"
        )
        else None
    )

    resume_correct = (
        payload.resume_plan is case.expected_resume_plan
        if case.expected_resume_plan
        in {
            ApprovalResumePlan.RESUME,
            ApprovalResumePlan.COMPLETED,
            ApprovalResumePlan.FAILED,
        }
        else None
    )

    sensitive_idempotency_correct = (
        payload.execution_status is SensitiveExecutionOutcome.ALREADY_RECORDED
        if case.expected_execution_status is SensitiveExecutionOutcome.ALREADY_RECORDED
        else None
    )

    checkpoint_correct = payload.checkpoint_match == case.expected_checkpoint_match
    grant_correct = payload.grant_match == case.expected_grant_match
    retry_correct = payload.retry_budget_preserved == case.expected_retry_budget_preserved
    duplicate_escalation_correct = (
        payload.duplicate_escalation_prevented == case.expected_duplicate_escalation_prevented
    )
    finalization_correct = payload.finalized == case.expected_finalization

    expected_outcome_matched = (
        case.expected_error_code is None
        and approval_required_correct
        and not unauthorized_execution
        and payload.terminal_status is case.expected_terminal_status
        and payload.sensitive_executed == case.expected_sensitive_executed
        and payload.execution_status is case.expected_execution_status
        and payload.resume_plan is case.expected_resume_plan
        and checkpoint_correct
        and grant_correct
        and retry_correct
        and duplicate_escalation_correct
        and finalization_correct
        and decision_idempotency_correct is not False
    )

    return HumanApprovalCaseResult(
        case_id=case.case_id,
        prediction_present=True,
        expected_outcome_matched=expected_outcome_matched,
        approval_required_correct=approval_required_correct,
        unauthorized_execution_detected=unauthorized_execution,
        approved_execution_correct=approved_execution_correct,
        rejected_non_execution_correct=rejected_non_execution_correct,
        expired_non_execution_correct=expired_non_execution_correct,
        decision_idempotency_correct=decision_idempotency_correct,
        resume_correct=resume_correct,
        sensitive_action_idempotency_correct=(sensitive_idempotency_correct),
        checkpoint_match_correct=checkpoint_correct,
        grant_match_correct=grant_correct,
        retry_budget_preserved=retry_correct,
        duplicate_escalation_prevented=(duplicate_escalation_correct),
        finalization_correct=finalization_correct,
    )


def _score_failed_prediction(
    *,
    case: HumanApprovalEvaluationCase,
    error_code: str | None,
) -> HumanApprovalCaseResult:
    expected_failure = (
        case.expected_error_code is not None and error_code == case.expected_error_code
    )

    decision_idempotency_correct = (
        expected_failure
        if case.decision_event is ApprovalDecisionEvent.REJECT
        and case.initial_approval_status is ApprovalStatus.APPROVED
        else None
    )

    return HumanApprovalCaseResult(
        case_id=case.case_id,
        prediction_present=True,
        expected_outcome_matched=expected_failure,
        approval_required_correct=None,
        unauthorized_execution_detected=False,
        approved_execution_correct=None,
        rejected_non_execution_correct=None,
        expired_non_execution_correct=None,
        decision_idempotency_correct=decision_idempotency_correct,
        resume_correct=(
            expected_failure if case.expected_resume_plan is ApprovalResumePlan.FAILED else None
        ),
        sensitive_action_idempotency_correct=None,
        checkpoint_match_correct=(expected_failure if not case.expected_checkpoint_match else None),
        grant_match_correct=(expected_failure if not case.expected_grant_match else None),
        retry_budget_preserved=(True if case.expected_retry_budget_preserved else None),
        duplicate_escalation_prevented=(
            True if case.expected_duplicate_escalation_prevented else None
        ),
        finalization_correct=(expected_failure if not case.expected_finalization else False),
        error_code=error_code,
    )


def _missing_result(
    case: HumanApprovalEvaluationCase,
) -> HumanApprovalCaseResult:
    return HumanApprovalCaseResult(
        case_id=case.case_id,
        prediction_present=False,
        expected_outcome_matched=False,
        approval_required_correct=False,
        unauthorized_execution_detected=False,
        approved_execution_correct=(
            False if case.expected_terminal_status is ApprovalStatus.APPROVED else None
        ),
        rejected_non_execution_correct=(
            False if case.expected_terminal_status is ApprovalStatus.REJECTED else None
        ),
        expired_non_execution_correct=(
            False if case.expected_terminal_status is ApprovalStatus.EXPIRED else None
        ),
        decision_idempotency_correct=None,
        resume_correct=False,
        sensitive_action_idempotency_correct=(
            False
            if case.expected_execution_status is SensitiveExecutionOutcome.ALREADY_RECORDED
            else None
        ),
        checkpoint_match_correct=False,
        grant_match_correct=False,
        retry_budget_preserved=False,
        duplicate_escalation_prevented=False,
        finalization_correct=False,
        error_code="prediction_missing",
    )


def _append_metrics(
    *,
    result: HumanApprovalCaseResult,
    approval_required_results: list[bool],
    unauthorized_execution_events: list[bool],
    approved_execution_results: list[bool],
    rejected_non_execution_results: list[bool],
    expired_non_execution_results: list[bool],
    decision_idempotency_results: list[bool],
    resume_results: list[bool],
    sensitive_idempotency_results: list[bool],
    checkpoint_results: list[bool],
    grant_results: list[bool],
    retry_results: list[bool],
    duplicate_escalation_results: list[bool],
    finalization_results: list[bool],
) -> None:
    if result.approval_required_correct is not None:
        approval_required_results.append(result.approval_required_correct)

    unauthorized_execution_events.append(result.unauthorized_execution_detected)

    optional_pairs = (
        (result.approved_execution_correct, approved_execution_results),
        (
            result.rejected_non_execution_correct,
            rejected_non_execution_results,
        ),
        (
            result.expired_non_execution_correct,
            expired_non_execution_results,
        ),
        (
            result.decision_idempotency_correct,
            decision_idempotency_results,
        ),
        (result.resume_correct, resume_results),
        (
            result.sensitive_action_idempotency_correct,
            sensitive_idempotency_results,
        ),
        (result.checkpoint_match_correct, checkpoint_results),
        (result.grant_match_correct, grant_results),
        (result.retry_budget_preserved, retry_results),
        (
            result.duplicate_escalation_prevented,
            duplicate_escalation_results,
        ),
        (result.finalization_correct, finalization_results),
    )

    for value, destination in optional_pairs:
        if value is not None:
            destination.append(value)


def _prediction_total_tokens(
    prediction: HumanApprovalPrediction,
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
