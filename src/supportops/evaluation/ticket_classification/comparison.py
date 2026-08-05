"""Deterministic paired comparison for ticket-classification prompts."""

from __future__ import annotations

import json
from decimal import Decimal
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    ValidationError,
    model_validator,
)

from supportops.evaluation.contracts.artifacts import (
    write_canonical_json_atomically,
)
from supportops.evaluation.contracts.hashing import sha256_hexdigest
from supportops.evaluation.contracts.manifest import EvaluationRunStatus
from supportops.evaluation.ticket_classification.dataset import (
    TicketClassificationEvaluationDataset,
)
from supportops.evaluation.ticket_classification.evaluator import (
    DEFAULT_TICKET_CLASSIFICATION_RELEASE_GATE_PROFILE,
    TicketClassificationEvaluationReport,
    TicketClassificationGateOutcome,
    TicketClassificationReleaseGateDefinition,
    TicketClassificationReleaseGateEvaluation,
    TicketClassificationReleaseGateProfile,
    TicketClassificationReleaseGateResult,
    evaluate_ticket_classification_predictions,
    evaluate_ticket_classification_release_gates,
)
from supportops.evaluation.ticket_classification.predictions import (
    TicketClassificationPredictionSet,
)
from supportops.evaluation.ticket_classification.split_manifest import (
    TicketClassificationSplitManifest,
)

Sha256Hex = Annotated[
    str,
    StringConstraints(pattern=r"^[0-9a-f]{64}$"),
]
NonEmptyString = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1),
]
_RATE_ONE = Decimal("1.000000")
_RATE_ZERO = Decimal("0.000000")


class TicketClassificationPairedComparisonError(ValueError):
    """Raised when paired classification evidence cannot be compared safely."""


class TicketClassificationComparisonEvidenceKind(StrEnum):
    """Authority represented by paired prediction evidence."""

    STATIC_FIXTURE = "static_fixture"
    PROVIDER = "provider"


class TicketClassificationPairedGateStatus(StrEnum):
    """Aggregate result for paired classification release gates."""

    PASSED = "passed"
    FAILED = "failed"
    INCOMPLETE = "incomplete"


class TicketClassificationMetricDelta(BaseModel):
    """Baseline, candidate, and signed delta for one deterministic metric."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    baseline_value: Decimal
    candidate_value: Decimal
    delta: Decimal

    @model_validator(mode="after")
    def validate_delta(self) -> Self:
        if self.delta != self.candidate_value - self.baseline_value:
            raise ValueError("Metric delta must equal candidate minus baseline.")
        return self


class TicketClassificationOptionalMetricDelta(BaseModel):
    """Signed operational delta that may remain unknown."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    baseline_value: Decimal | None
    candidate_value: Decimal | None
    delta: Decimal | None
    unknown_reason: NonEmptyString | None = None

    @model_validator(mode="after")
    def validate_optional_delta(self) -> Self:
        values = (
            self.baseline_value,
            self.candidate_value,
            self.delta,
        )
        if all(value is not None for value in values):
            assert self.baseline_value is not None
            assert self.candidate_value is not None
            assert self.delta is not None
            if self.delta != self.candidate_value - self.baseline_value:
                raise ValueError("Operational delta must equal candidate minus baseline.")
            if self.unknown_reason is not None:
                raise ValueError("Known operational deltas cannot define unknown_reason.")
            return self

        if any(value is not None for value in values):
            raise ValueError("Operational delta values must be all known or all unknown.")
        if self.unknown_reason is None:
            raise ValueError("Unknown operational deltas require unknown_reason.")
        return self


class TicketClassificationPairedGateEvaluation(BaseModel):
    """Complete candidate gate evaluation with paired metrics resolved."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    profile_id: NonEmptyString
    profile_version: int = Field(ge=1)
    candidate_report_content_hash: Sha256Hex
    gate_results: tuple[TicketClassificationReleaseGateResult, ...]
    blocking_failure_count: int = Field(ge=0)
    not_applicable_count: int = Field(ge=0)
    status: TicketClassificationPairedGateStatus
    content_hash: Sha256Hex

    @model_validator(mode="after")
    def validate_gate_evaluation(self) -> Self:
        if self.blocking_failure_count != sum(
            result.blocking and result.outcome is TicketClassificationGateOutcome.FAILED
            for result in self.gate_results
        ):
            raise ValueError("Paired gate blocking failure count is inconsistent.")
        if self.not_applicable_count != sum(
            result.outcome is TicketClassificationGateOutcome.NOT_APPLICABLE
            for result in self.gate_results
        ):
            raise ValueError("Paired gate not-applicable count is inconsistent.")
        expected_status = _aggregate_paired_gate_status(self.gate_results)
        if self.status is not expected_status:
            raise ValueError("Paired gate status is inconsistent.")
        content = self.model_dump(
            mode="python",
            exclude={"content_hash"},
        )
        if self.content_hash != sha256_hexdigest(content):
            raise ValueError("Paired gate evaluation hash does not match its content.")
        return self


class TicketClassificationPairedComparisonContent(BaseModel):
    """Canonical paired comparison content before hashing."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    comparison_id: NonEmptyString
    comparison_version: int = Field(ge=1)
    schema_version: NonEmptyString
    evidence_kind: TicketClassificationComparisonEvidenceKind

    dataset_id: NonEmptyString
    dataset_version: int = Field(ge=1)
    dataset_content_hash: Sha256Hex
    split_manifest_id: NonEmptyString
    split_manifest_version: int = Field(ge=1)
    split_manifest_content_hash: Sha256Hex

    provider: NonEmptyString
    model: NonEmptyString

    baseline_prompt_id: NonEmptyString
    baseline_prompt_version: int = Field(ge=1)
    baseline_prompt_content_hash: Sha256Hex
    baseline_predictions_content_hash: Sha256Hex
    baseline_report_content_hash: Sha256Hex

    candidate_prompt_id: NonEmptyString
    candidate_prompt_version: int = Field(ge=1)
    candidate_prompt_content_hash: Sha256Hex
    candidate_predictions_content_hash: Sha256Hex
    candidate_report_content_hash: Sha256Hex

    case_count: int = Field(ge=1)

    structured_label_exact_match: TicketClassificationMetricDelta
    category_accuracy: TicketClassificationMetricDelta
    intent_accuracy: TicketClassificationMetricDelta
    urgency_accuracy: TicketClassificationMetricDelta
    sentiment_accuracy: TicketClassificationMetricDelta
    human_review_accuracy: TicketClassificationMetricDelta
    prediction_coverage: TicketClassificationMetricDelta

    human_review_false_negative_delta: int
    human_review_false_positive_delta: int
    failed_prediction_count_delta: int

    average_total_tokens: TicketClassificationOptionalMetricDelta
    average_estimated_cost_usd: TicketClassificationOptionalMetricDelta
    average_latency_ms: TicketClassificationOptionalMetricDelta

    improved_case_ids: tuple[str, ...]
    regressed_case_ids: tuple[str, ...]
    unchanged_case_ids: tuple[str, ...]
    holdout_regressed_case_ids: tuple[str, ...]
    safety_gate_regressed_case_ids: tuple[str, ...]

    gate_evaluation: TicketClassificationPairedGateEvaluation
    run_status: EvaluationRunStatus

    @model_validator(mode="after")
    def validate_comparison_content(self) -> Self:
        if self.baseline_prompt_id != self.candidate_prompt_id:
            raise ValueError("Paired prompts must share one prompt_id.")
        if self.baseline_prompt_version == self.candidate_prompt_version:
            raise ValueError("Paired prompts must use distinct versions.")
        if self.baseline_prompt_content_hash == self.candidate_prompt_content_hash:
            raise ValueError("Paired prompts must use distinct content hashes.")

        partition = self.improved_case_ids + self.regressed_case_ids + self.unchanged_case_ids
        if len(partition) != self.case_count:
            raise ValueError("Case outcome partition must cover the full dataset.")
        if len(partition) != len(set(partition)):
            raise ValueError("Case outcome partition must not contain duplicates.")
        if not set(self.holdout_regressed_case_ids) <= set(self.regressed_case_ids):
            raise ValueError("Holdout regressions must be included in regressed_case_ids.")
        if not set(self.safety_gate_regressed_case_ids) <= set(self.regressed_case_ids):
            raise ValueError("Safety-gate regressions must be included in regressed_case_ids.")
        expected_run_status = (
            EvaluationRunStatus.INCOMPLETE
            if self.gate_evaluation.status is TicketClassificationPairedGateStatus.INCOMPLETE
            else EvaluationRunStatus.COMPLETE
        )
        if self.run_status is not expected_run_status:
            raise ValueError("Comparison run status is inconsistent.")
        return self


class TicketClassificationPairedComparison(TicketClassificationPairedComparisonContent):
    """Immutable paired classification comparison artifact."""

    comparison_content_hash: Sha256Hex

    @model_validator(mode="after")
    def validate_content_hash(self) -> Self:
        content = TicketClassificationPairedComparisonContent.model_validate(
            self.model_dump(
                mode="python",
                exclude={"comparison_content_hash"},
            )
        )
        if self.comparison_content_hash != sha256_hexdigest(content):
            raise ValueError("Paired comparison hash does not match canonical content.")
        return self


def compare_ticket_classification_prediction_sets(
    *,
    dataset: TicketClassificationEvaluationDataset,
    split_manifest: TicketClassificationSplitManifest,
    baseline_predictions: TicketClassificationPredictionSet,
    candidate_predictions: TicketClassificationPredictionSet,
    evidence_kind: TicketClassificationComparisonEvidenceKind,
    profile: TicketClassificationReleaseGateProfile = (
        DEFAULT_TICKET_CLASSIFICATION_RELEASE_GATE_PROFILE
    ),
) -> TicketClassificationPairedComparison:
    """Compare two prompt versions over the same immutable evidence."""

    split_manifest.validate_dataset_binding(
        dataset_id=dataset.dataset_id,
        dataset_version=dataset.version,
        dataset_hash=dataset.content_hash,
        dataset_case_ids=tuple(case.case_id for case in dataset.cases),
    )
    _validate_pairing(
        dataset=dataset,
        baseline_predictions=baseline_predictions,
        candidate_predictions=candidate_predictions,
    )

    baseline_report = evaluate_ticket_classification_predictions(
        dataset=dataset,
        predictions=baseline_predictions,
    )
    candidate_report = evaluate_ticket_classification_predictions(
        dataset=dataset,
        predictions=candidate_predictions,
    )
    candidate_standalone_gates = evaluate_ticket_classification_release_gates(
        candidate_report,
        profile=profile,
    )
    paired_gate_evaluation = _evaluate_paired_release_gates(
        baseline_report=baseline_report,
        candidate_report=candidate_report,
        candidate_standalone_gates=candidate_standalone_gates,
        profile=profile,
    )

    improved_case_ids, regressed_case_ids, unchanged_case_ids = _partition_case_outcomes(
        baseline_report=baseline_report,
        candidate_report=candidate_report,
    )
    holdout_case_ids = set(split_manifest.assignments.holdout)
    safety_gate_case_ids = set(split_manifest.assignments.safety_gate)

    content = TicketClassificationPairedComparisonContent(
        comparison_id="ticket-classification-prompt-v1-v2",
        comparison_version=1,
        schema_version="ticket-classification-paired-comparison-v1",
        evidence_kind=evidence_kind,
        dataset_id=dataset.dataset_id,
        dataset_version=dataset.version,
        dataset_content_hash=dataset.content_hash,
        split_manifest_id=split_manifest.split_manifest_id,
        split_manifest_version=split_manifest.split_manifest_version,
        split_manifest_content_hash=split_manifest.content_hash(),
        provider=baseline_report.provider,
        model=baseline_report.model,
        baseline_prompt_id=baseline_report.prompt_id,
        baseline_prompt_version=baseline_report.prompt_version,
        baseline_prompt_content_hash=baseline_report.prompt_content_hash,
        baseline_predictions_content_hash=baseline_predictions.content_hash,
        baseline_report_content_hash=baseline_report.report_content_hash,
        candidate_prompt_id=candidate_report.prompt_id,
        candidate_prompt_version=candidate_report.prompt_version,
        candidate_prompt_content_hash=candidate_report.prompt_content_hash,
        candidate_predictions_content_hash=candidate_predictions.content_hash,
        candidate_report_content_hash=candidate_report.report_content_hash,
        case_count=dataset.case_count,
        structured_label_exact_match=_metric_delta(
            baseline_report.structured_label_exact_match.rate,
            candidate_report.structured_label_exact_match.rate,
        ),
        category_accuracy=_metric_delta(
            baseline_report.category_accuracy.rate,
            candidate_report.category_accuracy.rate,
        ),
        intent_accuracy=_metric_delta(
            baseline_report.intent_accuracy.rate,
            candidate_report.intent_accuracy.rate,
        ),
        urgency_accuracy=_metric_delta(
            baseline_report.urgency_accuracy.rate,
            candidate_report.urgency_accuracy.rate,
        ),
        sentiment_accuracy=_metric_delta(
            baseline_report.sentiment_accuracy.rate,
            candidate_report.sentiment_accuracy.rate,
        ),
        human_review_accuracy=_metric_delta(
            baseline_report.human_review_accuracy.rate,
            candidate_report.human_review_accuracy.rate,
        ),
        prediction_coverage=_metric_delta(
            _prediction_coverage_rate(baseline_report),
            _prediction_coverage_rate(candidate_report),
        ),
        human_review_false_negative_delta=(
            candidate_report.human_review.false_negative_count
            - baseline_report.human_review.false_negative_count
        ),
        human_review_false_positive_delta=(
            candidate_report.human_review.false_positive_count
            - baseline_report.human_review.false_positive_count
        ),
        failed_prediction_count_delta=(
            candidate_report.failed_prediction_count - baseline_report.failed_prediction_count
        ),
        average_total_tokens=_known_optional_delta(
            baseline_value=(
                baseline_report.token_averages.average_total_tokens
                if baseline_report.token_averages.unknown_total_token_count == 0
                else None
            ),
            candidate_value=(
                candidate_report.token_averages.average_total_tokens
                if candidate_report.token_averages.unknown_total_token_count == 0
                else None
            ),
            unknown_reason=(
                "Mean total-token comparison is incomplete because at least "
                "one prediction has unknown token usage."
            ),
        ),
        average_estimated_cost_usd=_known_optional_delta(
            baseline_value=(
                _average_known_cost(baseline_report)
                if baseline_report.unknown_pricing_count == 0
                else None
            ),
            candidate_value=(
                _average_known_cost(candidate_report)
                if candidate_report.unknown_pricing_count == 0
                else None
            ),
            unknown_reason=(
                "Mean estimated-cost comparison is incomplete because pricing "
                "is unknown for at least one invocation."
            ),
        ),
        average_latency_ms=_known_optional_delta(
            baseline_value=(
                baseline_report.latency.average_latency_ms
                if baseline_report.latency.unknown_latency_count == 0
                else None
            ),
            candidate_value=(
                candidate_report.latency.average_latency_ms
                if candidate_report.latency.unknown_latency_count == 0
                else None
            ),
            unknown_reason=(
                "Mean latency comparison is incomplete because at least one "
                "prediction has unknown latency."
            ),
        ),
        improved_case_ids=improved_case_ids,
        regressed_case_ids=regressed_case_ids,
        unchanged_case_ids=unchanged_case_ids,
        holdout_regressed_case_ids=tuple(
            case_id for case_id in regressed_case_ids if case_id in holdout_case_ids
        ),
        safety_gate_regressed_case_ids=tuple(
            case_id for case_id in regressed_case_ids if case_id in safety_gate_case_ids
        ),
        gate_evaluation=paired_gate_evaluation,
        run_status=(
            EvaluationRunStatus.INCOMPLETE
            if paired_gate_evaluation.status is TicketClassificationPairedGateStatus.INCOMPLETE
            else EvaluationRunStatus.COMPLETE
        ),
    )
    return TicketClassificationPairedComparison(
        **content.model_dump(),
        comparison_content_hash=sha256_hexdigest(content),
    )


def load_ticket_classification_paired_comparison(
    path: Path,
) -> TicketClassificationPairedComparison:
    """Load and validate a paired comparison artifact."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise TicketClassificationPairedComparisonError(
            "Paired classification comparison could not be read."
        ) from error

    try:
        return TicketClassificationPairedComparison.model_validate(payload)
    except ValidationError as error:
        raise TicketClassificationPairedComparisonError(
            "Paired classification comparison does not match the contract."
        ) from error


def write_ticket_classification_paired_comparison(
    path: Path,
    comparison: TicketClassificationPairedComparison,
) -> None:
    """Write one canonical paired comparison through an atomic replacement."""

    write_canonical_json_atomically(
        path,
        comparison.model_dump(mode="python"),
    )


def _validate_pairing(
    *,
    dataset: TicketClassificationEvaluationDataset,
    baseline_predictions: TicketClassificationPredictionSet,
    candidate_predictions: TicketClassificationPredictionSet,
) -> None:
    dataset_case_ids = tuple(case.case_id for case in dataset.cases)
    baseline_case_ids = tuple(prediction.case_id for prediction in baseline_predictions.predictions)
    candidate_case_ids = tuple(
        prediction.case_id for prediction in candidate_predictions.predictions
    )
    if baseline_case_ids != dataset_case_ids:
        raise TicketClassificationPairedComparisonError(
            "Baseline prediction order must exactly match dataset order."
        )
    if candidate_case_ids != dataset_case_ids:
        raise TicketClassificationPairedComparisonError(
            "Candidate prediction order must exactly match dataset order."
        )

    baseline_identity = _prediction_identity(baseline_predictions)
    candidate_identity = _prediction_identity(candidate_predictions)
    if baseline_identity[0] != candidate_identity[0]:
        raise TicketClassificationPairedComparisonError(
            "Paired predictions must share one prompt_id."
        )
    if baseline_identity[1] == candidate_identity[1]:
        raise TicketClassificationPairedComparisonError(
            "Paired predictions must use distinct prompt versions."
        )
    if baseline_identity[2] == candidate_identity[2]:
        raise TicketClassificationPairedComparisonError(
            "Paired predictions must use distinct prompt hashes."
        )
    if baseline_identity[3:] != candidate_identity[3:]:
        raise TicketClassificationPairedComparisonError(
            "Paired predictions must share provider and model identity."
        )


def _prediction_identity(
    predictions: TicketClassificationPredictionSet,
) -> tuple[str, int, str, str, str]:
    identities = {
        (
            prediction.provenance.prompt_id,
            prediction.provenance.prompt_version,
            prediction.provenance.prompt_content_hash,
            prediction.provenance.provider,
            prediction.provenance.model,
        )
        for prediction in predictions.predictions
    }
    if len(identities) != 1:
        raise TicketClassificationPairedComparisonError(
            "Each prediction artifact must contain one provenance identity."
        )
    return next(iter(identities))


def _partition_case_outcomes(
    *,
    baseline_report: TicketClassificationEvaluationReport,
    candidate_report: TicketClassificationEvaluationReport,
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    candidate_by_case_id = {case.case_id: case for case in candidate_report.cases}
    improved: list[str] = []
    regressed: list[str] = []
    unchanged: list[str] = []

    for baseline_case in baseline_report.cases:
        candidate_case = candidate_by_case_id[baseline_case.case_id]
        if (
            not baseline_case.structured_label_exact_match
            and candidate_case.structured_label_exact_match
        ):
            improved.append(baseline_case.case_id)
        elif (
            baseline_case.structured_label_exact_match
            and not candidate_case.structured_label_exact_match
        ):
            regressed.append(baseline_case.case_id)
        else:
            unchanged.append(baseline_case.case_id)

    return tuple(improved), tuple(regressed), tuple(unchanged)


def _evaluate_paired_release_gates(
    *,
    baseline_report: TicketClassificationEvaluationReport,
    candidate_report: TicketClassificationEvaluationReport,
    candidate_standalone_gates: TicketClassificationReleaseGateEvaluation,
    profile: TicketClassificationReleaseGateProfile,
) -> TicketClassificationPairedGateEvaluation:
    standalone_by_gate_id = {
        result.gate_id: result for result in candidate_standalone_gates.gate_results
    }
    gate_results = tuple(
        _paired_gate_result(
            gate=gate,
            baseline_report=baseline_report,
            candidate_report=candidate_report,
            standalone_result=standalone_by_gate_id[gate.gate_id],
        )
        for gate in profile.gates
    )
    blocking_failure_count = sum(
        result.blocking and result.outcome is TicketClassificationGateOutcome.FAILED
        for result in gate_results
    )
    not_applicable_count = sum(
        result.outcome is TicketClassificationGateOutcome.NOT_APPLICABLE for result in gate_results
    )
    content = {
        "profile_id": profile.profile_id,
        "profile_version": profile.profile_version,
        "candidate_report_content_hash": (candidate_report.report_content_hash),
        "gate_results": gate_results,
        "blocking_failure_count": blocking_failure_count,
        "not_applicable_count": not_applicable_count,
        "status": _aggregate_paired_gate_status(gate_results),
    }
    return TicketClassificationPairedGateEvaluation(
        **content,
        content_hash=sha256_hexdigest(content),
    )


def _paired_gate_result(
    *,
    gate: TicketClassificationReleaseGateDefinition,
    baseline_report: TicketClassificationEvaluationReport,
    candidate_report: TicketClassificationEvaluationReport,
    standalone_result: TicketClassificationReleaseGateResult,
) -> TicketClassificationReleaseGateResult:
    gate_id = gate.gate_id
    if gate_id == "classification.structured-label-non-regression":
        return _binary_gate_result(
            gate=gate,
            passed=(
                candidate_report.structured_label_exact_match.rate
                >= baseline_report.structured_label_exact_match.rate
            ),
            reason=("candidate structured-label exact-match rate is not lower than baseline"),
        )
    if gate_id == "classification.category-accuracy-non-regression":
        return _binary_gate_result(
            gate=gate,
            passed=(
                candidate_report.category_accuracy.rate >= baseline_report.category_accuracy.rate
            ),
            reason="candidate category accuracy is not lower than baseline",
        )
    if gate_id == "classification.target-metric-improvement":
        return _binary_gate_result(
            gate=gate,
            passed=(
                candidate_report.structured_label_exact_match.rate
                > baseline_report.structured_label_exact_match.rate
            ),
            reason=(
                "candidate structured-label exact-match rate is strictly greater than baseline"
            ),
        )
    if gate_id == "classification.mean-token-increase":
        return _non_increase_gate_result(
            gate=gate,
            delta=(
                candidate_report.token_averages.average_total_tokens
                - baseline_report.token_averages.average_total_tokens
            )
            if (
                baseline_report.token_averages.unknown_total_token_count == 0
                and candidate_report.token_averages.unknown_total_token_count == 0
            )
            else None,
            unknown_reason=(
                "mean total-token increase is unknown because token usage is incomplete"
            ),
        )
    if gate_id == "classification.mean-cost-increase":
        return _non_increase_gate_result(
            gate=gate,
            delta=(_average_known_cost(candidate_report) - _average_known_cost(baseline_report))
            if (
                baseline_report.unknown_pricing_count == 0
                and candidate_report.unknown_pricing_count == 0
            )
            else None,
            unknown_reason=(
                "mean estimated-cost increase is unknown because pricing is incomplete"
            ),
        )
    if gate_id == "classification.mean-latency-increase":
        return _non_increase_gate_result(
            gate=gate,
            delta=(
                candidate_report.latency.average_latency_ms
                - baseline_report.latency.average_latency_ms
            )
            if (
                baseline_report.latency.unknown_latency_count == 0
                and candidate_report.latency.unknown_latency_count == 0
            )
            else None,
            unknown_reason=(
                "mean latency increase is unknown because latency evidence is incomplete"
            ),
        )
    return standalone_result


def _binary_gate_result(
    *,
    gate: TicketClassificationReleaseGateDefinition,
    passed: bool,
    reason: str,
) -> TicketClassificationReleaseGateResult:
    return TicketClassificationReleaseGateResult(
        gate_id=gate.gate_id,
        category=gate.category,
        outcome=(
            TicketClassificationGateOutcome.PASSED
            if passed
            else TicketClassificationGateOutcome.FAILED
        ),
        blocking=gate.blocking,
        actual_value=_RATE_ONE if passed else _RATE_ZERO,
        operator=gate.operator,
        threshold_value=gate.threshold_value,
        metric_name=gate.metric_name,
        reason=reason,
    )


def _non_increase_gate_result(
    *,
    gate: TicketClassificationReleaseGateDefinition,
    delta: Decimal | None,
    unknown_reason: str,
) -> TicketClassificationReleaseGateResult:
    if delta is None:
        return TicketClassificationReleaseGateResult(
            gate_id=gate.gate_id,
            category=gate.category,
            outcome=TicketClassificationGateOutcome.NOT_APPLICABLE,
            blocking=gate.blocking,
            actual_value=None,
            operator=gate.operator,
            threshold_value=gate.threshold_value,
            metric_name=gate.metric_name,
            reason=unknown_reason,
        )

    positive_increase = max(delta, _RATE_ZERO)
    return TicketClassificationReleaseGateResult(
        gate_id=gate.gate_id,
        category=gate.category,
        outcome=(
            TicketClassificationGateOutcome.PASSED
            if positive_increase == gate.threshold_value
            else TicketClassificationGateOutcome.FAILED
        ),
        blocking=gate.blocking,
        actual_value=positive_increase,
        operator=gate.operator,
        threshold_value=gate.threshold_value,
        metric_name=gate.metric_name,
        reason=(f"{gate.metric_name} positive increase is {positive_increase}"),
    )


def _aggregate_paired_gate_status(
    gate_results: tuple[TicketClassificationReleaseGateResult, ...],
) -> TicketClassificationPairedGateStatus:
    blocking_results = tuple(result for result in gate_results if result.blocking)
    if any(result.outcome is TicketClassificationGateOutcome.FAILED for result in blocking_results):
        return TicketClassificationPairedGateStatus.FAILED
    if any(
        result.outcome is TicketClassificationGateOutcome.NOT_APPLICABLE
        for result in blocking_results
    ):
        return TicketClassificationPairedGateStatus.INCOMPLETE
    return TicketClassificationPairedGateStatus.PASSED


def _metric_delta(
    baseline_value: Decimal,
    candidate_value: Decimal,
) -> TicketClassificationMetricDelta:
    return TicketClassificationMetricDelta(
        baseline_value=baseline_value,
        candidate_value=candidate_value,
        delta=candidate_value - baseline_value,
    )


def _known_optional_delta(
    *,
    baseline_value: Decimal | None,
    candidate_value: Decimal | None,
    unknown_reason: str,
) -> TicketClassificationOptionalMetricDelta:
    if baseline_value is None or candidate_value is None:
        return TicketClassificationOptionalMetricDelta(
            baseline_value=None,
            candidate_value=None,
            delta=None,
            unknown_reason=unknown_reason,
        )
    return TicketClassificationOptionalMetricDelta(
        baseline_value=baseline_value,
        candidate_value=candidate_value,
        delta=candidate_value - baseline_value,
    )


def _prediction_coverage_rate(
    report: TicketClassificationEvaluationReport,
) -> Decimal:
    covered_count = sum(case.prediction_status in {"succeeded", "failed"} for case in report.cases)
    return (Decimal(covered_count) / Decimal(report.case_count)).quantize(Decimal("0.000001"))


def _average_known_cost(
    report: TicketClassificationEvaluationReport,
) -> Decimal:
    return (report.known_estimated_total_cost_usd / Decimal(report.case_count)).quantize(
        Decimal("0.000001")
    )
