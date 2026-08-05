"""Unit tests for domain regression release-gate profiles."""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path

from supportops.evaluation.contracts.predictions import EvaluationPredictionStatus
from supportops.evaluation.controlled_support.dataset import load_controlled_support_dataset
from supportops.evaluation.controlled_support.evaluator import (
    evaluate_controlled_support_predictions,
)
from supportops.evaluation.controlled_support.models import (
    ControlledSupportEvaluationReport,
)
from supportops.evaluation.controlled_support.models import (
    CountRateMetric as ControlledCountRateMetric,
)
from supportops.evaluation.controlled_support.predictions import (
    load_controlled_support_predictions,
)
from supportops.evaluation.human_approval.dataset import load_human_approval_dataset
from supportops.evaluation.human_approval.evaluator import evaluate_human_approval_predictions
from supportops.evaluation.human_approval.models import (
    CountRateMetric as ApprovalCountRateMetric,
)
from supportops.evaluation.human_approval.models import (
    HumanApprovalEvaluationReport,
)
from supportops.evaluation.human_approval.predictions import load_human_approval_predictions
from supportops.evaluation.regression.gates import (
    CONTROLLED_SUPPORT_RELEASE_GATE_PROFILE_ID,
    CONTROLLED_SUPPORT_RELEASE_GATE_PROFILE_VERSION,
    HUMAN_APPROVAL_RELEASE_GATE_PROFILE_ID,
    HUMAN_APPROVAL_RELEASE_GATE_PROFILE_VERSION,
    SEMANTIC_RETRIEVAL_RELEASE_GATE_PROFILE_ID,
    SEMANTIC_RETRIEVAL_RELEASE_GATE_PROFILE_VERSION,
    evaluate_controlled_support_release_gates,
    evaluate_human_approval_release_gates,
    evaluate_semantic_retrieval_release_gates,
)
from supportops.evaluation.regression.models import (
    RegressionAggregateStatus,
    RegressionDomainProfileResult,
    RegressionGateOutcome,
    RegressionGateResult,
)
from supportops.evaluation.semantic_retrieval.dataset import load_semantic_retrieval_dataset
from supportops.evaluation.semantic_retrieval.evaluator import (
    evaluate_semantic_retrieval_predictions,
)
from supportops.evaluation.semantic_retrieval.models import (
    CountRateMetric as RetrievalCountRateMetric,
)
from supportops.evaluation.semantic_retrieval.models import (
    SemanticRetrievalEvaluationReport,
)
from supportops.evaluation.semantic_retrieval.predictions import (
    load_semantic_retrieval_predictions,
)

PROJECT_ROOT = Path(__file__).resolve().parents[4]

SEMANTIC_DATASET = (
    PROJECT_ROOT / "evals" / "semantic-retrieval" / "datasets" / "semantic-retrieval-eval-v1.jsonl"
)
SEMANTIC_PREDICTIONS = (
    PROJECT_ROOT
    / "evals"
    / "semantic-retrieval"
    / "predictions"
    / "semantic-retrieval-eval-v1.static.jsonl"
)
CONTROLLED_DATASET = (
    PROJECT_ROOT / "evals" / "controlled-support" / "datasets" / "controlled-support-eval-v1.jsonl"
)
CONTROLLED_PREDICTIONS = (
    PROJECT_ROOT
    / "evals"
    / "controlled-support"
    / "predictions"
    / "controlled-support-eval-v1.static.jsonl"
)
APPROVAL_DATASET = (
    PROJECT_ROOT / "evals" / "human-approval" / "datasets" / "human-approval-eval-v1.jsonl"
)
APPROVAL_PREDICTIONS = (
    PROJECT_ROOT
    / "evals"
    / "human-approval"
    / "predictions"
    / "human-approval-eval-v1.static.jsonl"
)


def _semantic_report() -> SemanticRetrievalEvaluationReport:
    dataset = load_semantic_retrieval_dataset(SEMANTIC_DATASET)
    predictions, prediction_hash = load_semantic_retrieval_predictions(SEMANTIC_PREDICTIONS)
    return evaluate_semantic_retrieval_predictions(
        dataset=dataset,
        predictions=predictions,
        prediction_hash=prediction_hash,
    )


def _controlled_report() -> ControlledSupportEvaluationReport:
    dataset = load_controlled_support_dataset(CONTROLLED_DATASET)
    predictions, prediction_hash = load_controlled_support_predictions(CONTROLLED_PREDICTIONS)
    return evaluate_controlled_support_predictions(
        dataset=dataset,
        predictions=predictions,
        prediction_hash=prediction_hash,
    )


def _approval_report() -> HumanApprovalEvaluationReport:
    dataset = load_human_approval_dataset(APPROVAL_DATASET)
    predictions, prediction_hash = load_human_approval_predictions(APPROVAL_PREDICTIONS)
    return evaluate_human_approval_predictions(
        dataset=dataset,
        predictions=predictions,
        prediction_hash=prediction_hash,
    )


def _gate_by_id(profile: RegressionDomainProfileResult, gate_id: str) -> RegressionGateResult:
    for result in profile.gate_results:
        if result.gate_id == gate_id:
            return result
    raise AssertionError(f"missing gate: {gate_id}")


def test_semantic_retrieval_profile_identity_and_order() -> None:
    profile = evaluate_semantic_retrieval_release_gates(_semantic_report())

    assert profile.profile_id == SEMANTIC_RETRIEVAL_RELEASE_GATE_PROFILE_ID
    assert profile.profile_version == SEMANTIC_RETRIEVAL_RELEASE_GATE_PROFILE_VERSION
    assert tuple(result.gate_id for result in profile.gate_results) == (
        "retrieval.workspace-isolation",
        "retrieval.no-result-accuracy",
        "retrieval.citation-resolution",
        "retrieval.prediction-coverage",
        "retrieval.ranking-non-regression",
        "retrieval.recall-non-regression",
        "retrieval.mean-latency-increase",
        "retrieval.mean-token-increase",
        "retrieval.mean-cost-increase",
    )


def test_controlled_support_profile_identity_and_order() -> None:
    profile = evaluate_controlled_support_release_gates(_controlled_report())

    assert profile.profile_id == CONTROLLED_SUPPORT_RELEASE_GATE_PROFILE_ID
    assert profile.profile_version == CONTROLLED_SUPPORT_RELEASE_GATE_PROFILE_VERSION
    assert tuple(result.gate_id for result in profile.gate_results) == (
        "controlled-support.forbidden-tool-execution",
        "controlled-support.repeated-tool-acceptance",
        "controlled-support.citation-validity",
        "controlled-support.workspace-isolation",
        "controlled-support.expected-outcome-accuracy",
        "controlled-support.prediction-coverage",
        "controlled-support.recommended-action-non-regression",
        "controlled-support.grounded-abstention-non-regression",
        "controlled-support.mean-tool-call-increase",
        "controlled-support.mean-llm-invocation-increase",
        "controlled-support.mean-token-increase",
        "controlled-support.mean-cost-increase",
    )


def test_human_approval_profile_identity_and_order() -> None:
    profile = evaluate_human_approval_release_gates(_approval_report())

    assert profile.profile_id == HUMAN_APPROVAL_RELEASE_GATE_PROFILE_ID
    assert profile.profile_version == HUMAN_APPROVAL_RELEASE_GATE_PROFILE_VERSION
    assert tuple(result.gate_id for result in profile.gate_results) == (
        "human-approval.unauthorized-sensitive-execution",
        "human-approval.rejected-non-execution",
        "human-approval.expired-non-execution",
        "human-approval.checkpoint-match",
        "human-approval.grant-match",
        "human-approval.approval-decision-idempotency",
        "human-approval.sensitive-action-idempotency",
        "human-approval.retry-budget-preservation",
        "human-approval.duplicate-escalation-prevention",
        "human-approval.prediction-coverage",
        "human-approval.approved-execution-non-regression",
        "human-approval.resume-success-non-regression",
        "human-approval.mean-latency-increase",
        "human-approval.mean-token-increase",
        "human-approval.mean-cost-increase",
    )


def test_standalone_fixture_profiles_are_incomplete() -> None:
    semantic = evaluate_semantic_retrieval_release_gates(_semantic_report())
    controlled = evaluate_controlled_support_release_gates(_controlled_report())
    approval = evaluate_human_approval_release_gates(_approval_report())

    assert semantic.status is RegressionAggregateStatus.INCOMPLETE
    assert controlled.status is RegressionAggregateStatus.INCOMPLETE
    assert approval.status is RegressionAggregateStatus.INCOMPLETE

    for profile in (semantic, controlled, approval):
        assert profile.blocking_failure_count == 0
        assert profile.not_applicable_count > 0
        for result in profile.gate_results:
            if result.outcome is RegressionGateOutcome.NOT_APPLICABLE:
                assert result.blocking is True
                if "non-regression" in result.gate_id or "increase" in result.gate_id:
                    assert result.reason == "Paired baseline evidence is required."


def test_zero_denominator_rate_gate_is_not_applicable() -> None:
    report = _semantic_report().model_copy(
        update={
            "workspace_isolation_rate": RetrievalCountRateMetric(
                numerator_count=0,
                denominator_count=0,
                rate=None,
            ),
        }
    )
    profile = evaluate_semantic_retrieval_release_gates(report)
    gate = _gate_by_id(profile, "retrieval.workspace-isolation")

    assert gate.outcome is RegressionGateOutcome.NOT_APPLICABLE
    assert gate.blocking is True
    assert gate.actual_value is None
    assert profile.status is RegressionAggregateStatus.INCOMPLETE


def test_coverage_counts_failed_predictions_as_present() -> None:
    dataset = load_controlled_support_dataset(CONTROLLED_DATASET)
    predictions, prediction_hash = load_controlled_support_predictions(CONTROLLED_PREDICTIONS)
    assert any(prediction.status is EvaluationPredictionStatus.FAILED for prediction in predictions)

    report = evaluate_controlled_support_predictions(
        dataset=dataset,
        predictions=predictions,
        prediction_hash=prediction_hash,
    )
    profile = evaluate_controlled_support_release_gates(report)
    coverage = _gate_by_id(profile, "controlled-support.prediction-coverage")

    assert all(case.prediction_present for case in report.case_results)
    assert coverage.outcome is RegressionGateOutcome.PASSED
    assert coverage.actual_value == Decimal("1.000000")


def test_missing_predictions_fail_coverage() -> None:
    dataset = load_semantic_retrieval_dataset(SEMANTIC_DATASET)
    predictions, prediction_hash = load_semantic_retrieval_predictions(SEMANTIC_PREDICTIONS)
    report = evaluate_semantic_retrieval_predictions(
        dataset=dataset,
        predictions=predictions[:-1],
        prediction_hash=prediction_hash,
    )
    profile = evaluate_semantic_retrieval_release_gates(report)
    coverage = _gate_by_id(profile, "retrieval.prediction-coverage")

    assert coverage.outcome is RegressionGateOutcome.FAILED
    assert coverage.actual_value == Decimal("0.900000")
    assert profile.status is RegressionAggregateStatus.FAILED


def test_retrieval_workspace_isolation_failure() -> None:
    report = _semantic_report().model_copy(
        update={
            "workspace_isolation_rate": RetrievalCountRateMetric(
                numerator_count=0,
                denominator_count=1,
                rate=Decimal("0.000000"),
            ),
        }
    )
    profile = evaluate_semantic_retrieval_release_gates(report)
    gate = _gate_by_id(profile, "retrieval.workspace-isolation")

    assert gate.outcome is RegressionGateOutcome.FAILED
    assert gate.blocking is True
    assert profile.blocking_failure_count == 1
    assert profile.status is RegressionAggregateStatus.FAILED


def test_controlled_support_forbidden_execution_failure() -> None:
    report = _controlled_report().model_copy(
        update={
            "forbidden_tool_call_rate": ControlledCountRateMetric(
                numerator_count=1,
                denominator_count=1,
                rate=Decimal("1.000000"),
            ),
        }
    )
    profile = evaluate_controlled_support_release_gates(report)
    gate = _gate_by_id(profile, "controlled-support.forbidden-tool-execution")

    assert gate.outcome is RegressionGateOutcome.FAILED
    assert gate.blocking is True
    assert profile.blocking_failure_count == 1
    assert profile.status is RegressionAggregateStatus.FAILED


def test_controlled_support_repeated_acceptance_failure() -> None:
    report = _controlled_report().model_copy(
        update={
            "repeated_tool_call_rate": ControlledCountRateMetric(
                numerator_count=1,
                denominator_count=1,
                rate=Decimal("1.000000"),
            ),
        }
    )
    profile = evaluate_controlled_support_release_gates(report)
    gate = _gate_by_id(profile, "controlled-support.repeated-tool-acceptance")

    assert gate.outcome is RegressionGateOutcome.FAILED
    assert gate.blocking is True
    assert profile.status is RegressionAggregateStatus.FAILED


def test_controlled_support_invalid_citation_failure() -> None:
    report = _controlled_report().model_copy(
        update={
            "citation_validity_rate": ControlledCountRateMetric(
                numerator_count=0,
                denominator_count=1,
                rate=Decimal("0.000000"),
            ),
        }
    )
    profile = evaluate_controlled_support_release_gates(report)
    gate = _gate_by_id(profile, "controlled-support.citation-validity")

    assert gate.outcome is RegressionGateOutcome.FAILED
    assert gate.blocking is True


def test_approval_unauthorized_execution_failure() -> None:
    report = _approval_report().model_copy(
        update={
            "unauthorized_sensitive_execution_rate": ApprovalCountRateMetric(
                numerator_count=1,
                denominator_count=1,
                rate=Decimal("1.000000"),
            ),
        }
    )
    profile = evaluate_human_approval_release_gates(report)
    gate = _gate_by_id(profile, "human-approval.unauthorized-sensitive-execution")

    assert gate.outcome is RegressionGateOutcome.FAILED
    assert gate.blocking is True
    assert profile.status is RegressionAggregateStatus.FAILED


def test_approval_rejected_and_expired_non_execution_failures() -> None:
    report = _approval_report().model_copy(
        update={
            "rejected_non_execution_rate": ApprovalCountRateMetric(
                numerator_count=0,
                denominator_count=1,
                rate=Decimal("0.000000"),
            ),
            "expired_non_execution_rate": ApprovalCountRateMetric(
                numerator_count=0,
                denominator_count=1,
                rate=Decimal("0.000000"),
            ),
        }
    )
    profile = evaluate_human_approval_release_gates(report)
    rejected = _gate_by_id(profile, "human-approval.rejected-non-execution")
    expired = _gate_by_id(profile, "human-approval.expired-non-execution")

    assert rejected.outcome is RegressionGateOutcome.FAILED
    assert rejected.blocking is True
    assert expired.outcome is RegressionGateOutcome.FAILED
    assert expired.blocking is True
    assert profile.status is RegressionAggregateStatus.FAILED


def test_approval_checkpoint_and_grant_mismatch_failures() -> None:
    report = _approval_report().model_copy(
        update={
            "checkpoint_approval_match_rate": ApprovalCountRateMetric(
                numerator_count=0,
                denominator_count=1,
                rate=Decimal("0.000000"),
            ),
            "grant_match_rate": ApprovalCountRateMetric(
                numerator_count=0,
                denominator_count=1,
                rate=Decimal("0.000000"),
            ),
        }
    )
    profile = evaluate_human_approval_release_gates(report)
    checkpoint = _gate_by_id(profile, "human-approval.checkpoint-match")
    grant = _gate_by_id(profile, "human-approval.grant-match")

    assert checkpoint.outcome is RegressionGateOutcome.FAILED
    assert checkpoint.blocking is True
    assert grant.outcome is RegressionGateOutcome.FAILED
    assert grant.blocking is True
    assert profile.status is RegressionAggregateStatus.FAILED


def test_gate_profile_hashes_are_deterministic() -> None:
    first = evaluate_semantic_retrieval_release_gates(_semantic_report())
    second = evaluate_semantic_retrieval_release_gates(_semantic_report())

    assert first.content_hash == second.content_hash
    assert first.model_dump() == second.model_dump()
