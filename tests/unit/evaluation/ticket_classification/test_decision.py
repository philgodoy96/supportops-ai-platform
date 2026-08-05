"""Unit tests for ticket-classification prompt decisions."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

import pytest
from pydantic import ValidationError

from supportops.evaluation.contracts.manifest import EvaluationRunStatus
from supportops.evaluation.ticket_classification.comparison import (
    TicketClassificationComparisonEvidenceKind,
    TicketClassificationPairedComparison,
    TicketClassificationPairedGateStatus,
    load_ticket_classification_paired_comparison,
)
from supportops.evaluation.ticket_classification.decision import (
    TicketClassificationPromptDecisionError,
    TicketClassificationPromptDecisionOutcome,
    TicketClassificationPromptDecisionReview,
    decide_ticket_classification_prompt_candidate,
    load_ticket_classification_prompt_decision,
    write_ticket_classification_prompt_decision,
)

PROJECT_ROOT = Path(__file__).resolve().parents[4]

COMPARISON_PATH = (
    PROJECT_ROOT
    / "evals"
    / "ticket-classification"
    / "comparisons"
    / "ticket-classification-prompt-v1-v2.static.json"
)
DECISION_PATH = (
    PROJECT_ROOT
    / "evals"
    / "ticket-classification"
    / "decisions"
    / "ticket-classification-prompt-v2-decision.static.json"
)

DECISION_HASH = "69df2e12af3700261e5b3dcbbb2ecc5491807786db6ce9ee8f7384b146631d40"


def _static_review() -> TicketClassificationPromptDecisionReview:
    return TicketClassificationPromptDecisionReview(
        decision_rationale=(
            "Static fixtures demonstrate deterministic comparison and "
            "decision behavior but do not provide provider-backed quality "
            "or efficiency evidence for runtime adoption."
        ),
        known_trade_offs=(
            "Static predictions are synthetic contract evidence.",
            "Mean token and cost comparisons remain incomplete.",
            ("The runtime remains pinned to ticket-classification prompt version 1."),
        ),
        evidence_references=(
            ("evals/ticket-classification/analyses/classification-prompt-v1-failure-analysis.json"),
            (
                "evals/ticket-classification/predictions/"
                "ticket-classification-eval-v1.prompt-v1.static.jsonl"
            ),
            (
                "evals/ticket-classification/predictions/"
                "ticket-classification-eval-v1.prompt-v2.static.jsonl"
            ),
            (
                "evals/ticket-classification/comparisons/"
                "ticket-classification-prompt-v1-v2.static.json"
            ),
        ),
        approved_for_runtime_adoption=False,
    )


def _approved_provider_review() -> TicketClassificationPromptDecisionReview:
    return TicketClassificationPromptDecisionReview(
        reviewer_id="staff-ai-reviewer",
        reviewed_at=datetime(
            2026,
            8,
            5,
            18,
            0,
            tzinfo=UTC,
        ),
        decision_rationale=(
            "Provider-backed evidence is complete and all safety, "
            "reliability, quality, and efficiency gates pass."
        ),
        known_trade_offs=("Runtime adoption remains a separate repository change.",),
        evidence_references=(
            "artifacts/evaluation/ticket-classification/provider-comparison.json",
        ),
        approved_for_runtime_adoption=True,
    )


def _unapproved_provider_review() -> TicketClassificationPromptDecisionReview:
    return TicketClassificationPromptDecisionReview(
        reviewer_id="staff-ai-reviewer",
        reviewed_at=datetime(
            2026,
            8,
            5,
            18,
            0,
            tzinfo=UTC,
        ),
        decision_rationale=("Provider evidence was reviewed without approving runtime adoption."),
        known_trade_offs=("Runtime adoption remains pending.",),
        evidence_references=(
            "artifacts/evaluation/ticket-classification/provider-comparison.json",
        ),
        approved_for_runtime_adoption=False,
    )


def _load_static_comparison() -> TicketClassificationPairedComparison:
    return load_ticket_classification_paired_comparison(COMPARISON_PATH)


def _provider_ready_comparison() -> TicketClassificationPairedComparison:
    comparison = _load_static_comparison()
    ready_gates = comparison.gate_evaluation.model_copy(
        update={
            "blocking_failure_count": 0,
            "not_applicable_count": 0,
            "status": TicketClassificationPairedGateStatus.PASSED,
        }
    )
    return comparison.model_copy(
        update={
            "evidence_kind": (TicketClassificationComparisonEvidenceKind.PROVIDER),
            "gate_evaluation": ready_gates,
            "run_status": EvaluationRunStatus.COMPLETE,
        }
    )


def test_committed_static_decision_matches_builder() -> None:
    comparison = _load_static_comparison()
    committed = load_ticket_classification_prompt_decision(DECISION_PATH)
    rebuilt = decide_ticket_classification_prompt_candidate(
        comparison=comparison,
        review=_static_review(),
    )

    assert committed == rebuilt
    assert committed.decision_content_hash == DECISION_HASH


def test_static_evidence_is_always_inconclusive() -> None:
    decision = decide_ticket_classification_prompt_candidate(
        comparison=_load_static_comparison(),
        review=_static_review(),
    )

    assert decision.outcome is (TicketClassificationPromptDecisionOutcome.INCONCLUSIVE)
    assert decision.run_status is EvaluationRunStatus.INCOMPLETE
    assert decision.blocking_reasons == ()
    assert decision.review.approved_for_runtime_adoption is False
    assert decision.separate_runtime_adoption_required is True


def test_static_evidence_cannot_approve_runtime_adoption() -> None:
    with pytest.raises(
        TicketClassificationPromptDecisionError,
        match="Static fixture evidence cannot approve",
    ):
        decide_ticket_classification_prompt_candidate(
            comparison=_load_static_comparison(),
            review=_approved_provider_review(),
        )


def test_complete_provider_evidence_promotes_only_with_approval() -> None:
    decision = decide_ticket_classification_prompt_candidate(
        comparison=_provider_ready_comparison(),
        review=_approved_provider_review(),
    )

    assert decision.outcome is (TicketClassificationPromptDecisionOutcome.PROMOTED)
    assert decision.run_status is EvaluationRunStatus.COMPLETE
    assert decision.blocking_reasons == ()
    assert decision.review.approved_for_runtime_adoption is True


def test_complete_provider_evidence_without_approval_is_inconclusive() -> None:
    decision = decide_ticket_classification_prompt_candidate(
        comparison=_provider_ready_comparison(),
        review=_unapproved_provider_review(),
    )

    assert decision.outcome is (TicketClassificationPromptDecisionOutcome.INCONCLUSIVE)
    assert decision.run_status is EvaluationRunStatus.INCOMPLETE


def test_incomplete_provider_evidence_is_inconclusive() -> None:
    comparison = _load_static_comparison().model_copy(
        update={
            "evidence_kind": (TicketClassificationComparisonEvidenceKind.PROVIDER),
        }
    )

    decision = decide_ticket_classification_prompt_candidate(
        comparison=comparison,
        review=_unapproved_provider_review(),
    )

    assert decision.outcome is (TicketClassificationPromptDecisionOutcome.INCONCLUSIVE)
    assert decision.run_status is EvaluationRunStatus.INCOMPLETE


def test_blocking_provider_regressions_are_rejected() -> None:
    comparison = _provider_ready_comparison()
    failed_gates = comparison.gate_evaluation.model_copy(
        update={
            "blocking_failure_count": 1,
            "status": TicketClassificationPairedGateStatus.FAILED,
        }
    )
    blocking_comparison = comparison.model_copy(
        update={
            "gate_evaluation": failed_gates,
            "human_review_false_negative_delta": 1,
            "prediction_coverage": (
                comparison.prediction_coverage.model_copy(
                    update={
                        "candidate_value": Decimal("0.958333"),
                        "delta": Decimal("-0.041667"),
                    }
                )
            ),
            "failed_prediction_count_delta": 1,
            "safety_gate_regressed_case_ids": ("security-exposed-api-key-012",),
        }
    )

    decision = decide_ticket_classification_prompt_candidate(
        comparison=blocking_comparison,
        review=_unapproved_provider_review(),
    )

    assert decision.outcome is (TicketClassificationPromptDecisionOutcome.REJECTED)
    assert decision.run_status is EvaluationRunStatus.COMPLETE
    assert decision.blocking_reasons == (
        "The candidate failed one or more blocking release gates.",
        ("Human-review false negatives increased relative to baseline."),
        "Prediction coverage decreased relative to baseline.",
        "Failed prediction count increased relative to baseline.",
        "One or more safety-gate cases regressed.",
    )


def test_blocking_regressions_cannot_be_approved() -> None:
    comparison = _provider_ready_comparison().model_copy(
        update={
            "human_review_false_negative_delta": 1,
        }
    )

    with pytest.raises(
        TicketClassificationPromptDecisionError,
        match="blocking regressions cannot be approved",
    ):
        decide_ticket_classification_prompt_candidate(
            comparison=comparison,
            review=_approved_provider_review(),
        )


def test_review_requires_reviewer_and_timestamp_together() -> None:
    with pytest.raises(
        ValidationError,
        match="must be provided together",
    ):
        TicketClassificationPromptDecisionReview(
            reviewer_id="reviewer-without-time",
            reviewed_at=None,
            decision_rationale="The review is incomplete.",
            known_trade_offs=("Runtime adoption is pending.",),
            evidence_references=("comparison.json",),
            approved_for_runtime_adoption=False,
        )


def test_runtime_approval_requires_identified_reviewer() -> None:
    with pytest.raises(
        ValidationError,
        match="requires an identified reviewer",
    ):
        TicketClassificationPromptDecisionReview(
            decision_rationale="Approval without ownership is invalid.",
            known_trade_offs=("Runtime adoption is separate.",),
            evidence_references=("comparison.json",),
            approved_for_runtime_adoption=True,
        )


def test_atomic_writer_round_trips_decision(
    tmp_path: Path,
) -> None:
    decision = decide_ticket_classification_prompt_candidate(
        comparison=_load_static_comparison(),
        review=_static_review(),
    )
    output_path = tmp_path / "decision.json"

    write_ticket_classification_prompt_decision(
        output_path,
        decision,
    )

    assert load_ticket_classification_prompt_decision(output_path) == decision
    assert not tuple(tmp_path.glob("*.tmp"))


def test_rejects_tampered_decision_hash(
    tmp_path: Path,
) -> None:
    payload = json.loads(DECISION_PATH.read_text(encoding="utf-8"))
    payload["decision_content_hash"] = "0" * 64
    tampered_path = tmp_path / "tampered-decision.json"
    tampered_path.write_text(
        json.dumps(payload),
        encoding="utf-8",
    )

    with pytest.raises(
        TicketClassificationPromptDecisionError,
        match="does not match the contract",
    ):
        load_ticket_classification_prompt_decision(tampered_path)


def test_decision_does_not_change_runtime_prompt_selection() -> None:
    from supportops.ai.prompts.ticket_classification_v1 import (
        TICKET_CLASSIFICATION_PROMPT_VERSION,
    )

    decision = load_ticket_classification_prompt_decision(DECISION_PATH)

    assert TICKET_CLASSIFICATION_PROMPT_VERSION == 1
    assert decision.candidate_prompt_version == 2
    assert decision.separate_runtime_adoption_required is True
