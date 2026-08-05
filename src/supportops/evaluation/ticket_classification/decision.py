"""Safety-first prompt decision contracts for ticket classification."""

from __future__ import annotations

import json
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Literal, Self

from pydantic import (
    AwareDatetime,
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
from supportops.evaluation.ticket_classification.comparison import (
    TicketClassificationComparisonEvidenceKind,
    TicketClassificationPairedComparison,
    TicketClassificationPairedGateStatus,
)

NonEmptyString = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1),
]
Sha256Hex = Annotated[
    str,
    StringConstraints(pattern=r"^[0-9a-f]{64}$"),
]


class TicketClassificationPromptDecisionError(ValueError):
    """Raised when a prompt decision cannot be trusted."""


class TicketClassificationPromptDecisionOutcome(StrEnum):
    """Governed outcome for one candidate prompt version."""

    PROMOTED = "promoted"
    REJECTED = "rejected"
    INCONCLUSIVE = "inconclusive"


class TicketClassificationPromptDecisionReview(BaseModel):
    """Lightweight human review attached to one prompt decision."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    reviewer_id: NonEmptyString | None = None
    reviewed_at: AwareDatetime | None = None
    decision_rationale: NonEmptyString
    known_trade_offs: tuple[NonEmptyString, ...] = Field(min_length=1)
    evidence_references: tuple[NonEmptyString, ...] = Field(min_length=1)
    approved_for_runtime_adoption: bool

    @model_validator(mode="after")
    def validate_review(self) -> Self:
        if (self.reviewer_id is None) != (self.reviewed_at is None):
            raise ValueError("Reviewer identity and review timestamp must be provided together.")
        if len(self.known_trade_offs) != len(set(self.known_trade_offs)):
            raise ValueError("Known trade-offs must be unique.")
        if len(self.evidence_references) != len(set(self.evidence_references)):
            raise ValueError("Evidence references must be unique.")
        if self.approved_for_runtime_adoption and self.reviewer_id is None:
            raise ValueError("Runtime adoption approval requires an identified reviewer.")
        return self


class TicketClassificationPromptDecisionContent(BaseModel):
    """Canonical prompt decision content before hashing."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    decision_id: NonEmptyString
    decision_version: int = Field(ge=1)
    schema_version: Literal["ticket-classification-prompt-decision-v1"]

    comparison_id: NonEmptyString
    comparison_version: int = Field(ge=1)
    comparison_content_hash: Sha256Hex
    evidence_kind: TicketClassificationComparisonEvidenceKind

    baseline_prompt_id: NonEmptyString
    baseline_prompt_version: int = Field(ge=1)
    baseline_prompt_content_hash: Sha256Hex

    candidate_prompt_id: NonEmptyString
    candidate_prompt_version: int = Field(ge=1)
    candidate_prompt_content_hash: Sha256Hex

    outcome: TicketClassificationPromptDecisionOutcome
    run_status: EvaluationRunStatus
    blocking_reasons: tuple[NonEmptyString, ...]
    review: TicketClassificationPromptDecisionReview

    separate_runtime_adoption_required: Literal[True] = True

    @model_validator(mode="after")
    def validate_decision_content(self) -> Self:
        if self.baseline_prompt_id != self.candidate_prompt_id:
            raise ValueError("Baseline and candidate decisions must share one prompt_id.")
        if self.baseline_prompt_version == self.candidate_prompt_version:
            raise ValueError("Baseline and candidate prompt versions must be distinct.")
        if self.baseline_prompt_content_hash == self.candidate_prompt_content_hash:
            raise ValueError("Baseline and candidate prompt hashes must be distinct.")
        if len(self.blocking_reasons) != len(set(self.blocking_reasons)):
            raise ValueError("Blocking reasons must be unique.")

        if self.evidence_kind is TicketClassificationComparisonEvidenceKind.STATIC_FIXTURE:
            if self.outcome is not (TicketClassificationPromptDecisionOutcome.INCONCLUSIVE):
                raise ValueError(
                    "Static fixture evidence can only produce an inconclusive "
                    "runtime-adoption decision."
                )
            if self.review.approved_for_runtime_adoption:
                raise ValueError("Static fixture evidence cannot approve runtime adoption.")

        if self.outcome is TicketClassificationPromptDecisionOutcome.PROMOTED:
            if self.evidence_kind is not (TicketClassificationComparisonEvidenceKind.PROVIDER):
                raise ValueError("Promotion requires provider-backed evidence.")
            if self.blocking_reasons:
                raise ValueError("Promoted decisions cannot contain blocking reasons.")
            if not self.review.approved_for_runtime_adoption:
                raise ValueError("Promoted decisions require human runtime-adoption approval.")
            if self.run_status is not EvaluationRunStatus.COMPLETE:
                raise ValueError("Promoted decisions must have complete run status.")

        if self.outcome is TicketClassificationPromptDecisionOutcome.REJECTED:
            if not self.blocking_reasons:
                raise ValueError("Rejected decisions require at least one blocking reason.")
            if self.review.approved_for_runtime_adoption:
                raise ValueError("Rejected decisions cannot approve runtime adoption.")
            if self.run_status is not EvaluationRunStatus.COMPLETE:
                raise ValueError("Rejected decisions must have complete run status.")

        if self.outcome is TicketClassificationPromptDecisionOutcome.INCONCLUSIVE:
            if self.review.approved_for_runtime_adoption:
                raise ValueError("Inconclusive decisions cannot approve runtime adoption.")
            if self.run_status is not EvaluationRunStatus.INCOMPLETE:
                raise ValueError("Inconclusive decisions must have incomplete run status.")

        return self


class TicketClassificationPromptDecision(TicketClassificationPromptDecisionContent):
    """Immutable prompt decision artifact."""

    decision_content_hash: Sha256Hex

    @model_validator(mode="after")
    def validate_content_hash(self) -> Self:
        content = TicketClassificationPromptDecisionContent.model_validate(
            self.model_dump(
                mode="python",
                exclude={"decision_content_hash"},
            )
        )
        if self.decision_content_hash != sha256_hexdigest(content):
            raise ValueError("Prompt decision hash does not match canonical content.")
        return self


def decide_ticket_classification_prompt_candidate(
    *,
    comparison: TicketClassificationPairedComparison,
    review: TicketClassificationPromptDecisionReview,
) -> TicketClassificationPromptDecision:
    """Produce a safety-first decision without changing runtime selection."""

    blocking_reasons = _collect_blocking_reasons(comparison)

    if comparison.evidence_kind is TicketClassificationComparisonEvidenceKind.STATIC_FIXTURE:
        if review.approved_for_runtime_adoption:
            raise TicketClassificationPromptDecisionError(
                "Static fixture evidence cannot approve runtime adoption."
            )
        outcome = TicketClassificationPromptDecisionOutcome.INCONCLUSIVE
        run_status = EvaluationRunStatus.INCOMPLETE
    elif blocking_reasons:
        if review.approved_for_runtime_adoption:
            raise TicketClassificationPromptDecisionError(
                "A candidate with blocking regressions cannot be approved."
            )
        outcome = TicketClassificationPromptDecisionOutcome.REJECTED
        run_status = EvaluationRunStatus.COMPLETE
    elif (
        comparison.run_status is EvaluationRunStatus.INCOMPLETE
        or comparison.gate_evaluation.status is TicketClassificationPairedGateStatus.INCOMPLETE
    ):
        if review.approved_for_runtime_adoption:
            raise TicketClassificationPromptDecisionError(
                "Incomplete evidence cannot approve runtime adoption."
            )
        outcome = TicketClassificationPromptDecisionOutcome.INCONCLUSIVE
        run_status = EvaluationRunStatus.INCOMPLETE
    elif review.approved_for_runtime_adoption:
        outcome = TicketClassificationPromptDecisionOutcome.PROMOTED
        run_status = EvaluationRunStatus.COMPLETE
    else:
        outcome = TicketClassificationPromptDecisionOutcome.INCONCLUSIVE
        run_status = EvaluationRunStatus.INCOMPLETE

    content = TicketClassificationPromptDecisionContent(
        decision_id="ticket-classification-prompt-v2-adoption",
        decision_version=1,
        schema_version="ticket-classification-prompt-decision-v1",
        comparison_id=comparison.comparison_id,
        comparison_version=comparison.comparison_version,
        comparison_content_hash=comparison.comparison_content_hash,
        evidence_kind=comparison.evidence_kind,
        baseline_prompt_id=comparison.baseline_prompt_id,
        baseline_prompt_version=comparison.baseline_prompt_version,
        baseline_prompt_content_hash=(comparison.baseline_prompt_content_hash),
        candidate_prompt_id=comparison.candidate_prompt_id,
        candidate_prompt_version=comparison.candidate_prompt_version,
        candidate_prompt_content_hash=(comparison.candidate_prompt_content_hash),
        outcome=outcome,
        run_status=run_status,
        blocking_reasons=blocking_reasons,
        review=review,
        separate_runtime_adoption_required=True,
    )
    return TicketClassificationPromptDecision(
        **content.model_dump(),
        decision_content_hash=sha256_hexdigest(content),
    )


def load_ticket_classification_prompt_decision(
    path: Path,
) -> TicketClassificationPromptDecision:
    """Load and validate one committed prompt decision."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise TicketClassificationPromptDecisionError(
            "Ticket classification prompt decision could not be read."
        ) from error

    try:
        return TicketClassificationPromptDecision.model_validate(payload)
    except ValidationError as error:
        raise TicketClassificationPromptDecisionError(
            "Ticket classification prompt decision does not match the contract."
        ) from error


def write_ticket_classification_prompt_decision(
    path: Path,
    decision: TicketClassificationPromptDecision,
) -> None:
    """Write one canonical prompt decision through atomic replacement."""

    write_canonical_json_atomically(
        path,
        decision.model_dump(mode="python"),
    )


def _collect_blocking_reasons(
    comparison: TicketClassificationPairedComparison,
) -> tuple[str, ...]:
    reasons: list[str] = []

    if comparison.gate_evaluation.blocking_failure_count > 0:
        reasons.append("The candidate failed one or more blocking release gates.")
    if comparison.human_review_false_negative_delta > 0:
        reasons.append("Human-review false negatives increased relative to baseline.")
    if comparison.prediction_coverage.delta < 0:
        reasons.append("Prediction coverage decreased relative to baseline.")
    if comparison.failed_prediction_count_delta > 0:
        reasons.append("Failed prediction count increased relative to baseline.")
    if comparison.safety_gate_regressed_case_ids:
        reasons.append("One or more safety-gate cases regressed.")

    return tuple(reasons)
