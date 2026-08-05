from __future__ import annotations

from decimal import Decimal
from enum import StrEnum
from typing import Annotated, Self
from uuid import UUID

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    model_validator,
)

NonEmptyString = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1),
]
CaseId = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        min_length=1,
        pattern=r"^[a-z0-9]+(?:-[a-z0-9]+)*$",
    ),
]
Sha256Hex = Annotated[
    str,
    StringConstraints(pattern=r"^[0-9a-f]{64}$"),
]


class HumanApprovalDatasetSource(StrEnum):
    SYNTHETIC = "synthetic"


class ApprovalStatus(StrEnum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXPIRED = "expired"


class ApprovalDecisionEvent(StrEnum):
    APPROVE = "approve"
    REJECT = "reject"
    EXPIRE = "expire"


class SensitiveExecutionOutcome(StrEnum):
    APPLIED = "applied"
    ALREADY_RECORDED = "already_recorded"
    NOT_EXECUTED = "not_executed"


class ApprovalResumePlan(StrEnum):
    CONTINUE = "continue"
    RESUME = "resume"
    COMPLETED = "completed"
    FAILED = "failed"


class HumanApprovalEvaluationCase(BaseModel):
    """One immutable human-approval regression scenario."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    dataset_id: NonEmptyString
    dataset_version: int = Field(ge=1)
    schema_version: NonEmptyString

    case_id: CaseId
    description: NonEmptyString
    source: HumanApprovalDatasetSource
    tags: tuple[NonEmptyString, ...] = Field(min_length=1)
    risk_tags: tuple[NonEmptyString, ...] = ()

    workspace_id: UUID
    workflow_name: NonEmptyString
    workflow_version: NonEmptyString

    requires_approval: bool
    initial_approval_status: ApprovalStatus | None
    decision_event: ApprovalDecisionEvent | None
    expected_terminal_status: ApprovalStatus | None

    expected_sensitive_executed: bool
    expected_execution_status: SensitiveExecutionOutcome
    expected_resume_plan: ApprovalResumePlan

    expected_checkpoint_match: bool
    expected_grant_match: bool
    expected_retry_budget_preserved: bool
    expected_duplicate_escalation_prevented: bool
    expected_finalization: bool
    expected_error_code: NonEmptyString | None

    @model_validator(mode="after")
    def validate_case(self) -> Self:
        self._require_unique("tags", self.tags)
        self._require_unique("risk_tags", self.risk_tags)

        if not self.requires_approval and (
            self.initial_approval_status is not None
            or self.decision_event is not None
            or self.expected_terminal_status is not None
        ):
            raise ValueError("approval-independent cases cannot declare approval state")

        if (
            not self.expected_sensitive_executed
            and self.expected_execution_status is not SensitiveExecutionOutcome.NOT_EXECUTED
        ):
            raise ValueError("non-executed cases must use not_executed status")

        if (
            self.expected_sensitive_executed
            and self.expected_execution_status is SensitiveExecutionOutcome.NOT_EXECUTED
        ):
            raise ValueError("executed cases cannot use not_executed status")

        if (
            self.expected_terminal_status in {ApprovalStatus.REJECTED, ApprovalStatus.EXPIRED}
            and self.expected_sensitive_executed
        ):
            raise ValueError("rejected or expired approval cannot execute")

        if self.expected_error_code is not None and self.expected_finalization:
            raise ValueError("failed cases cannot declare successful finalization")

        return self

    @staticmethod
    def _require_unique(name: str, values: tuple[object, ...]) -> None:
        if len(values) != len(set(values)):
            raise ValueError(f"{name} must contain unique values")


class HumanApprovalEvaluationDataset(BaseModel):
    """Validated human-approval dataset and deterministic hash."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    dataset_id: NonEmptyString
    dataset_version: int = Field(ge=1)
    schema_version: NonEmptyString
    source: HumanApprovalDatasetSource
    workflow_name: NonEmptyString
    workflow_version: NonEmptyString
    cases: tuple[HumanApprovalEvaluationCase, ...] = Field(min_length=1)
    content_hash: Sha256Hex

    @model_validator(mode="after")
    def validate_dataset(self) -> Self:
        case_ids = tuple(case.case_id for case in self.cases)
        if len(case_ids) != len(set(case_ids)):
            raise ValueError("dataset contains duplicate case IDs")

        for case in self.cases:
            if case.dataset_id != self.dataset_id:
                raise ValueError("case dataset_id does not match dataset")
            if case.dataset_version != self.dataset_version:
                raise ValueError("case dataset_version does not match dataset")
            if case.schema_version != self.schema_version:
                raise ValueError("case schema_version does not match dataset")
            if case.source is not self.source:
                raise ValueError("case source does not match dataset")
            if case.workflow_name != self.workflow_name:
                raise ValueError("case workflow_name does not match dataset")
            if case.workflow_version != self.workflow_version:
                raise ValueError("case workflow_version does not match dataset")

        return self


class HumanApprovalPredictionPayload(BaseModel):
    """Typed static outcome for one approval workflow scenario."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    requires_approval: bool
    terminal_status: ApprovalStatus | None
    sensitive_executed: bool
    execution_status: SensitiveExecutionOutcome
    resume_plan: ApprovalResumePlan

    checkpoint_match: bool
    grant_match: bool
    retry_budget_preserved: bool
    duplicate_escalation_prevented: bool
    decision_idempotent: bool
    unauthorized_execution_detected: bool
    finalized: bool

    @model_validator(mode="after")
    def validate_execution(self) -> Self:
        if (
            not self.sensitive_executed
            and self.execution_status is not SensitiveExecutionOutcome.NOT_EXECUTED
        ):
            raise ValueError("non-executed predictions must use not_executed")

        if (
            self.sensitive_executed
            and self.execution_status is SensitiveExecutionOutcome.NOT_EXECUTED
        ):
            raise ValueError("executed predictions cannot use not_executed")

        return self


class CountRateMetric(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    numerator_count: int = Field(ge=0)
    denominator_count: int = Field(ge=0)
    rate: Decimal | None


class MeanMetric(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    total: Decimal
    known_count: int = Field(ge=0)
    unknown_count: int = Field(ge=0)
    average: Decimal | None


class HumanApprovalCaseResult(BaseModel):
    """Case-level approval evaluation evidence."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    case_id: CaseId
    prediction_present: bool
    expected_outcome_matched: bool

    approval_required_correct: bool | None
    unauthorized_execution_detected: bool
    approved_execution_correct: bool | None
    rejected_non_execution_correct: bool | None
    expired_non_execution_correct: bool | None
    decision_idempotency_correct: bool | None
    resume_correct: bool | None
    sensitive_action_idempotency_correct: bool | None
    checkpoint_match_correct: bool | None
    grant_match_correct: bool | None
    retry_budget_preserved: bool | None
    duplicate_escalation_prevented: bool | None
    finalization_correct: bool | None

    error_code: NonEmptyString | None = None


class HumanApprovalEvaluationReport(BaseModel):
    """Deterministic approval evaluation report."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    dataset_id: NonEmptyString
    dataset_version: int = Field(ge=1)
    schema_version: NonEmptyString
    workflow_name: NonEmptyString
    workflow_version: NonEmptyString

    dataset_hash: Sha256Hex
    prediction_hash: Sha256Hex
    case_count: int = Field(ge=1)

    expected_outcome_accuracy: CountRateMetric
    approval_required_accuracy: CountRateMetric
    unauthorized_sensitive_execution_rate: CountRateMetric
    approved_execution_success_rate: CountRateMetric
    rejected_non_execution_rate: CountRateMetric
    expired_non_execution_rate: CountRateMetric
    approval_decision_idempotency_rate: CountRateMetric
    resume_success_rate: CountRateMetric
    sensitive_action_idempotency_rate: CountRateMetric
    checkpoint_approval_match_rate: CountRateMetric
    grant_match_rate: CountRateMetric
    retry_budget_preservation_rate: CountRateMetric
    duplicate_escalation_prevention_rate: CountRateMetric
    successful_finalization_rate: CountRateMetric

    average_latency_ms: MeanMetric
    average_total_tokens: MeanMetric
    estimated_cost_usd: MeanMetric

    case_results: tuple[HumanApprovalCaseResult, ...]
    report_content_hash: Sha256Hex
