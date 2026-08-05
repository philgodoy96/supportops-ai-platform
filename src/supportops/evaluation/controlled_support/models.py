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


class ControlledSupportDatasetSource(StrEnum):
    """Supported provenance for controlled-support evaluation data."""

    SYNTHETIC = "synthetic"


class ControlledSupportRecommendedAction(StrEnum):
    """Application-owned recommendation actions."""

    RESPOND = "respond"
    REQUEST_MORE_INFORMATION = "request_more_information"
    RECOMMEND_ESCALATION = "recommend_escalation"


class ControlledSupportEvaluationCase(BaseModel):
    """One immutable controlled-support regression scenario."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    dataset_id: NonEmptyString
    dataset_version: int = Field(ge=1)
    schema_version: NonEmptyString

    case_id: CaseId
    description: NonEmptyString
    source: ControlledSupportDatasetSource
    tags: tuple[NonEmptyString, ...] = Field(min_length=1)
    risk_tags: tuple[NonEmptyString, ...] = ()

    workspace_id: UUID
    workflow_name: NonEmptyString
    workflow_version: NonEmptyString

    required_tool_calls: tuple[NonEmptyString, ...] = ()
    forbidden_tool_calls: tuple[NonEmptyString, ...] = ()
    expected_tool_sequence: tuple[NonEmptyString, ...] = ()
    allow_repeated_tool_calls: bool

    expected_step_limit_violation: bool
    expected_recommended_action: ControlledSupportRecommendedAction | None
    expected_requires_human_review: bool | None
    expected_evidence_sufficient: bool | None
    expected_citation_chunk_ids: tuple[UUID, ...] = ()
    expected_completion: bool
    expected_error_code: NonEmptyString | None

    @model_validator(mode="after")
    def validate_case(self) -> Self:
        self._require_unique("tags", self.tags)
        self._require_unique("risk_tags", self.risk_tags)
        self._require_unique(
            "required_tool_calls",
            self.required_tool_calls,
        )
        self._require_unique(
            "forbidden_tool_calls",
            self.forbidden_tool_calls,
        )
        self._require_unique(
            "expected_citation_chunk_ids",
            self.expected_citation_chunk_ids,
        )

        overlapping_tools = set(self.required_tool_calls) & set(self.forbidden_tool_calls)
        if overlapping_tools:
            raise ValueError("required and forbidden tool calls must not overlap")

        if self.expected_completion and self.expected_error_code is not None:
            raise ValueError("completed cases cannot declare an expected error")

        if not self.expected_completion and self.expected_error_code is None:
            raise ValueError("non-completing cases must declare an expected error")

        return self

    @staticmethod
    def _require_unique(name: str, values: tuple[object, ...]) -> None:
        if len(values) != len(set(values)):
            raise ValueError(f"{name} must contain unique values")


class ControlledSupportEvaluationDataset(BaseModel):
    """Validated controlled-support dataset and deterministic hash."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    dataset_id: NonEmptyString
    dataset_version: int = Field(ge=1)
    schema_version: NonEmptyString
    source: ControlledSupportDatasetSource
    workflow_name: NonEmptyString
    workflow_version: NonEmptyString
    cases: tuple[ControlledSupportEvaluationCase, ...] = Field(min_length=1)
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


class ControlledSupportToolCallPrediction(BaseModel):
    """One proposed or accepted tool call in an evaluation trace."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    tool_name: NonEmptyString
    accepted: bool
    rejection_code: NonEmptyString | None = None

    @model_validator(mode="after")
    def validate_rejection(self) -> Self:
        if self.accepted and self.rejection_code is not None:
            raise ValueError("accepted tool calls cannot declare a rejection code")

        if not self.accepted and self.rejection_code is None:
            raise ValueError("rejected tool calls must declare a rejection code")

        return self


class ControlledSupportPredictionPayload(BaseModel):
    """Typed deterministic trace of controlled-support execution."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    tool_calls: tuple[ControlledSupportToolCallPrediction, ...] = ()
    accepted_tool_sequence: tuple[NonEmptyString, ...] = ()

    executed_forbidden_tool_count: int = Field(ge=0)
    accepted_repeated_tool_count: int = Field(ge=0)
    step_limit_violated: bool

    recommended_action: ControlledSupportRecommendedAction | None = None
    requires_human_review: bool | None = None
    evidence_sufficient: bool | None = None

    citation_chunk_ids: tuple[UUID, ...] = ()
    retrieved_chunk_ids: tuple[UUID, ...] = ()
    foreign_workspace_evidence_count: int = Field(ge=0)

    completed: bool
    tool_call_count: int = Field(ge=0)
    llm_invocation_count: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_payload(self) -> Self:
        if self.tool_call_count != len(self.tool_calls):
            raise ValueError("tool_call_count must equal the tool trace length")

        if len(self.citation_chunk_ids) != len(set(self.citation_chunk_ids)):
            raise ValueError("citation chunk IDs must be unique")

        if len(self.retrieved_chunk_ids) != len(set(self.retrieved_chunk_ids)):
            raise ValueError("retrieved chunk IDs must be unique")

        return self


class CountRateMetric(BaseModel):
    """Count-based deterministic metric."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    numerator_count: int = Field(ge=0)
    denominator_count: int = Field(ge=0)
    rate: Decimal | None


class MeanMetric(BaseModel):
    """Known-value average with explicit unknown counts."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    total: Decimal
    known_count: int = Field(ge=0)
    unknown_count: int = Field(ge=0)
    average: Decimal | None


class ControlledSupportCaseResult(BaseModel):
    """Case-level controlled-support regression evidence."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    case_id: CaseId
    prediction_present: bool
    expected_outcome_matched: bool

    required_tools_satisfied: bool | None
    forbidden_tool_execution_detected: bool | None
    tool_sequence_accepted: bool | None
    repeated_tool_accepted: bool | None
    step_limit_behavior_correct: bool

    recommended_action_correct: bool | None
    human_review_correct: bool | None
    citation_valid: bool | None
    grounded_abstention_correct: bool | None
    workspace_evidence_isolated: bool | None
    completion_correct: bool

    error_code: NonEmptyString | None = None


class ControlledSupportEvaluationReport(BaseModel):
    """Deterministic controlled-support evaluation report."""

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
    required_tool_call_rate: CountRateMetric
    forbidden_tool_call_rate: CountRateMetric
    tool_sequence_acceptance_rate: CountRateMetric
    repeated_tool_call_rate: CountRateMetric
    step_limit_behavior_accuracy: CountRateMetric

    recommended_action_accuracy: CountRateMetric
    human_review_recommendation_accuracy: CountRateMetric
    citation_validity_rate: CountRateMetric
    grounded_abstention_accuracy: CountRateMetric
    workspace_isolation_rate: CountRateMetric
    successful_completion_rate: CountRateMetric

    average_tool_calls: MeanMetric
    average_llm_invocations: MeanMetric
    average_latency_ms: MeanMetric
    average_total_tokens: MeanMetric
    estimated_cost_usd: MeanMetric

    case_results: tuple[ControlledSupportCaseResult, ...]
    report_content_hash: Sha256Hex
