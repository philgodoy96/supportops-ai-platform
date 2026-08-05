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


class EvaluationDatasetSource(StrEnum):
    """Supported provenance for committed evaluation data."""

    SYNTHETIC = "synthetic"


class SemanticRetrievalEvaluationCase(BaseModel):
    """One immutable semantic-retrieval evaluation scenario."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    dataset_id: NonEmptyString
    dataset_version: int = Field(ge=1)
    schema_version: NonEmptyString

    case_id: CaseId
    description: NonEmptyString
    source: EvaluationDatasetSource

    tags: tuple[NonEmptyString, ...] = Field(min_length=1)
    risk_tags: tuple[NonEmptyString, ...] = ()

    workspace_id: UUID
    query: NonEmptyString
    top_k: int = Field(ge=1, le=20)

    expected_document_ids: tuple[UUID, ...] = ()
    expected_chunk_ids: tuple[UUID, ...] = ()
    expected_no_result: bool
    expected_workspace_id: UUID
    expected_citation_chunk_ids: tuple[UUID, ...] = ()
    expected_searched_version_count: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def validate_case_contract(self) -> Self:
        self._require_unique("tags", self.tags)
        self._require_unique("risk_tags", self.risk_tags)
        self._require_unique(
            "expected_document_ids",
            self.expected_document_ids,
        )
        self._require_unique("expected_chunk_ids", self.expected_chunk_ids)
        self._require_unique(
            "expected_citation_chunk_ids",
            self.expected_citation_chunk_ids,
        )

        if self.expected_workspace_id != self.workspace_id:
            raise ValueError("expected_workspace_id must match the case workspace_id")

        if self.expected_no_result and (
            self.expected_document_ids
            or self.expected_chunk_ids
            or self.expected_citation_chunk_ids
        ):
            raise ValueError("no-result cases cannot declare expected evidence")

        if not set(self.expected_citation_chunk_ids).issubset(self.expected_chunk_ids):
            raise ValueError("expected citation chunks must be expected retrieval chunks")

        return self

    @staticmethod
    def _require_unique(name: str, values: tuple[object, ...]) -> None:
        if len(values) != len(set(values)):
            raise ValueError(f"{name} must contain unique values")


class SemanticRetrievalEvaluationDataset(BaseModel):
    """Validated semantic-retrieval dataset and deterministic hash."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    dataset_id: NonEmptyString
    dataset_version: int = Field(ge=1)
    schema_version: NonEmptyString
    source: EvaluationDatasetSource
    cases: tuple[SemanticRetrievalEvaluationCase, ...] = Field(min_length=1)
    content_hash: Sha256Hex

    @model_validator(mode="after")
    def validate_dataset_consistency(self) -> Self:
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

        return self


class SemanticRetrievalEvidencePrediction(BaseModel):
    """One ranked evidence identity emitted by retrieval."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    rank: int = Field(ge=1)
    workspace_id: UUID
    document_id: UUID
    document_version_id: UUID
    chunk_id: UUID
    ordinal: int = Field(ge=0)
    score: Decimal
    content_sha256: Sha256Hex
    citation_resolved: bool = True


class SemanticRetrievalPredictionPayload(BaseModel):
    """Typed output captured from one semantic-retrieval execution."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    searched_version_count: int = Field(ge=0)
    evidence: tuple[SemanticRetrievalEvidencePrediction, ...] = ()
    filtered_duplicate_count: int = Field(default=0, ge=0)
    filtered_cross_workspace_count: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def validate_rank_order(self) -> Self:
        expected_ranks = tuple(range(1, len(self.evidence) + 1))
        actual_ranks = tuple(item.rank for item in self.evidence)

        if actual_ranks != expected_ranks:
            raise ValueError("evidence ranks must be contiguous and start at one")

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


class SemanticRetrievalCaseResult(BaseModel):
    """Traceable case-level deterministic scoring result."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    case_id: CaseId
    prediction_present: bool
    prediction_succeeded: bool

    document_hit: bool | None
    chunk_hit: bool | None
    reciprocal_rank: Decimal | None
    recall_at_k: Decimal | None

    no_result_correct: bool
    workspace_isolated: bool
    citations_resolved: bool | None

    error_code: NonEmptyString | None = None


class SemanticRetrievalEvaluationReport(BaseModel):
    """Deterministic semantic-retrieval evaluation evidence."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    dataset_id: NonEmptyString
    dataset_version: int = Field(ge=1)
    schema_version: NonEmptyString
    dataset_hash: Sha256Hex
    prediction_hash: Sha256Hex

    case_count: int = Field(ge=1)

    document_hit_rate_at_k: CountRateMetric
    chunk_hit_rate_at_k: CountRateMetric
    mean_reciprocal_rank: MeanMetric
    recall_at_k: MeanMetric
    no_result_accuracy: CountRateMetric
    workspace_isolation_rate: CountRateMetric
    citation_resolution_rate: CountRateMetric

    average_latency_ms: MeanMetric
    average_query_tokens: MeanMetric
    estimated_query_cost_usd: MeanMetric

    case_results: tuple[SemanticRetrievalCaseResult, ...]
    report_content_hash: Sha256Hex
