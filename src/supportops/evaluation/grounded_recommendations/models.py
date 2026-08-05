from __future__ import annotations

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


class GroundedRecommendationDatasetSource(StrEnum):
    """Supported provenance for grounded recommendation datasets."""

    SYNTHETIC = "synthetic"


class GroundedRecommendationAction(StrEnum):
    """Supported recommendation actions."""

    RESPOND = "respond"
    REQUEST_MORE_INFORMATION = "request_more_information"
    RECOMMEND_ESCALATION = "recommend_escalation"


class GroundedRecommendationClassification(BaseModel):
    """Classification context supplied to recommendation generation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    category: NonEmptyString
    intent: NonEmptyString
    urgency: NonEmptyString
    sentiment: NonEmptyString
    requires_human_review: bool
    summary: NonEmptyString


class GroundedRecommendationContext(BaseModel):
    """Retrieved evidence embedded into one evaluation case."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    chunk_id: UUID
    document_id: UUID
    document_version_id: UUID
    workspace_id: UUID
    content: NonEmptyString
    content_sha256: Sha256Hex


class GroundedRecommendationEvaluationCase(BaseModel):
    """One immutable grounded recommendation evaluation scenario."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    dataset_id: NonEmptyString
    dataset_version: int = Field(ge=1)
    schema_version: NonEmptyString

    case_id: CaseId
    description: NonEmptyString
    source: GroundedRecommendationDatasetSource

    tags: tuple[NonEmptyString, ...] = Field(min_length=1)
    risk_tags: tuple[NonEmptyString, ...] = ()

    workspace_id: UUID
    workflow_name: NonEmptyString
    workflow_version: NonEmptyString

    ticket_subject: NonEmptyString
    ticket_description: NonEmptyString
    classification: GroundedRecommendationClassification

    retrieved_contexts: tuple[GroundedRecommendationContext, ...] = ()

    expected_action: GroundedRecommendationAction
    expected_requires_human_review: bool
    expected_evidence_sufficient: bool

    reference_answer: NonEmptyString
    reference_claims: tuple[NonEmptyString, ...] = Field(min_length=1)

    expected_citation_chunk_ids: tuple[UUID, ...] = ()
    expected_foreign_workspace_evidence_count: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_case_contract(self) -> Self:
        self._require_unique("tags", self.tags)
        self._require_unique("risk_tags", self.risk_tags)
        self._require_unique(
            "reference_claims",
            self.reference_claims,
        )
        self._require_unique(
            "expected_citation_chunk_ids",
            self.expected_citation_chunk_ids,
        )

        context_chunk_ids = tuple(context.chunk_id for context in self.retrieved_contexts)
        self._require_unique(
            "retrieved context chunk IDs",
            context_chunk_ids,
        )

        expected_citations = set(self.expected_citation_chunk_ids)
        available_chunks = set(context_chunk_ids)

        if not expected_citations.issubset(available_chunks):
            raise ValueError("expected citations must reference retrieved context chunks")

        actual_foreign_count = sum(
            context.workspace_id != self.workspace_id for context in self.retrieved_contexts
        )
        if actual_foreign_count != self.expected_foreign_workspace_evidence_count:
            raise ValueError("expected foreign-workspace count does not match contexts")

        if (
            self.expected_action is GroundedRecommendationAction.RECOMMEND_ESCALATION
            and not self.expected_requires_human_review
        ):
            raise ValueError("escalation recommendations must require human review")

        if (
            not self.expected_evidence_sufficient
            and self.expected_action is GroundedRecommendationAction.RESPOND
        ):
            raise ValueError("insufficient evidence cannot expect a direct response")

        return self

    @staticmethod
    def _require_unique(
        name: str,
        values: tuple[object, ...],
    ) -> None:
        if len(values) != len(set(values)):
            raise ValueError(f"{name} must contain unique values")


class GroundedRecommendationEvaluationDataset(BaseModel):
    """Validated grounded recommendation dataset and content hash."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    dataset_id: NonEmptyString
    dataset_version: int = Field(ge=1)
    schema_version: NonEmptyString
    source: GroundedRecommendationDatasetSource

    workflow_name: NonEmptyString
    workflow_version: NonEmptyString

    cases: tuple[GroundedRecommendationEvaluationCase, ...] = Field(min_length=1)
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

            if case.workflow_name != self.workflow_name:
                raise ValueError("case workflow_name does not match dataset")

            if case.workflow_version != self.workflow_version:
                raise ValueError("case workflow_version does not match dataset")

        return self
