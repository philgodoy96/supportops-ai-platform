from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Annotated, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    model_validator,
)

from supportops.evaluation.contracts.hashing import (
    canonical_json_bytes,
    sha256_bytes,
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


class GroundedRecommendationHumanReviewScale(BaseModel):
    """Scoring scale for qualitative grounded recommendation review."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    minimum: int
    maximum: int
    labels: dict[int, NonEmptyString]

    @model_validator(mode="after")
    def validate_scale(self) -> Self:
        if self.minimum >= self.maximum:
            raise ValueError("review scale minimum must be lower than maximum")

        expected_scores = set(range(self.minimum, self.maximum + 1))
        if set(self.labels) != expected_scores:
            raise ValueError("review scale labels must cover every score")

        return self


class GroundedRecommendationHumanReviewDimension(BaseModel):
    """One qualitative review dimension."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    dimension: NonEmptyString
    description: NonEmptyString
    blocking_when: NonEmptyString


class GroundedRecommendationHumanReviewPolicy(BaseModel):
    """Operational policy for human review evidence."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    review_all_cases: bool
    required_note_for_score_at_or_below: int
    required_note_for_safety_concern: bool
    required_evidence_reference: bool
    blocking_issue_requires_second_reviewer: bool
    unresolved_blocking_disagreement_outcome: NonEmptyString


class GroundedRecommendationHumanReviewRubric(BaseModel):
    """Committed human review rubric."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    rubric_id: NonEmptyString
    rubric_version: int = Field(ge=1)
    schema_version: NonEmptyString

    scale: GroundedRecommendationHumanReviewScale
    dimensions: tuple[
        GroundedRecommendationHumanReviewDimension,
        ...,
    ] = Field(min_length=1)

    review_policy: GroundedRecommendationHumanReviewPolicy
    review_record_fields: tuple[NonEmptyString, ...] = Field(min_length=1)

    content_hash: Sha256Hex

    @model_validator(mode="after")
    def validate_rubric(self) -> Self:
        dimension_names = tuple(dimension.dimension for dimension in self.dimensions)
        if len(dimension_names) != len(set(dimension_names)):
            raise ValueError("review dimensions must be unique")

        if len(self.review_record_fields) != len(set(self.review_record_fields)):
            raise ValueError("review record fields must be unique")

        required_fields = {
            "reviewer_id",
            "reviewed_at",
            "case_id",
            "dimension_scores",
            "evidence_references",
            "notes",
            "blocking_issue",
        }
        if not required_fields.issubset(set(self.review_record_fields)):
            raise ValueError("review record fields are incomplete")

        threshold = self.review_policy.required_note_for_score_at_or_below
        if not self.scale.minimum <= threshold <= self.scale.maximum:
            raise ValueError("required note threshold must be within the scale")

        return self


class GroundedRecommendationHumanReviewRecord(BaseModel):
    """One completed qualitative review record."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    reviewer_id: NonEmptyString
    reviewed_at: datetime
    case_id: CaseId

    dimension_scores: dict[NonEmptyString, int]
    evidence_references: tuple[NonEmptyString, ...]
    notes: NonEmptyString | None = None
    blocking_issue: bool

    @model_validator(mode="after")
    def validate_review_record(self) -> Self:
        if not self.dimension_scores:
            raise ValueError("dimension scores must not be empty")

        return self


class GroundedRecommendationHumanReviewRubricError(ValueError):
    """Raised when the human review rubric is invalid."""


def load_grounded_recommendation_human_review_rubric(
    path: Path,
) -> GroundedRecommendationHumanReviewRubric:
    """Load, validate, and hash the committed review rubric."""

    try:
        payload = json.loads(path.read_bytes())
    except (OSError, json.JSONDecodeError) as exc:
        raise GroundedRecommendationHumanReviewRubricError(
            f"invalid human review rubric: {exc}"
        ) from exc

    if not isinstance(payload, dict):
        raise GroundedRecommendationHumanReviewRubricError(
            "invalid human review rubric: payload must be a JSON object"
        )

    declared_content = {key: value for key, value in payload.items() if key != "content_hash"}

    try:
        return GroundedRecommendationHumanReviewRubric.model_validate(
            {
                **declared_content,
                "content_hash": sha256_bytes(canonical_json_bytes(declared_content)),
            }
        )
    except ValueError as exc:
        raise GroundedRecommendationHumanReviewRubricError(
            f"invalid human review rubric: {exc}"
        ) from exc
