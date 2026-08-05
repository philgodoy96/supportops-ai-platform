from __future__ import annotations

import json
from decimal import Decimal
from enum import StrEnum
from pathlib import Path
from typing import Annotated

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
from supportops.evaluation.grounded_recommendations.ragas_adapter import (
    RagasMetricName,
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


class RagasMetricStatus(StrEnum):
    """Outcome status for one model-based evaluation metric."""

    SUCCEEDED = "succeeded"
    FAILED = "failed"
    NOT_APPLICABLE = "not_applicable"


class GroundedRecommendationRagasMetricScore(BaseModel):
    """One normalized RAGAS metric outcome."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    metric: RagasMetricName
    status: RagasMetricStatus
    score: Decimal | None = Field(default=None, ge=0, le=1)
    error_code: NonEmptyString | None = None
    reason: NonEmptyString | None = None

    @model_validator(mode="after")
    def validate_outcome(self) -> GroundedRecommendationRagasMetricScore:
        if self.status is RagasMetricStatus.SUCCEEDED:
            if self.score is None:
                raise ValueError("a succeeded metric must include a score")
            if self.error_code is not None:
                raise ValueError("a succeeded metric cannot include an error code")

        if self.status is RagasMetricStatus.FAILED:
            if self.score is not None:
                raise ValueError("a failed metric cannot include a score")
            if self.error_code is None:
                raise ValueError("a failed metric must include an error code")

        if self.status is RagasMetricStatus.NOT_APPLICABLE:
            if self.score is not None:
                raise ValueError("a not-applicable metric cannot include a score")
            if self.reason is None:
                raise ValueError("a not-applicable metric must include a reason")

        return self


class GroundedRecommendationRagasCaseScore(BaseModel):
    """Normalized model-based scores for one recommendation case."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    case_id: CaseId
    metrics: tuple[
        GroundedRecommendationRagasMetricScore,
        ...,
    ] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_metric_uniqueness(
        self,
    ) -> GroundedRecommendationRagasCaseScore:
        metric_names = tuple(metric.metric for metric in self.metrics)

        if len(metric_names) != len(set(metric_names)):
            raise ValueError("a case cannot contain duplicate RAGAS metrics")

        return self


class GroundedRecommendationRagasScoreArtifact(BaseModel):
    """Validated RAGAS score artifact with canonical hash."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    case_scores: tuple[
        GroundedRecommendationRagasCaseScore,
        ...,
    ] = Field(min_length=1)
    content_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_case_uniqueness(
        self,
    ) -> GroundedRecommendationRagasScoreArtifact:
        case_ids = tuple(case_score.case_id for case_score in self.case_scores)

        if len(case_ids) != len(set(case_ids)):
            raise ValueError("RAGAS score artifact contains duplicate case IDs")

        return self


class GroundedRecommendationRagasScoreError(ValueError):
    """Raised when a RAGAS score artifact is invalid."""


def load_grounded_recommendation_ragas_scores(
    path: Path,
) -> GroundedRecommendationRagasScoreArtifact:
    """Load and canonically hash normalized RAGAS JSONL scores."""

    case_scores: list[GroundedRecommendationRagasCaseScore] = []
    canonical_lines: list[bytes] = []
    seen_case_ids: set[str] = set()

    with path.open("r", encoding="utf-8") as score_file:
        for line_number, raw_line in enumerate(
            score_file,
            start=1,
        ):
            if not raw_line.strip():
                continue

            try:
                payload = json.loads(raw_line)
                case_score = GroundedRecommendationRagasCaseScore.model_validate(payload)
            except (json.JSONDecodeError, ValueError) as exc:
                raise GroundedRecommendationRagasScoreError(
                    f"invalid RAGAS score line {line_number}: {exc}"
                ) from exc

            if case_score.case_id in seen_case_ids:
                raise GroundedRecommendationRagasScoreError(
                    f"duplicate RAGAS score case_id: {case_score.case_id}"
                )

            seen_case_ids.add(case_score.case_id)
            case_scores.append(case_score)
            canonical_lines.append(
                canonical_json_bytes(
                    case_score.model_dump(
                        mode="json",
                        exclude_none=False,
                    )
                )
                + b"\n"
            )

    if not case_scores:
        raise GroundedRecommendationRagasScoreError("RAGAS score artifact must not be empty")

    return GroundedRecommendationRagasScoreArtifact(
        case_scores=tuple(case_scores),
        content_hash=sha256_bytes(b"".join(canonical_lines)),
    )
