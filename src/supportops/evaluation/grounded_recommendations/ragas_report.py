from __future__ import annotations

from decimal import ROUND_HALF_UP, Decimal
from typing import Annotated

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
)

from supportops.evaluation.contracts.hashing import (
    sha256_hexdigest,
)
from supportops.evaluation.grounded_recommendations.models import (
    GroundedRecommendationEvaluationDataset,
)
from supportops.evaluation.grounded_recommendations.ragas_adapter import (
    RagasMetricName,
)
from supportops.evaluation.grounded_recommendations.ragas_scores import (
    GroundedRecommendationRagasScoreArtifact,
    RagasMetricStatus,
)

NonEmptyString = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1),
]

Sha256Hex = Annotated[
    str,
    StringConstraints(pattern=r"^[0-9a-f]{64}$"),
]

_QUANTUM = Decimal("0.000001")


class GroundedRecommendationRagasMetricAggregate(BaseModel):
    """Aggregate outcome for one RAGAS metric."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    metric: RagasMetricName
    succeeded_count: int = Field(ge=0)
    failed_count: int = Field(ge=0)
    not_applicable_count: int = Field(ge=0)
    average_score: Decimal | None = Field(
        default=None,
        ge=0,
        le=1,
    )


class GroundedRecommendationRagasReport(BaseModel):
    """Offline aggregate of previously generated RAGAS scores."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    dataset_id: NonEmptyString
    dataset_version: int = Field(ge=1)
    dataset_hash: Sha256Hex
    score_artifact_hash: Sha256Hex

    case_count: int = Field(ge=1)
    scored_case_count: int = Field(ge=0)
    missing_case_count: int = Field(ge=0)
    unknown_case_count: int = Field(ge=0)

    metric_aggregates: tuple[
        GroundedRecommendationRagasMetricAggregate,
        ...,
    ]

    report_content_hash: Sha256Hex


class GroundedRecommendationRagasReportError(ValueError):
    """Raised when offline RAGAS aggregation cannot complete."""


def build_grounded_recommendation_ragas_report(
    *,
    dataset: GroundedRecommendationEvaluationDataset,
    scores: GroundedRecommendationRagasScoreArtifact,
) -> GroundedRecommendationRagasReport:
    """Aggregate existing score artifacts without provider calls."""

    dataset_case_ids = {case.case_id for case in dataset.cases}
    score_case_ids = {case_score.case_id for case_score in scores.case_scores}

    unknown_case_ids = sorted(score_case_ids - dataset_case_ids)
    known_case_ids = score_case_ids & dataset_case_ids
    missing_case_ids = dataset_case_ids - score_case_ids

    aggregates = tuple(
        _aggregate_metric(
            metric=metric,
            scores=scores,
            dataset_case_ids=dataset_case_ids,
        )
        for metric in RagasMetricName
    )

    report_without_hash = {
        "dataset_id": dataset.dataset_id,
        "dataset_version": dataset.dataset_version,
        "dataset_hash": dataset.content_hash,
        "score_artifact_hash": scores.content_hash,
        "case_count": len(dataset.cases),
        "scored_case_count": len(known_case_ids),
        "missing_case_count": len(missing_case_ids),
        "unknown_case_count": len(unknown_case_ids),
        "metric_aggregates": aggregates,
    }

    return GroundedRecommendationRagasReport(
        **report_without_hash,
        report_content_hash=sha256_hexdigest(report_without_hash),
    )


def _aggregate_metric(
    *,
    metric: RagasMetricName,
    scores: GroundedRecommendationRagasScoreArtifact,
    dataset_case_ids: set[str],
) -> GroundedRecommendationRagasMetricAggregate:
    matching_scores = [
        metric_score
        for case_score in scores.case_scores
        if case_score.case_id in dataset_case_ids
        for metric_score in case_score.metrics
        if metric_score.metric is metric
    ]

    succeeded_scores = [
        metric_score.score
        for metric_score in matching_scores
        if (metric_score.status is RagasMetricStatus.SUCCEEDED and metric_score.score is not None)
    ]

    failed_count = sum(
        metric_score.status is RagasMetricStatus.FAILED for metric_score in matching_scores
    )
    not_applicable_count = sum(
        metric_score.status is RagasMetricStatus.NOT_APPLICABLE for metric_score in matching_scores
    )

    total = sum(
        succeeded_scores,
        start=Decimal("0"),
    )

    average = (
        (total / Decimal(len(succeeded_scores))).quantize(
            _QUANTUM,
            rounding=ROUND_HALF_UP,
        )
        if succeeded_scores
        else None
    )

    return GroundedRecommendationRagasMetricAggregate(
        metric=metric,
        succeeded_count=len(succeeded_scores),
        failed_count=failed_count,
        not_applicable_count=not_applicable_count,
        average_score=average,
    )
