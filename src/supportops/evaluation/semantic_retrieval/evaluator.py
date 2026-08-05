from __future__ import annotations

from decimal import ROUND_HALF_UP, Decimal
from uuid import UUID

from supportops.evaluation.contracts.hashing import sha256_hexdigest
from supportops.evaluation.contracts.predictions import (
    EvaluationPredictionStatus,
)
from supportops.evaluation.semantic_retrieval.models import (
    CountRateMetric,
    MeanMetric,
    SemanticRetrievalCaseResult,
    SemanticRetrievalEvaluationCase,
    SemanticRetrievalEvaluationDataset,
    SemanticRetrievalEvaluationReport,
    SemanticRetrievalEvidencePrediction,
)
from supportops.evaluation.semantic_retrieval.predictions import (
    SemanticRetrievalPrediction,
)

_METRIC_QUANTUM = Decimal("0.000001")


class SemanticRetrievalEvaluationError(ValueError):
    """Raised when semantic-retrieval scoring cannot be completed."""


def evaluate_semantic_retrieval_predictions(
    *,
    dataset: SemanticRetrievalEvaluationDataset,
    predictions: tuple[SemanticRetrievalPrediction, ...],
    prediction_hash: str,
) -> SemanticRetrievalEvaluationReport:
    """Score static retrieval predictions against a versioned dataset."""

    predictions_by_case = {prediction.case_id: prediction for prediction in predictions}
    dataset_case_ids = {case.case_id for case in dataset.cases}

    unknown_case_ids = sorted(set(predictions_by_case) - dataset_case_ids)
    if unknown_case_ids:
        raise SemanticRetrievalEvaluationError(
            "unknown prediction case IDs: " + ", ".join(unknown_case_ids)
        )

    case_results: list[SemanticRetrievalCaseResult] = []

    document_hits: list[bool] = []
    chunk_hits: list[bool] = []
    reciprocal_ranks: list[Decimal] = []
    recalls: list[Decimal] = []
    no_result_results: list[bool] = []
    workspace_results: list[bool] = []
    citation_results: list[bool] = []

    latency_values: list[Decimal] = []
    unknown_latency_count = 0

    token_values: list[Decimal] = []
    unknown_token_count = 0

    cost_values: list[Decimal] = []
    unknown_cost_count = 0

    for case in dataset.cases:
        prediction = predictions_by_case.get(case.case_id)

        if prediction is None:
            case_results.append(_missing_case_result(case))
            _record_missing_case_metrics(
                case=case,
                document_hits=document_hits,
                chunk_hits=chunk_hits,
                reciprocal_ranks=reciprocal_ranks,
                recalls=recalls,
                no_result_results=no_result_results,
                workspace_results=workspace_results,
                citation_results=citation_results,
            )
            unknown_latency_count += 1
            unknown_token_count += 1
            unknown_cost_count += 1
            continue

        if prediction.latency_ms is None:
            unknown_latency_count += 1
        else:
            latency_values.append(Decimal(prediction.latency_ms))

        if prediction.embedding_tokens is None:
            unknown_token_count += 1
        else:
            token_values.append(Decimal(prediction.embedding_tokens))

        if prediction.estimated_cost_usd is None:
            unknown_cost_count += 1
        else:
            cost_values.append(prediction.estimated_cost_usd)

        if prediction.status is EvaluationPredictionStatus.FAILED or prediction.payload is None:
            case_results.append(
                _failed_case_result(
                    case,
                    error_code=prediction.error_code,
                )
            )
            _record_missing_case_metrics(
                case=case,
                document_hits=document_hits,
                chunk_hits=chunk_hits,
                reciprocal_ranks=reciprocal_ranks,
                recalls=recalls,
                no_result_results=no_result_results,
                workspace_results=workspace_results,
                citation_results=citation_results,
            )
            continue

        result = _score_case(case, prediction.payload.evidence)
        case_results.append(result)

        if result.document_hit is not None:
            document_hits.append(result.document_hit)
        if result.chunk_hit is not None:
            chunk_hits.append(result.chunk_hit)
        if result.reciprocal_rank is not None:
            reciprocal_ranks.append(result.reciprocal_rank)
        if result.recall_at_k is not None:
            recalls.append(result.recall_at_k)
        no_result_results.append(result.no_result_correct)
        workspace_results.append(result.workspace_isolated)
        if result.citations_resolved is not None:
            citation_results.append(result.citations_resolved)

    report_without_hash = {
        "dataset_id": dataset.dataset_id,
        "dataset_version": dataset.dataset_version,
        "schema_version": dataset.schema_version,
        "dataset_hash": dataset.content_hash,
        "prediction_hash": prediction_hash,
        "case_count": len(dataset.cases),
        "document_hit_rate_at_k": _count_rate(document_hits),
        "chunk_hit_rate_at_k": _count_rate(chunk_hits),
        "mean_reciprocal_rank": _mean_metric(
            reciprocal_ranks,
            unknown_count=0,
        ),
        "recall_at_k": _mean_metric(recalls, unknown_count=0),
        "no_result_accuracy": _count_rate(no_result_results),
        "workspace_isolation_rate": _count_rate(workspace_results),
        "citation_resolution_rate": _count_rate(citation_results),
        "average_latency_ms": _mean_metric(
            latency_values,
            unknown_count=unknown_latency_count,
        ),
        "average_query_tokens": _mean_metric(
            token_values,
            unknown_count=unknown_token_count,
        ),
        "estimated_query_cost_usd": _mean_metric(
            cost_values,
            unknown_count=unknown_cost_count,
        ),
        "case_results": tuple(case_results),
    }

    report_hash = sha256_hexdigest(report_without_hash)

    return SemanticRetrievalEvaluationReport(
        **report_without_hash,
        report_content_hash=report_hash,
    )


def _score_case(
    case: SemanticRetrievalEvaluationCase,
    evidence: tuple[SemanticRetrievalEvidencePrediction, ...],
) -> SemanticRetrievalCaseResult:
    effective_evidence = _deduplicate_and_limit(
        evidence,
        top_k=case.top_k,
    )

    retrieved_document_ids = {item.document_id for item in effective_evidence}
    retrieved_chunk_ids = {item.chunk_id for item in effective_evidence}

    document_hit = (
        bool(set(case.expected_document_ids) & retrieved_document_ids)
        if case.expected_document_ids
        else None
    )
    chunk_hit = (
        bool(set(case.expected_chunk_ids) & retrieved_chunk_ids)
        if case.expected_chunk_ids
        else None
    )

    reciprocal_rank = _reciprocal_rank(
        expected_chunk_ids=set(case.expected_chunk_ids),
        evidence=effective_evidence,
    )
    recall_at_k = _recall_at_k(
        expected_chunk_ids=set(case.expected_chunk_ids),
        retrieved_chunk_ids=retrieved_chunk_ids,
    )

    no_result_correct = (
        len(effective_evidence) == 0 if case.expected_no_result else len(effective_evidence) > 0
    )

    workspace_isolated = all(
        item.workspace_id == case.expected_workspace_id for item in effective_evidence
    )

    citations_resolved: bool | None
    if case.expected_citation_chunk_ids:
        expected_citations = set(case.expected_citation_chunk_ids)
        returned_by_chunk = {item.chunk_id: item for item in effective_evidence}
        citations_resolved = all(
            chunk_id in returned_by_chunk
            and returned_by_chunk[chunk_id].citation_resolved
            and returned_by_chunk[chunk_id].workspace_id == case.expected_workspace_id
            for chunk_id in expected_citations
        )
    elif case.expected_no_result:
        citations_resolved = None
    else:
        citations_resolved = all(item.citation_resolved for item in effective_evidence)

    return SemanticRetrievalCaseResult(
        case_id=case.case_id,
        prediction_present=True,
        prediction_succeeded=True,
        document_hit=document_hit,
        chunk_hit=chunk_hit,
        reciprocal_rank=reciprocal_rank,
        recall_at_k=recall_at_k,
        no_result_correct=no_result_correct,
        workspace_isolated=workspace_isolated,
        citations_resolved=citations_resolved,
    )


def _deduplicate_and_limit(
    evidence: tuple[SemanticRetrievalEvidencePrediction, ...],
    *,
    top_k: int,
) -> tuple[SemanticRetrievalEvidencePrediction, ...]:
    deduplicated: list[SemanticRetrievalEvidencePrediction] = []
    seen_chunk_ids: set[UUID] = set()

    for item in evidence:
        if item.chunk_id in seen_chunk_ids:
            continue
        seen_chunk_ids.add(item.chunk_id)
        deduplicated.append(item)
        if len(deduplicated) == top_k:
            break

    return tuple(deduplicated)


def _reciprocal_rank(
    *,
    expected_chunk_ids: set[UUID],
    evidence: tuple[SemanticRetrievalEvidencePrediction, ...],
) -> Decimal | None:
    if not expected_chunk_ids:
        return None

    for effective_rank, item in enumerate(evidence, start=1):
        if item.chunk_id in expected_chunk_ids:
            return _quantize(Decimal(1) / Decimal(effective_rank))

    return Decimal("0.000000")


def _recall_at_k(
    *,
    expected_chunk_ids: set[UUID],
    retrieved_chunk_ids: set[UUID],
) -> Decimal | None:
    if not expected_chunk_ids:
        return None

    matched_count = len(expected_chunk_ids & retrieved_chunk_ids)
    return _quantize(Decimal(matched_count) / Decimal(len(expected_chunk_ids)))


def _record_missing_case_metrics(
    *,
    case: SemanticRetrievalEvaluationCase,
    document_hits: list[bool],
    chunk_hits: list[bool],
    reciprocal_ranks: list[Decimal],
    recalls: list[Decimal],
    no_result_results: list[bool],
    workspace_results: list[bool],
    citation_results: list[bool],
) -> None:
    if case.expected_document_ids:
        document_hits.append(False)
    if case.expected_chunk_ids:
        chunk_hits.append(False)
        reciprocal_ranks.append(Decimal("0.000000"))
        recalls.append(Decimal("0.000000"))
    no_result_results.append(False)
    workspace_results.append(False)
    if case.expected_citation_chunk_ids:
        citation_results.append(False)


def _missing_case_result(
    case: SemanticRetrievalEvaluationCase,
) -> SemanticRetrievalCaseResult:
    return SemanticRetrievalCaseResult(
        case_id=case.case_id,
        prediction_present=False,
        prediction_succeeded=False,
        document_hit=False if case.expected_document_ids else None,
        chunk_hit=False if case.expected_chunk_ids else None,
        reciprocal_rank=(Decimal("0.000000") if case.expected_chunk_ids else None),
        recall_at_k=(Decimal("0.000000") if case.expected_chunk_ids else None),
        no_result_correct=False,
        workspace_isolated=False,
        citations_resolved=(False if case.expected_citation_chunk_ids else None),
        error_code="prediction_missing",
    )


def _failed_case_result(
    case: SemanticRetrievalEvaluationCase,
    *,
    error_code: str | None,
) -> SemanticRetrievalCaseResult:
    return SemanticRetrievalCaseResult(
        case_id=case.case_id,
        prediction_present=True,
        prediction_succeeded=False,
        document_hit=False if case.expected_document_ids else None,
        chunk_hit=False if case.expected_chunk_ids else None,
        reciprocal_rank=(Decimal("0.000000") if case.expected_chunk_ids else None),
        recall_at_k=(Decimal("0.000000") if case.expected_chunk_ids else None),
        no_result_correct=False,
        workspace_isolated=False,
        citations_resolved=(False if case.expected_citation_chunk_ids else None),
        error_code=error_code,
    )


def _count_rate(values: list[bool]) -> CountRateMetric:
    numerator = sum(values)
    denominator = len(values)

    rate = _quantize(Decimal(numerator) / Decimal(denominator)) if denominator else None

    return CountRateMetric(
        numerator_count=numerator,
        denominator_count=denominator,
        rate=rate,
    )


def _mean_metric(
    values: list[Decimal],
    *,
    unknown_count: int,
) -> MeanMetric:
    total = sum(values, start=Decimal("0"))
    average = _quantize(total / Decimal(len(values))) if values else None

    return MeanMetric(
        total=total,
        known_count=len(values),
        unknown_count=unknown_count,
        average=average,
    )


def _quantize(value: Decimal) -> Decimal:
    return value.quantize(
        _METRIC_QUANTUM,
        rounding=ROUND_HALF_UP,
    )
