from decimal import Decimal
from pathlib import Path

import pytest

from supportops.evaluation.contracts.predictions import (
    EvaluationPredictionEnvelope,
    EvaluationPredictionStatus,
)
from supportops.evaluation.semantic_retrieval.dataset import (
    load_semantic_retrieval_dataset,
)
from supportops.evaluation.semantic_retrieval.evaluator import (
    SemanticRetrievalEvaluationError,
    evaluate_semantic_retrieval_predictions,
)
from supportops.evaluation.semantic_retrieval.models import (
    SemanticRetrievalPredictionPayload,
)
from supportops.evaluation.semantic_retrieval.predictions import (
    load_semantic_retrieval_predictions,
)

PROJECT_ROOT = Path(__file__).resolve().parents[4]
DATASET_PATH = (
    PROJECT_ROOT / "evals" / "semantic-retrieval" / "datasets" / "semantic-retrieval-eval-v1.jsonl"
)
PREDICTIONS_PATH = (
    PROJECT_ROOT
    / "evals"
    / "semantic-retrieval"
    / "predictions"
    / "semantic-retrieval-eval-v1.static.jsonl"
)

PREDICTION_HASH = "5df6acdc47130f020b905b144d2a2ee0f3485b4d01db43f1b4ed3245b08e3655"


def test_static_predictions_produce_expected_metrics() -> None:
    dataset = load_semantic_retrieval_dataset(DATASET_PATH)
    predictions, prediction_hash = load_semantic_retrieval_predictions(PREDICTIONS_PATH)

    report = evaluate_semantic_retrieval_predictions(
        dataset=dataset,
        predictions=predictions,
        prediction_hash=prediction_hash,
    )

    assert prediction_hash == PREDICTION_HASH
    assert report.document_hit_rate_at_k.rate == Decimal("1.000000")
    assert report.chunk_hit_rate_at_k.rate == Decimal("1.000000")
    assert report.mean_reciprocal_rank.average == Decimal("0.937500")
    assert report.recall_at_k.average == Decimal("1.000000")
    assert report.no_result_accuracy.rate == Decimal("1.000000")
    assert report.workspace_isolation_rate.rate == Decimal("1.000000")
    assert report.citation_resolution_rate.rate == Decimal("1.000000")

    assert report.average_latency_ms.average == Decimal("13.500000")
    assert report.average_query_tokens.average == Decimal("8.666667")
    assert report.estimated_query_cost_usd.total == Decimal("0.000078")
    assert report.estimated_query_cost_usd.unknown_count == 1


def test_report_hash_is_deterministic() -> None:
    dataset = load_semantic_retrieval_dataset(DATASET_PATH)
    predictions, prediction_hash = load_semantic_retrieval_predictions(PREDICTIONS_PATH)

    first = evaluate_semantic_retrieval_predictions(
        dataset=dataset,
        predictions=predictions,
        prediction_hash=prediction_hash,
    )
    second = evaluate_semantic_retrieval_predictions(
        dataset=dataset,
        predictions=predictions,
        prediction_hash=prediction_hash,
    )

    assert first.report_content_hash == second.report_content_hash


def test_missing_prediction_remains_visible_as_failure() -> None:
    dataset = load_semantic_retrieval_dataset(DATASET_PATH)
    predictions, prediction_hash = load_semantic_retrieval_predictions(PREDICTIONS_PATH)

    report = evaluate_semantic_retrieval_predictions(
        dataset=dataset,
        predictions=predictions[:-1],
        prediction_hash=prediction_hash,
    )

    missing = report.case_results[-1]

    assert missing.prediction_present is False
    assert missing.error_code == "prediction_missing"
    assert report.workspace_isolation_rate.rate == Decimal("0.900000")


def test_failed_prediction_retains_known_usage() -> None:
    dataset = load_semantic_retrieval_dataset(DATASET_PATH)
    predictions, prediction_hash = load_semantic_retrieval_predictions(PREDICTIONS_PATH)

    failed = EvaluationPredictionEnvelope[SemanticRetrievalPredictionPayload](
        case_id=predictions[0].case_id,
        status=EvaluationPredictionStatus.FAILED,
        error_code="retrieval_failed",
        latency_ms=25,
        embedding_tokens=12,
        estimated_cost_usd=Decimal("0.000012"),
    )

    report = evaluate_semantic_retrieval_predictions(
        dataset=dataset,
        predictions=(failed, *predictions[1:]),
        prediction_hash=prediction_hash,
    )

    case_result = report.case_results[0]

    assert case_result.prediction_succeeded is False
    assert case_result.document_hit is False
    assert report.average_latency_ms.known_count == 10
    assert report.average_query_tokens.known_count == 9


def test_unknown_prediction_case_id_is_rejected() -> None:
    dataset = load_semantic_retrieval_dataset(DATASET_PATH)
    predictions, prediction_hash = load_semantic_retrieval_predictions(PREDICTIONS_PATH)

    unknown = predictions[0].model_copy(update={"case_id": "unknown-retrieval-case-999"})

    with pytest.raises(
        SemanticRetrievalEvaluationError,
        match="unknown prediction case IDs",
    ):
        evaluate_semantic_retrieval_predictions(
            dataset=dataset,
            predictions=(*predictions, unknown),
            prediction_hash=prediction_hash,
        )


def test_duplicate_chunks_count_once_for_recall() -> None:
    dataset = load_semantic_retrieval_dataset(DATASET_PATH)
    predictions, prediction_hash = load_semantic_retrieval_predictions(PREDICTIONS_PATH)

    target_index = 7
    target = predictions[target_index]
    assert target.payload is not None

    duplicated_payload = target.payload.model_copy(
        update={
            "evidence": (
                target.payload.evidence[0],
                target.payload.evidence[0].model_copy(update={"rank": 2}),
                target.payload.evidence[1].model_copy(update={"rank": 3}),
            )
        }
    )
    duplicated = target.model_copy(update={"payload": duplicated_payload})

    modified_predictions = list(predictions)
    modified_predictions[target_index] = duplicated

    report = evaluate_semantic_retrieval_predictions(
        dataset=dataset,
        predictions=tuple(modified_predictions),
        prediction_hash=prediction_hash,
    )

    result = report.case_results[target_index]

    assert result.recall_at_k == Decimal("1.000000")
    assert result.reciprocal_rank == Decimal("1.000000")
