from pathlib import Path

import pytest
from pydantic import ValidationError

from supportops.evaluation.semantic_retrieval.dataset import (
    load_semantic_retrieval_dataset,
)
from supportops.evaluation.semantic_retrieval.models import (
    SemanticRetrievalEvaluationCase,
)

PROJECT_ROOT = Path(__file__).resolve().parents[4]
DATASET_PATH = (
    PROJECT_ROOT / "evals" / "semantic-retrieval" / "datasets" / "semantic-retrieval-eval-v1.jsonl"
)

DATASET_HASH = "6667624d6cc122c86eb5b0634aae9fc609298d76aca20fd5a689f10f7dced039"


def test_committed_dataset_is_complete_and_immutable() -> None:
    dataset = load_semantic_retrieval_dataset(DATASET_PATH)

    assert dataset.dataset_id == "semantic-retrieval-eval"
    assert dataset.dataset_version == 1
    assert dataset.schema_version == "semantic-retrieval-eval-v1"
    assert dataset.source.value == "synthetic"
    assert len(dataset.cases) == 10
    assert len({case.case_id for case in dataset.cases}) == 10
    assert dataset.content_hash == DATASET_HASH


def test_dataset_contains_required_safety_scenarios() -> None:
    dataset = load_semantic_retrieval_dataset(DATASET_PATH)

    case_ids = {case.case_id for case in dataset.cases}

    assert "retrieval-no-active-documents-006" in case_ids
    assert "retrieval-cross-workspace-candidate-007" in case_ids
    assert "retrieval-duplicate-candidate-008" in case_ids
    assert "retrieval-prompt-injection-content-010" in case_ids


def test_no_result_case_rejects_expected_evidence() -> None:
    dataset = load_semantic_retrieval_dataset(DATASET_PATH)
    payload = dataset.cases[0].model_dump(mode="json")
    payload["expected_no_result"] = True

    with pytest.raises(
        ValidationError,
        match="no-result cases cannot declare expected evidence",
    ):
        SemanticRetrievalEvaluationCase.model_validate(payload)


def test_expected_citations_must_be_expected_chunks() -> None:
    dataset = load_semantic_retrieval_dataset(DATASET_PATH)
    payload = dataset.cases[0].model_dump(mode="json")
    payload["expected_citation_chunk_ids"] = ["ffffffff-ffff-4fff-8fff-ffffffffffff"]

    with pytest.raises(
        ValidationError,
        match="expected citation chunks",
    ):
        SemanticRetrievalEvaluationCase.model_validate(payload)
