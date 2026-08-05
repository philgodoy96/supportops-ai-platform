from pathlib import Path

import pytest
from pydantic import ValidationError

from supportops.evaluation.grounded_recommendations.dataset import (
    GroundedRecommendationDatasetError,
    load_grounded_recommendation_dataset,
)
from supportops.evaluation.grounded_recommendations.models import (
    GroundedRecommendationEvaluationCase,
)

PROJECT_ROOT = Path(__file__).resolve().parents[4]

DATASET_PATH = (
    PROJECT_ROOT
    / "evals"
    / "grounded-recommendations"
    / "datasets"
    / "grounded-recommendations-eval-v1.jsonl"
)

DATASET_HASH = "99a7f6cbdb68feb92d3fde32adc43e20893c9e855f299d35e970d166411662cd"


def test_committed_dataset_is_complete_and_immutable() -> None:
    dataset = load_grounded_recommendation_dataset(DATASET_PATH)

    assert dataset.dataset_id == "grounded-recommendations-eval"
    assert dataset.dataset_version == 1
    assert dataset.schema_version == "grounded-recommendations-eval-v1"
    assert dataset.source.value == "synthetic"
    assert dataset.workflow_name == "ticket-processing"
    assert dataset.workflow_version == "controlled-support-v1"

    assert len(dataset.cases) == 14
    assert len({case.case_id for case in dataset.cases}) == 14
    assert dataset.content_hash == DATASET_HASH


def test_dataset_contains_required_grounding_and_safety_cases() -> None:
    dataset = load_grounded_recommendation_dataset(DATASET_PATH)

    case_ids = {case.case_id for case in dataset.cases}

    assert "grounded-recommendation-fully-grounded-001" in case_ids
    assert "grounded-recommendation-unsupported-claim-003" in case_ids
    assert "grounded-recommendation-invalid-citation-006" in case_ids
    assert "grounded-recommendation-cross-workspace-009" in case_ids
    assert "grounded-recommendation-ticket-injection-010" in case_ids
    assert "grounded-recommendation-document-injection-011" in case_ids
    assert "grounded-recommendation-human-review-012" in case_ids


def test_context_content_is_committed_with_identity() -> None:
    dataset = load_grounded_recommendation_dataset(DATASET_PATH)

    contexts = tuple(context for case in dataset.cases for context in case.retrieved_contexts)

    assert contexts
    assert all(context.content for context in contexts)
    assert all(len(context.content_sha256) == 64 for context in contexts)


def test_cross_workspace_case_declares_exact_foreign_count() -> None:
    dataset = load_grounded_recommendation_dataset(DATASET_PATH)

    case = next(
        case
        for case in dataset.cases
        if case.case_id == "grounded-recommendation-cross-workspace-009"
    )

    actual_foreign_count = sum(
        context.workspace_id != case.workspace_id for context in case.retrieved_contexts
    )

    assert actual_foreign_count == 1
    assert case.expected_foreign_workspace_evidence_count == 1


def test_expected_citation_must_reference_retrieved_context() -> None:
    dataset = load_grounded_recommendation_dataset(DATASET_PATH)

    payload = dataset.cases[0].model_dump(mode="json")
    payload["expected_citation_chunk_ids"] = ["ffffffff-ffff-4fff-8fff-ffffffffffff"]

    with pytest.raises(
        ValidationError,
        match="expected citations must reference retrieved context chunks",
    ):
        GroundedRecommendationEvaluationCase.model_validate(payload)


def test_foreign_workspace_count_must_match_contexts() -> None:
    dataset = load_grounded_recommendation_dataset(DATASET_PATH)

    payload = dataset.cases[0].model_dump(mode="json")
    payload["expected_foreign_workspace_evidence_count"] = 1

    with pytest.raises(
        ValidationError,
        match="foreign-workspace count",
    ):
        GroundedRecommendationEvaluationCase.model_validate(payload)


def test_escalation_requires_human_review() -> None:
    dataset = load_grounded_recommendation_dataset(DATASET_PATH)

    escalation_case = next(
        case for case in dataset.cases if case.expected_action.value == "recommend_escalation"
    )

    payload = escalation_case.model_dump(mode="json")
    payload["expected_requires_human_review"] = False

    with pytest.raises(
        ValidationError,
        match="escalation recommendations must require human review",
    ):
        GroundedRecommendationEvaluationCase.model_validate(payload)


def test_insufficient_evidence_cannot_expect_direct_response() -> None:
    dataset = load_grounded_recommendation_dataset(DATASET_PATH)

    abstention_case = next(case for case in dataset.cases if not case.expected_evidence_sufficient)

    payload = abstention_case.model_dump(mode="json")
    payload["expected_action"] = "respond"

    with pytest.raises(
        ValidationError,
        match="insufficient evidence cannot expect a direct response",
    ):
        GroundedRecommendationEvaluationCase.model_validate(payload)


def test_empty_dataset_is_rejected(tmp_path: Path) -> None:
    empty_dataset = tmp_path / "empty.jsonl"
    empty_dataset.write_text("", encoding="utf-8")

    with pytest.raises(
        GroundedRecommendationDatasetError,
        match="dataset must not be empty",
    ):
        load_grounded_recommendation_dataset(empty_dataset)


def test_invalid_json_reports_line_number(tmp_path: Path) -> None:
    invalid_dataset = tmp_path / "invalid.jsonl"
    invalid_dataset.write_text(
        '{"dataset_id": "valid"}\nnot-json\n',
        encoding="utf-8",
    )

    with pytest.raises(
        GroundedRecommendationDatasetError,
        match="invalid dataset line 1",
    ):
        load_grounded_recommendation_dataset(invalid_dataset)
