from pathlib import Path

import pytest
from pydantic import ValidationError

from supportops.evaluation.controlled_support.dataset import (
    load_controlled_support_dataset,
)
from supportops.evaluation.controlled_support.models import (
    ControlledSupportEvaluationCase,
)

PROJECT_ROOT = Path(__file__).resolve().parents[4]
DATASET_PATH = (
    PROJECT_ROOT / "evals" / "controlled-support" / "datasets" / "controlled-support-eval-v1.jsonl"
)

DATASET_HASH = "51789ed08f967d31275d4a583b81a74cd4a6766ba03e6afa1949bc70f3a059b9"


def test_committed_dataset_is_complete_and_immutable() -> None:
    dataset = load_controlled_support_dataset(DATASET_PATH)

    assert dataset.dataset_id == "controlled-support-eval"
    assert dataset.dataset_version == 1
    assert dataset.schema_version == "controlled-support-eval-v1"
    assert dataset.workflow_name == "ticket-processing"
    assert dataset.workflow_version == "controlled-support-v1"
    assert len(dataset.cases) == 14
    assert len({case.case_id for case in dataset.cases}) == 14
    assert dataset.content_hash == DATASET_HASH


def test_dataset_contains_required_safety_cases() -> None:
    dataset = load_controlled_support_dataset(DATASET_PATH)
    case_ids = {case.case_id for case in dataset.cases}

    assert "controlled-support-forbidden-sensitive-tool-003" in case_ids
    assert "controlled-support-repeated-tool-attempt-005" in case_ids
    assert "controlled-support-invalid-citation-009" in case_ids
    assert "controlled-support-ticket-prompt-injection-013" in case_ids
    assert "controlled-support-document-prompt-injection-014" in case_ids


def test_completed_case_rejects_expected_error() -> None:
    dataset = load_controlled_support_dataset(DATASET_PATH)
    payload = dataset.cases[0].model_dump(mode="json")
    payload["expected_error_code"] = "unexpected_error"

    with pytest.raises(
        ValidationError,
        match="completed cases cannot declare an expected error",
    ):
        ControlledSupportEvaluationCase.model_validate(payload)


def test_required_and_forbidden_tools_cannot_overlap() -> None:
    dataset = load_controlled_support_dataset(DATASET_PATH)
    payload = dataset.cases[0].model_dump(mode="json")
    payload["forbidden_tool_calls"] = ["search_knowledge"]

    with pytest.raises(
        ValidationError,
        match="must not overlap",
    ):
        ControlledSupportEvaluationCase.model_validate(payload)
