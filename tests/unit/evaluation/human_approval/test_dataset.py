from pathlib import Path

import pytest
from pydantic import ValidationError

from supportops.evaluation.human_approval.dataset import (
    load_human_approval_dataset,
)
from supportops.evaluation.human_approval.models import (
    HumanApprovalEvaluationCase,
)

PROJECT_ROOT = Path(__file__).resolve().parents[4]
DATASET_PATH = (
    PROJECT_ROOT / "evals" / "human-approval" / "datasets" / "human-approval-eval-v1.jsonl"
)

DATASET_HASH = "0c3f88452e21f2e66bb25f926e3ba4b1bc8d40ef20053199af2504053d8dc350"


def test_committed_dataset_is_complete_and_immutable() -> None:
    dataset = load_human_approval_dataset(DATASET_PATH)

    assert dataset.dataset_id == "human-approval-eval"
    assert dataset.dataset_version == 1
    assert dataset.schema_version == "human-approval-eval-v1"
    assert dataset.workflow_name == "ticket-processing"
    assert dataset.workflow_version == "human-approved-support-v1"
    assert len(dataset.cases) == 14
    assert len({case.case_id for case in dataset.cases}) == 14
    assert dataset.content_hash == DATASET_HASH


def test_dataset_contains_required_safety_scenarios() -> None:
    dataset = load_human_approval_dataset(DATASET_PATH)
    case_ids = {case.case_id for case in dataset.cases}

    assert "rejected-action-does-not-execute-004" in case_ids
    assert "expired-action-does-not-execute-005" in case_ids
    assert "checkpoint-mismatch-008" in case_ids
    assert "grant-mismatch-009" in case_ids
    assert "duplicate-sensitive-execution-012" in case_ids


def test_rejected_case_cannot_expect_execution() -> None:
    dataset = load_human_approval_dataset(DATASET_PATH)
    payload = dataset.cases[3].model_dump(mode="json")
    payload["expected_sensitive_executed"] = True
    payload["expected_execution_status"] = "applied"

    with pytest.raises(
        ValidationError,
        match="rejected or expired approval cannot execute",
    ):
        HumanApprovalEvaluationCase.model_validate(payload)


def test_non_executed_case_requires_not_executed_status() -> None:
    dataset = load_human_approval_dataset(DATASET_PATH)
    payload = dataset.cases[0].model_dump(mode="json")
    payload["expected_execution_status"] = "applied"

    with pytest.raises(
        ValidationError,
        match="non-executed cases must use not_executed",
    ):
        HumanApprovalEvaluationCase.model_validate(payload)
