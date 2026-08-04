from datetime import datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from supportops.evaluation.contracts.manifest import (
    EvaluationManifest,
    EvaluationRunStatus,
    EvaluationSplit,
)

DATASET_HASH = "a" * 64
PROMPT_HASH = "b" * 64
PREDICTION_HASH = "c" * 64
SPLIT_HASH = "d" * 64


def build_manifest(**overrides: object) -> EvaluationManifest:
    values: dict[str, object] = {
        "evaluation_id": "ticket-classification-baseline",
        "evaluation_version": 1,
        "dataset_id": "ticket-classification-eval",
        "dataset_version": 1,
        "dataset_hash": DATASET_HASH,
        "split_manifest_id": "ticket-classification-splits",
        "split_manifest_version": 1,
        "split_manifest_hash": SPLIT_HASH,
        "split": EvaluationSplit.DEVELOPMENT,
        "system_provider": "openai",
        "system_model": "example-system-model",
        "workflow_name": None,
        "workflow_version": None,
        "prompt_id": "ticket-classification",
        "prompt_version": 1,
        "prompt_hash": PROMPT_HASH,
        "schema_version": "ticket-classification-v1",
        "embedding_provider": None,
        "embedding_model": None,
        "embedding_dimensions": None,
        "retrieval_profile": None,
        "evaluator_provider": None,
        "evaluator_model": None,
        "evaluator_embedding_model": None,
        "ragas_version": None,
        "pricing_catalog_version": "supportops-pricing-2026-08-01",
        "capture_timestamp": datetime(
            2026,
            8,
            4,
            20,
            0,
            tzinfo=timezone(timedelta(hours=-3)),
        ),
        "git_commit": "1" * 40,
        "prediction_hash": PREDICTION_HASH,
        "run_status": EvaluationRunStatus.COMPLETE,
    }
    values.update(overrides)
    return EvaluationManifest.model_validate(values)


def test_manifest_serializes_explicit_nulls_and_normalizes_timestamp() -> None:
    manifest = build_manifest()

    payload = manifest.canonical_payload()

    assert payload["workflow_name"] is None
    assert payload["evaluator_provider"] is None
    assert payload["capture_timestamp"] == "2026-08-04T23:00:00Z"


def test_manifest_content_hash_is_deterministic() -> None:
    first = build_manifest()
    second = build_manifest()

    assert first.content_hash() == second.content_hash()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("prompt_version", None),
        ("prompt_hash", None),
        ("prompt_id", None),
    ],
)
def test_manifest_rejects_partial_prompt_provenance(
    field: str,
    value: object,
) -> None:
    with pytest.raises(ValidationError, match="prompt provenance"):
        build_manifest(**{field: value})


def test_manifest_rejects_split_without_split_manifest() -> None:
    with pytest.raises(ValidationError, match="split requires"):
        build_manifest(
            split_manifest_id=None,
            split_manifest_version=None,
            split_manifest_hash=None,
        )


def test_manifest_rejects_naive_capture_timestamp() -> None:
    with pytest.raises(ValidationError):
        build_manifest(capture_timestamp=datetime(2026, 8, 4, 23, 0))
