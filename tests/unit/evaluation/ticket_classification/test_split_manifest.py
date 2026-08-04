from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from supportops.evaluation.contracts.manifest import EvaluationSplit
from supportops.evaluation.ticket_classification.split_manifest import (
    SplitManifestCaseAllocationError,
    SplitManifestDatasetMismatchError,
    TicketClassificationSplitManifest,
    load_ticket_classification_split_manifest,
)

PROJECT_ROOT = Path(__file__).resolve().parents[4]

DATASET_PATH = (
    PROJECT_ROOT
    / "evals"
    / "ticket-classification"
    / "datasets"
    / "ticket-classification-eval-v1.jsonl"
)
SPLIT_MANIFEST_PATH = (
    PROJECT_ROOT
    / "evals"
    / "ticket-classification"
    / "splits"
    / "ticket-classification-eval-v1-splits-v1.json"
)

DATASET_ID = "ticket-classification-eval"
DATASET_VERSION = 1
DATASET_HASH = "a42445dff9ded6c5d7f73c3f2704cc065a445c06ebb1a1a4ad36fa46dcce984b"
SPLIT_MANIFEST_HASH = "5f78658a3bce4146a243b7fe316efd96db96bc464edefb99aa9722586f571d69"


def load_dataset_case_ids() -> tuple[str, ...]:
    with DATASET_PATH.open("r", encoding="utf-8") as dataset_file:
        return tuple(json.loads(line)["case_id"] for line in dataset_file if line.strip())


def load_committed_manifest() -> TicketClassificationSplitManifest:
    return load_ticket_classification_split_manifest(SPLIT_MANIFEST_PATH)


def test_committed_split_manifest_has_stable_identity_and_hash() -> None:
    manifest = load_committed_manifest()

    assert manifest.split_manifest_id == "ticket-classification-eval-splits"
    assert manifest.split_manifest_version == 1
    assert manifest.dataset_id == DATASET_ID
    assert manifest.dataset_version == DATASET_VERSION
    assert manifest.dataset_hash == DATASET_HASH
    assert manifest.source.value == "synthetic"
    assert manifest.content_hash() == SPLIT_MANIFEST_HASH


def test_committed_split_manifest_has_expected_split_sizes() -> None:
    manifest = load_committed_manifest()

    assert len(manifest.assignments.development) == 12
    assert len(manifest.assignments.holdout) == 8
    assert len(manifest.assignments.safety_gate) == 4
    assert len(manifest.assignments.all_case_ids()) == 24


def test_committed_split_manifest_exactly_covers_dataset() -> None:
    manifest = load_committed_manifest()
    dataset_case_ids = load_dataset_case_ids()

    manifest.validate_dataset_binding(
        dataset_id=DATASET_ID,
        dataset_version=DATASET_VERSION,
        dataset_hash=DATASET_HASH,
        dataset_case_ids=dataset_case_ids,
    )

    assert set(manifest.assignments.all_case_ids()) == set(dataset_case_ids)


def test_committed_safety_gate_allocation_is_frozen() -> None:
    manifest = load_committed_manifest()

    assert manifest.assignments.safety_gate == (
        "service-incident-global-outage-003",
        "security-exposed-api-key-012",
        "security-prompt-injection-019",
        "security-data-deletion-request-024",
    )


def test_split_lookup_returns_explicit_assignment() -> None:
    manifest = load_committed_manifest()

    assert (
        manifest.assignments.split_for_case("billing-angry-low-impact-007")
        is EvaluationSplit.DEVELOPMENT
    )
    assert (
        manifest.assignments.split_for_case("account-access-executive-lockout-020")
        is EvaluationSplit.HOLDOUT
    )
    assert (
        manifest.assignments.split_for_case("security-exposed-api-key-012")
        is EvaluationSplit.SAFETY_GATE
    )


def test_split_lookup_rejects_unknown_case_id() -> None:
    manifest = load_committed_manifest()

    with pytest.raises(KeyError, match="is not allocated"):
        manifest.assignments.split_for_case("unknown-case-999")


def test_split_manifest_rejects_duplicate_case_in_one_split() -> None:
    manifest = load_committed_manifest()
    payload = manifest.model_dump(mode="json")
    payload["assignments"]["development"].append(payload["assignments"]["development"][0])

    with pytest.raises(ValidationError, match="duplicate case IDs"):
        TicketClassificationSplitManifest.model_validate(payload)


def test_split_manifest_rejects_cross_split_duplicate() -> None:
    manifest = load_committed_manifest()
    payload = manifest.model_dump(mode="json")
    payload["assignments"]["holdout"].append(payload["assignments"]["development"][0])

    with pytest.raises(ValidationError, match="allocated to both"):
        TicketClassificationSplitManifest.model_validate(payload)


def test_dataset_binding_rejects_provenance_mismatch() -> None:
    manifest = load_committed_manifest()

    with pytest.raises(
        SplitManifestDatasetMismatchError,
        match="dataset_hash",
    ):
        manifest.validate_dataset_binding(
            dataset_id=DATASET_ID,
            dataset_version=DATASET_VERSION,
            dataset_hash="0" * 64,
            dataset_case_ids=load_dataset_case_ids(),
        )


def test_dataset_binding_rejects_missing_case_allocation() -> None:
    manifest = load_committed_manifest()
    dataset_case_ids = (*load_dataset_case_ids(), "new-case-999")

    with pytest.raises(
        SplitManifestCaseAllocationError,
        match="missing case IDs: new-case-999",
    ):
        manifest.validate_dataset_binding(
            dataset_id=DATASET_ID,
            dataset_version=DATASET_VERSION,
            dataset_hash=DATASET_HASH,
            dataset_case_ids=dataset_case_ids,
        )


def test_dataset_binding_rejects_unknown_manifest_case() -> None:
    manifest = load_committed_manifest()
    dataset_case_ids = tuple(
        case_id for case_id in load_dataset_case_ids() if case_id != "billing-positive-feedback-023"
    )

    with pytest.raises(
        SplitManifestCaseAllocationError,
        match="unknown case IDs: billing-positive-feedback-023",
    ):
        manifest.validate_dataset_binding(
            dataset_id=DATASET_ID,
            dataset_version=DATASET_VERSION,
            dataset_hash=DATASET_HASH,
            dataset_case_ids=dataset_case_ids,
        )


def test_dataset_binding_rejects_duplicate_dataset_case_ids() -> None:
    manifest = load_committed_manifest()
    dataset_case_ids = load_dataset_case_ids()

    with pytest.raises(
        SplitManifestCaseAllocationError,
        match="dataset_case_ids contains duplicate",
    ):
        manifest.validate_dataset_binding(
            dataset_id=DATASET_ID,
            dataset_version=DATASET_VERSION,
            dataset_hash=DATASET_HASH,
            dataset_case_ids=(*dataset_case_ids, dataset_case_ids[0]),
        )
