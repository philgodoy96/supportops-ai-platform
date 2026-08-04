from __future__ import annotations

import json
from collections.abc import Iterable
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    model_validator,
)

from supportops.evaluation.contracts.hashing import sha256_hexdigest
from supportops.evaluation.contracts.manifest import EvaluationSplit

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
Sha256Hex = Annotated[
    str,
    StringConstraints(pattern=r"^[0-9a-f]{64}$"),
]


class EvaluationDatasetSource(StrEnum):
    """Supported provenance declarations for committed evaluation datasets."""

    SYNTHETIC = "synthetic"


class SplitManifestError(ValueError):
    """Base error for split-manifest validation failures."""


class SplitManifestDatasetMismatchError(SplitManifestError):
    """Raised when split-manifest provenance does not match its dataset."""


class SplitManifestCaseAllocationError(SplitManifestError):
    """Raised when split assignments do not exactly cover a dataset."""


class TicketClassificationSplitAssignments(BaseModel):
    """Immutable case allocation for ticket-classification evaluation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    development: tuple[CaseId, ...] = Field(min_length=1)
    holdout: tuple[CaseId, ...] = Field(min_length=1)
    safety_gate: tuple[CaseId, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_unique_case_allocation(self) -> Self:
        allocations = {
            EvaluationSplit.DEVELOPMENT: self.development,
            EvaluationSplit.HOLDOUT: self.holdout,
            EvaluationSplit.SAFETY_GATE: self.safety_gate,
        }

        for split, case_ids in allocations.items():
            if len(case_ids) != len(set(case_ids)):
                raise ValueError(f"{split.value} contains duplicate case IDs")

        seen: dict[str, EvaluationSplit] = {}
        for split, case_ids in allocations.items():
            for case_id in case_ids:
                previous_split = seen.get(case_id)
                if previous_split is not None:
                    raise ValueError(
                        f"case_id {case_id!r} is allocated to both "
                        f"{previous_split.value!r} and {split.value!r}"
                    )
                seen[case_id] = split

        return self

    def case_ids_for(self, split: EvaluationSplit) -> tuple[str, ...]:
        """Return case IDs assigned to one evaluation split."""

        if split is EvaluationSplit.DEVELOPMENT:
            return self.development
        if split is EvaluationSplit.HOLDOUT:
            return self.holdout
        return self.safety_gate

    def all_case_ids(self) -> tuple[str, ...]:
        """Return all assigned case IDs in split declaration order."""

        return self.development + self.holdout + self.safety_gate

    def split_for_case(self, case_id: str) -> EvaluationSplit:
        """Return the split assigned to a known case ID."""

        for split in EvaluationSplit:
            if case_id in self.case_ids_for(split):
                return split

        raise KeyError(f"case_id {case_id!r} is not allocated")


class TicketClassificationSplitManifest(BaseModel):
    """Versioned sidecar split allocation for one immutable dataset."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    split_manifest_id: NonEmptyString
    split_manifest_version: int = Field(ge=1)

    dataset_id: NonEmptyString
    dataset_version: int = Field(ge=1)
    dataset_hash: Sha256Hex

    description: NonEmptyString
    source: EvaluationDatasetSource
    assignments: TicketClassificationSplitAssignments

    def canonical_payload(self) -> dict[str, object]:
        """Return deterministic manifest content with explicit fields."""

        return dict(self.model_dump(mode="json", exclude_none=False))

    def content_hash(self) -> str:
        """Return the deterministic hash of the split manifest."""

        return sha256_hexdigest(self.canonical_payload())

    def validate_dataset_binding(
        self,
        *,
        dataset_id: str,
        dataset_version: int,
        dataset_hash: str,
        dataset_case_ids: Iterable[str],
    ) -> None:
        """Validate provenance and exact case coverage against a dataset."""

        provenance_mismatches: list[str] = []

        if self.dataset_id != dataset_id:
            provenance_mismatches.append(
                f"dataset_id expected {self.dataset_id!r}, received {dataset_id!r}"
            )
        if self.dataset_version != dataset_version:
            provenance_mismatches.append(
                f"dataset_version expected {self.dataset_version}, received {dataset_version}"
            )
        if self.dataset_hash != dataset_hash:
            provenance_mismatches.append(
                f"dataset_hash expected {self.dataset_hash!r}, received {dataset_hash!r}"
            )

        if provenance_mismatches:
            raise SplitManifestDatasetMismatchError("; ".join(provenance_mismatches))

        expected_case_ids = tuple(dataset_case_ids)
        if len(expected_case_ids) != len(set(expected_case_ids)):
            raise SplitManifestCaseAllocationError("dataset_case_ids contains duplicate case IDs")

        expected_case_id_set = set(expected_case_ids)
        assigned_case_id_set = set(self.assignments.all_case_ids())

        missing_case_ids = sorted(expected_case_id_set - assigned_case_id_set)
        unknown_case_ids = sorted(assigned_case_id_set - expected_case_id_set)

        if missing_case_ids or unknown_case_ids:
            details: list[str] = []
            if missing_case_ids:
                details.append(f"missing case IDs: {', '.join(missing_case_ids)}")
            if unknown_case_ids:
                details.append(f"unknown case IDs: {', '.join(unknown_case_ids)}")

            raise SplitManifestCaseAllocationError("; ".join(details))


def load_ticket_classification_split_manifest(
    path: Path,
) -> TicketClassificationSplitManifest:
    """Load and validate a ticket-classification split manifest."""

    with path.open("r", encoding="utf-8") as manifest_file:
        payload = json.load(manifest_file)

    return TicketClassificationSplitManifest.model_validate(payload)
