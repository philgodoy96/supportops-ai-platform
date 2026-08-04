from __future__ import annotations

from datetime import UTC
from enum import StrEnum
from typing import Annotated, Self

from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)

from supportops.evaluation.contracts.hashing import sha256_hexdigest

NonEmptyString = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1),
]
Sha256Hex = Annotated[
    str,
    StringConstraints(pattern=r"^[0-9a-f]{64}$"),
]
GitCommitHash = Annotated[
    str,
    StringConstraints(pattern=r"^[0-9a-f]{7,64}$"),
]


class EvaluationRunStatus(StrEnum):
    """Completion state for an evaluation execution."""

    COMPLETE = "complete"
    INCOMPLETE = "incomplete"
    FAILED = "failed"


class EvaluationSplit(StrEnum):
    """Versioned evaluation split assigned to a case."""

    DEVELOPMENT = "development"
    HOLDOUT = "holdout"
    SAFETY_GATE = "safety_gate"


class EvaluationManifest(BaseModel):
    """Repository-owned provenance for one evaluation execution."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    evaluation_id: NonEmptyString
    evaluation_version: int = Field(ge=1)

    dataset_id: NonEmptyString
    dataset_version: int = Field(ge=1)
    dataset_hash: Sha256Hex

    split_manifest_id: NonEmptyString | None = None
    split_manifest_version: int | None = Field(default=None, ge=1)
    split_manifest_hash: Sha256Hex | None = None
    split: EvaluationSplit | None = None

    system_provider: NonEmptyString | None = None
    system_model: NonEmptyString | None = None

    workflow_name: NonEmptyString | None = None
    workflow_version: NonEmptyString | None = None

    prompt_id: NonEmptyString | None = None
    prompt_version: int | None = Field(default=None, ge=1)
    prompt_hash: Sha256Hex | None = None

    schema_version: NonEmptyString

    embedding_provider: NonEmptyString | None = None
    embedding_model: NonEmptyString | None = None
    embedding_dimensions: int | None = Field(default=None, ge=1)
    retrieval_profile: NonEmptyString | None = None

    evaluator_provider: NonEmptyString | None = None
    evaluator_model: NonEmptyString | None = None
    evaluator_embedding_model: NonEmptyString | None = None
    ragas_version: NonEmptyString | None = None

    pricing_catalog_version: NonEmptyString
    capture_timestamp: AwareDatetime
    git_commit: GitCommitHash
    prediction_hash: Sha256Hex | None = None
    run_status: EvaluationRunStatus

    @field_validator("capture_timestamp")
    @classmethod
    def normalize_capture_timestamp(cls, value: AwareDatetime) -> AwareDatetime:
        """Normalize timestamps so equivalent instants serialize identically."""

        return value.astimezone(UTC)

    @model_validator(mode="after")
    def validate_provenance_groups(self) -> Self:
        self._require_all_or_none(
            "split manifest provenance",
            (
                self.split_manifest_id,
                self.split_manifest_version,
                self.split_manifest_hash,
            ),
        )
        self._require_all_or_none(
            "system model provenance",
            (self.system_provider, self.system_model),
        )
        self._require_all_or_none(
            "workflow provenance",
            (self.workflow_name, self.workflow_version),
        )
        self._require_all_or_none(
            "prompt provenance",
            (self.prompt_id, self.prompt_version, self.prompt_hash),
        )
        self._require_all_or_none(
            "embedding provenance",
            (self.embedding_provider, self.embedding_model),
        )
        self._require_all_or_none(
            "evaluator model provenance",
            (self.evaluator_provider, self.evaluator_model),
        )

        if self.split is not None and self.split_manifest_id is None:
            raise ValueError("split requires complete split manifest provenance")
        if self.embedding_dimensions is not None and self.embedding_model is None:
            raise ValueError("embedding_dimensions requires complete embedding provenance")
        if self.evaluator_embedding_model is not None and self.evaluator_provider is None:
            raise ValueError("evaluator_embedding_model requires evaluator provider provenance")
        if self.ragas_version is not None and self.evaluator_provider is None:
            raise ValueError("ragas_version requires evaluator model provenance")

        return self

    def canonical_payload(self) -> dict[str, object]:
        """Return the manifest with explicit nulls and deterministic field values."""

        return dict(self.model_dump(mode="json", exclude_none=False))

    def content_hash(self) -> str:
        """Return the deterministic hash of the complete manifest."""

        return sha256_hexdigest(self.canonical_payload())

    @staticmethod
    def _require_all_or_none(
        group_name: str,
        values: tuple[object | None, ...],
    ) -> None:
        present_count = sum(value is not None for value in values)
        if present_count not in (0, len(values)):
            raise ValueError(f"{group_name} must be provided together")
