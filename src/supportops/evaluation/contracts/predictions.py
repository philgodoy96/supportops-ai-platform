from __future__ import annotations

from decimal import Decimal
from enum import StrEnum
from typing import Annotated, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    model_validator,
)

from supportops.evaluation.contracts.hashing import sha256_hexdigest

NonEmptyString = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1),
]


class EvaluationPredictionStatus(StrEnum):
    """Execution outcome for one evaluation case prediction."""

    SUCCEEDED = "succeeded"
    FAILED = "failed"


class EvaluationPredictionEnvelope[PayloadT: BaseModel](BaseModel):
    """Shared execution metadata around a typed domain prediction payload."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    case_id: NonEmptyString
    status: EvaluationPredictionStatus
    payload: PayloadT | None = None

    latency_ms: int | None = Field(default=None, ge=0)
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    embedding_tokens: int | None = Field(default=None, ge=0)
    estimated_cost_usd: Decimal | None = Field(default=None, ge=0)

    error_code: NonEmptyString | None = None
    trace_id: NonEmptyString | None = None

    @model_validator(mode="after")
    def validate_status_contract(self) -> Self:
        if self.status is EvaluationPredictionStatus.SUCCEEDED:
            if self.payload is None:
                raise ValueError("succeeded predictions require a payload")
            if self.error_code is not None:
                raise ValueError("succeeded predictions cannot include error_code")
        else:
            if self.payload is not None:
                raise ValueError("failed predictions cannot include a payload")
            if self.error_code is None:
                raise ValueError("failed predictions require error_code")

        return self

    def canonical_payload(self) -> dict[str, object]:
        """Return the prediction envelope with explicit nulls."""

        return dict(self.model_dump(mode="json", exclude_none=False))

    def content_hash(self) -> str:
        """Return the deterministic hash of the prediction envelope."""

        return sha256_hexdigest(self.canonical_payload())
