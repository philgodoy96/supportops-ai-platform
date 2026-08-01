"""Validated prediction artifacts for ticket-classification evaluation."""

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Annotated, Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    TypeAdapter,
    ValidationError,
    model_validator,
)

from supportops.ai.gateway.errors import LLMErrorCode
from supportops.ai.gateway.results import LLMInvocationStatus
from supportops.ai.schemas.ticket_classification import (
    TicketClassificationResult,
)
from supportops.evaluation.ticket_classification.models import (
    EvaluationCaseId,
)

EvaluationProvenanceText = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        min_length=1,
        max_length=255,
    ),
]

EvaluationContentHash = Annotated[
    str,
    StringConstraints(
        pattern=r"^[a-f0-9]{64}$",
    ),
]


class TicketClassificationPredictionError(ValueError):
    """Raised when a prediction artifact cannot be trusted."""


class DuplicateTicketClassificationPredictionError(
    TicketClassificationPredictionError,
):
    """Raised when a case ID occurs more than once."""


class TicketClassificationPredictionProvenance(BaseModel):
    """Prompt and runtime identity shared by one prediction."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    prompt_id: EvaluationProvenanceText
    prompt_version: Annotated[int, Field(gt=0)]
    prompt_content_hash: EvaluationContentHash
    provider: EvaluationProvenanceText
    model: EvaluationProvenanceText


class TicketClassificationPredictionUsage(BaseModel):
    """Provider-reported usage for one logical invocation."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    input_tokens: Annotated[int, Field(ge=0)] | None = None
    cached_input_tokens: Annotated[int, Field(ge=0)] | None = None
    output_tokens: Annotated[int, Field(ge=0)] | None = None
    reasoning_tokens: Annotated[int, Field(ge=0)] | None = None
    total_tokens: Annotated[int, Field(ge=0)] | None = None

    @model_validator(mode="after")
    def validate_token_relationships(self) -> Self:
        """Preserve the production token-usage invariants."""

        if (
            self.cached_input_tokens is not None
            and self.input_tokens is not None
            and self.cached_input_tokens > self.input_tokens
        ):
            raise ValueError(
                "cached_input_tokens cannot exceed input_tokens.",
            )

        if (
            self.reasoning_tokens is not None
            and self.output_tokens is not None
            and self.reasoning_tokens > self.output_tokens
        ):
            raise ValueError(
                "reasoning_tokens cannot exceed output_tokens.",
            )

        if (
            self.input_tokens is not None
            and self.output_tokens is not None
            and self.total_tokens is not None
            and self.total_tokens != self.input_tokens + self.output_tokens
        ):
            raise ValueError(
                "total_tokens must equal input_tokens plus output_tokens.",
            )

        return self


class TicketClassificationPredictionCost(BaseModel):
    """Application-estimated cost for one logical invocation."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    pricing_catalog_version: EvaluationProvenanceText
    pricing_found: bool
    estimated_input_cost_usd: Decimal | None = None
    estimated_cached_input_cost_usd: Decimal | None = None
    estimated_output_cost_usd: Decimal | None = None
    estimated_total_cost_usd: Decimal | None = None

    @model_validator(mode="after")
    def validate_cost_relationships(self) -> Self:
        """Reject invented or internally inconsistent cost data."""

        components = (
            self.estimated_input_cost_usd,
            self.estimated_cached_input_cost_usd,
            self.estimated_output_cost_usd,
        )

        for component in components:
            if component is not None and component < 0:
                raise ValueError(
                    "Estimated cost components must be non-negative.",
                )

        if self.estimated_total_cost_usd is not None and self.estimated_total_cost_usd < 0:
            raise ValueError(
                "Estimated total cost must be non-negative.",
            )

        if not self.pricing_found:
            if any(component is not None for component in components):
                raise ValueError(
                    "Unknown pricing cannot define cost components.",
                )

            if self.estimated_total_cost_usd is not None:
                raise ValueError(
                    "Unknown pricing cannot define total cost.",
                )

            return self

        if all(component is not None for component in components):
            expected_total = sum(
                (component for component in components if component is not None),
                start=Decimal("0"),
            )

            if self.estimated_total_cost_usd != expected_total:
                raise ValueError(
                    "Estimated total cost must equal the stored cost components.",
                )
        elif self.estimated_total_cost_usd is not None:
            raise ValueError(
                "Total cost must remain unknown when a required component is unknown.",
            )

        return self


class TicketClassificationPredictionInvocation(BaseModel):
    """One logical invocation recorded by an evaluation predictor."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    invocation_sequence: Annotated[int, Field(gt=0)]
    status: LLMInvocationStatus
    provider: EvaluationProvenanceText
    model: EvaluationProvenanceText
    usage: TicketClassificationPredictionUsage | None
    cost: TicketClassificationPredictionCost
    latency_ms: Annotated[int, Field(ge=0)]
    error_code: LLMErrorCode | None

    @model_validator(mode="after")
    def validate_status_error_relationship(self) -> Self:
        """Require safe errors only for failed invocations."""

        if self.status is LLMInvocationStatus.SUCCEEDED:
            if self.error_code is not None:
                raise ValueError(
                    "Successful invocations cannot define an error_code.",
                )
        elif self.error_code is None:
            raise ValueError(
                "Failed invocations require an error_code.",
            )

        return self


class _TicketClassificationPredictionBase(BaseModel):
    """Shared contract for successful and failed case predictions."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    case_id: EvaluationCaseId
    provenance: TicketClassificationPredictionProvenance
    invocations: Annotated[
        tuple[TicketClassificationPredictionInvocation, ...],
        Field(min_length=1),
    ]

    @model_validator(mode="after")
    def validate_invocation_history(self) -> Self:
        """Require contiguous traces from one configured runtime."""

        expected_sequences = tuple(
            range(
                1,
                len(self.invocations) + 1,
            ),
        )
        actual_sequences = tuple(invocation.invocation_sequence for invocation in self.invocations)

        if actual_sequences != expected_sequences:
            raise ValueError(
                "Prediction invocation sequences must be contiguous, ordered, and start at one.",
            )

        for invocation in self.invocations:
            if invocation.provider != self.provenance.provider:
                raise ValueError(
                    "Prediction invocation provider must match prediction provenance.",
                )

            if invocation.model != self.provenance.model:
                raise ValueError(
                    "Prediction invocation model must match prediction provenance.",
                )

        return self


class TicketClassificationSuccessfulPrediction(
    _TicketClassificationPredictionBase,
):
    """Accepted structured output for one dataset case."""

    status: Literal["succeeded"]
    output: TicketClassificationResult

    @model_validator(mode="after")
    def validate_successful_history(self) -> Self:
        """Require exactly one final successful invocation."""

        successful_invocations = tuple(
            invocation
            for invocation in self.invocations
            if invocation.status is LLMInvocationStatus.SUCCEEDED
        )

        if len(successful_invocations) != 1:
            raise ValueError(
                "Successful predictions require exactly one successful invocation.",
            )

        if (
            successful_invocations[0].invocation_sequence
            != self.invocations[-1].invocation_sequence
        ):
            raise ValueError(
                "The successful invocation must be the final prediction invocation.",
            )

        return self


class TicketClassificationFailedPrediction(
    _TicketClassificationPredictionBase,
):
    """Normalized final failure for one dataset case."""

    status: Literal["failed"]
    error_code: LLMErrorCode

    @model_validator(mode="after")
    def validate_failed_history(self) -> Self:
        """Require a failed final trace matching the case failure."""

        if any(
            invocation.status is LLMInvocationStatus.SUCCEEDED for invocation in self.invocations
        ):
            raise ValueError(
                "Failed predictions cannot contain a successful invocation.",
            )

        if self.invocations[-1].error_code is not self.error_code:
            raise ValueError(
                "Prediction error_code must match the final invocation error_code.",
            )

        return self


TicketClassificationEvaluationPrediction = Annotated[
    TicketClassificationSuccessfulPrediction | TicketClassificationFailedPrediction,
    Field(discriminator="status"),
]

_PREDICTION_ADAPTER: TypeAdapter[TicketClassificationEvaluationPrediction] = TypeAdapter(
    TicketClassificationEvaluationPrediction,
)


@dataclass(frozen=True, slots=True)
class TicketClassificationPredictionSet:
    """Immutable prediction collection with canonical provenance."""

    content_hash: str
    predictions: tuple[
        TicketClassificationEvaluationPrediction,
        ...,
    ]

    def __post_init__(self) -> None:
        if not self.predictions:
            raise ValueError(
                "Ticket classification predictions must contain at least one case.",
            )

        duplicate_case_id = _find_duplicate_case_id(
            self.predictions,
        )
        if duplicate_case_id is not None:
            raise DuplicateTicketClassificationPredictionError(
                f"Duplicate ticket classification prediction case ID: {duplicate_case_id}.",
            )

        expected_hash = compute_ticket_classification_predictions_content_hash(
            self.predictions,
        )
        if self.content_hash != expected_hash:
            raise ValueError(
                "Prediction content_hash does not match the canonical prediction content.",
            )


def load_ticket_classification_predictions(
    path: Path,
) -> TicketClassificationPredictionSet:
    """Load and validate one JSONL prediction artifact."""

    try:
        raw_content = path.read_text(
            encoding="utf-8",
        )
    except (OSError, UnicodeError) as error:
        raise TicketClassificationPredictionError(
            "Ticket classification predictions could not be read.",
        ) from error

    lines = raw_content.splitlines()
    if not lines:
        raise TicketClassificationPredictionError(
            "Ticket classification predictions must contain at least one case.",
        )

    predictions: list[TicketClassificationEvaluationPrediction] = []
    seen_case_ids: set[str] = set()

    for line_number, line in enumerate(
        lines,
        start=1,
    ):
        if not line.strip():
            raise TicketClassificationPredictionError(
                f"Ticket classification prediction line {line_number} must not be blank.",
            )

        try:
            prediction = _PREDICTION_ADAPTER.validate_json(
                line,
            )
        except ValidationError as error:
            raise TicketClassificationPredictionError(
                "Ticket classification prediction line "
                f"{line_number} does not match "
                "the prediction contract.",
            ) from error

        if prediction.case_id in seen_case_ids:
            raise DuplicateTicketClassificationPredictionError(
                f"Duplicate ticket classification prediction case ID: {prediction.case_id}.",
            )

        seen_case_ids.add(prediction.case_id)
        predictions.append(prediction)

    immutable_predictions = tuple(predictions)

    return TicketClassificationPredictionSet(
        content_hash=(
            compute_ticket_classification_predictions_content_hash(
                immutable_predictions,
            )
        ),
        predictions=immutable_predictions,
    )


def compute_ticket_classification_predictions_content_hash(
    predictions: Sequence[TicketClassificationEvaluationPrediction],
) -> str:
    """Hash canonical JSONL while preserving artifact order."""

    canonical_content = "".join(
        f"{_canonical_prediction_json(prediction)}\n" for prediction in predictions
    )

    return hashlib.sha256(
        canonical_content.encode("utf-8"),
    ).hexdigest()


def _canonical_prediction_json(
    prediction: TicketClassificationEvaluationPrediction,
) -> str:
    return json.dumps(
        prediction.model_dump(
            mode="json",
        ),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _find_duplicate_case_id(
    predictions: Sequence[TicketClassificationEvaluationPrediction],
) -> str | None:
    seen_case_ids: set[str] = set()

    for prediction in predictions:
        if prediction.case_id in seen_case_ids:
            return prediction.case_id

        seen_case_ids.add(prediction.case_id)

    return None
