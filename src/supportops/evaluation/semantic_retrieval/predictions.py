from __future__ import annotations

import json
from pathlib import Path

from pydantic import TypeAdapter

from supportops.evaluation.contracts.hashing import (
    canonical_json_bytes,
    sha256_bytes,
)
from supportops.evaluation.contracts.predictions import (
    EvaluationPredictionEnvelope,
)
from supportops.evaluation.semantic_retrieval.models import (
    SemanticRetrievalPredictionPayload,
)

SemanticRetrievalPrediction = EvaluationPredictionEnvelope[SemanticRetrievalPredictionPayload]

_PREDICTION_ADAPTER = TypeAdapter(SemanticRetrievalPrediction)


class SemanticRetrievalPredictionError(ValueError):
    """Raised when semantic-retrieval predictions are invalid."""


def load_semantic_retrieval_predictions(
    path: Path,
) -> tuple[tuple[SemanticRetrievalPrediction, ...], str]:
    """Load and hash typed semantic-retrieval prediction JSONL."""

    predictions: list[SemanticRetrievalPrediction] = []
    canonical_lines: list[bytes] = []
    seen_case_ids: set[str] = set()

    with path.open("r", encoding="utf-8") as prediction_file:
        for line_number, raw_line in enumerate(prediction_file, start=1):
            if not raw_line.strip():
                continue

            try:
                payload = json.loads(raw_line)
                prediction = _PREDICTION_ADAPTER.validate_python(payload)
            except (json.JSONDecodeError, ValueError) as exc:
                raise SemanticRetrievalPredictionError(
                    f"invalid prediction line {line_number}: {exc}"
                ) from exc

            if prediction.case_id in seen_case_ids:
                raise SemanticRetrievalPredictionError(
                    f"duplicate prediction case_id: {prediction.case_id}"
                )

            seen_case_ids.add(prediction.case_id)
            predictions.append(prediction)
            canonical_lines.append(
                canonical_json_bytes(prediction.model_dump(mode="json", exclude_none=False)) + b"\n"
            )

    if not predictions:
        raise SemanticRetrievalPredictionError("prediction set must not be empty")

    return (
        tuple(predictions),
        sha256_bytes(b"".join(canonical_lines)),
    )
