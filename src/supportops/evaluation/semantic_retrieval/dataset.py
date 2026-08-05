from __future__ import annotations

import json
from pathlib import Path

from supportops.evaluation.contracts.hashing import (
    canonical_json_bytes,
    sha256_bytes,
)
from supportops.evaluation.semantic_retrieval.models import (
    SemanticRetrievalEvaluationCase,
    SemanticRetrievalEvaluationDataset,
)


class SemanticRetrievalDatasetError(ValueError):
    """Raised when a semantic-retrieval dataset is invalid."""


def load_semantic_retrieval_dataset(
    path: Path,
) -> SemanticRetrievalEvaluationDataset:
    """Load, validate, and hash a canonical JSONL evaluation dataset."""

    cases: list[SemanticRetrievalEvaluationCase] = []
    canonical_lines: list[bytes] = []

    with path.open("r", encoding="utf-8") as dataset_file:
        for line_number, raw_line in enumerate(dataset_file, start=1):
            if not raw_line.strip():
                continue

            try:
                payload = json.loads(raw_line)
                case = SemanticRetrievalEvaluationCase.model_validate(payload)
            except (json.JSONDecodeError, ValueError) as exc:
                raise SemanticRetrievalDatasetError(
                    f"invalid dataset line {line_number}: {exc}"
                ) from exc

            cases.append(case)
            canonical_lines.append(
                canonical_json_bytes(case.model_dump(mode="json", exclude_none=False)) + b"\n"
            )

    if not cases:
        raise SemanticRetrievalDatasetError("dataset must not be empty")

    first = cases[0]
    content_hash = sha256_bytes(b"".join(canonical_lines))

    return SemanticRetrievalEvaluationDataset(
        dataset_id=first.dataset_id,
        dataset_version=first.dataset_version,
        schema_version=first.schema_version,
        source=first.source,
        cases=tuple(cases),
        content_hash=content_hash,
    )
