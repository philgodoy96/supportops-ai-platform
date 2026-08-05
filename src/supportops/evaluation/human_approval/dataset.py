from __future__ import annotations

import json
from pathlib import Path

from supportops.evaluation.contracts.hashing import (
    canonical_json_bytes,
    sha256_bytes,
)
from supportops.evaluation.human_approval.models import (
    HumanApprovalEvaluationCase,
    HumanApprovalEvaluationDataset,
)


class HumanApprovalDatasetError(ValueError):
    """Raised when human-approval evaluation data is invalid."""


def load_human_approval_dataset(
    path: Path,
) -> HumanApprovalEvaluationDataset:
    """Load, validate, and hash canonical JSONL approval cases."""

    cases: list[HumanApprovalEvaluationCase] = []
    canonical_lines: list[bytes] = []

    with path.open("r", encoding="utf-8") as dataset_file:
        for line_number, raw_line in enumerate(dataset_file, start=1):
            if not raw_line.strip():
                continue

            try:
                payload = json.loads(raw_line)
                case = HumanApprovalEvaluationCase.model_validate(payload)
            except (json.JSONDecodeError, ValueError) as exc:
                raise HumanApprovalDatasetError(
                    f"invalid dataset line {line_number}: {exc}"
                ) from exc

            cases.append(case)
            canonical_lines.append(
                canonical_json_bytes(case.model_dump(mode="json", exclude_none=False)) + b"\n"
            )

    if not cases:
        raise HumanApprovalDatasetError("dataset must not be empty")

    first = cases[0]

    return HumanApprovalEvaluationDataset(
        dataset_id=first.dataset_id,
        dataset_version=first.dataset_version,
        schema_version=first.schema_version,
        source=first.source,
        workflow_name=first.workflow_name,
        workflow_version=first.workflow_version,
        cases=tuple(cases),
        content_hash=sha256_bytes(b"".join(canonical_lines)),
    )
