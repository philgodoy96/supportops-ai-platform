"""Canonical loading and provenance for classification datasets."""

import hashlib
import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from pydantic import ValidationError

from supportops.evaluation.ticket_classification.models import (
    TicketClassificationEvaluationCase,
)

TICKET_CLASSIFICATION_EVALUATION_DATASET_ID = "ticket-classification-eval"
TICKET_CLASSIFICATION_EVALUATION_DATASET_VERSION = 1

_DATASET_IDENTIFIER_PATTERN = re.compile(
    r"^[a-z0-9]+(?:-[a-z0-9]+)*$",
)


class TicketClassificationDatasetError(ValueError):
    """Raised when an evaluation dataset cannot be trusted."""


class DuplicateTicketClassificationEvaluationCaseError(
    TicketClassificationDatasetError,
):
    """Raised when one case ID appears more than once."""


@dataclass(frozen=True, slots=True)
class TicketClassificationEvaluationDataset:
    """Immutable validated dataset with deterministic provenance."""

    dataset_id: str
    version: int
    content_hash: str
    cases: tuple[TicketClassificationEvaluationCase, ...]

    def __post_init__(self) -> None:
        _validate_dataset_identifier(self.dataset_id)

        if self.version <= 0:
            raise ValueError(
                "Dataset version must be positive.",
            )

        if not self.cases:
            raise ValueError(
                "Ticket classification dataset must contain at least one case.",
            )

        duplicate_case_id = _find_duplicate_case_id(
            self.cases,
        )
        if duplicate_case_id is not None:
            raise (
                DuplicateTicketClassificationEvaluationCaseError(
                    f"Duplicate ticket classification evaluation case ID: {duplicate_case_id}.",
                )
            )

        expected_content_hash = compute_ticket_classification_dataset_content_hash(
            self.cases,
        )
        if self.content_hash != expected_content_hash:
            raise ValueError(
                "Dataset content_hash does not match the canonical case content.",
            )

    @property
    def case_count(self) -> int:
        """Return the number of validated evaluation cases."""

        return len(self.cases)


def load_ticket_classification_dataset(
    path: Path,
    *,
    dataset_id: str,
    version: int,
) -> TicketClassificationEvaluationDataset:
    """Load and validate one JSONL evaluation dataset."""

    try:
        raw_content = path.read_text(
            encoding="utf-8",
        )
    except (OSError, UnicodeError) as error:
        raise TicketClassificationDatasetError(
            "Ticket classification dataset could not be read.",
        ) from error

    lines = raw_content.splitlines()
    if not lines:
        raise TicketClassificationDatasetError(
            "Ticket classification dataset must contain at least one case.",
        )

    cases: list[TicketClassificationEvaluationCase] = []
    seen_case_ids: set[str] = set()

    for line_number, line in enumerate(
        lines,
        start=1,
    ):
        if not line.strip():
            raise TicketClassificationDatasetError(
                f"Ticket classification dataset line {line_number} must not be blank.",
            )

        try:
            payload = json.loads(line)
        except json.JSONDecodeError as error:
            raise TicketClassificationDatasetError(
                f"Ticket classification dataset line {line_number} is not valid JSON.",
            ) from error

        try:
            case = TicketClassificationEvaluationCase.model_validate(payload)
        except ValidationError as error:
            raise TicketClassificationDatasetError(
                "Ticket classification dataset line "
                f"{line_number} does not match "
                "the evaluation case contract.",
            ) from error

        if case.case_id in seen_case_ids:
            raise (
                DuplicateTicketClassificationEvaluationCaseError(
                    f"Duplicate ticket classification evaluation case ID: {case.case_id}.",
                )
            )

        seen_case_ids.add(case.case_id)
        cases.append(case)

    immutable_cases = tuple(cases)

    return TicketClassificationEvaluationDataset(
        dataset_id=dataset_id,
        version=version,
        content_hash=(
            compute_ticket_classification_dataset_content_hash(
                immutable_cases,
            )
        ),
        cases=immutable_cases,
    )


def compute_ticket_classification_dataset_content_hash(
    cases: Sequence[TicketClassificationEvaluationCase],
) -> str:
    """Hash canonical JSONL content while preserving case order."""

    canonical_content = "".join(f"{_canonical_case_json(case)}\n" for case in cases)

    return hashlib.sha256(
        canonical_content.encode("utf-8"),
    ).hexdigest()


def _canonical_case_json(
    case: TicketClassificationEvaluationCase,
) -> str:
    return json.dumps(
        case.model_dump(
            mode="json",
        ),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _find_duplicate_case_id(
    cases: Sequence[TicketClassificationEvaluationCase],
) -> str | None:
    seen_case_ids: set[str] = set()

    for case in cases:
        if case.case_id in seen_case_ids:
            return case.case_id

        seen_case_ids.add(case.case_id)

    return None


def _validate_dataset_identifier(
    value: str,
) -> None:
    if not value:
        raise ValueError(
            "Dataset ID is required.",
        )

    if value != value.strip():
        raise ValueError(
            "Dataset ID must not contain surrounding whitespace.",
        )

    if len(value) > 128:
        raise ValueError(
            "Dataset ID exceeds the maximum length.",
        )

    if _DATASET_IDENTIFIER_PATTERN.fullmatch(value) is None:
        raise ValueError(
            "Dataset ID must use lowercase kebab-case.",
        )
