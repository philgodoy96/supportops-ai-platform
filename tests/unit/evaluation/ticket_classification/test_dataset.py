"""Unit tests for canonical classification dataset loading."""

import json
from collections.abc import Sequence
from pathlib import Path

import pytest

from supportops.ai.schemas.ticket_classification import (
    TicketCategory,
    TicketIntent,
    TicketSentiment,
    TicketUrgency,
)
from supportops.evaluation.ticket_classification.dataset import (
    TICKET_CLASSIFICATION_EVALUATION_DATASET_ID,
    TICKET_CLASSIFICATION_EVALUATION_DATASET_VERSION,
    DuplicateTicketClassificationEvaluationCaseError,
    TicketClassificationDatasetError,
    load_ticket_classification_dataset,
)

_EXPECTED_V1_CONTENT_HASH = "a42445dff9ded6c5d7f73c3f2704cc065a445c06ebb1a1a4ad36fa46dcce984b"


def _case_payload(
    *,
    case_id: str = "billing-duplicate-charge-001",
) -> dict[str, object]:
    return {
        "case_id": case_id,
        "tags": [
            "billing",
            "individual-impact",
        ],
        "ticket": {
            "subject": "Duplicated invoice charge",
            "description": ("The latest invoice contains the same charge twice."),
        },
        "expected": {
            "category": "billing",
            "intent": "ask_question",
            "urgency": "normal",
            "sentiment": "neutral",
            "requires_human_review": False,
            "schema_version": "ticket-classification-v1",
        },
    }


def _write_jsonl(
    path: Path,
    payloads: Sequence[dict[str, object]],
) -> None:
    path.write_text(
        "\n".join(json.dumps(payload) for payload in payloads) + "\n",
        encoding="utf-8",
    )


def test_loads_valid_jsonl_with_deterministic_provenance(
    tmp_path: Path,
) -> None:
    path = tmp_path / "dataset.jsonl"
    _write_jsonl(
        path,
        (_case_payload(),),
    )

    dataset = load_ticket_classification_dataset(
        path,
        dataset_id="ticket-classification-eval",
        version=1,
    )

    assert dataset.dataset_id == ("ticket-classification-eval")
    assert dataset.version == 1
    assert dataset.case_count == 1
    assert dataset.cases[0].case_id == ("billing-duplicate-charge-001")
    assert len(dataset.content_hash) == 64


def test_content_hash_ignores_json_key_order_and_spacing(
    tmp_path: Path,
) -> None:
    first_path = tmp_path / "first.jsonl"
    second_path = tmp_path / "second.jsonl"

    payload = _case_payload()
    reversed_payload = {
        "expected": payload["expected"],
        "ticket": payload["ticket"],
        "tags": payload["tags"],
        "case_id": payload["case_id"],
    }

    first_path.write_text(
        json.dumps(
            payload,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    second_path.write_text(
        json.dumps(
            reversed_payload,
            separators=(", ", ": "),
            sort_keys=False,
        )
        + "\n",
        encoding="utf-8",
    )

    first = load_ticket_classification_dataset(
        first_path,
        dataset_id="ticket-classification-eval",
        version=1,
    )
    second = load_ticket_classification_dataset(
        second_path,
        dataset_id="ticket-classification-eval",
        version=1,
    )

    assert first.content_hash == second.content_hash


def test_case_order_is_part_of_dataset_hash(
    tmp_path: Path,
) -> None:
    first_path = tmp_path / "first.jsonl"
    second_path = tmp_path / "second.jsonl"

    first_case = _case_payload(
        case_id="billing-case-001",
    )
    second_case = _case_payload(
        case_id="billing-case-002",
    )

    _write_jsonl(
        first_path,
        (
            first_case,
            second_case,
        ),
    )
    _write_jsonl(
        second_path,
        (
            second_case,
            first_case,
        ),
    )

    first = load_ticket_classification_dataset(
        first_path,
        dataset_id="ticket-classification-eval",
        version=1,
    )
    second = load_ticket_classification_dataset(
        second_path,
        dataset_id="ticket-classification-eval",
        version=1,
    )

    assert first.content_hash != second.content_hash


def test_rejects_duplicate_case_ids(
    tmp_path: Path,
) -> None:
    path = tmp_path / "dataset.jsonl"
    _write_jsonl(
        path,
        (
            _case_payload(),
            _case_payload(),
        ),
    )

    with pytest.raises(
        DuplicateTicketClassificationEvaluationCaseError,
        match=("Duplicate ticket classification evaluation case ID"),
    ):
        load_ticket_classification_dataset(
            path,
            dataset_id="ticket-classification-eval",
            version=1,
        )


def test_rejects_invalid_case_contract_with_line_number(
    tmp_path: Path,
) -> None:
    path = tmp_path / "dataset.jsonl"
    path.write_text(
        '{"case_id":"incomplete-case"}\n',
        encoding="utf-8",
    )

    with pytest.raises(
        TicketClassificationDatasetError,
        match="line 1 does not match",
    ):
        load_ticket_classification_dataset(
            path,
            dataset_id="ticket-classification-eval",
            version=1,
        )

    with pytest.raises(
        TicketClassificationDatasetError,
        match="line 1 does not match",
    ):
        load_ticket_classification_dataset(
            path,
            dataset_id="ticket-classification-eval",
            version=1,
        )


def test_rejects_invalid_second_json_line(
    tmp_path: Path,
) -> None:
    path = tmp_path / "dataset.jsonl"
    path.write_text(
        json.dumps(_case_payload()) + "\n" + '{"case_id":\n',
        encoding="utf-8",
    )

    with pytest.raises(
        TicketClassificationDatasetError,
        match="line 2 is not valid JSON",
    ):
        load_ticket_classification_dataset(
            path,
            dataset_id="ticket-classification-eval",
            version=1,
        )


def test_rejects_blank_lines(
    tmp_path: Path,
) -> None:
    path = tmp_path / "dataset.jsonl"
    path.write_text(
        json.dumps(_case_payload()) + "\n\n",
        encoding="utf-8",
    )

    with pytest.raises(
        TicketClassificationDatasetError,
        match="line 2 must not be blank",
    ):
        load_ticket_classification_dataset(
            path,
            dataset_id="ticket-classification-eval",
            version=1,
        )


def test_rejects_empty_dataset(
    tmp_path: Path,
) -> None:
    path = tmp_path / "dataset.jsonl"
    path.write_text(
        "",
        encoding="utf-8",
    )

    with pytest.raises(
        TicketClassificationDatasetError,
        match="must contain at least one case",
    ):
        load_ticket_classification_dataset(
            path,
            dataset_id="ticket-classification-eval",
            version=1,
        )


def test_rejects_unreadable_dataset(
    tmp_path: Path,
) -> None:
    path = tmp_path / "missing.jsonl"

    with pytest.raises(
        TicketClassificationDatasetError,
        match="could not be read",
    ):
        load_ticket_classification_dataset(
            path,
            dataset_id="ticket-classification-eval",
            version=1,
        )


@pytest.mark.parametrize(
    ("dataset_id", "version"),
    [
        (
            "",
            1,
        ),
        (
            "Ticket Classification",
            1,
        ),
        (
            " ticket-classification-eval",
            1,
        ),
        (
            "ticket-classification-eval",
            0,
        ),
    ],
)
def test_rejects_invalid_dataset_identity(
    tmp_path: Path,
    dataset_id: str,
    version: int,
) -> None:
    path = tmp_path / "dataset.jsonl"
    _write_jsonl(
        path,
        (_case_payload(),),
    )

    with pytest.raises(ValueError):
        load_ticket_classification_dataset(
            path,
            dataset_id=dataset_id,
            version=version,
        )


def test_committed_v1_dataset_is_complete_and_immutable() -> None:
    repository_root = Path(__file__).resolve().parents[4]
    path = (
        repository_root
        / "evals"
        / "ticket-classification"
        / "datasets"
        / "ticket-classification-eval-v1.jsonl"
    )

    dataset = load_ticket_classification_dataset(
        path,
        dataset_id=(TICKET_CLASSIFICATION_EVALUATION_DATASET_ID),
        version=(TICKET_CLASSIFICATION_EVALUATION_DATASET_VERSION),
    )

    assert dataset.case_count == 24
    assert dataset.content_hash == (_EXPECTED_V1_CONTENT_HASH)

    assert {case.expected.category for case in dataset.cases} == set(TicketCategory)

    assert {case.expected.intent for case in dataset.cases} == set(TicketIntent)

    assert {case.expected.urgency for case in dataset.cases} == set(TicketUrgency)

    assert {case.expected.sentiment for case in dataset.cases} == set(TicketSentiment)

    assert {case.expected.requires_human_review for case in dataset.cases} == {
        False,
        True,
    }

    tags = {tag for case in dataset.cases for tag in case.tags}

    assert {
        "ambiguous",
        "emotion-vs-impact",
        "human-review",
        "prompt-injection",
        "security",
        "untrusted-input",
    } <= tags
