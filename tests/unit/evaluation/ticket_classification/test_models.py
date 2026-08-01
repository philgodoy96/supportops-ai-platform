"""Unit tests for classification evaluation case models."""

from copy import deepcopy

import pytest
from pydantic import ValidationError

from supportops.ai.schemas.ticket_classification import (
    TICKET_CLASSIFICATION_SCHEMA_VERSION,
    TicketCategory,
    TicketIntent,
    TicketSentiment,
    TicketUrgency,
)
from supportops.evaluation.ticket_classification.models import (
    TicketClassificationEvaluationCase,
)


def _case_payload() -> dict[str, object]:
    return {
        "case_id": "billing-duplicate-charge-001",
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
            "schema_version": (TICKET_CLASSIFICATION_SCHEMA_VERSION),
        },
    }


def test_evaluation_case_reuses_production_taxonomy() -> None:
    case = TicketClassificationEvaluationCase.model_validate(
        _case_payload(),
    )

    assert case.case_id == "billing-duplicate-charge-001"
    assert case.tags == (
        "billing",
        "individual-impact",
    )
    assert case.expected.category is TicketCategory.BILLING
    assert case.expected.intent is TicketIntent.ASK_QUESTION
    assert case.expected.urgency is TicketUrgency.NORMAL
    assert case.expected.sentiment is TicketSentiment.NEUTRAL
    assert case.expected.requires_human_review is False
    assert case.expected.schema_version == (TICKET_CLASSIFICATION_SCHEMA_VERSION)


@pytest.mark.parametrize(
    "case_id",
    [
        "",
        "Billing-Case",
        "billing_case",
        " billing-case",
        "billing-case ",
    ],
)
def test_evaluation_case_requires_kebab_case_id(
    case_id: str,
) -> None:
    payload = _case_payload()
    payload["case_id"] = case_id

    with pytest.raises(ValidationError):
        TicketClassificationEvaluationCase.model_validate(
            payload,
        )


def test_evaluation_case_requires_unique_tags() -> None:
    payload = _case_payload()
    payload["tags"] = [
        "billing",
        "billing",
    ]

    with pytest.raises(
        ValidationError,
        match="Evaluation case tags must be unique",
    ):
        TicketClassificationEvaluationCase.model_validate(
            payload,
        )


def test_evaluation_case_rejects_unknown_fields() -> None:
    payload = _case_payload()
    payload["unexpected"] = "value"

    with pytest.raises(ValidationError):
        TicketClassificationEvaluationCase.model_validate(
            payload,
        )


def test_evaluation_ticket_requires_non_empty_content() -> None:
    payload = _case_payload()
    ticket = deepcopy(payload["ticket"])

    assert isinstance(ticket, dict)

    ticket["subject"] = "   "
    payload["ticket"] = ticket

    with pytest.raises(ValidationError):
        TicketClassificationEvaluationCase.model_validate(
            payload,
        )


def test_expected_labels_require_current_schema_version() -> None:
    payload = _case_payload()
    expected = deepcopy(payload["expected"])

    assert isinstance(expected, dict)

    expected["schema_version"] = "ticket-classification-v2"
    payload["expected"] = expected

    with pytest.raises(ValidationError):
        TicketClassificationEvaluationCase.model_validate(
            payload,
        )


def test_human_review_requires_strict_boolean() -> None:
    payload = _case_payload()
    expected = deepcopy(payload["expected"])

    assert isinstance(expected, dict)

    expected["requires_human_review"] = "false"
    payload["expected"] = expected

    with pytest.raises(ValidationError):
        TicketClassificationEvaluationCase.model_validate(
            payload,
        )
