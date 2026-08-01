"""Unit tests for the ticket classification output schema."""

from collections.abc import Mapping
from typing import Any

import pytest
from pydantic import ValidationError

from supportops.ai.schemas.ticket_classification import (
    TICKET_CLASSIFICATION_SCHEMA_VERSION,
    TicketCategory,
    TicketClassificationResult,
    TicketIntent,
    TicketSentiment,
    TicketUrgency,
)


def _valid_payload() -> dict[str, object]:
    return {
        "category": "billing",
        "intent": "ask_question",
        "urgency": "normal",
        "sentiment": "neutral",
        "requires_human_review": False,
        "summary": "The customer is asking about an invoice charge.",
        "schema_version": TICKET_CLASSIFICATION_SCHEMA_VERSION,
    }


def _replace(
    payload: Mapping[str, object],
    **changes: object,
) -> dict[str, object]:
    return {
        **payload,
        **changes,
    }


@pytest.mark.parametrize(
    "category",
    list(TicketCategory),
)
def test_accepts_every_supported_category(
    category: TicketCategory,
) -> None:
    result = TicketClassificationResult.model_validate(
        _replace(_valid_payload(), category=category.value),
    )

    assert result.category is category


@pytest.mark.parametrize(
    "intent",
    list(TicketIntent),
)
def test_accepts_every_supported_intent(
    intent: TicketIntent,
) -> None:
    result = TicketClassificationResult.model_validate(
        _replace(_valid_payload(), intent=intent.value),
    )

    assert result.intent is intent


@pytest.mark.parametrize(
    "urgency",
    list(TicketUrgency),
)
def test_accepts_every_supported_urgency(
    urgency: TicketUrgency,
) -> None:
    result = TicketClassificationResult.model_validate(
        _replace(_valid_payload(), urgency=urgency.value),
    )

    assert result.urgency is urgency


@pytest.mark.parametrize(
    "sentiment",
    list(TicketSentiment),
)
def test_accepts_every_supported_sentiment(
    sentiment: TicketSentiment,
) -> None:
    result = TicketClassificationResult.model_validate(
        _replace(_valid_payload(), sentiment=sentiment.value),
    )

    assert result.sentiment is sentiment


@pytest.mark.parametrize(
    ("field_name", "invalid_value"),
    [
        ("category", "unknown_category"),
        ("intent", "unknown_intent"),
        ("urgency", "emergency"),
        ("sentiment", "angry"),
    ],
)
def test_rejects_values_outside_the_bounded_taxonomy(
    field_name: str,
    invalid_value: str,
) -> None:
    with pytest.raises(ValidationError):
        TicketClassificationResult.model_validate(
            _replace(
                _valid_payload(),
                **{field_name: invalid_value},
            ),
        )


def test_normalizes_summary_whitespace() -> None:
    result = TicketClassificationResult.model_validate(
        _replace(
            _valid_payload(),
            summary="  The customer cannot access the account.  ",
        ),
    )

    assert result.summary == "The customer cannot access the account."


@pytest.mark.parametrize(
    "invalid_summary",
    [
        "",
        " ",
        "\n\t",
        "x" * 501,
    ],
)
def test_rejects_invalid_summary(
    invalid_summary: str,
) -> None:
    with pytest.raises(ValidationError):
        TicketClassificationResult.model_validate(
            _replace(
                _valid_payload(),
                summary=invalid_summary,
            ),
        )


def test_accepts_summary_at_maximum_length() -> None:
    result = TicketClassificationResult.model_validate(
        _replace(
            _valid_payload(),
            summary="x" * 500,
        ),
    )

    assert len(result.summary) == 500


@pytest.mark.parametrize(
    "requires_human_review",
    [
        True,
        False,
    ],
)
def test_accepts_explicit_human_review_boolean(
    requires_human_review: bool,
) -> None:
    result = TicketClassificationResult.model_validate(
        _replace(
            _valid_payload(),
            requires_human_review=requires_human_review,
        ),
    )

    assert result.requires_human_review is requires_human_review


@pytest.mark.parametrize(
    "invalid_human_review_value",
    [
        "true",
        "false",
        1,
        0,
    ],
)
def test_rejects_coerced_human_review_values(
    invalid_human_review_value: Any,
) -> None:
    with pytest.raises(ValidationError):
        TicketClassificationResult.model_validate(
            _replace(
                _valid_payload(),
                requires_human_review=invalid_human_review_value,
            ),
        )


def test_requires_explicit_schema_version() -> None:
    payload = _valid_payload()
    del payload["schema_version"]

    with pytest.raises(ValidationError):
        TicketClassificationResult.model_validate(payload)


def test_rejects_unsupported_schema_version() -> None:
    with pytest.raises(ValidationError):
        TicketClassificationResult.model_validate(
            _replace(
                _valid_payload(),
                schema_version="ticket-classification-v2",
            ),
        )


def test_rejects_additional_fields() -> None:
    with pytest.raises(ValidationError):
        TicketClassificationResult.model_validate(
            _replace(
                _valid_payload(),
                confidence=0.98,
            ),
        )


def test_does_not_silently_coerce_unknown_category_to_other() -> None:
    with pytest.raises(ValidationError):
        TicketClassificationResult.model_validate(
            _replace(
                _valid_payload(),
                category="customer_problem",
            ),
        )


def test_validated_result_is_immutable() -> None:
    result = TicketClassificationResult.model_validate(
        _valid_payload(),
    )

    with pytest.raises(ValidationError):
        result.summary = "Changed summary."  # type: ignore[misc]
