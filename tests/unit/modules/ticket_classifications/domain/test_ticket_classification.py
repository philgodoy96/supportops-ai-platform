"""Unit tests for durable accepted ticket classifications."""

from dataclasses import FrozenInstanceError, replace
from datetime import UTC, datetime, timedelta
from typing import cast
from uuid import uuid4

import pytest

from supportops.ai.schemas.ticket_classification import (
    TICKET_CLASSIFICATION_SCHEMA_VERSION,
    TicketCategory,
    TicketClassificationSchemaVersion,
    TicketIntent,
    TicketSentiment,
    TicketUrgency,
)
from supportops.modules.ticket_classifications.domain.models import (
    TicketClassification,
)

_NOW = datetime(
    2026,
    8,
    1,
    17,
    0,
    tzinfo=UTC,
)
_PROMPT_HASH = "a" * 64


def _classification() -> TicketClassification:
    return TicketClassification.create(
        workspace_id=uuid4(),
        ticket_id=uuid4(),
        agent_run_id=uuid4(),
        category=TicketCategory.BILLING,
        intent=TicketIntent.ASK_QUESTION,
        urgency=TicketUrgency.NORMAL,
        sentiment=TicketSentiment.NEUTRAL,
        requires_human_review=False,
        summary="The customer is asking about an invoice.",
        schema_version=(TICKET_CLASSIFICATION_SCHEMA_VERSION),
        prompt_id="ticket-classification",
        prompt_version=1,
        prompt_content_hash=_PROMPT_HASH,
        provider="openai",
        model="gpt-5-nano",
        accepted_invocation_sequence=1,
        now=_NOW,
    )


def test_create_builds_immutable_accepted_classification() -> None:
    workspace_id = uuid4()
    ticket_id = uuid4()
    agent_run_id = uuid4()

    classification = TicketClassification.create(
        workspace_id=workspace_id,
        ticket_id=ticket_id,
        agent_run_id=agent_run_id,
        category=TicketCategory.SECURITY,
        intent=TicketIntent.REPORT_INCIDENT,
        urgency=TicketUrgency.HIGH,
        sentiment=TicketSentiment.NEGATIVE,
        requires_human_review=True,
        summary="  Possible unauthorized account activity.  ",
        schema_version=(TICKET_CLASSIFICATION_SCHEMA_VERSION),
        prompt_id="ticket-classification",
        prompt_version=1,
        prompt_content_hash=_PROMPT_HASH,
        provider="openai",
        model="gpt-5-nano",
        accepted_invocation_sequence=2,
        now=_NOW,
    )

    assert classification.workspace_id == workspace_id
    assert classification.ticket_id == ticket_id
    assert classification.agent_run_id == agent_run_id
    assert classification.category is TicketCategory.SECURITY
    assert classification.intent is TicketIntent.REPORT_INCIDENT
    assert classification.urgency is TicketUrgency.HIGH
    assert classification.sentiment is TicketSentiment.NEGATIVE
    assert classification.requires_human_review is True
    assert classification.summary == ("Possible unauthorized account activity.")
    assert classification.prompt_version == 1
    assert classification.accepted_invocation_sequence == 2
    assert classification.created_at == _NOW
    assert classification.updated_at == _NOW

    with pytest.raises(FrozenInstanceError):
        classification.summary = "Changed summary."  # type: ignore[misc]


@pytest.mark.parametrize(
    "prompt_content_hash",
    [
        "",
        "a" * 63,
        "a" * 65,
        "A" * 64,
        "z" * 64,
    ],
)
def test_rejects_invalid_prompt_content_hash(
    prompt_content_hash: str,
) -> None:
    with pytest.raises(
        ValueError,
        match="lowercase SHA-256",
    ):
        replace(
            _classification(),
            prompt_content_hash=prompt_content_hash,
        )


@pytest.mark.parametrize(
    "prompt_version",
    [
        0,
        -1,
    ],
)
def test_rejects_non_positive_prompt_version(
    prompt_version: int,
) -> None:
    with pytest.raises(
        ValueError,
        match="prompt_version must be positive",
    ):
        replace(
            _classification(),
            prompt_version=prompt_version,
        )


@pytest.mark.parametrize(
    "accepted_invocation_sequence",
    [
        0,
        -1,
    ],
)
def test_rejects_non_positive_accepted_invocation_sequence(
    accepted_invocation_sequence: int,
) -> None:
    with pytest.raises(
        ValueError,
        match=("accepted_invocation_sequence must be positive"),
    ):
        replace(
            _classification(),
            accepted_invocation_sequence=(accepted_invocation_sequence),
        )


def test_rejects_unsupported_schema_version() -> None:
    unsupported_version = cast(
        TicketClassificationSchemaVersion,
        "ticket-classification-v2",
    )

    with pytest.raises(
        ValueError,
        match="schema_version must be",
    ):
        replace(
            _classification(),
            schema_version=unsupported_version,
        )


def test_rejects_non_boolean_human_review_value() -> None:
    with pytest.raises(
        ValueError,
        match="must be a boolean",
    ):
        replace(
            _classification(),
            requires_human_review=cast(bool, 1),
        )


def test_rejects_raw_category_string() -> None:
    with pytest.raises(
        ValueError,
        match="category must use the supported taxonomy",
    ):
        replace(
            _classification(),
            category=cast(TicketCategory, "billing"),
        )


def test_rejects_raw_intent_string() -> None:
    with pytest.raises(
        ValueError,
        match="intent must use the supported taxonomy",
    ):
        replace(
            _classification(),
            intent=cast(TicketIntent, "ask_question"),
        )


def test_rejects_raw_urgency_string() -> None:
    with pytest.raises(
        ValueError,
        match="urgency must use the supported taxonomy",
    ):
        replace(
            _classification(),
            urgency=cast(TicketUrgency, "normal"),
        )


def test_rejects_raw_sentiment_string() -> None:
    with pytest.raises(
        ValueError,
        match="sentiment must use the supported taxonomy",
    ):
        replace(
            _classification(),
            sentiment=cast(TicketSentiment, "neutral"),
        )


@pytest.mark.parametrize(
    "summary",
    [
        "",
        " ",
        " surrounding whitespace ",
        "x" * 501,
    ],
)
def test_rejects_invalid_summary(
    summary: str,
) -> None:
    with pytest.raises(ValueError):
        replace(
            _classification(),
            summary=summary,
        )


def test_rejects_modified_timestamp_for_immutable_record() -> None:
    with pytest.raises(
        ValueError,
        match="updated_at must equal created_at",
    ):
        replace(
            _classification(),
            updated_at=_NOW + timedelta(seconds=1),
        )


def test_rejects_non_utc_timestamp() -> None:
    with pytest.raises(
        ValueError,
        match="created_at must be a UTC-aware timestamp",
    ):
        replace(
            _classification(),
            created_at=datetime(
                2026,
                8,
                1,
                17,
                0,
            ),
            updated_at=datetime(
                2026,
                8,
                1,
                17,
                0,
            ),
        )
