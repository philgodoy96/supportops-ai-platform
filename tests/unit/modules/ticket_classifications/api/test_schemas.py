"""Unit tests for classification inspection HTTP schemas."""

from datetime import UTC, datetime
from uuid import UUID

from supportops.ai.schemas.ticket_classification import (
    TICKET_CLASSIFICATION_SCHEMA_VERSION,
    TicketCategory,
    TicketIntent,
    TicketSentiment,
    TicketUrgency,
)
from supportops.modules.ticket_classifications.api.schemas import (
    TicketClassificationResponse,
)
from supportops.modules.ticket_classifications.domain.models import (
    TicketClassification,
)

_NOW = datetime(
    2026,
    8,
    1,
    21,
    0,
    tzinfo=UTC,
)
_WORKSPACE_ID = UUID(
    "11111111-1111-4111-8111-111111111111",
)
_TICKET_ID = UUID(
    "22222222-2222-4222-8222-222222222222",
)
_AGENT_RUN_ID = UUID(
    "33333333-3333-4333-8333-333333333333",
)
_INVOCATION_ID = UUID(
    "44444444-4444-4444-8444-444444444444",
)
_CLASSIFICATION_ID = UUID(
    "55555555-5555-4555-8555-555555555555",
)
_PROMPT_HASH = "a" * 64


def _classification() -> TicketClassification:
    return TicketClassification.create(
        classification_id=_CLASSIFICATION_ID,
        workspace_id=_WORKSPACE_ID,
        ticket_id=_TICKET_ID,
        agent_run_id=_AGENT_RUN_ID,
        accepted_llm_invocation_id=_INVOCATION_ID,
        category=TicketCategory.BILLING,
        intent=TicketIntent.ASK_QUESTION,
        urgency=TicketUrgency.NORMAL,
        sentiment=TicketSentiment.NEUTRAL,
        requires_human_review=False,
        summary=("The customer is asking about a duplicated invoice charge."),
        schema_version=(TICKET_CLASSIFICATION_SCHEMA_VERSION),
        prompt_id="ticket-classification",
        prompt_version=1,
        prompt_content_hash=_PROMPT_HASH,
        provider="mock",
        model="mock-ticket-classifier-v1",
        now=_NOW,
    )


def test_classification_response_projects_public_provenance() -> None:
    response = TicketClassificationResponse.from_domain(
        _classification(),
    )

    assert response.id == _CLASSIFICATION_ID
    assert response.workspace_id == _WORKSPACE_ID
    assert response.ticket_id == _TICKET_ID
    assert response.agent_run_id == _AGENT_RUN_ID
    assert response.accepted_invocation_id == (_INVOCATION_ID)
    assert response.category is TicketCategory.BILLING
    assert response.intent is TicketIntent.ASK_QUESTION
    assert response.urgency is TicketUrgency.NORMAL
    assert response.sentiment is TicketSentiment.NEUTRAL
    assert response.requires_human_review is False
    assert response.schema_version == (TICKET_CLASSIFICATION_SCHEMA_VERSION)
    assert response.prompt.id == "ticket-classification"
    assert response.prompt.version == 1
    assert response.prompt.content_hash == _PROMPT_HASH
    assert response.provider == "mock"
    assert response.model == "mock-ticket-classifier-v1"
    assert response.created_at == _NOW


def test_classification_response_uses_public_field_names() -> None:
    payload = TicketClassificationResponse.from_domain(
        _classification(),
    ).model_dump(
        mode="json",
    )

    assert payload["accepted_invocation_id"] == str(
        _INVOCATION_ID,
    )
    assert "accepted_llm_invocation_id" not in payload
    assert "updated_at" not in payload


def test_classification_response_contains_no_provider_internal_data() -> None:
    payload = TicketClassificationResponse.from_domain(
        _classification(),
    ).model_dump(
        mode="json",
    )

    assert "provider_request_id" not in payload
    assert "raw_prompt" not in payload
    assert "raw_response" not in payload
    assert "lease_token" not in payload
    assert "execution_request_id" not in payload
