"""Unit tests for safe classification inspection projections."""

from dataclasses import replace
from datetime import UTC, datetime
from decimal import Decimal
from uuid import UUID

import pytest

from supportops.ai.gateway.results import (
    LLMInvocationStatus,
)
from supportops.ai.schemas.ticket_classification import (
    TICKET_CLASSIFICATION_SCHEMA_VERSION,
    TicketCategory,
    TicketIntent,
    TicketSentiment,
    TicketUrgency,
)
from supportops.modules.ticket_classifications.domain.inspection import (
    AgentRunClassificationReference,
    LLMInvocationInspection,
)
from supportops.modules.ticket_classifications.domain.models import (
    LLMInvocation,
    TicketClassification,
)

_NOW = datetime(
    2026,
    8,
    1,
    20,
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
_ATTEMPT_ID = UUID(
    "44444444-4444-4444-8444-444444444444",
)
_INVOCATION_ID = UUID(
    "55555555-5555-4555-8555-555555555555",
)
_CLASSIFICATION_ID = UUID(
    "66666666-6666-4666-8666-666666666666",
)
_PROMPT_HASH = "a" * 64
_ZERO_COST = Decimal("0.000000000000")


def _invocation() -> LLMInvocation:
    return LLMInvocation.create(
        invocation_id=_INVOCATION_ID,
        workspace_id=_WORKSPACE_ID,
        ticket_id=_TICKET_ID,
        agent_run_id=_AGENT_RUN_ID,
        agent_run_attempt_id=_ATTEMPT_ID,
        invocation_sequence=1,
        status=LLMInvocationStatus.SUCCEEDED,
        provider="mock",
        model="mock-ticket-classifier-v1",
        provider_request_id="internal-provider-request",
        prompt_id="ticket-classification",
        prompt_version=1,
        prompt_content_hash=_PROMPT_HASH,
        schema_version=(TICKET_CLASSIFICATION_SCHEMA_VERSION),
        input_tokens=100,
        cached_input_tokens=0,
        output_tokens=20,
        reasoning_tokens=None,
        total_tokens=120,
        pricing_catalog_version="pricing-v1",
        pricing_found=True,
        estimated_input_cost_usd=_ZERO_COST,
        estimated_cached_input_cost_usd=_ZERO_COST,
        estimated_output_cost_usd=_ZERO_COST,
        estimated_total_cost_usd=_ZERO_COST,
        latency_ms=25,
        error_code=None,
        now=_NOW,
    )


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
        summary="The customer is asking about a billing charge.",
        schema_version=(TICKET_CLASSIFICATION_SCHEMA_VERSION),
        prompt_id="ticket-classification",
        prompt_version=1,
        prompt_content_hash=_PROMPT_HASH,
        provider="mock",
        model="mock-ticket-classifier-v1",
        now=_NOW,
    )


def test_classification_reference_preserves_minimal_identity() -> None:
    reference = AgentRunClassificationReference.from_domain(
        _classification(),
    )

    assert reference.id == _CLASSIFICATION_ID
    assert reference.schema_version == (TICKET_CLASSIFICATION_SCHEMA_VERSION)
    assert reference.created_at == _NOW


def test_invocation_projection_preserves_safe_provenance() -> None:
    inspection = LLMInvocationInspection.from_domain(
        invocation=_invocation(),
        attempt_number=2,
    )

    assert inspection.id == _INVOCATION_ID
    assert inspection.workspace_id == _WORKSPACE_ID
    assert inspection.ticket_id == _TICKET_ID
    assert inspection.agent_run_id == _AGENT_RUN_ID
    assert inspection.agent_run_attempt_id == _ATTEMPT_ID
    assert inspection.attempt_number == 2
    assert inspection.invocation_sequence == 1
    assert inspection.status is (LLMInvocationStatus.SUCCEEDED)
    assert inspection.provider == "mock"
    assert inspection.model == "mock-ticket-classifier-v1"
    assert inspection.input_tokens == 100
    assert inspection.total_tokens == 120
    assert inspection.estimated_total_cost_usd == _ZERO_COST
    assert inspection.error_code is None


def test_invocation_projection_excludes_provider_request_id() -> None:
    inspection = LLMInvocationInspection.from_domain(
        invocation=_invocation(),
        attempt_number=1,
    )

    assert not hasattr(
        inspection,
        "provider_request_id",
    )


@pytest.mark.parametrize(
    "attempt_number",
    [
        0,
        -1,
    ],
)
def test_invocation_projection_requires_positive_attempt_number(
    attempt_number: int,
) -> None:
    with pytest.raises(
        ValueError,
        match="attempt_number must be positive",
    ):
        LLMInvocationInspection.from_domain(
            invocation=_invocation(),
            attempt_number=attempt_number,
        )


def test_invocation_projection_requires_positive_sequence() -> None:
    inspection = LLMInvocationInspection.from_domain(
        invocation=_invocation(),
        attempt_number=1,
    )

    with pytest.raises(
        ValueError,
        match="invocation_sequence must be positive",
    ):
        replace(
            inspection,
            invocation_sequence=0,
        )
