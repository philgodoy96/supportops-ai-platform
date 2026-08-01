"""Unit tests for fenced classification persistence commands."""

from dataclasses import replace
from datetime import UTC, datetime
from decimal import Decimal
from uuid import UUID

import pytest

from supportops.ai.gateway.errors import LLMErrorCode
from supportops.ai.gateway.results import LLMInvocationStatus
from supportops.ai.schemas.ticket_classification import (
    TICKET_CLASSIFICATION_SCHEMA_VERSION,
    TicketCategory,
    TicketIntent,
    TicketSentiment,
    TicketUrgency,
)
from supportops.modules.ticket_classifications.application.persistence import (
    PersistClassificationExecutionCommand,
)
from supportops.modules.ticket_classifications.domain.models import (
    LLMInvocation,
    TicketClassification,
)

_NOW = datetime(
    2026,
    8,
    1,
    18,
    30,
    tzinfo=UTC,
)
_WORKSPACE_ID = UUID(
    "3457a529-f42e-458f-ac50-feb40716f694",
)
_TICKET_ID = UUID(
    "1b433d26-d418-4461-b595-b44d0923b8cf",
)
_AGENT_RUN_ID = UUID(
    "a3a2452b-40f4-42d7-a298-18ab2f2e3c80",
)
_ATTEMPT_ID = UUID(
    "e146b761-29cd-4e2e-a35e-b294bd7728dc",
)
_LEASE_TOKEN = UUID(
    "98e033ef-6f8a-44d8-8fd3-073470eeec43",
)
_INVOCATION_ID = UUID(
    "96327fb4-4bb7-4d7d-9e71-86428d03ba25",
)
_CLASSIFICATION_ID = UUID(
    "c37072d6-fd2e-4814-86cc-4b89c4768424",
)
_PROMPT_HASH = "a" * 64
_ZERO_COST = Decimal("0.000000000000")


def _successful_invocation() -> LLMInvocation:
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
        provider_request_id="mock-request-1",
        prompt_id="ticket-classification",
        prompt_version=1,
        prompt_content_hash=_PROMPT_HASH,
        schema_version=TICKET_CLASSIFICATION_SCHEMA_VERSION,
        input_tokens=120,
        cached_input_tokens=0,
        output_tokens=24,
        reasoning_tokens=None,
        total_tokens=144,
        pricing_catalog_version=("supportops-pricing-2026-08-01"),
        pricing_found=True,
        estimated_input_cost_usd=_ZERO_COST,
        estimated_cached_input_cost_usd=_ZERO_COST,
        estimated_output_cost_usd=_ZERO_COST,
        estimated_total_cost_usd=_ZERO_COST,
        latency_ms=10,
        error_code=None,
        now=_NOW,
    )


def _failed_invocation() -> LLMInvocation:
    return LLMInvocation.create(
        invocation_id=_INVOCATION_ID,
        workspace_id=_WORKSPACE_ID,
        ticket_id=_TICKET_ID,
        agent_run_id=_AGENT_RUN_ID,
        agent_run_attempt_id=_ATTEMPT_ID,
        invocation_sequence=1,
        status=LLMInvocationStatus.TIMED_OUT,
        provider="openai",
        model="gpt-5-nano",
        provider_request_id=None,
        prompt_id="ticket-classification",
        prompt_version=1,
        prompt_content_hash=_PROMPT_HASH,
        schema_version=TICKET_CLASSIFICATION_SCHEMA_VERSION,
        input_tokens=None,
        cached_input_tokens=None,
        output_tokens=None,
        reasoning_tokens=None,
        total_tokens=None,
        pricing_catalog_version=("supportops-pricing-2026-08-01"),
        pricing_found=True,
        estimated_input_cost_usd=None,
        estimated_cached_input_cost_usd=None,
        estimated_output_cost_usd=None,
        estimated_total_cost_usd=None,
        latency_ms=12_000,
        error_code=LLMErrorCode.TIMEOUT,
        now=_NOW,
    )


def _classification() -> TicketClassification:
    return TicketClassification.create(
        classification_id=_CLASSIFICATION_ID,
        workspace_id=_WORKSPACE_ID,
        ticket_id=_TICKET_ID,
        agent_run_id=_AGENT_RUN_ID,
        accepted_llm_invocation_id=_INVOCATION_ID,
        category=TicketCategory.OTHER,
        intent=TicketIntent.OTHER,
        urgency=TicketUrgency.NORMAL,
        sentiment=TicketSentiment.NEUTRAL,
        requires_human_review=False,
        summary="The ticket received the deterministic mock classification.",
        schema_version=TICKET_CLASSIFICATION_SCHEMA_VERSION,
        prompt_id="ticket-classification",
        prompt_version=1,
        prompt_content_hash=_PROMPT_HASH,
        provider="mock",
        model="mock-ticket-classifier-v1",
        now=_NOW,
    )


def _success_command() -> PersistClassificationExecutionCommand:
    return PersistClassificationExecutionCommand(
        workspace_id=_WORKSPACE_ID,
        ticket_id=_TICKET_ID,
        agent_run_id=_AGENT_RUN_ID,
        agent_run_attempt_id=_ATTEMPT_ID,
        lease_token=_LEASE_TOKEN,
        persisted_at=_NOW,
        invocations=(_successful_invocation(),),
        classification=_classification(),
    )


def test_accepts_consistent_success_result() -> None:
    command = _success_command()

    assert len(command.invocations) == 1
    assert command.classification is not None
    assert command.classification.accepted_llm_invocation_id == command.invocations[0].id


def test_accepts_failure_invocations_without_classification() -> None:
    command = PersistClassificationExecutionCommand(
        workspace_id=_WORKSPACE_ID,
        ticket_id=_TICKET_ID,
        agent_run_id=_AGENT_RUN_ID,
        agent_run_attempt_id=_ATTEMPT_ID,
        lease_token=_LEASE_TOKEN,
        persisted_at=_NOW,
        invocations=(_failed_invocation(),),
        classification=None,
    )

    assert command.classification is None


def test_requires_at_least_one_invocation() -> None:
    with pytest.raises(
        ValueError,
        match="At least one LLM invocation",
    ):
        PersistClassificationExecutionCommand(
            workspace_id=_WORKSPACE_ID,
            ticket_id=_TICKET_ID,
            agent_run_id=_AGENT_RUN_ID,
            agent_run_attempt_id=_ATTEMPT_ID,
            lease_token=_LEASE_TOKEN,
            persisted_at=_NOW,
            invocations=(),
            classification=None,
        )


def test_requires_contiguous_ordered_sequences() -> None:
    invocation = replace(
        _failed_invocation(),
        invocation_sequence=2,
    )

    with pytest.raises(
        ValueError,
        match="contiguous, ordered, and start at one",
    ):
        PersistClassificationExecutionCommand(
            workspace_id=_WORKSPACE_ID,
            ticket_id=_TICKET_ID,
            agent_run_id=_AGENT_RUN_ID,
            agent_run_attempt_id=_ATTEMPT_ID,
            lease_token=_LEASE_TOKEN,
            persisted_at=_NOW,
            invocations=(invocation,),
            classification=None,
        )


@pytest.mark.parametrize(
    ("field_name", "message"),
    [
        (
            "workspace_id",
            "Invocation workspace ownership",
        ),
        (
            "ticket_id",
            "Invocation ticket ownership",
        ),
        (
            "agent_run_id",
            "Invocation AgentRun ownership",
        ),
        (
            "agent_run_attempt_id",
            "Invocation AgentRunAttempt ownership",
        ),
    ],
)
def test_rejects_invocation_ownership_mismatch(
    field_name: str,
    message: str,
) -> None:
    mismatched_id = UUID(
        "86f263eb-0dd1-4185-be6d-8f73f9d3635d",
    )
    base = _failed_invocation()
    if field_name == "workspace_id":
        invocation = replace(
            base,
            workspace_id=mismatched_id,
        )
    elif field_name == "ticket_id":
        invocation = replace(
            base,
            ticket_id=mismatched_id,
        )
    elif field_name == "agent_run_id":
        invocation = replace(
            base,
            agent_run_id=mismatched_id,
        )
    else:
        invocation = replace(
            base,
            agent_run_attempt_id=mismatched_id,
        )

    with pytest.raises(
        ValueError,
        match=message,
    ):
        PersistClassificationExecutionCommand(
            workspace_id=_WORKSPACE_ID,
            ticket_id=_TICKET_ID,
            agent_run_id=_AGENT_RUN_ID,
            agent_run_attempt_id=_ATTEMPT_ID,
            lease_token=_LEASE_TOKEN,
            persisted_at=_NOW,
            invocations=(invocation,),
            classification=None,
        )


def test_successful_invocation_requires_classification() -> None:
    with pytest.raises(
        ValueError,
        match="requires an accepted classification",
    ):
        PersistClassificationExecutionCommand(
            workspace_id=_WORKSPACE_ID,
            ticket_id=_TICKET_ID,
            agent_run_id=_AGENT_RUN_ID,
            agent_run_attempt_id=_ATTEMPT_ID,
            lease_token=_LEASE_TOKEN,
            persisted_at=_NOW,
            invocations=(_successful_invocation(),),
            classification=None,
        )


def test_classification_must_reference_successful_invocation() -> None:
    classification = replace(
        _classification(),
        accepted_llm_invocation_id=UUID(
            "6421cbaa-f6aa-45e5-a5a0-32ef34222b35",
        ),
    )

    with pytest.raises(
        ValueError,
        match="must reference the successful invocation",
    ):
        replace(
            _success_command(),
            classification=classification,
        )


@pytest.mark.parametrize(
    "field_name",
    [
        "prompt_id",
        "prompt_version",
        "prompt_content_hash",
        "schema_version",
        "provider",
        "model",
    ],
)
def test_classification_must_match_accepted_provenance(
    field_name: str,
) -> None:
    base = _successful_invocation()
    if field_name == "prompt_id":
        invocation = replace(
            base,
            prompt_id="another-prompt",
        )
    elif field_name == "prompt_version":
        invocation = replace(
            base,
            prompt_version=2,
        )
    elif field_name == "prompt_content_hash":
        invocation = replace(
            base,
            prompt_content_hash="b" * 64,
        )
    elif field_name == "schema_version":
        invocation = replace(
            base,
            schema_version="another-schema",
        )
    elif field_name == "provider":
        invocation = replace(
            base,
            provider="openai",
        )
    else:
        invocation = replace(
            base,
            model="gpt-5-nano",
        )

    with pytest.raises(
        ValueError,
        match=f"share {field_name} provenance",
    ):
        replace(
            _success_command(),
            invocations=(invocation,),
        )


def test_requires_utc_persistence_timestamp() -> None:
    with pytest.raises(
        ValueError,
        match="persisted_at must be a UTC-aware timestamp",
    ):
        replace(
            _success_command(),
            persisted_at=datetime(
                2026,
                8,
                1,
                18,
                30,
            ),
        )
