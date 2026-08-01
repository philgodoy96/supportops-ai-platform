"""Unit tests for ticket-classification SQLAlchemy mappings."""

from datetime import UTC, datetime
from decimal import Decimal
from typing import cast
from uuid import UUID

from sqlalchemy import (
    CheckConstraint,
    ForeignKeyConstraint,
    Numeric,
    Table,
    UniqueConstraint,
)

from supportops.ai.gateway.errors import LLMErrorCode
from supportops.ai.gateway.results import LLMInvocationStatus
from supportops.ai.schemas.ticket_classification import (
    TICKET_CLASSIFICATION_SCHEMA_VERSION,
    TicketCategory,
    TicketIntent,
    TicketSentiment,
    TicketUrgency,
)
from supportops.modules.ticket_classifications.domain.models import (
    LLMInvocation,
    TicketClassification,
)
from supportops.modules.ticket_classifications.infrastructure.models import (
    LLMInvocationRecord,
    TicketClassificationRecord,
)

_NOW = datetime(
    2026,
    8,
    1,
    18,
    0,
    tzinfo=UTC,
)

_WORKSPACE_ID = UUID(
    "8fbd2f2d-873a-4594-8e43-82282cdbff37",
)
_TICKET_ID = UUID(
    "423078b7-66b9-486d-bc8c-0d46b0573280",
)
_AGENT_RUN_ID = UUID(
    "73d8af09-16af-4ad5-b5c4-29a3cf50a849",
)
_ATTEMPT_ID = UUID(
    "a8ca5405-e778-444e-9e76-af997c3c511f",
)
_INVOCATION_ID = UUID(
    "db4b5923-3866-44a5-a284-e596277b2469",
)
_CLASSIFICATION_ID = UUID(
    "77eb5781-14b8-4ace-8d5f-b49867c490dc",
)
_PROMPT_HASH = "a" * 64


def _successful_invocation() -> LLMInvocation:
    return LLMInvocation.create(
        invocation_id=_INVOCATION_ID,
        workspace_id=_WORKSPACE_ID,
        ticket_id=_TICKET_ID,
        agent_run_id=_AGENT_RUN_ID,
        agent_run_attempt_id=_ATTEMPT_ID,
        invocation_sequence=1,
        status=LLMInvocationStatus.SUCCEEDED,
        provider="openai",
        model="gpt-5-nano",
        provider_request_id="req_test_1",
        prompt_id="ticket-classification",
        prompt_version=1,
        prompt_content_hash=_PROMPT_HASH,
        schema_version=TICKET_CLASSIFICATION_SCHEMA_VERSION,
        input_tokens=100,
        cached_input_tokens=20,
        output_tokens=25,
        reasoning_tokens=5,
        total_tokens=125,
        pricing_catalog_version=("supportops-pricing-2026-08-01"),
        pricing_found=True,
        estimated_input_cost_usd=Decimal(
            "0.000004000000",
        ),
        estimated_cached_input_cost_usd=Decimal(
            "0.000000100000",
        ),
        estimated_output_cost_usd=Decimal(
            "0.000010000000",
        ),
        estimated_total_cost_usd=Decimal(
            "0.000014100000",
        ),
        latency_ms=125,
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
        summary="The customer is asking about an invoice.",
        schema_version=TICKET_CLASSIFICATION_SCHEMA_VERSION,
        prompt_id="ticket-classification",
        prompt_version=1,
        prompt_content_hash=_PROMPT_HASH,
        provider="openai",
        model="gpt-5-nano",
        now=_NOW,
    )


def test_llm_invocation_record_round_trip_preserves_success() -> None:
    invocation = _successful_invocation()

    record = LLMInvocationRecord.from_domain(invocation)
    restored = record.to_domain()

    assert restored == invocation
    assert record.status == "succeeded"
    assert record.error_code is None
    assert record.provider_request_id == "req_test_1"
    assert record.estimated_total_cost_usd == Decimal(
        "0.000014100000",
    )


def test_llm_invocation_record_round_trip_preserves_failure() -> None:
    invocation = LLMInvocation.create(
        invocation_id=_INVOCATION_ID,
        workspace_id=_WORKSPACE_ID,
        ticket_id=_TICKET_ID,
        agent_run_id=_AGENT_RUN_ID,
        agent_run_attempt_id=_ATTEMPT_ID,
        invocation_sequence=1,
        status=LLMInvocationStatus.TIMED_OUT,
        provider="openai",
        model="gpt-5-nano",
        provider_request_id="req_timeout_1",
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

    record = LLMInvocationRecord.from_domain(invocation)
    restored = record.to_domain()

    assert restored == invocation
    assert record.status == "timed_out"
    assert record.error_code == "llm_timeout"


def test_llm_invocation_record_preserves_unknown_pricing() -> None:
    invocation = LLMInvocation.create(
        invocation_id=_INVOCATION_ID,
        workspace_id=_WORKSPACE_ID,
        ticket_id=_TICKET_ID,
        agent_run_id=_AGENT_RUN_ID,
        agent_run_attempt_id=_ATTEMPT_ID,
        invocation_sequence=1,
        status=LLMInvocationStatus.SUCCEEDED,
        provider="future-provider",
        model="future-model",
        provider_request_id=None,
        prompt_id="ticket-classification",
        prompt_version=1,
        prompt_content_hash=_PROMPT_HASH,
        schema_version=TICKET_CLASSIFICATION_SCHEMA_VERSION,
        input_tokens=100,
        cached_input_tokens=0,
        output_tokens=20,
        reasoning_tokens=None,
        total_tokens=120,
        pricing_catalog_version=("supportops-pricing-2026-08-01"),
        pricing_found=False,
        estimated_input_cost_usd=None,
        estimated_cached_input_cost_usd=None,
        estimated_output_cost_usd=None,
        estimated_total_cost_usd=None,
        latency_ms=100,
        error_code=None,
        now=_NOW,
    )

    restored = LLMInvocationRecord.from_domain(
        invocation,
    ).to_domain()

    assert restored == invocation
    assert restored.pricing_found is False
    assert restored.estimated_total_cost_usd is None


def test_ticket_classification_record_round_trip() -> None:
    classification = _classification()

    record = TicketClassificationRecord.from_domain(
        classification,
    )
    restored = record.to_domain()

    assert restored == classification
    assert record.category == "billing"
    assert record.intent == "ask_question"
    assert record.urgency == "normal"
    assert record.sentiment == "neutral"
    assert record.accepted_llm_invocation_id == _INVOCATION_ID


def test_llm_invocation_metadata_declares_expected_constraints() -> None:
    table = cast(
        Table,
        LLMInvocationRecord.__table__,
    )

    constraint_names = {
        constraint.name
        for constraint in table.constraints
        if isinstance(
            constraint,
            (
                CheckConstraint,
                UniqueConstraint,
                ForeignKeyConstraint,
            ),
        )
    }

    assert {
        "fk_llm_invocations_workspace_ticket_agent_run",
        "fk_llm_invocations_agent_run_attempt",
        "uq_llm_invocations_attempt_sequence",
        "uq_llm_invocations_run_id",
        "ck_llm_invocations_llm_invocation_sequence_positive",
        "ck_llm_invocations_llm_invocation_status",
        "ck_llm_invocations_llm_invocation_provider_format",
        "ck_llm_invocations_llm_invocation_model_format",
        ("ck_llm_invocations_llm_invocation_provider_request_id_format"),
        "ck_llm_invocations_llm_invocation_prompt_id_format",
        ("ck_llm_invocations_llm_invocation_prompt_version_positive"),
        ("ck_llm_invocations_llm_invocation_prompt_content_hash"),
        ("ck_llm_invocations_llm_invocation_schema_version_format"),
        ("ck_llm_invocations_llm_invocation_pricing_catalog_version_format"),
        "ck_llm_invocations_llm_invocation_tokens_non_negative",
        "ck_llm_invocations_llm_invocation_cached_input_limit",
        ("ck_llm_invocations_llm_invocation_reasoning_token_limit"),
        ("ck_llm_invocations_llm_invocation_total_token_consistency"),
        "ck_llm_invocations_llm_invocation_costs_non_negative",
        "ck_llm_invocations_llm_invocation_pricing_state",
        ("ck_llm_invocations_llm_invocation_latency_non_negative"),
        "ck_llm_invocations_llm_invocation_error_code",
        "ck_llm_invocations_llm_invocation_error_state",
    }.issubset(constraint_names)


def test_classification_metadata_declares_expected_constraints() -> None:
    table = cast(
        Table,
        TicketClassificationRecord.__table__,
    )

    constraint_names = {
        constraint.name
        for constraint in table.constraints
        if isinstance(
            constraint,
            (
                CheckConstraint,
                UniqueConstraint,
                ForeignKeyConstraint,
            ),
        )
    }

    assert {
        ("fk_ticket_classifications_workspace_ticket_agent_run"),
        ("fk_ticket_classifications_accepted_invocation"),
        "uq_ticket_classifications_agent_run",
        ("uq_ticket_classifications_accepted_invocation"),
        ("ck_ticket_classifications_ticket_classification_category"),
        ("ck_ticket_classifications_ticket_classification_intent"),
        ("ck_ticket_classifications_ticket_classification_urgency"),
        ("ck_ticket_classifications_ticket_classification_sentiment"),
        ("ck_ticket_classifications_ticket_classification_summary_format"),
        ("ck_ticket_classifications_ticket_classification_schema_version"),
        ("ck_ticket_classifications_ticket_classification_prompt_id_format"),
        ("ck_ticket_classifications_ticket_classification_prompt_version_positive"),
        ("ck_ticket_classifications_ticket_classification_prompt_content_hash"),
        ("ck_ticket_classifications_ticket_classification_provider_format"),
        ("ck_ticket_classifications_ticket_classification_model_format"),
        ("ck_ticket_classifications_ticket_classification_immutable_timestamp"),
    }.issubset(constraint_names)


def test_invocation_metadata_declares_query_index() -> None:
    table = cast(
        Table,
        LLMInvocationRecord.__table__,
    )

    assert {index.name for index in table.indexes} == {
        "ix_llm_invocations_workspace_run_created_id",
    }


def test_classification_metadata_declares_query_index() -> None:
    table = cast(
        Table,
        TicketClassificationRecord.__table__,
    )

    assert {index.name for index in table.indexes} == {
        ("ix_ticket_classifications_workspace_ticket_created_id"),
    }


def test_cost_columns_use_decimal_numeric_type() -> None:
    table = cast(
        Table,
        LLMInvocationRecord.__table__,
    )

    cost_columns = (
        table.c.estimated_input_cost_usd,
        table.c.estimated_cached_input_cost_usd,
        table.c.estimated_output_cost_usd,
        table.c.estimated_total_cost_usd,
    )

    for column in cost_columns:
        column_type = column.type

        assert isinstance(column_type, Numeric)
        assert column_type.precision == 20
        assert column_type.scale == 12
        assert column_type.asdecimal is True


def test_invocation_table_excludes_raw_model_content() -> None:
    table = cast(
        Table,
        LLMInvocationRecord.__table__,
    )
    column_names = set(table.c.keys())

    assert "ticket_subject" not in column_names
    assert "ticket_description" not in column_names
    assert "prompt" not in column_names
    assert "raw_response" not in column_names
    assert "error_summary" not in column_names
    assert "chain_of_thought" not in column_names
