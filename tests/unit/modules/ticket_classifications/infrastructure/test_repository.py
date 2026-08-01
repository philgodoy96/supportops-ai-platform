"""Unit tests for fenced classification repository behavior."""

from datetime import UTC, datetime
from decimal import Decimal
from typing import cast
from unittest.mock import MagicMock
from uuid import UUID

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

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
    ClassificationPersistenceResult,
    PersistClassificationExecutionCommand,
)
from supportops.modules.ticket_classifications.domain.models import (
    LLMInvocation,
    TicketClassification,
)
from supportops.modules.ticket_classifications.infrastructure.models import (
    LLMInvocationRecord,
    TicketClassificationRecord,
)
from supportops.modules.ticket_classifications.infrastructure.repository import (
    SqlAlchemyClassificationPersistenceRepository,
)

_NOW = datetime(
    2026,
    8,
    1,
    19,
    0,
    tzinfo=UTC,
)
_WORKSPACE_ID = UUID(
    "b62ab5ab-c8e4-4e35-aa15-620d93e311cc",
)
_TICKET_ID = UUID(
    "cd38c1ad-83ae-4c78-921c-d14de044a718",
)
_AGENT_RUN_ID = UUID(
    "01b266e7-b993-417d-9252-a18669f662c1",
)
_ATTEMPT_ID = UUID(
    "964df1c4-bf15-4bb4-8d99-c793025ad84b",
)
_LEASE_TOKEN = UUID(
    "779c0df4-5356-4786-b170-83838119d62a",
)
_INVOCATION_ID = UUID(
    "10bea479-d193-4bf3-a6f9-bef35862956c",
)
_CLASSIFICATION_ID = UUID(
    "51db6adf-b78a-44fa-b1a4-39019090dff9",
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


def _failure_command() -> PersistClassificationExecutionCommand:
    return PersistClassificationExecutionCommand(
        workspace_id=_WORKSPACE_ID,
        ticket_id=_TICKET_ID,
        agent_run_id=_AGENT_RUN_ID,
        agent_run_attempt_id=_ATTEMPT_ID,
        lease_token=_LEASE_TOKEN,
        persisted_at=_NOW,
        invocations=(_failed_invocation(),),
        classification=None,
    )


def _session() -> MagicMock:
    return MagicMock(
        spec=AsyncSession,
    )


def _scalar_result(value: object | None) -> MagicMock:
    result = MagicMock()
    result.scalar_one_or_none.return_value = value
    return result


def _scalars_result(values: list[object]) -> MagicMock:
    result = MagicMock()
    result.scalars.return_value.all.return_value = values
    return result


async def test_add_flushes_classification_record() -> None:
    session = _session()
    repository = SqlAlchemyClassificationPersistenceRepository(
        cast(AsyncSession, session),
    )

    await repository.add(_classification())

    session.add.assert_called_once()
    added_record = session.add.call_args.args[0]

    assert isinstance(
        added_record,
        TicketClassificationRecord,
    )
    session.flush.assert_awaited_once()


async def test_add_many_skips_empty_sequence() -> None:
    session = _session()
    repository = SqlAlchemyClassificationPersistenceRepository(
        cast(AsyncSession, session),
    )

    await repository.add_many(())

    session.add_all.assert_not_called()
    session.flush.assert_not_awaited()


async def test_get_returns_workspace_scoped_classification() -> None:
    session = _session()
    record = TicketClassificationRecord.from_domain(
        _classification(),
    )
    session.execute.return_value = _scalar_result(record)
    repository = SqlAlchemyClassificationPersistenceRepository(
        cast(AsyncSession, session),
    )

    result = await repository.get_by_agent_run_id(
        workspace_id=_WORKSPACE_ID,
        agent_run_id=_AGENT_RUN_ID,
    )

    assert result == _classification()


async def test_fenced_success_persists_invocation_and_classification() -> None:
    session = _session()
    session.execute.side_effect = (
        _scalar_result(_AGENT_RUN_ID),
        _scalar_result(_ATTEMPT_ID),
        _scalar_result(None),
        _scalars_result([]),
    )
    repository = SqlAlchemyClassificationPersistenceRepository(
        cast(AsyncSession, session),
    )

    result = await repository.persist_fenced(
        _success_command(),
    )

    assert result is ClassificationPersistenceResult.APPLIED
    session.add_all.assert_called_once()
    invocation_records = session.add_all.call_args.args[0]
    assert len(invocation_records) == 1
    assert isinstance(
        invocation_records[0],
        LLMInvocationRecord,
    )

    session.add.assert_called_once()
    assert isinstance(
        session.add.call_args.args[0],
        TicketClassificationRecord,
    )
    assert session.flush.await_count == 2


async def test_fenced_persistence_rejects_lost_run_lease() -> None:
    session = _session()
    session.execute.side_effect = (_scalar_result(None),)
    repository = SqlAlchemyClassificationPersistenceRepository(
        cast(AsyncSession, session),
    )

    result = await repository.persist_fenced(
        _success_command(),
    )

    assert result is ClassificationPersistenceResult.LEASE_LOST
    session.add_all.assert_not_called()
    session.add.assert_not_called()
    session.flush.assert_not_awaited()


async def test_fenced_persistence_rejects_inactive_attempt() -> None:
    session = _session()
    session.execute.side_effect = (
        _scalar_result(_AGENT_RUN_ID),
        _scalar_result(None),
    )
    repository = SqlAlchemyClassificationPersistenceRepository(
        cast(AsyncSession, session),
    )

    result = await repository.persist_fenced(
        _success_command(),
    )

    assert result is ClassificationPersistenceResult.LEASE_LOST
    session.add_all.assert_not_called()
    session.add.assert_not_called()


async def test_existing_classification_is_idempotent() -> None:
    session = _session()
    session.execute.side_effect = (
        _scalar_result(_AGENT_RUN_ID),
        _scalar_result(_ATTEMPT_ID),
        _scalar_result(
            TicketClassificationRecord.from_domain(
                _classification(),
            ),
        ),
    )
    repository = SqlAlchemyClassificationPersistenceRepository(
        cast(AsyncSession, session),
    )

    result = await repository.persist_fenced(
        _success_command(),
    )

    assert result is (ClassificationPersistenceResult.ALREADY_CLASSIFIED)
    session.add_all.assert_not_called()
    session.add.assert_not_called()
    session.flush.assert_not_awaited()


async def test_existing_failure_invocation_is_idempotent() -> None:
    invocation = _failed_invocation()
    existing_record = LLMInvocationRecord.from_domain(
        invocation,
    )
    session = _session()
    session.execute.side_effect = (
        _scalar_result(_AGENT_RUN_ID),
        _scalar_result(_ATTEMPT_ID),
        _scalar_result(None),
        _scalars_result(
            [
                existing_record,
            ],
        ),
    )
    repository = SqlAlchemyClassificationPersistenceRepository(
        cast(AsyncSession, session),
    )

    result = await repository.persist_fenced(
        _failure_command(),
    )

    assert result is (ClassificationPersistenceResult.ALREADY_RECORDED)
    session.add_all.assert_not_called()
    session.add.assert_not_called()
    session.flush.assert_not_awaited()


async def test_conflicting_invocation_sequence_is_internal_error() -> None:
    conflicting_invocation = LLMInvocationRecord.from_domain(
        _failed_invocation(),
    )
    conflicting_invocation.id = UUID(
        "aad1d151-cd7b-42a8-94b0-822d1061a0d3",
    )
    session = _session()
    session.execute.side_effect = (
        _scalar_result(_AGENT_RUN_ID),
        _scalar_result(_ATTEMPT_ID),
        _scalar_result(None),
        _scalars_result(
            [
                conflicting_invocation,
            ],
        ),
    )
    repository = SqlAlchemyClassificationPersistenceRepository(
        cast(AsyncSession, session),
    )

    with pytest.raises(
        RuntimeError,
        match="already persisted with different invocation data",
    ):
        await repository.persist_fenced(
            _failure_command(),
        )

    session.add_all.assert_not_called()
    session.add.assert_not_called()
    session.flush.assert_not_awaited()
