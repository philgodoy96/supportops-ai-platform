"""Unit tests for SQLAlchemy classification inspection queries."""

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import cast
from unittest.mock import MagicMock
from uuid import UUID

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

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
from supportops.modules.ticket_classifications.infrastructure.models import (
    LLMInvocationRecord,
    TicketClassificationRecord,
)
from supportops.modules.ticket_classifications.infrastructure.repository import (
    SqlAlchemyTicketClassificationQueryRepository,
)

_NOW = datetime(
    2026,
    8,
    1,
    20,
    45,
    tzinfo=UTC,
)
_WORKSPACE_ID = UUID(
    "11111111-aaaa-4111-8111-111111111111",
)
_TICKET_ID = UUID(
    "22222222-bbbb-4222-8222-222222222222",
)
_AGENT_RUN_ID = UUID(
    "33333333-cccc-4333-8333-333333333333",
)
_ATTEMPT_ID = UUID(
    "44444444-dddd-4444-8444-444444444444",
)
_INVOCATION_ID = UUID(
    "55555555-eeee-4555-8555-555555555555",
)
_CLASSIFICATION_ID = UUID(
    "66666666-ffff-4666-8666-666666666666",
)
_PROMPT_HASH = "a" * 64
_ZERO_COST = Decimal("0.000000000000")


def _classification(
    *,
    classification_id: UUID = _CLASSIFICATION_ID,
    created_at: datetime = _NOW,
) -> TicketClassification:
    return TicketClassification.create(
        classification_id=classification_id,
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
        now=created_at,
    )


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
        provider_request_id=("internal-provider-request-id"),
        prompt_id="ticket-classification",
        prompt_version=1,
        prompt_content_hash=_PROMPT_HASH,
        schema_version=(TICKET_CLASSIFICATION_SCHEMA_VERSION),
        input_tokens=120,
        cached_input_tokens=0,
        output_tokens=24,
        reasoning_tokens=None,
        total_tokens=144,
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


def _session() -> MagicMock:
    return MagicMock(
        spec=AsyncSession,
    )


def _scalar_result(
    value: object | None,
) -> MagicMock:
    result = MagicMock()
    result.scalar_one_or_none.return_value = value
    return result


def _row_result(
    value: object | None,
) -> MagicMock:
    result = MagicMock()
    result.one_or_none.return_value = value
    return result


def _scalars_result(
    values: list[object],
) -> MagicMock:
    result = MagicMock()
    result.scalars.return_value.all.return_value = values
    return result


def _rows_result(
    values: list[object],
) -> MagicMock:
    result = MagicMock()
    result.all.return_value = values
    return result


def _executed_statement(
    session: MagicMock,
) -> object:
    return session.execute.await_args.args[0]


def _normalized_sql(
    statement: object,
) -> str:
    return " ".join(str(statement).split())


async def test_get_returns_workspace_scoped_classification() -> None:
    classification = _classification()
    record = TicketClassificationRecord.from_domain(
        classification,
    )
    session = _session()
    session.execute.return_value = _scalar_result(
        record,
    )
    repository = SqlAlchemyTicketClassificationQueryRepository(
        cast(AsyncSession, session),
    )

    result = await repository.get(
        workspace_id=_WORKSPACE_ID,
        classification_id=_CLASSIFICATION_ID,
    )

    assert result == classification

    statement = _executed_statement(session)
    sql = _normalized_sql(statement)

    assert "ticket_classifications.workspace_id" in sql
    assert "ticket_classifications.id" in sql


async def test_get_returns_none_when_scope_does_not_match() -> None:
    session = _session()
    session.execute.return_value = _scalar_result(
        None,
    )
    repository = SqlAlchemyTicketClassificationQueryRepository(
        cast(AsyncSession, session),
    )

    result = await repository.get(
        workspace_id=_WORKSPACE_ID,
        classification_id=_CLASSIFICATION_ID,
    )

    assert result is None


async def test_get_by_agent_run_returns_scoped_classification() -> None:
    classification = _classification()
    session = _session()
    session.execute.return_value = _scalar_result(
        TicketClassificationRecord.from_domain(
            classification,
        ),
    )
    repository = SqlAlchemyTicketClassificationQueryRepository(
        cast(AsyncSession, session),
    )

    result = await repository.get_by_agent_run_id(
        workspace_id=_WORKSPACE_ID,
        agent_run_id=_AGENT_RUN_ID,
    )

    assert result == classification

    statement = _executed_statement(session)
    sql = _normalized_sql(statement)

    assert "ticket_classifications.workspace_id" in sql
    assert "ticket_classifications.agent_run_id" in sql


async def test_reference_query_returns_lightweight_projection() -> None:
    session = _session()
    session.execute.return_value = _row_result(
        (
            _CLASSIFICATION_ID,
            TICKET_CLASSIFICATION_SCHEMA_VERSION,
            _NOW,
        ),
    )
    repository = SqlAlchemyTicketClassificationQueryRepository(
        cast(AsyncSession, session),
    )

    result = await repository.get_reference_by_agent_run_id(
        workspace_id=_WORKSPACE_ID,
        agent_run_id=_AGENT_RUN_ID,
    )

    assert result == AgentRunClassificationReference(
        id=_CLASSIFICATION_ID,
        schema_version=(TICKET_CLASSIFICATION_SCHEMA_VERSION),
        created_at=_NOW,
    )

    statement = _executed_statement(session)
    sql = _normalized_sql(statement)

    assert "ticket_classifications.summary" not in sql
    assert "ticket_classifications.accepted_llm_invocation_id" not in sql
    assert "ticket_classifications.id" in sql
    assert "ticket_classifications.schema_version" in sql
    assert "ticket_classifications.created_at" in sql
    assert "ticket_classifications.workspace_id" in sql
    assert "ticket_classifications.agent_run_id" in sql


async def test_reference_query_returns_none() -> None:
    session = _session()
    session.execute.return_value = _row_result(
        None,
    )
    repository = SqlAlchemyTicketClassificationQueryRepository(
        cast(AsyncSession, session),
    )

    result = await repository.get_reference_by_agent_run_id(
        workspace_id=_WORKSPACE_ID,
        agent_run_id=_AGENT_RUN_ID,
    )

    assert result is None


async def test_ticket_list_returns_domain_entities() -> None:
    newest = _classification()
    older_id = UUID(
        "77777777-1111-4777-8777-777777777777",
    )
    older = _classification(
        classification_id=older_id,
        created_at=_NOW - timedelta(minutes=5),
    )
    session = _session()
    session.execute.return_value = _scalars_result(
        [
            TicketClassificationRecord.from_domain(
                newest,
            ),
            TicketClassificationRecord.from_domain(
                older,
            ),
        ],
    )
    repository = SqlAlchemyTicketClassificationQueryRepository(
        cast(AsyncSession, session),
    )

    result = await repository.list_by_ticket(
        workspace_id=_WORKSPACE_ID,
        ticket_id=_TICKET_ID,
        limit=21,
    )

    assert result == (
        newest,
        older,
    )

    statement = _executed_statement(session)
    sql = _normalized_sql(statement)

    assert "ticket_classifications.workspace_id" in sql
    assert "ticket_classifications.ticket_id" in sql
    assert "ORDER BY ticket_classifications.created_at DESC, ticket_classifications.id DESC" in sql
    assert "LIMIT" in sql


async def test_ticket_list_applies_keyset_position() -> None:
    session = _session()
    session.execute.return_value = _scalars_result(
        [],
    )
    repository = SqlAlchemyTicketClassificationQueryRepository(
        cast(AsyncSession, session),
    )

    result = await repository.list_by_ticket(
        workspace_id=_WORKSPACE_ID,
        ticket_id=_TICKET_ID,
        limit=20,
        after_created_at=_NOW,
        after_classification_id=_CLASSIFICATION_ID,
    )

    assert result == ()

    statement = _executed_statement(session)
    sql = _normalized_sql(statement)

    assert "(ticket_classifications.created_at, ticket_classifications.id) <" in sql
    assert "ORDER BY ticket_classifications.created_at DESC, ticket_classifications.id DESC" in sql


@pytest.mark.parametrize(
    "limit",
    [
        0,
        -1,
    ],
)
async def test_ticket_list_requires_positive_limit(
    limit: int,
) -> None:
    session = _session()
    repository = SqlAlchemyTicketClassificationQueryRepository(
        cast(AsyncSession, session),
    )

    with pytest.raises(
        ValueError,
        match=("Ticket classification list limit must be positive"),
    ):
        await repository.list_by_ticket(
            workspace_id=_WORKSPACE_ID,
            ticket_id=_TICKET_ID,
            limit=limit,
        )

    session.execute.assert_not_awaited()


@pytest.mark.parametrize(
    (
        "after_created_at",
        "after_classification_id",
    ),
    [
        (
            _NOW,
            None,
        ),
        (
            None,
            _CLASSIFICATION_ID,
        ),
    ],
)
async def test_ticket_list_requires_complete_keyset(
    after_created_at: datetime | None,
    after_classification_id: UUID | None,
) -> None:
    session = _session()
    repository = SqlAlchemyTicketClassificationQueryRepository(
        cast(AsyncSession, session),
    )

    with pytest.raises(
        ValueError,
        match=("Ticket classification pagination position requires both timestamp and ID"),
    ):
        await repository.list_by_ticket(
            workspace_id=_WORKSPACE_ID,
            ticket_id=_TICKET_ID,
            limit=20,
            after_created_at=after_created_at,
            after_classification_id=(after_classification_id),
        )

    session.execute.assert_not_awaited()


async def test_invocation_list_returns_safe_attempt_projection() -> None:
    invocation = _invocation()
    record = LLMInvocationRecord.from_domain(
        invocation,
    )
    session = _session()
    session.execute.return_value = _rows_result(
        [
            (
                record,
                2,
            ),
        ],
    )
    repository = SqlAlchemyTicketClassificationQueryRepository(
        cast(AsyncSession, session),
    )

    result = await repository.list_invocations_by_agent_run(
        workspace_id=_WORKSPACE_ID,
        agent_run_id=_AGENT_RUN_ID,
    )

    assert len(result) == 1

    inspection = result[0]

    assert isinstance(
        inspection,
        LLMInvocationInspection,
    )
    assert inspection.id == _INVOCATION_ID
    assert inspection.attempt_number == 2
    assert inspection.invocation_sequence == 1
    assert inspection.provider == "mock"
    assert inspection.model == ("mock-ticket-classifier-v1")
    assert inspection.total_tokens == 144
    assert inspection.estimated_total_cost_usd == _ZERO_COST
    assert not hasattr(
        inspection,
        "provider_request_id",
    )

    statement = _executed_statement(session)
    sql = _normalized_sql(statement)

    assert "JOIN agent_run_attempts" in sql
    assert "agent_run_attempts.id = llm_invocations.agent_run_attempt_id" in sql
    assert "agent_run_attempts.agent_run_id = llm_invocations.agent_run_id" in sql
    assert "llm_invocations.workspace_id" in sql
    assert "llm_invocations.agent_run_id" in sql
    assert (
        "ORDER BY "
        "agent_run_attempts.attempt_number ASC, "
        "llm_invocations.invocation_sequence ASC" in sql
    )


async def test_invocation_list_returns_empty_tuple() -> None:
    session = _session()
    session.execute.return_value = _rows_result(
        [],
    )
    repository = SqlAlchemyTicketClassificationQueryRepository(
        cast(AsyncSession, session),
    )

    result = await repository.list_invocations_by_agent_run(
        workspace_id=_WORKSPACE_ID,
        agent_run_id=_AGENT_RUN_ID,
    )

    assert result == ()


async def test_query_repository_never_flushes_or_writes() -> None:
    session = _session()
    session.execute.side_effect = (
        _scalar_result(None),
        _row_result(None),
        _scalars_result([]),
        _rows_result([]),
    )
    repository = SqlAlchemyTicketClassificationQueryRepository(
        cast(AsyncSession, session),
    )

    await repository.get(
        workspace_id=_WORKSPACE_ID,
        classification_id=_CLASSIFICATION_ID,
    )
    await repository.get_reference_by_agent_run_id(
        workspace_id=_WORKSPACE_ID,
        agent_run_id=_AGENT_RUN_ID,
    )
    await repository.list_by_ticket(
        workspace_id=_WORKSPACE_ID,
        ticket_id=_TICKET_ID,
        limit=20,
    )
    await repository.list_invocations_by_agent_run(
        workspace_id=_WORKSPACE_ID,
        agent_run_id=_AGENT_RUN_ID,
    )

    session.add.assert_not_called()
    session.add_all.assert_not_called()
    session.flush.assert_not_awaited()
    session.commit.assert_not_awaited()
    session.rollback.assert_not_awaited()
