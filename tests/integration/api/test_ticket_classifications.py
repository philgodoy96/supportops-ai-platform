"""Integration tests for classification and LLM invocation inspection."""

from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal
from typing import cast
from uuid import UUID, uuid4

import pytest
from httpx import AsyncClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from supportops.ai.gateway.errors import LLMErrorCode
from supportops.ai.gateway.results import LLMInvocationStatus
from supportops.ai.pricing.catalog import (
    PRICING_CATALOG_VERSION,
)
from supportops.ai.schemas.ticket_classification import (
    TICKET_CLASSIFICATION_SCHEMA_VERSION,
    TicketCategory,
    TicketIntent,
    TicketSentiment,
    TicketUrgency,
)
from supportops.modules.agent_runs.domain.models import (
    INITIAL_TICKET_PROCESSING_WORKFLOW_NAME,
    TICKET_CLASSIFICATION_WORKFLOW_VERSION,
    AgentRun,
    AgentRunAttempt,
    AgentRunAttemptOutcome,
    AgentRunStatus,
)
from supportops.modules.agent_runs.infrastructure.models import (
    AgentRunAttemptRecord,
    AgentRunRecord,
)
from supportops.modules.ticket_classifications.domain.models import (
    LLMInvocation,
    TicketClassification,
)
from supportops.modules.ticket_classifications.infrastructure.models import (
    LLMInvocationRecord,
    TicketClassificationRecord,
)

pytestmark = pytest.mark.integration

_PROMPT_HASH = "a" * 64
_MODEL = "mock-ticket-classifier-v1"
_ZERO_COST = Decimal("0.000000000000")


@dataclass(frozen=True, slots=True)
class PersistedClassificationFixture:
    """Identifiers produced by one integration-test classification history."""

    agent_run_id: UUID
    classification_id: UUID
    accepted_invocation_id: UUID
    classification_created_at: datetime
    invocation_ids: tuple[UUID, ...]


async def _create_workspace(
    client: AsyncClient,
    *,
    name: str,
    slug: str,
) -> dict[str, str]:
    response = await client.post(
        "/api/v1/workspaces",
        json={
            "name": name,
            "slug": slug,
        },
    )

    assert response.status_code == 201

    return cast(
        dict[str, str],
        response.json(),
    )


async def _create_ticket_and_initial_run(
    client: AsyncClient,
    session: AsyncSession,
    *,
    workspace_id: str,
    subject: str = "Duplicated invoice charge",
) -> tuple[dict[str, object], AgentRunRecord]:
    response = await client.post(
        f"/api/v1/workspaces/{workspace_id}/tickets",
        json={
            "subject": subject,
            "description": ("The latest invoice contains the same subscription charge twice."),
        },
    )

    assert response.status_code == 201

    ticket = cast(
        dict[str, object],
        response.json(),
    )
    result = await session.execute(
        select(AgentRunRecord).where(
            AgentRunRecord.ticket_id
            == UUID(
                cast(
                    str,
                    ticket["id"],
                ),
            ),
        ),
    )

    return (
        ticket,
        result.scalar_one(),
    )


def _completed_attempt(
    *,
    agent_run_id: UUID,
    attempt_number: int,
    started_at: datetime,
    finished_at: datetime,
    outcome: AgentRunAttemptOutcome,
    error_code: str | None = None,
    error_summary: str | None = None,
) -> AgentRunAttempt:
    return AgentRunAttempt(
        id=uuid4(),
        agent_run_id=agent_run_id,
        attempt_number=attempt_number,
        worker_id=f"inspection-worker-{attempt_number}",
        lease_token=uuid4(),
        execution_request_id=uuid4(),
        started_at=started_at,
        finished_at=finished_at,
        outcome=outcome,
        error_code=error_code,
        error_summary=error_summary,
    )


def _invocation(
    *,
    workspace_id: UUID,
    ticket_id: UUID,
    agent_run_id: UUID,
    attempt_id: UUID,
    invocation_sequence: int,
    status: LLMInvocationStatus,
    error_code: LLMErrorCode | None,
    created_at: datetime,
    include_usage: bool,
) -> LLMInvocation:
    invocation_id = uuid4()

    return LLMInvocation.create(
        invocation_id=invocation_id,
        workspace_id=workspace_id,
        ticket_id=ticket_id,
        agent_run_id=agent_run_id,
        agent_run_attempt_id=attempt_id,
        invocation_sequence=invocation_sequence,
        status=status,
        provider="mock",
        model=_MODEL,
        provider_request_id=(f"internal-provider-request-{invocation_id}"),
        prompt_id="ticket-classification",
        prompt_version=1,
        prompt_content_hash=_PROMPT_HASH,
        schema_version=(TICKET_CLASSIFICATION_SCHEMA_VERSION),
        input_tokens=120 if include_usage else None,
        cached_input_tokens=0 if include_usage else None,
        output_tokens=24 if include_usage else None,
        reasoning_tokens=None,
        total_tokens=144 if include_usage else None,
        pricing_catalog_version=PRICING_CATALOG_VERSION,
        pricing_found=True,
        estimated_input_cost_usd=(_ZERO_COST if include_usage else None),
        estimated_cached_input_cost_usd=(_ZERO_COST if include_usage else None),
        estimated_output_cost_usd=(_ZERO_COST if include_usage else None),
        estimated_total_cost_usd=(_ZERO_COST if include_usage else None),
        latency_ms=25 if include_usage else 12_000,
        error_code=error_code,
        now=created_at,
    )


def _classification(
    *,
    classification_id: UUID,
    workspace_id: UUID,
    ticket_id: UUID,
    agent_run_id: UUID,
    accepted_invocation_id: UUID,
    created_at: datetime,
    category: TicketCategory = TicketCategory.BILLING,
    intent: TicketIntent = TicketIntent.ASK_QUESTION,
    urgency: TicketUrgency = TicketUrgency.NORMAL,
    sentiment: TicketSentiment = TicketSentiment.NEUTRAL,
    requires_human_review: bool = False,
) -> TicketClassification:
    return TicketClassification.create(
        classification_id=classification_id,
        workspace_id=workspace_id,
        ticket_id=ticket_id,
        agent_run_id=agent_run_id,
        accepted_llm_invocation_id=(accepted_invocation_id),
        category=category,
        intent=intent,
        urgency=urgency,
        sentiment=sentiment,
        requires_human_review=(requires_human_review),
        summary=("The customer is asking about a duplicated subscription charge."),
        schema_version=(TICKET_CLASSIFICATION_SCHEMA_VERSION),
        prompt_id="ticket-classification",
        prompt_version=1,
        prompt_content_hash=_PROMPT_HASH,
        provider="mock",
        model=_MODEL,
        now=created_at,
    )


def _mark_run_succeeded(
    *,
    record: AgentRunRecord,
    attempt_count: int,
    first_started_at: datetime,
    completed_at: datetime,
) -> None:
    record.status = AgentRunStatus.SUCCEEDED.value
    record.attempt_count = attempt_count
    record.first_started_at = first_started_at
    record.completed_at = completed_at
    record.updated_at = completed_at
    record.lease_owner = None
    record.lease_token = None
    record.lease_expires_at = None
    record.last_error_code = None
    record.last_error_summary = None


async def _persist_single_classification(
    session: AsyncSession,
    *,
    run_record: AgentRunRecord,
    classification_created_at: datetime,
    classification_id: UUID | None = None,
) -> PersistedClassificationFixture:
    started_at = classification_created_at - timedelta(seconds=5)
    completed_at = classification_created_at + timedelta(seconds=1)

    _mark_run_succeeded(
        record=run_record,
        attempt_count=1,
        first_started_at=started_at,
        completed_at=completed_at,
    )

    attempt = _completed_attempt(
        agent_run_id=run_record.id,
        attempt_number=1,
        started_at=started_at,
        finished_at=completed_at,
        outcome=AgentRunAttemptOutcome.SUCCEEDED,
    )
    invocation = _invocation(
        workspace_id=run_record.workspace_id,
        ticket_id=run_record.ticket_id,
        agent_run_id=run_record.id,
        attempt_id=attempt.id,
        invocation_sequence=1,
        status=LLMInvocationStatus.SUCCEEDED,
        error_code=None,
        created_at=(classification_created_at - timedelta(seconds=1)),
        include_usage=True,
    )
    classification = _classification(
        classification_id=(classification_id or uuid4()),
        workspace_id=run_record.workspace_id,
        ticket_id=run_record.ticket_id,
        agent_run_id=run_record.id,
        accepted_invocation_id=invocation.id,
        created_at=classification_created_at,
    )

    session.add(
        AgentRunAttemptRecord.from_domain(
            attempt,
        ),
    )
    await session.flush()

    session.add(
        LLMInvocationRecord.from_domain(
            invocation,
        ),
    )
    await session.flush()

    session.add(
        TicketClassificationRecord.from_domain(
            classification,
        ),
    )
    await session.commit()

    return PersistedClassificationFixture(
        agent_run_id=run_record.id,
        classification_id=classification.id,
        accepted_invocation_id=invocation.id,
        classification_created_at=(classification.created_at),
        invocation_ids=(invocation.id,),
    )


async def _persist_retry_classification(
    session: AsyncSession,
    *,
    run_record: AgentRunRecord,
    classification_created_at: datetime,
) -> PersistedClassificationFixture:
    first_started_at = classification_created_at - timedelta(seconds=30)
    first_finished_at = classification_created_at - timedelta(seconds=20)
    second_started_at = classification_created_at - timedelta(seconds=10)
    completed_at = classification_created_at + timedelta(seconds=1)

    _mark_run_succeeded(
        record=run_record,
        attempt_count=2,
        first_started_at=first_started_at,
        completed_at=completed_at,
    )

    first_attempt = _completed_attempt(
        agent_run_id=run_record.id,
        attempt_number=1,
        started_at=first_started_at,
        finished_at=first_finished_at,
        outcome=(AgentRunAttemptOutcome.RETRYABLE_FAILURE),
        error_code=LLMErrorCode.TIMEOUT.value,
        error_summary=("The LLM provider request exceeded its configured timeout."),
    )
    second_attempt = _completed_attempt(
        agent_run_id=run_record.id,
        attempt_number=2,
        started_at=second_started_at,
        finished_at=completed_at,
        outcome=AgentRunAttemptOutcome.SUCCEEDED,
    )

    timed_out_invocation = _invocation(
        workspace_id=run_record.workspace_id,
        ticket_id=run_record.ticket_id,
        agent_run_id=run_record.id,
        attempt_id=first_attempt.id,
        invocation_sequence=1,
        status=LLMInvocationStatus.TIMED_OUT,
        error_code=LLMErrorCode.TIMEOUT,
        created_at=(first_started_at + timedelta(seconds=1)),
        include_usage=False,
    )
    incomplete_invocation = _invocation(
        workspace_id=run_record.workspace_id,
        ticket_id=run_record.ticket_id,
        agent_run_id=run_record.id,
        attempt_id=second_attempt.id,
        invocation_sequence=1,
        status=LLMInvocationStatus.INCOMPLETE,
        error_code=LLMErrorCode.INCOMPLETE_RESPONSE,
        created_at=(second_started_at + timedelta(seconds=1)),
        include_usage=False,
    )
    successful_invocation = _invocation(
        workspace_id=run_record.workspace_id,
        ticket_id=run_record.ticket_id,
        agent_run_id=run_record.id,
        attempt_id=second_attempt.id,
        invocation_sequence=2,
        status=LLMInvocationStatus.SUCCEEDED,
        error_code=None,
        created_at=(second_started_at + timedelta(seconds=2)),
        include_usage=True,
    )
    classification = _classification(
        classification_id=uuid4(),
        workspace_id=run_record.workspace_id,
        ticket_id=run_record.ticket_id,
        agent_run_id=run_record.id,
        accepted_invocation_id=successful_invocation.id,
        created_at=classification_created_at,
        category=TicketCategory.SECURITY,
        intent=TicketIntent.REPORT_INCIDENT,
        urgency=TicketUrgency.HIGH,
        sentiment=TicketSentiment.NEGATIVE,
        requires_human_review=True,
    )

    session.add_all(
        [
            AgentRunAttemptRecord.from_domain(
                first_attempt,
            ),
            AgentRunAttemptRecord.from_domain(
                second_attempt,
            ),
        ],
    )
    await session.flush()

    session.add_all(
        [
            LLMInvocationRecord.from_domain(
                timed_out_invocation,
            ),
            LLMInvocationRecord.from_domain(
                incomplete_invocation,
            ),
            LLMInvocationRecord.from_domain(
                successful_invocation,
            ),
        ],
    )
    await session.flush()

    session.add(
        TicketClassificationRecord.from_domain(
            classification,
        ),
    )
    await session.commit()

    return PersistedClassificationFixture(
        agent_run_id=run_record.id,
        classification_id=classification.id,
        accepted_invocation_id=(successful_invocation.id),
        classification_created_at=(classification.created_at),
        invocation_ids=(
            timed_out_invocation.id,
            incomplete_invocation.id,
            successful_invocation.id,
        ),
    )


async def _persist_historical_classification(
    session: AsyncSession,
    *,
    workspace_id: UUID,
    ticket_id: UUID,
    classification_created_at: datetime,
    trigger_key: str,
) -> PersistedClassificationFixture:
    run_created_at = classification_created_at - timedelta(seconds=10)
    first_started_at = classification_created_at - timedelta(seconds=5)
    completed_at = classification_created_at + timedelta(seconds=1)
    agent_run = AgentRun(
        id=uuid4(),
        workspace_id=workspace_id,
        ticket_id=ticket_id,
        workflow_name=(INITIAL_TICKET_PROCESSING_WORKFLOW_NAME),
        workflow_version=(TICKET_CLASSIFICATION_WORKFLOW_VERSION),
        trigger_key=trigger_key,
        status=AgentRunStatus.SUCCEEDED,
        available_at=run_created_at,
        attempt_count=1,
        retryable_failure_count=0,
        max_retryable_failures=3,
        lease_owner=None,
        lease_token=None,
        lease_expires_at=None,
        first_started_at=first_started_at,
        completed_at=completed_at,
        last_error_code=None,
        last_error_summary=None,
        ingestion_request_id=uuid4(),
        correlation_id=uuid4(),
        created_at=run_created_at,
        updated_at=completed_at,
    )
    run_record = AgentRunRecord.from_domain(
        agent_run,
    )

    session.add(run_record)
    await session.flush()

    return await _persist_single_classification(
        session,
        run_record=run_record,
        classification_created_at=(classification_created_at),
    )


async def test_get_classification_returns_public_provenance(
    integration_client: AsyncClient,
    postgresql_session: AsyncSession,
    clean_business_tables: None,
) -> None:
    workspace = await _create_workspace(
        integration_client,
        name="Classification Inspection",
        slug="classification-inspection",
    )
    ticket, run_record = await _create_ticket_and_initial_run(
        integration_client,
        postgresql_session,
        workspace_id=workspace["id"],
    )
    fixture = await _persist_single_classification(
        postgresql_session,
        run_record=run_record,
        classification_created_at=(run_record.created_at + timedelta(minutes=1)),
    )

    response = await integration_client.get(
        (
            f"/api/v1/workspaces/{workspace['id']}"
            f"/ticket-classifications/"
            f"{fixture.classification_id}"
        ),
    )

    assert response.status_code == 200

    payload = response.json()

    assert payload["id"] == str(
        fixture.classification_id,
    )
    assert payload["workspace_id"] == workspace["id"]
    assert payload["ticket_id"] == ticket["id"]
    assert payload["agent_run_id"] == str(
        fixture.agent_run_id,
    )
    assert payload["accepted_invocation_id"] == str(
        fixture.accepted_invocation_id,
    )
    assert payload["category"] == "billing"
    assert payload["intent"] == "ask_question"
    assert payload["urgency"] == "normal"
    assert payload["sentiment"] == "neutral"
    assert payload["requires_human_review"] is False
    assert payload["schema_version"] == (TICKET_CLASSIFICATION_SCHEMA_VERSION)
    assert payload["prompt"] == {
        "id": "ticket-classification",
        "version": 1,
        "content_hash": _PROMPT_HASH,
    }
    assert payload["provider"] == "mock"
    assert payload["model"] == _MODEL
    assert payload["created_at"] is not None

    serialized_payload = str(payload)

    assert "provider_request_id" not in serialized_payload
    assert "raw_prompt" not in serialized_payload
    assert "raw_response" not in serialized_payload
    assert "lease_token" not in serialized_payload
    assert "execution_request_id" not in serialized_payload
    assert "updated_at" not in payload


async def test_classification_detail_hides_cross_workspace_resource(
    integration_client: AsyncClient,
    postgresql_session: AsyncSession,
    clean_business_tables: None,
) -> None:
    workspace_a = await _create_workspace(
        integration_client,
        name="Classification Workspace A",
        slug="classification-workspace-a",
    )
    workspace_b = await _create_workspace(
        integration_client,
        name="Classification Workspace B",
        slug="classification-workspace-b",
    )
    _, run_record = await _create_ticket_and_initial_run(
        integration_client,
        postgresql_session,
        workspace_id=workspace_a["id"],
    )
    fixture = await _persist_single_classification(
        postgresql_session,
        run_record=run_record,
        classification_created_at=(run_record.created_at + timedelta(minutes=1)),
    )

    response = await integration_client.get(
        (
            f"/api/v1/workspaces/{workspace_b['id']}"
            f"/ticket-classifications/"
            f"{fixture.classification_id}"
        ),
    )

    assert response.status_code == 404
    assert response.json() == {
        "error": {
            "code": "ticket_classification_not_found",
            "message": ("Ticket classification was not found."),
            "request_id": response.headers["X-Request-ID"],
        },
    }


async def test_ticket_classification_history_is_keyset_paginated(
    integration_client: AsyncClient,
    postgresql_session: AsyncSession,
    clean_business_tables: None,
) -> None:
    workspace = await _create_workspace(
        integration_client,
        name="Classification History",
        slug="classification-history",
    )
    ticket, initial_run = await _create_ticket_and_initial_run(
        integration_client,
        postgresql_session,
        workspace_id=workspace["id"],
    )
    workspace_id = UUID(workspace["id"])
    ticket_id = UUID(
        cast(
            str,
            ticket["id"],
        ),
    )
    base_time = initial_run.created_at

    oldest = await _persist_historical_classification(
        postgresql_session,
        workspace_id=workspace_id,
        ticket_id=ticket_id,
        classification_created_at=(base_time + timedelta(minutes=1)),
        trigger_key="inspection-history-001",
    )
    middle = await _persist_historical_classification(
        postgresql_session,
        workspace_id=workspace_id,
        ticket_id=ticket_id,
        classification_created_at=(base_time + timedelta(minutes=2)),
        trigger_key="inspection-history-002",
    )
    newest = await _persist_historical_classification(
        postgresql_session,
        workspace_id=workspace_id,
        ticket_id=ticket_id,
        classification_created_at=(base_time + timedelta(minutes=3)),
        trigger_key="inspection-history-003",
    )

    first_page_response = await integration_client.get(
        (f"/api/v1/workspaces/{workspace['id']}/tickets/{ticket['id']}/classifications"),
        params={
            "page_size": 2,
        },
    )

    assert first_page_response.status_code == 200

    first_page = first_page_response.json()

    assert [item["id"] for item in first_page["items"]] == [
        str(newest.classification_id),
        str(middle.classification_id),
    ]
    assert first_page["next_cursor"] is not None

    second_page_response = await integration_client.get(
        (f"/api/v1/workspaces/{workspace['id']}/tickets/{ticket['id']}/classifications"),
        params={
            "page_size": 2,
            "cursor": first_page["next_cursor"],
        },
    )

    assert second_page_response.status_code == 200

    second_page = second_page_response.json()

    assert [item["id"] for item in second_page["items"]] == [
        str(oldest.classification_id),
    ]
    assert second_page["next_cursor"] is None

    all_ids = [item["id"] for item in (first_page["items"] + second_page["items"])]

    assert len(all_ids) == 3
    assert len(set(all_ids)) == 3


async def test_ticket_classification_history_hides_cross_workspace_ticket(
    integration_client: AsyncClient,
    postgresql_session: AsyncSession,
    clean_business_tables: None,
) -> None:
    workspace_a = await _create_workspace(
        integration_client,
        name="Ticket Workspace A",
        slug="ticket-workspace-a",
    )
    workspace_b = await _create_workspace(
        integration_client,
        name="Ticket Workspace B",
        slug="ticket-workspace-b",
    )
    ticket, _ = await _create_ticket_and_initial_run(
        integration_client,
        postgresql_session,
        workspace_id=workspace_a["id"],
    )

    response = await integration_client.get(
        (f"/api/v1/workspaces/{workspace_b['id']}/tickets/{ticket['id']}/classifications"),
    )

    assert response.status_code == 404
    assert response.json()["error"]["code"] == ("ticket_not_found")


async def test_ticket_classification_history_rejects_invalid_cursor(
    integration_client: AsyncClient,
    postgresql_session: AsyncSession,
    clean_business_tables: None,
) -> None:
    workspace = await _create_workspace(
        integration_client,
        name="Invalid Cursor",
        slug="invalid-classification-cursor",
    )
    ticket, _ = await _create_ticket_and_initial_run(
        integration_client,
        postgresql_session,
        workspace_id=workspace["id"],
    )

    response = await integration_client.get(
        (f"/api/v1/workspaces/{workspace['id']}/tickets/{ticket['id']}/classifications"),
        params={
            "cursor": "not-a-valid-cursor",
        },
    )

    assert response.status_code == 400
    assert response.json() == {
        "error": {
            "code": "invalid_pagination_cursor",
            "message": "Pagination cursor is invalid.",
            "request_id": response.headers["X-Request-ID"],
        },
    }


async def test_agent_run_detail_includes_classification_reference(
    integration_client: AsyncClient,
    postgresql_session: AsyncSession,
    clean_business_tables: None,
) -> None:
    workspace = await _create_workspace(
        integration_client,
        name="AgentRun Classification",
        slug="agent-run-classification",
    )
    _, run_record = await _create_ticket_and_initial_run(
        integration_client,
        postgresql_session,
        workspace_id=workspace["id"],
    )
    fixture = await _persist_single_classification(
        postgresql_session,
        run_record=run_record,
        classification_created_at=(run_record.created_at + timedelta(minutes=1)),
    )

    response = await integration_client.get(
        (f"/api/v1/workspaces/{workspace['id']}/agent-runs/{fixture.agent_run_id}"),
    )

    assert response.status_code == 200

    payload = response.json()

    assert payload["status"] == "succeeded"
    assert payload["classification"] == {
        "id": str(fixture.classification_id),
        "schema_version": (TICKET_CLASSIFICATION_SCHEMA_VERSION),
        "created_at": (fixture.classification_created_at.isoformat().replace("+00:00", "Z")),
    }


async def test_agent_run_without_classification_returns_null_reference(
    integration_client: AsyncClient,
    postgresql_session: AsyncSession,
    clean_business_tables: None,
) -> None:
    workspace = await _create_workspace(
        integration_client,
        name="Unclassified AgentRun",
        slug="unclassified-agent-run",
    )
    _, run_record = await _create_ticket_and_initial_run(
        integration_client,
        postgresql_session,
        workspace_id=workspace["id"],
    )

    response = await integration_client.get(
        (f"/api/v1/workspaces/{workspace['id']}/agent-runs/{run_record.id}"),
    )

    assert response.status_code == 200
    assert response.json()["classification"] is None


async def test_agent_run_invocations_return_ordered_safe_history(
    integration_client: AsyncClient,
    postgresql_session: AsyncSession,
    clean_business_tables: None,
) -> None:
    workspace = await _create_workspace(
        integration_client,
        name="Invocation Inspection",
        slug="invocation-inspection",
    )
    _, run_record = await _create_ticket_and_initial_run(
        integration_client,
        postgresql_session,
        workspace_id=workspace["id"],
    )
    fixture = await _persist_retry_classification(
        postgresql_session,
        run_record=run_record,
        classification_created_at=(run_record.created_at + timedelta(minutes=1)),
    )

    response = await integration_client.get(
        (f"/api/v1/workspaces/{workspace['id']}/agent-runs/{fixture.agent_run_id}/llm-invocations"),
    )

    assert response.status_code == 200

    items = response.json()["items"]

    assert [
        (
            item["attempt_number"],
            item["invocation_sequence"],
        )
        for item in items
    ] == [
        (
            1,
            1,
        ),
        (
            2,
            1,
        ),
        (
            2,
            2,
        ),
    ]
    assert [item["id"] for item in items] == [
        str(invocation_id) for invocation_id in fixture.invocation_ids
    ]

    timed_out, incomplete, succeeded = items

    assert timed_out["status"] == "timed_out"
    assert timed_out["error_code"] == "llm_timeout"
    assert timed_out["usage"] is None
    assert timed_out["estimated_cost"] == {
        "pricing_catalog_version": (PRICING_CATALOG_VERSION),
        "pricing_found": True,
        "input_cost_usd": None,
        "cached_input_cost_usd": None,
        "output_cost_usd": None,
        "total_cost_usd": None,
    }

    assert incomplete["status"] == "incomplete"
    assert incomplete["error_code"] == ("llm_incomplete_response")
    assert incomplete["usage"] is None

    assert succeeded["status"] == "succeeded"
    assert succeeded["error_code"] is None
    assert succeeded["provider"] == "mock"
    assert succeeded["model"] == _MODEL
    assert succeeded["prompt"] == {
        "id": "ticket-classification",
        "version": 1,
        "content_hash": _PROMPT_HASH,
    }
    assert succeeded["schema_version"] == (TICKET_CLASSIFICATION_SCHEMA_VERSION)
    assert succeeded["usage"] == {
        "input_tokens": 120,
        "cached_input_tokens": 0,
        "output_tokens": 24,
        "reasoning_tokens": None,
        "total_tokens": 144,
    }
    assert succeeded["estimated_cost"]["pricing_catalog_version"] == PRICING_CATALOG_VERSION
    assert succeeded["estimated_cost"]["pricing_found"] is True
    assert Decimal(
        succeeded["estimated_cost"]["total_cost_usd"],
    ) == Decimal("0")

    for item in items:
        serialized_item = str(item)

        assert "workspace_id" not in item
        assert "ticket_id" not in item
        assert "agent_run_id" not in item
        assert "provider_request_id" not in serialized_item
        assert "raw_prompt" not in serialized_item
        assert "raw_response" not in serialized_item
        assert "lease_token" not in serialized_item
        assert "execution_request_id" not in serialized_item
        assert "worker_id" not in serialized_item


async def test_agent_run_invocation_history_supports_empty_result(
    integration_client: AsyncClient,
    postgresql_session: AsyncSession,
    clean_business_tables: None,
) -> None:
    workspace = await _create_workspace(
        integration_client,
        name="Empty Invocation History",
        slug="empty-invocation-history",
    )
    _, run_record = await _create_ticket_and_initial_run(
        integration_client,
        postgresql_session,
        workspace_id=workspace["id"],
    )

    response = await integration_client.get(
        (f"/api/v1/workspaces/{workspace['id']}/agent-runs/{run_record.id}/llm-invocations"),
    )

    assert response.status_code == 200
    assert response.json() == {
        "items": [],
    }


async def test_agent_run_invocation_history_hides_cross_workspace_run(
    integration_client: AsyncClient,
    postgresql_session: AsyncSession,
    clean_business_tables: None,
) -> None:
    workspace_a = await _create_workspace(
        integration_client,
        name="Invocation Workspace A",
        slug="invocation-workspace-a",
    )
    workspace_b = await _create_workspace(
        integration_client,
        name="Invocation Workspace B",
        slug="invocation-workspace-b",
    )
    _, run_record = await _create_ticket_and_initial_run(
        integration_client,
        postgresql_session,
        workspace_id=workspace_a["id"],
    )

    response = await integration_client.get(
        (f"/api/v1/workspaces/{workspace_b['id']}/agent-runs/{run_record.id}/llm-invocations"),
    )

    assert response.status_code == 404
    assert response.json()["error"]["code"] == ("agent_run_not_found")
