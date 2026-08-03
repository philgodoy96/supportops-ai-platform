"""Integration tests for controlled-support inspection API."""

from datetime import datetime, timedelta
from decimal import Decimal
from typing import cast
from uuid import UUID, uuid4

import pytest
from httpx import AsyncClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from supportops.agent_tools.domain.audit import (
    AgentToolCall,
    AgentToolCallStatus,
)
from supportops.agent_tools.domain.contracts import (
    ToolSafetyLevel,
)
from supportops.agent_tools.infrastructure.models import (
    AgentToolCallRecord,
)
from supportops.ai.gateway.results import (
    LLMInvocationStatus,
)
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
    TICKET_CLASSIFICATION_WORKFLOW_VERSION,
    AgentRunAttempt,
    AgentRunAttemptOutcome,
    AgentRunStatus,
)
from supportops.modules.agent_runs.infrastructure.models import (
    AgentRunAttemptRecord,
    AgentRunRecord,
)
from supportops.modules.support_recommendations.domain.models import (
    SupportRecommendation,
    SupportRecommendationAction,
)
from supportops.modules.support_recommendations.infrastructure.models import (
    SupportRecommendationRecord,
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

    return cast(dict[str, str], response.json())


async def _create_ticket(
    client: AsyncClient,
    session: AsyncSession,
    *,
    workspace_id: str,
) -> tuple[dict[str, object], AgentRunRecord]:
    response = await client.post(
        f"/api/v1/workspaces/{workspace_id}/tickets",
        json={
            "subject": "Unable to reset account access",
            "description": (
                "The customer cannot complete the documented account recovery procedure."
            ),
        },
    )

    assert response.status_code == 201

    ticket = cast(dict[str, object], response.json())
    result = await session.execute(
        select(AgentRunRecord).where(
            AgentRunRecord.ticket_id == UUID(cast(str, ticket["id"])),
        )
    )
    agent_run = result.scalar_one()

    return ticket, agent_run


def _inspection_path(
    *,
    workspace_id: str,
    ticket_id: str,
    agent_run_id: UUID,
) -> str:
    return (
        f"/api/v1/workspaces/{workspace_id}"
        f"/tickets/{ticket_id}"
        f"/agent-runs/{agent_run_id}/inspection"
    )


def _completed_attempt(
    *,
    agent_run_id: UUID,
    attempt_id: UUID,
    lease_token: UUID,
    now: datetime,
) -> AgentRunAttempt:
    return AgentRunAttempt(
        id=attempt_id,
        agent_run_id=agent_run_id,
        attempt_number=1,
        worker_id="inspection-worker",
        lease_token=lease_token,
        execution_request_id=uuid4(),
        started_at=now,
        finished_at=now + timedelta(seconds=5),
        outcome=AgentRunAttemptOutcome.SUCCEEDED,
        error_code=None,
        error_summary=None,
    )


def _failed_attempt(
    *,
    agent_run_id: UUID,
    attempt_id: UUID,
    lease_token: UUID,
    now: datetime,
) -> AgentRunAttempt:
    return AgentRunAttempt(
        id=attempt_id,
        agent_run_id=agent_run_id,
        attempt_number=1,
        worker_id="inspection-worker",
        lease_token=lease_token,
        execution_request_id=uuid4(),
        started_at=now,
        finished_at=now + timedelta(seconds=5),
        outcome=AgentRunAttemptOutcome.TERMINAL_FAILURE,
        error_code="synthetic_terminal_failure",
        error_summary=("The synthetic workflow reached a terminal failure."),
    )


def _active_attempt(
    *,
    agent_run_id: UUID,
    attempt_id: UUID,
    lease_token: UUID,
    now: datetime,
) -> AgentRunAttempt:
    return AgentRunAttempt.start(
        agent_run_id=agent_run_id,
        attempt_number=1,
        worker_id="inspection-worker",
        lease_token=lease_token,
        execution_request_id=uuid4(),
        attempt_id=attempt_id,
        now=now,
    )


def _invocation(
    *,
    workspace_id: UUID,
    ticket_id: UUID,
    agent_run_id: UUID,
    attempt_id: UUID,
    sequence: int,
    prompt_id: str,
    schema_version: str,
    cost: Decimal,
    now: datetime,
) -> LLMInvocation:
    return LLMInvocation.create(
        workspace_id=workspace_id,
        ticket_id=ticket_id,
        agent_run_id=agent_run_id,
        agent_run_attempt_id=attempt_id,
        invocation_sequence=sequence,
        status=LLMInvocationStatus.SUCCEEDED,
        provider="mock",
        model="mock-support-model-v1",
        provider_request_id=(f"private-provider-request-{sequence}"),
        prompt_id=prompt_id,
        prompt_version=1,
        prompt_content_hash=(str(sequence) * 64),
        schema_version=schema_version,
        input_tokens=10 * sequence,
        cached_input_tokens=sequence,
        output_tokens=5 * sequence,
        reasoning_tokens=sequence,
        total_tokens=15 * sequence,
        pricing_catalog_version=(PRICING_CATALOG_VERSION),
        pricing_found=True,
        estimated_input_cost_usd=cost,
        estimated_cached_input_cost_usd=Decimal("0"),
        estimated_output_cost_usd=Decimal("0"),
        estimated_total_cost_usd=cost,
        latency_ms=20 * sequence,
        error_code=None,
        invocation_id=uuid4(),
        now=now + timedelta(seconds=sequence),
    )


def _service_status_tool_call(
    *,
    workspace_id: UUID,
    ticket_id: UUID,
    agent_run_id: UUID,
    attempt_id: UUID,
    sequence: int,
    now: datetime,
) -> AgentToolCall:
    started_at = now + timedelta(seconds=sequence)

    return AgentToolCall.create_terminal(
        tool_call_id=uuid4(),
        workspace_id=workspace_id,
        ticket_id=ticket_id,
        agent_run_id=agent_run_id,
        agent_run_attempt_id=attempt_id,
        sequence=sequence,
        provider_tool_call_id=(f"private-provider-tool-call-{sequence}"),
        tool_name="lookup_service_status",
        tool_version=1,
        safety_level=ToolSafetyLevel.READ_ONLY,
        status=AgentToolCallStatus.SUCCEEDED,
        input_fingerprint=("a" * 64 if sequence == 1 else "b" * 64),
        safe_input={
            "service_name": ("payments-api" if sequence == 1 else "identity-api"),
        },
        safe_output={
            "service_name": ("payments-api" if sequence == 1 else "identity-api"),
            "status": ("operational" if sequence == 1 else "degraded"),
            "incident_reference": (None if sequence == 1 else "incident-local-001"),
            "has_incident": sequence == 2,
            "source": "deterministic_catalog",
        },
        latency_ms=sequence,
        error_code=None,
        started_at=started_at,
        finished_at=started_at,
    )


async def _persist_running_run(
    session: AsyncSession,
    *,
    record: AgentRunRecord,
) -> None:
    attempt_id = uuid4()
    lease_token = uuid4()
    now = record.created_at

    record.status = AgentRunStatus.RUNNING.value
    record.attempt_count = 1
    record.lease_owner = "inspection-worker"
    record.lease_token = lease_token
    record.lease_expires_at = now + timedelta(minutes=5)
    record.first_started_at = now
    record.updated_at = now

    session.add(
        AgentRunAttemptRecord.from_domain(
            _active_attempt(
                agent_run_id=record.id,
                attempt_id=attempt_id,
                lease_token=lease_token,
                now=now,
            )
        )
    )
    await session.commit()


async def _persist_failed_run(
    session: AsyncSession,
    *,
    record: AgentRunRecord,
) -> None:
    attempt_id = uuid4()
    lease_token = uuid4()
    now = record.created_at
    completed_at = now + timedelta(seconds=5)

    record.status = AgentRunStatus.FAILED.value
    record.attempt_count = 1
    record.first_started_at = now
    record.completed_at = completed_at
    record.last_error_code = "synthetic_terminal_failure"
    record.last_error_summary = "The synthetic workflow reached a terminal failure."
    record.updated_at = completed_at

    session.add(
        AgentRunAttemptRecord.from_domain(
            _failed_attempt(
                agent_run_id=record.id,
                attempt_id=attempt_id,
                lease_token=lease_token,
                now=now,
            )
        )
    )
    await session.commit()


async def _persist_completed_run(
    session: AsyncSession,
    *,
    record: AgentRunRecord,
) -> None:
    workspace_id = record.workspace_id
    ticket_id = record.ticket_id
    agent_run_id = record.id
    attempt_id = uuid4()
    lease_token = uuid4()
    now = record.created_at
    completed_at = now + timedelta(seconds=10)

    classification_invocation = _invocation(
        workspace_id=workspace_id,
        ticket_id=ticket_id,
        agent_run_id=agent_run_id,
        attempt_id=attempt_id,
        sequence=1,
        prompt_id="ticket-classification",
        schema_version=(TICKET_CLASSIFICATION_SCHEMA_VERSION),
        cost=Decimal("0.001000"),
        now=now,
    )
    decision_invocation = _invocation(
        workspace_id=workspace_id,
        ticket_id=ticket_id,
        agent_run_id=agent_run_id,
        attempt_id=attempt_id,
        sequence=2,
        prompt_id="support-action-decision",
        schema_version="provider-tool-decision-v1",
        cost=Decimal("0.002000"),
        now=now,
    )
    recommendation_invocation = _invocation(
        workspace_id=workspace_id,
        ticket_id=ticket_id,
        agent_run_id=agent_run_id,
        attempt_id=attempt_id,
        sequence=3,
        prompt_id="support-recommendation-draft",
        schema_version="support-recommendation-v1",
        cost=Decimal("0.003000"),
        now=now,
    )
    classification = TicketClassification.create(
        workspace_id=workspace_id,
        ticket_id=ticket_id,
        agent_run_id=agent_run_id,
        category=TicketCategory.ACCOUNT_ACCESS,
        intent=TicketIntent.REQUEST_ACCESS,
        urgency=TicketUrgency.NORMAL,
        sentiment=TicketSentiment.NEUTRAL,
        requires_human_review=False,
        summary=("The customer needs documented account recovery guidance."),
        schema_version=(TICKET_CLASSIFICATION_SCHEMA_VERSION),
        prompt_id="ticket-classification",
        prompt_version=1,
        prompt_content_hash="c" * 64,
        provider="mock",
        model="mock-support-model-v1",
        accepted_llm_invocation_id=(classification_invocation.id),
        classification_id=uuid4(),
        now=now + timedelta(seconds=1),
    )
    recommendation = SupportRecommendation.create(
        workspace_id=workspace_id,
        ticket_id=ticket_id,
        agent_run_id=agent_run_id,
        classification_id=classification.id,
        accepted_llm_invocation_id=(recommendation_invocation.id),
        recommended_action=(SupportRecommendationAction.RESPOND),
        response_text=("Follow the documented account recovery steps and verify access afterward."),
        requires_human_review=False,
        decision_summary=(
            "The persisted classification and service status support a direct response."
        ),
        prompt_id="support-recommendation-draft",
        prompt_version=1,
        prompt_content_hash="d" * 64,
        provider="mock",
        model="mock-support-model-v1",
        recommendation_id=uuid4(),
        now=now + timedelta(seconds=9),
    )
    tool_calls = (
        _service_status_tool_call(
            workspace_id=workspace_id,
            ticket_id=ticket_id,
            agent_run_id=agent_run_id,
            attempt_id=attempt_id,
            sequence=1,
            now=now,
        ),
        _service_status_tool_call(
            workspace_id=workspace_id,
            ticket_id=ticket_id,
            agent_run_id=agent_run_id,
            attempt_id=attempt_id,
            sequence=2,
            now=now,
        ),
    )

    record.status = AgentRunStatus.SUCCEEDED.value
    record.attempt_count = 1
    record.first_started_at = now
    record.completed_at = completed_at
    record.updated_at = completed_at

    session.add(
        AgentRunAttemptRecord.from_domain(
            _completed_attempt(
                agent_run_id=agent_run_id,
                attempt_id=attempt_id,
                lease_token=lease_token,
                now=now,
            )
        )
    )
    await session.flush()

    session.add_all(
        [
            LLMInvocationRecord.from_domain(classification_invocation),
            LLMInvocationRecord.from_domain(decision_invocation),
            LLMInvocationRecord.from_domain(recommendation_invocation),
        ]
    )
    await session.flush()

    session.add(TicketClassificationRecord.from_domain(classification))
    await session.flush()

    session.add_all(
        [
            AgentToolCallRecord.from_domain(tool_calls[1]),
            AgentToolCallRecord.from_domain(tool_calls[0]),
            SupportRecommendationRecord.from_domain(recommendation),
        ]
    )
    await session.commit()


def _collect_keys(value: object) -> set[str]:
    keys: set[str] = set()

    if isinstance(value, dict):
        for key, child in value.items():
            keys.add(str(key))
            keys.update(_collect_keys(child))
    elif isinstance(value, list):
        for child in value:
            keys.update(_collect_keys(child))

    return keys


async def test_returns_queued_partial_inspection(
    integration_client: AsyncClient,
    postgresql_session: AsyncSession,
    clean_business_tables: None,
) -> None:
    del clean_business_tables

    workspace = await _create_workspace(
        integration_client,
        name="Queued Inspection",
        slug="queued-inspection",
    )
    ticket, agent_run = await _create_ticket(
        integration_client,
        postgresql_session,
        workspace_id=workspace["id"],
    )

    response = await integration_client.get(
        _inspection_path(
            workspace_id=workspace["id"],
            ticket_id=cast(str, ticket["id"]),
            agent_run_id=agent_run.id,
        )
    )

    assert response.status_code == 200

    payload = response.json()

    assert payload["agent_run"]["status"] == "queued"
    assert payload["classification"] is None
    assert payload["tool_calls"] == []
    assert payload["recommendation"] is None
    assert payload["llm_invocations"] == []
    assert payload["llm_usage"]["invocation_count"] == 0


async def test_returns_running_partial_inspection_without_lease_data(
    integration_client: AsyncClient,
    postgresql_session: AsyncSession,
    clean_business_tables: None,
) -> None:
    del clean_business_tables

    workspace = await _create_workspace(
        integration_client,
        name="Running Inspection",
        slug="running-inspection",
    )
    ticket, agent_run = await _create_ticket(
        integration_client,
        postgresql_session,
        workspace_id=workspace["id"],
    )
    await _persist_running_run(
        postgresql_session,
        record=agent_run,
    )

    response = await integration_client.get(
        _inspection_path(
            workspace_id=workspace["id"],
            ticket_id=cast(str, ticket["id"]),
            agent_run_id=agent_run.id,
        )
    )

    assert response.status_code == 200

    payload = response.json()

    assert payload["agent_run"]["status"] == "running"
    assert payload["recommendation"] is None

    keys = _collect_keys(payload)

    assert "lease_owner" not in keys
    assert "lease_token" not in keys
    assert "lease_expires_at" not in keys
    assert "execution_request_id" not in keys


async def test_returns_failed_inspection_without_recommendation(
    integration_client: AsyncClient,
    postgresql_session: AsyncSession,
    clean_business_tables: None,
) -> None:
    del clean_business_tables

    workspace = await _create_workspace(
        integration_client,
        name="Failed Inspection",
        slug="failed-inspection",
    )
    ticket, agent_run = await _create_ticket(
        integration_client,
        postgresql_session,
        workspace_id=workspace["id"],
    )
    await _persist_failed_run(
        postgresql_session,
        record=agent_run,
    )

    response = await integration_client.get(
        _inspection_path(
            workspace_id=workspace["id"],
            ticket_id=cast(str, ticket["id"]),
            agent_run_id=agent_run.id,
        )
    )

    assert response.status_code == 200

    payload = response.json()

    assert payload["agent_run"]["status"] == "failed"
    assert payload["agent_run"]["last_error_code"] == ("synthetic_terminal_failure")
    assert payload["recommendation"] is None


async def test_returns_completed_ordered_inspection_and_hides_sensitive_data(
    integration_client: AsyncClient,
    postgresql_session: AsyncSession,
    clean_business_tables: None,
) -> None:
    del clean_business_tables

    workspace = await _create_workspace(
        integration_client,
        name="Completed Inspection",
        slug="completed-inspection",
    )
    ticket, agent_run = await _create_ticket(
        integration_client,
        postgresql_session,
        workspace_id=workspace["id"],
    )
    await _persist_completed_run(
        postgresql_session,
        record=agent_run,
    )

    response = await integration_client.get(
        _inspection_path(
            workspace_id=workspace["id"],
            ticket_id=cast(str, ticket["id"]),
            agent_run_id=agent_run.id,
        )
    )

    assert response.status_code == 200

    payload = response.json()

    assert payload["agent_run"]["status"] == "completed"
    assert payload["classification"]["category"] == ("account_access")
    assert payload["classification"]["intent"] == ("request_access")

    assert [item["sequence"] for item in payload["tool_calls"]] == [
        1,
        2,
    ]
    assert [item["result_summary"]["service_name"] for item in payload["tool_calls"]] == [
        "payments-api",
        "identity-api",
    ]

    assert [item["invocation_sequence"] for item in payload["llm_invocations"]] == [
        1,
        2,
        3,
    ]
    assert [item["prompt_id"] for item in payload["llm_invocations"]] == [
        "ticket-classification",
        "support-action-decision",
        "support-recommendation-draft",
    ]

    usage = payload["llm_usage"]

    assert usage["invocation_count"] == 3
    assert usage["successful_invocation_count"] == 3
    assert usage["failed_invocation_count"] == 0
    assert usage["input_tokens"] == 60
    assert usage["cached_input_tokens"] == 6
    assert usage["output_tokens"] == 30
    assert usage["reasoning_tokens"] == 6
    assert usage["total_tokens"] == 90
    assert Decimal(usage["estimated_cost_usd"]) == Decimal("0.006000")
    assert usage["unpriced_invocation_count"] == 0

    recommendation = payload["recommendation"]

    assert recommendation["recommended_action"] == "respond"
    assert recommendation["requires_human_review"] is False
    assert recommendation["citations"] == []

    private_keys = {
        "lease_owner",
        "lease_token",
        "lease_expires_at",
        "execution_request_id",
        "provider_request_id",
        "provider_tool_call_id",
        "safe_input",
        "safe_output",
        "input_fingerprint",
        "checkpoint_id",
        "checkpoint_ns",
        "checkpoint",
        "prompt_content",
        "raw_request",
        "raw_response",
    }

    assert _collect_keys(payload).isdisjoint(private_keys)


async def test_hides_cross_workspace_inspection_as_not_found(
    integration_client: AsyncClient,
    postgresql_session: AsyncSession,
    clean_business_tables: None,
) -> None:
    del clean_business_tables

    workspace_a = await _create_workspace(
        integration_client,
        name="Inspection Workspace A",
        slug="inspection-workspace-a",
    )
    workspace_b = await _create_workspace(
        integration_client,
        name="Inspection Workspace B",
        slug="inspection-workspace-b",
    )
    ticket, agent_run = await _create_ticket(
        integration_client,
        postgresql_session,
        workspace_id=workspace_a["id"],
    )

    response = await integration_client.get(
        _inspection_path(
            workspace_id=workspace_b["id"],
            ticket_id=cast(str, ticket["id"]),
            agent_run_id=agent_run.id,
        )
    )

    assert response.status_code == 404
    assert response.json() == {
        "error": {
            "code": ("controlled_support_inspection_not_found"),
            "message": ("The controlled support inspection was not found."),
            "request_id": response.headers["X-Request-ID"],
        },
    }


async def test_returns_conflict_for_unsupported_workflow(
    integration_client: AsyncClient,
    postgresql_session: AsyncSession,
    clean_business_tables: None,
) -> None:
    del clean_business_tables

    workspace = await _create_workspace(
        integration_client,
        name="Unsupported Inspection",
        slug="unsupported-inspection",
    )
    ticket, agent_run = await _create_ticket(
        integration_client,
        postgresql_session,
        workspace_id=workspace["id"],
    )

    agent_run.workflow_version = TICKET_CLASSIFICATION_WORKFLOW_VERSION
    await postgresql_session.commit()

    response = await integration_client.get(
        _inspection_path(
            workspace_id=workspace["id"],
            ticket_id=cast(str, ticket["id"]),
            agent_run_id=agent_run.id,
        )
    )

    assert response.status_code == 409
    assert response.json() == {
        "error": {
            "code": "unsupported_agent_run_inspection",
            "message": ("The AgentRun workflow does not support this inspection view."),
            "request_id": response.headers["X-Request-ID"],
        },
    }
