"""Integration tests for controlled-support inspection reads."""

from dataclasses import replace
from datetime import UTC, datetime, timedelta
from uuid import UUID

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from supportops.agent_graph.domain.state import (
    CONTROLLED_SUPPORT_WORKFLOW_VERSION,
)
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
from supportops.infrastructure.postgresql.transaction import (
    SqlAlchemyTransactionManager,
)
from supportops.modules.agent_runs.domain.models import (
    AgentRun,
    AgentRunAttempt,
    AgentRunAttemptOutcome,
    AgentRunStatus,
)
from supportops.modules.agent_runs.infrastructure.models import (
    AgentRunAttemptRecord,
)
from supportops.modules.agent_runs.infrastructure.repository import (
    SqlAlchemyAgentRunRepository,
)
from supportops.modules.controlled_support_inspection.application.repository import (
    ControlledSupportInspectionIdentity,
)
from supportops.modules.controlled_support_inspection.infrastructure.repository import (
    SqlAlchemyControlledSupportInspectionRepository,
)
from supportops.modules.ticket_classifications.domain.models import (
    LLMInvocation,
)
from supportops.modules.ticket_classifications.infrastructure.models import (
    LLMInvocationRecord,
)
from supportops.modules.tickets.domain.models import (
    Ticket,
)
from supportops.modules.tickets.infrastructure.repository import (
    SqlAlchemyTicketRepository,
)
from supportops.modules.workspaces.domain.models import (
    Workspace,
)
from supportops.modules.workspaces.infrastructure.repository import (
    SqlAlchemyWorkspaceRepository,
)

pytestmark = pytest.mark.integration

_WORKSPACE_ID = UUID("41000000-0000-4000-8000-000000000001")
_OTHER_WORKSPACE_ID = UUID("42000000-0000-4000-8000-000000000002")
_TICKET_ID = UUID("43000000-0000-4000-8000-000000000003")
_AGENT_RUN_ID = UUID("44000000-0000-4000-8000-000000000004")
_ATTEMPT_ONE_ID = UUID("45000000-0000-4000-8000-000000000005")
_ATTEMPT_TWO_ID = UUID("46000000-0000-4000-8000-000000000006")
_TOOL_ONE_ID = UUID("47000000-0000-4000-8000-000000000007")
_TOOL_TWO_ID = UUID("48000000-0000-4000-8000-000000000008")
_INVOCATION_ONE_ID = UUID("49000000-0000-4000-8000-000000000009")
_INVOCATION_TWO_ID = UUID("4a000000-0000-4000-8000-000000000010")

_CREATED_AT = datetime(
    2026,
    8,
    2,
    18,
    0,
    tzinfo=UTC,
)
_ATTEMPT_ONE_STARTED_AT = _CREATED_AT + timedelta(seconds=1)
_ATTEMPT_ONE_FINISHED_AT = _CREATED_AT + timedelta(seconds=2)
_ATTEMPT_TWO_STARTED_AT = _CREATED_AT + timedelta(seconds=3)
_ATTEMPT_TWO_FINISHED_AT = _CREATED_AT + timedelta(seconds=4)


def _failed_run(
    ticket: Ticket,
) -> AgentRun:
    initial = AgentRun.create_initial(
        agent_run_id=_AGENT_RUN_ID,
        workspace_id=_WORKSPACE_ID,
        ticket_id=_TICKET_ID,
        ingestion_request_id=(ticket.ingestion_request_id),
        correlation_id=ticket.correlation_id,
        workflow_version=(CONTROLLED_SUPPORT_WORKFLOW_VERSION),
        max_retryable_failures=3,
        now=_CREATED_AT,
    )

    return replace(
        initial,
        status=AgentRunStatus.FAILED,
        attempt_count=2,
        first_started_at=_ATTEMPT_ONE_STARTED_AT,
        completed_at=_ATTEMPT_TWO_FINISHED_AT,
        last_error_code="synthetic_terminal_failure",
        last_error_summary=("The synthetic integration workflow failed."),
        updated_at=_ATTEMPT_TWO_FINISHED_AT,
    )


def _attempts() -> tuple[
    AgentRunAttempt,
    AgentRunAttempt,
]:
    first = AgentRunAttempt(
        id=_ATTEMPT_ONE_ID,
        agent_run_id=_AGENT_RUN_ID,
        attempt_number=1,
        worker_id="inspection-worker-1",
        lease_token=UUID("4b000000-0000-4000-8000-000000000011"),
        execution_request_id=UUID("4c000000-0000-4000-8000-000000000012"),
        started_at=_ATTEMPT_ONE_STARTED_AT,
        finished_at=_ATTEMPT_ONE_FINISHED_AT,
        outcome=(AgentRunAttemptOutcome.RETRYABLE_FAILURE),
        error_code="synthetic_retryable_failure",
        error_summary=("The first synthetic attempt was retryable."),
    )
    second = AgentRunAttempt(
        id=_ATTEMPT_TWO_ID,
        agent_run_id=_AGENT_RUN_ID,
        attempt_number=2,
        worker_id="inspection-worker-2",
        lease_token=UUID("4d000000-0000-4000-8000-000000000013"),
        execution_request_id=UUID("4e000000-0000-4000-8000-000000000014"),
        started_at=_ATTEMPT_TWO_STARTED_AT,
        finished_at=_ATTEMPT_TWO_FINISHED_AT,
        outcome=(AgentRunAttemptOutcome.TERMINAL_FAILURE),
        error_code="synthetic_terminal_failure",
        error_summary=("The second synthetic attempt was terminal."),
    )

    return first, second


def _tool_call(
    *,
    tool_call_id: UUID,
    attempt_id: UUID,
    provider_call_id: str,
    started_at: datetime,
) -> AgentToolCall:
    return AgentToolCall.create_terminal(
        tool_call_id=tool_call_id,
        workspace_id=_WORKSPACE_ID,
        ticket_id=_TICKET_ID,
        agent_run_id=_AGENT_RUN_ID,
        agent_run_attempt_id=attempt_id,
        sequence=1,
        provider_tool_call_id=provider_call_id,
        tool_name="lookup_service_status",
        tool_version=1,
        safety_level=ToolSafetyLevel.READ_ONLY,
        status=AgentToolCallStatus.SUCCEEDED,
        input_fingerprint=("a" * 64 if attempt_id == _ATTEMPT_ONE_ID else "b" * 64),
        safe_input={
            "service_name": "payments-api",
        },
        safe_output={
            "service_name": "payments-api",
            "status": "operational",
            "incident_reference": None,
            "has_incident": False,
            "source": "deterministic_catalog",
        },
        latency_ms=1,
        error_code=None,
        started_at=started_at,
        finished_at=started_at,
    )


def _invocation(
    *,
    invocation_id: UUID,
    attempt_id: UUID,
    provider_request_id: str,
    created_at: datetime,
) -> LLMInvocation:
    return LLMInvocation.create(
        invocation_id=invocation_id,
        workspace_id=_WORKSPACE_ID,
        ticket_id=_TICKET_ID,
        agent_run_id=_AGENT_RUN_ID,
        agent_run_attempt_id=attempt_id,
        invocation_sequence=1,
        status=LLMInvocationStatus.SUCCEEDED,
        provider="mock",
        model="mock-support-model-v1",
        provider_request_id=provider_request_id,
        prompt_id="support-tool-decision",
        prompt_version=1,
        prompt_content_hash="c" * 64,
        schema_version="provider-tool-decision-v1",
        input_tokens=10,
        cached_input_tokens=0,
        output_tokens=5,
        reasoning_tokens=0,
        total_tokens=15,
        pricing_catalog_version=(PRICING_CATALOG_VERSION),
        pricing_found=False,
        estimated_input_cost_usd=None,
        estimated_cached_input_cost_usd=None,
        estimated_output_cost_usd=None,
        estimated_total_cost_usd=None,
        latency_ms=20,
        error_code=None,
        now=created_at,
    )


async def _persist_history(
    session: AsyncSession,
) -> None:
    workspace = Workspace(
        id=_WORKSPACE_ID,
        name="Controlled Inspection",
        slug="controlled-inspection",
        created_at=_CREATED_AT,
        updated_at=_CREATED_AT,
    )
    ticket = Ticket.create(
        ticket_id=_TICKET_ID,
        workspace_id=_WORKSPACE_ID,
        subject="Unable to reset account access",
        description=("The customer cannot complete the documented account recovery procedure."),
        external_reference=None,
        ingestion_request_id=UUID("4f000000-0000-4000-8000-000000000015"),
        correlation_id=UUID("50000000-0000-4000-8000-000000000016"),
        now=_CREATED_AT,
    )
    agent_run = _failed_run(ticket)
    first_attempt, second_attempt = _attempts()
    first_tool = _tool_call(
        tool_call_id=_TOOL_ONE_ID,
        attempt_id=_ATTEMPT_ONE_ID,
        provider_call_id="provider-call-attempt-1",
        started_at=_ATTEMPT_ONE_STARTED_AT,
    )
    second_tool = _tool_call(
        tool_call_id=_TOOL_TWO_ID,
        attempt_id=_ATTEMPT_TWO_ID,
        provider_call_id="provider-call-attempt-2",
        started_at=_ATTEMPT_TWO_STARTED_AT,
    )
    first_invocation = _invocation(
        invocation_id=_INVOCATION_ONE_ID,
        attempt_id=_ATTEMPT_ONE_ID,
        provider_request_id="provider-request-attempt-1",
        created_at=_ATTEMPT_ONE_STARTED_AT,
    )
    second_invocation = _invocation(
        invocation_id=_INVOCATION_TWO_ID,
        attempt_id=_ATTEMPT_TWO_ID,
        provider_request_id="provider-request-attempt-2",
        created_at=_ATTEMPT_TWO_STARTED_AT,
    )

    async with SqlAlchemyTransactionManager(session).transaction():
        await SqlAlchemyWorkspaceRepository(session).add(workspace)
        await SqlAlchemyTicketRepository(session).add(ticket)
        await SqlAlchemyAgentRunRepository(session).add(agent_run)

        session.add_all(
            [
                AgentRunAttemptRecord.from_domain(first_attempt),
                AgentRunAttemptRecord.from_domain(second_attempt),
            ]
        )
        await session.flush()

        session.add_all(
            [
                AgentToolCallRecord.from_domain(first_tool),
                AgentToolCallRecord.from_domain(second_tool),
                LLMInvocationRecord.from_domain(first_invocation),
                LLMInvocationRecord.from_domain(second_invocation),
            ]
        )


async def test_loads_attempt_scoped_history_in_order(
    postgresql_session: AsyncSession,
    clean_business_tables: None,
) -> None:
    del clean_business_tables

    await _persist_history(postgresql_session)
    repository = SqlAlchemyControlledSupportInspectionRepository(postgresql_session)
    transaction_manager = SqlAlchemyTransactionManager(postgresql_session)

    async with transaction_manager.transaction():
        data = await repository.get_inspection_data(
            ControlledSupportInspectionIdentity(
                workspace_id=_WORKSPACE_ID,
                ticket_id=_TICKET_ID,
                agent_run_id=_AGENT_RUN_ID,
            )
        )

    assert data is not None
    assert data.agent_run.id == _AGENT_RUN_ID
    assert tuple(attempt.attempt_number for attempt in data.attempts) == (
        1,
        2,
    )
    assert tuple(tool_call.proposed_by_agent_run_attempt_id for tool_call in data.tool_calls) == (
        _ATTEMPT_ONE_ID,
        _ATTEMPT_TWO_ID,
    )
    assert tuple(tool_call.executed_by_agent_run_attempt_id for tool_call in data.tool_calls) == (
        _ATTEMPT_ONE_ID,
        _ATTEMPT_TWO_ID,
    )
    assert tuple(invocation.agent_run_attempt_id for invocation in data.llm_invocations) == (
        _ATTEMPT_ONE_ID,
        _ATTEMPT_TWO_ID,
    )
    assert data.classification is None
    assert data.recommendation is None
    assert data.citations == ()


async def test_cross_workspace_lookup_returns_none(
    postgresql_session: AsyncSession,
    clean_business_tables: None,
) -> None:
    del clean_business_tables

    await _persist_history(postgresql_session)
    repository = SqlAlchemyControlledSupportInspectionRepository(postgresql_session)

    async with SqlAlchemyTransactionManager(postgresql_session).transaction():
        data = await repository.get_inspection_data(
            ControlledSupportInspectionIdentity(
                workspace_id=_OTHER_WORKSPACE_ID,
                ticket_id=_TICKET_ID,
                agent_run_id=_AGENT_RUN_ID,
            )
        )

    assert data is None


async def test_cross_ticket_lookup_returns_none(
    postgresql_session: AsyncSession,
    clean_business_tables: None,
) -> None:
    del clean_business_tables

    await _persist_history(postgresql_session)
    repository = SqlAlchemyControlledSupportInspectionRepository(postgresql_session)

    async with SqlAlchemyTransactionManager(postgresql_session).transaction():
        data = await repository.get_inspection_data(
            ControlledSupportInspectionIdentity(
                workspace_id=_WORKSPACE_ID,
                ticket_id=UUID("51000000-0000-4000-8000-000000000017"),
                agent_run_id=_AGENT_RUN_ID,
            )
        )

    assert data is None
