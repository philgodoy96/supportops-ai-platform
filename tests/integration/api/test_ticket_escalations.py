"""Integration tests for ticket escalation inspection."""

from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from uuid import UUID

import pytest
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from supportops.agent_tools.application.grant_persistence import (
    SensitiveExecutionGrantPersistenceResult,
)
from supportops.agent_tools.application.persistence import (
    AgentToolCallPersistenceResult,
    PersistAgentToolCallCommand,
)
from supportops.agent_tools.domain.audit import AgentToolCall
from supportops.agent_tools.domain.grants import SensitiveExecutionGrant
from supportops.agent_tools.infrastructure.grant_repository import (
    SqlAlchemySensitiveExecutionGrantRepository,
)
from supportops.agent_tools.infrastructure.repository import (
    SqlAlchemyAgentToolCallExecutionRepository,
)
from supportops.agent_tools.tools.escalate_ticket import (
    EscalateTicketInput,
)
from supportops.ai.gateway.errors import LLMErrorCode
from supportops.ai.gateway.results import LLMInvocationStatus
from supportops.ai.pricing.catalog import PRICING_CATALOG_VERSION
from supportops.ai.schemas.ticket_classification import (
    TICKET_CLASSIFICATION_SCHEMA_VERSION,
)
from supportops.infrastructure.postgresql.transaction import (
    SqlAlchemyTransactionManager,
)
from supportops.modules.agent_runs.domain.claiming import (
    ClaimAgentRunCommand,
)
from supportops.modules.agent_runs.domain.models import (
    DETERMINISTIC_BASELINE_WORKFLOW_VERSION,
    AgentRun,
)
from supportops.modules.agent_runs.infrastructure.repository import (
    SqlAlchemyAgentRunRepository,
)
from supportops.modules.approvals.domain.models import ApprovalRequest
from supportops.modules.approvals.domain.repositories import (
    ApprovalRequestPersistenceResult,
)
from supportops.modules.approvals.infrastructure.repository import (
    SqlAlchemyApprovalRequestRepository,
)
from supportops.modules.ticket_classifications.domain.models import (
    LLMInvocation,
)
from supportops.modules.ticket_classifications.infrastructure.models import (
    LLMInvocationRecord,
)
from supportops.modules.tickets.domain.escalation import TicketEscalation
from supportops.modules.tickets.domain.escalation_repositories import (
    TicketEscalationPersistenceResult,
)
from supportops.modules.tickets.domain.models import Ticket
from supportops.modules.tickets.infrastructure.escalation_repository import (
    SqlAlchemyTicketEscalationRepository,
)
from supportops.modules.tickets.infrastructure.repository import (
    SqlAlchemyTicketRepository,
)
from supportops.modules.workspaces.domain.models import Workspace
from supportops.modules.workspaces.infrastructure.repository import (
    SqlAlchemyWorkspaceRepository,
)

_WORKSPACE_ID = UUID("12000000-0000-4000-8000-000000000021")
_FOREIGN_WORKSPACE_ID = UUID("12000000-0000-4000-8000-000000000091")
_TICKET_ID = UUID("22000000-0000-4000-8000-000000000022")
_OTHER_TICKET_ID = UUID("22000000-0000-4000-8000-000000000032")
_FOREIGN_TICKET_ID = UUID("22000000-0000-4000-8000-000000000092")
_AGENT_RUN_ID = UUID("32000000-0000-4000-8000-000000000023")
_OTHER_AGENT_RUN_ID = UUID("32000000-0000-4000-8000-000000000033")
_FOREIGN_AGENT_RUN_ID = UUID("32000000-0000-4000-8000-000000000093")
_LEASE_TOKEN = UUID("42000000-0000-4000-8000-000000000024")
_OTHER_LEASE_TOKEN = UUID("42000000-0000-4000-8000-000000000034")
_FOREIGN_LEASE_TOKEN = UUID("42000000-0000-4000-8000-000000000094")
_EXECUTION_REQUEST_ID = UUID("52000000-0000-4000-8000-000000000025")
_OTHER_EXECUTION_REQUEST_ID = UUID("52000000-0000-4000-8000-000000000035")
_FOREIGN_EXECUTION_REQUEST_ID = UUID("52000000-0000-4000-8000-000000000095")
_TOOL_CALL_ID = UUID("62000000-0000-4000-8000-000000000026")
_OTHER_TOOL_CALL_ID = UUID("62000000-0000-4000-8000-000000000036")
_FOREIGN_TOOL_CALL_ID = UUID("62000000-0000-4000-8000-000000000096")
_INVOCATION_ID = UUID("82000000-0000-4000-8000-000000000028")
_OTHER_INVOCATION_ID = UUID("82000000-0000-4000-8000-000000000038")
_FOREIGN_INVOCATION_ID = UUID("82000000-0000-4000-8000-000000000098")
_APPROVAL_REQUEST_ID = UUID("92000000-0000-4000-8000-000000000029")
_OTHER_APPROVAL_REQUEST_ID = UUID("92000000-0000-4000-8000-000000000039")
_FOREIGN_APPROVAL_REQUEST_ID = UUID("92000000-0000-4000-8000-000000000099")
_GRANT_ID = UUID("a2000000-0000-4000-8000-00000000002a")
_OTHER_GRANT_ID = UUID("a2000000-0000-4000-8000-00000000003a")
_FOREIGN_GRANT_ID = UUID("a2000000-0000-4000-8000-00000000009a")
_ESCALATION_ID = UUID("d2000000-0000-4000-8000-00000000002d")
_OTHER_ESCALATION_ID = UUID("d2000000-0000-4000-8000-00000000003d")
_FOREIGN_ESCALATION_ID = UUID("d2000000-0000-4000-8000-00000000009d")
_DECISION_REQUEST_ID = UUID("b2000000-0000-4000-8000-00000000002b")
_OTHER_DECISION_REQUEST_ID = UUID("b2000000-0000-4000-8000-00000000003b")
_FOREIGN_DECISION_REQUEST_ID = UUID("b2000000-0000-4000-8000-00000000009b")
_DECISION_CORRELATION_ID = UUID("c2000000-0000-4000-8000-00000000002c")
_OTHER_DECISION_CORRELATION_ID = UUID("c2000000-0000-4000-8000-00000000003c")
_FOREIGN_DECISION_CORRELATION_ID = UUID("c2000000-0000-4000-8000-00000000009c")

_CREATED_AT = datetime(2026, 8, 3, 19, 0, tzinfo=UTC)
_CLAIMED_AT = _CREATED_AT + timedelta(minutes=1)
_LEASE_EXPIRES_AT = _CLAIMED_AT + timedelta(seconds=45)
_TOOL_PROPOSED_AT = _CLAIMED_AT + timedelta(seconds=1)
_INVOCATION_AT = _CLAIMED_AT + timedelta(seconds=2)
_APPROVAL_AT = _CLAIMED_AT + timedelta(seconds=3)
_EXPIRES_AT = _APPROVAL_AT + timedelta(hours=24)
_DECIDED_AT = _APPROVAL_AT + timedelta(minutes=5)
_GRANT_CREATED_AT = _DECIDED_AT + timedelta(minutes=1)
_ESCALATION_CREATED_AT = _GRANT_CREATED_AT + timedelta(minutes=1)


@dataclass(frozen=True, slots=True)
class PersistedTicketEscalationFixture:
    """Seeded escalations for workspace-scoped inspection API tests."""

    workspace_id: UUID
    ticket_id: UUID
    escalation: TicketEscalation
    foreign_workspace_id: UUID
    foreign_escalation: TicketEscalation


@pytest.fixture
async def persisted_ticket_escalation_fixture(
    postgresql_session_factory: async_sessionmaker[AsyncSession],
    clean_business_tables: None,
) -> AsyncIterator[PersistedTicketEscalationFixture]:
    """Persist one primary escalation plus a foreign-workspace sibling."""

    del clean_business_tables

    async with postgresql_session_factory() as session:
        escalation = await _seed_escalation_graph(
            session,
            workspace_id=_WORKSPACE_ID,
            workspace_name="Escalation Inspection Workspace",
            workspace_slug="escalation-inspection-workspace",
            ticket_id=_TICKET_ID,
            agent_run_id=_AGENT_RUN_ID,
            lease_token=_LEASE_TOKEN,
            execution_request_id=_EXECUTION_REQUEST_ID,
            tool_call_id=_TOOL_CALL_ID,
            invocation_id=_INVOCATION_ID,
            approval_request_id=_APPROVAL_REQUEST_ID,
            grant_id=_GRANT_ID,
            decision_request_id=_DECISION_REQUEST_ID,
            decision_correlation_id=_DECISION_CORRELATION_ID,
            escalation_id=_ESCALATION_ID,
            worker_id="escalation-inspection-worker-1",
            provider_tool_call_id="escalation-inspection-call-1",
            input_fingerprint="b" * 64,
            reason="Primary ticket requires escalation review.",
            ingestion_request_id=UUID("81200000-0000-4000-8000-000000000028"),
            correlation_id=UUID("82200000-0000-4000-8000-000000000029"),
            create_workspace=True,
        )
        await _seed_escalation_graph(
            session,
            workspace_id=_WORKSPACE_ID,
            workspace_name="Escalation Inspection Workspace",
            workspace_slug="escalation-inspection-workspace",
            ticket_id=_OTHER_TICKET_ID,
            agent_run_id=_OTHER_AGENT_RUN_ID,
            lease_token=_OTHER_LEASE_TOKEN,
            execution_request_id=_OTHER_EXECUTION_REQUEST_ID,
            tool_call_id=_OTHER_TOOL_CALL_ID,
            invocation_id=_OTHER_INVOCATION_ID,
            approval_request_id=_OTHER_APPROVAL_REQUEST_ID,
            grant_id=_OTHER_GRANT_ID,
            decision_request_id=_OTHER_DECISION_REQUEST_ID,
            decision_correlation_id=_OTHER_DECISION_CORRELATION_ID,
            escalation_id=_OTHER_ESCALATION_ID,
            worker_id="escalation-inspection-worker-2",
            provider_tool_call_id="escalation-inspection-call-2",
            input_fingerprint="e" * 64,
            reason="Sibling ticket escalation in the same workspace.",
            ingestion_request_id=UUID("81200000-0000-4000-8000-000000000038"),
            correlation_id=UUID("82200000-0000-4000-8000-000000000039"),
            create_workspace=False,
        )
        foreign_escalation = await _seed_escalation_graph(
            session,
            workspace_id=_FOREIGN_WORKSPACE_ID,
            workspace_name="Foreign Escalation Workspace",
            workspace_slug="foreign-escalation-workspace",
            ticket_id=_FOREIGN_TICKET_ID,
            agent_run_id=_FOREIGN_AGENT_RUN_ID,
            lease_token=_FOREIGN_LEASE_TOKEN,
            execution_request_id=_FOREIGN_EXECUTION_REQUEST_ID,
            tool_call_id=_FOREIGN_TOOL_CALL_ID,
            invocation_id=_FOREIGN_INVOCATION_ID,
            approval_request_id=_FOREIGN_APPROVAL_REQUEST_ID,
            grant_id=_FOREIGN_GRANT_ID,
            decision_request_id=_FOREIGN_DECISION_REQUEST_ID,
            decision_correlation_id=_FOREIGN_DECISION_CORRELATION_ID,
            escalation_id=_FOREIGN_ESCALATION_ID,
            worker_id="escalation-inspection-worker-foreign",
            provider_tool_call_id="escalation-inspection-call-foreign",
            input_fingerprint="2" * 64,
            reason="Foreign workspace escalation must stay hidden.",
            ingestion_request_id=UUID("81200000-0000-4000-8000-000000000098"),
            correlation_id=UUID("82200000-0000-4000-8000-000000000099"),
            create_workspace=True,
        )

    yield PersistedTicketEscalationFixture(
        workspace_id=_WORKSPACE_ID,
        ticket_id=_TICKET_ID,
        escalation=escalation,
        foreign_workspace_id=_FOREIGN_WORKSPACE_ID,
        foreign_escalation=foreign_escalation,
    )


async def _seed_escalation_graph(
    session: AsyncSession,
    *,
    workspace_id: UUID,
    workspace_name: str,
    workspace_slug: str,
    ticket_id: UUID,
    agent_run_id: UUID,
    lease_token: UUID,
    execution_request_id: UUID,
    tool_call_id: UUID,
    invocation_id: UUID,
    approval_request_id: UUID,
    grant_id: UUID,
    decision_request_id: UUID,
    decision_correlation_id: UUID,
    escalation_id: UUID,
    worker_id: str,
    provider_tool_call_id: str,
    input_fingerprint: str,
    reason: str,
    ingestion_request_id: UUID,
    correlation_id: UUID,
    create_workspace: bool,
) -> TicketEscalation:
    ticket = Ticket.create(
        ticket_id=ticket_id,
        workspace_id=workspace_id,
        subject="Needs escalation inspection",
        description=("The customer requested a policy-sensitive escalation."),
        external_reference=None,
        ingestion_request_id=ingestion_request_id,
        correlation_id=correlation_id,
        now=_CREATED_AT,
    )
    agent_run = AgentRun.create_initial(
        agent_run_id=agent_run_id,
        workspace_id=workspace_id,
        ticket_id=ticket_id,
        ingestion_request_id=ticket.ingestion_request_id,
        correlation_id=ticket.correlation_id,
        workflow_version=DETERMINISTIC_BASELINE_WORKFLOW_VERSION,
        max_retryable_failures=3,
        now=_CREATED_AT,
    )
    transaction_manager = SqlAlchemyTransactionManager(session)

    async with transaction_manager.transaction():
        if create_workspace:
            await SqlAlchemyWorkspaceRepository(session).add(
                Workspace(
                    id=workspace_id,
                    name=workspace_name,
                    slug=workspace_slug,
                    created_at=_CREATED_AT,
                    updated_at=_CREATED_AT,
                ),
            )
        await SqlAlchemyTicketRepository(session).add(ticket)
        await SqlAlchemyAgentRunRepository(session).add(agent_run)

    async with transaction_manager.transaction():
        claim = await SqlAlchemyAgentRunRepository(session).claim_next_available(
            ClaimAgentRunCommand(
                worker_id=worker_id,
                lease_token=lease_token,
                execution_request_id=execution_request_id,
                claimed_at=_CLAIMED_AT,
                lease_expires_at=_LEASE_EXPIRES_AT,
            ),
        )

    assert claim is not None
    assert claim.agent_run.id == agent_run_id

    tool_call = AgentToolCall.propose_for_approval(
        tool_call_id=tool_call_id,
        workspace_id=workspace_id,
        ticket_id=ticket_id,
        agent_run_id=claim.agent_run.id,
        proposed_by_agent_run_attempt_id=claim.attempt.id,
        sequence=1,
        provider_tool_call_id=provider_tool_call_id,
        tool_name="escalate_ticket",
        tool_version=1,
        input_fingerprint=input_fingerprint,
        safe_input={
            "target_queue": "support_operations",
            "reason": reason,
        },
        proposed_at=_TOOL_PROPOSED_AT,
    )
    invocation = LLMInvocation.create(
        invocation_id=invocation_id,
        workspace_id=workspace_id,
        ticket_id=ticket_id,
        agent_run_id=claim.agent_run.id,
        agent_run_attempt_id=claim.attempt.id,
        invocation_sequence=1,
        status=LLMInvocationStatus.TIMED_OUT,
        provider="mock",
        model="mock-support-v1",
        provider_request_id=f"mock-request-{tool_call_id}",
        prompt_id="controlled-support",
        prompt_version=1,
        prompt_content_hash="d" * 64,
        schema_version=TICKET_CLASSIFICATION_SCHEMA_VERSION,
        input_tokens=None,
        cached_input_tokens=None,
        output_tokens=None,
        reasoning_tokens=None,
        total_tokens=None,
        pricing_catalog_version=PRICING_CATALOG_VERSION,
        pricing_found=True,
        estimated_input_cost_usd=None,
        estimated_cached_input_cost_usd=None,
        estimated_output_cost_usd=None,
        estimated_total_cost_usd=None,
        latency_ms=12_000,
        error_code=LLMErrorCode.TIMEOUT,
        now=_INVOCATION_AT,
    )
    pending = ApprovalRequest.create_pending(
        tool_call=tool_call,
        requested_by_llm_invocation_id=invocation_id,
        request_reason="Requires human review before escalation.",
        expires_at=_EXPIRES_AT,
        approval_request_id=approval_request_id,
        now=_APPROVAL_AT,
    )
    approved = pending.approve(
        actor_reference="operator:alice",
        comment=None,
        request_id=decision_request_id,
        correlation_id=decision_correlation_id,
        decided_at=_DECIDED_AT,
    )
    grant = SensitiveExecutionGrant.create(
        approval_request=approved,
        tool_call=tool_call,
        executed_by_agent_run_attempt_id=claim.attempt.id,
        created_at=_GRANT_CREATED_AT,
        grant_id=grant_id,
    )
    escalation = TicketEscalation.create_from_grant(
        grant=grant,
        input_data=EscalateTicketInput.model_validate(
            dict(grant.granted_input),
        ),
        created_at=_ESCALATION_CREATED_AT,
        escalation_id=escalation_id,
    )

    async with transaction_manager.transaction():
        session.add(LLMInvocationRecord.from_domain(invocation))
        await session.flush()
        tool_result = await SqlAlchemyAgentToolCallExecutionRepository(
            session,
        ).persist_fenced(
            PersistAgentToolCallCommand(
                workspace_id=workspace_id,
                ticket_id=ticket_id,
                agent_run_id=claim.agent_run.id,
                agent_run_attempt_id=claim.attempt.id,
                lease_token=lease_token,
                persisted_at=_APPROVAL_AT,
                tool_call=tool_call,
            ),
        )
        assert tool_result is AgentToolCallPersistenceResult.APPLIED
        approval_result = await SqlAlchemyApprovalRequestRepository(
            session,
        ).persist_pending(pending)
        assert approval_result is ApprovalRequestPersistenceResult.APPLIED
        await SqlAlchemyApprovalRequestRepository(session).save(approved)
        grant_result = await SqlAlchemySensitiveExecutionGrantRepository(
            session,
        ).persist(grant)
        assert grant_result is SensitiveExecutionGrantPersistenceResult.APPLIED
        escalation_result = await SqlAlchemyTicketEscalationRepository(
            session,
        ).persist(escalation)
        assert escalation_result is TicketEscalationPersistenceResult.APPLIED

    return escalation


@pytest.mark.integration
async def test_list_escalations_is_workspace_scoped(
    integration_client: AsyncClient,
    persisted_ticket_escalation_fixture: PersistedTicketEscalationFixture,
) -> None:
    fixture = persisted_ticket_escalation_fixture

    response = await integration_client.get(
        (f"/api/v1/workspaces/{fixture.workspace_id}/ticket-escalations"),
    )

    assert response.status_code == 200
    ids = {item["id"] for item in response.json()["items"]}
    assert str(fixture.escalation.id) in ids
    assert str(fixture.foreign_escalation.id) not in ids


@pytest.mark.integration
async def test_list_escalations_filters_by_ticket(
    integration_client: AsyncClient,
    persisted_ticket_escalation_fixture: PersistedTicketEscalationFixture,
) -> None:
    fixture = persisted_ticket_escalation_fixture

    response = await integration_client.get(
        (f"/api/v1/workspaces/{fixture.workspace_id}/ticket-escalations"),
        params={"ticket_id": str(fixture.ticket_id)},
    )

    assert response.status_code == 200
    assert all(item["ticket_id"] == str(fixture.ticket_id) for item in response.json()["items"])


@pytest.mark.integration
async def test_foreign_workspace_detail_returns_404(
    integration_client: AsyncClient,
    persisted_ticket_escalation_fixture: PersistedTicketEscalationFixture,
) -> None:
    fixture = persisted_ticket_escalation_fixture

    response = await integration_client.get(
        (
            f"/api/v1/workspaces/{fixture.foreign_workspace_id}/"
            f"ticket-escalations/{fixture.escalation.id}"
        ),
    )

    assert response.status_code == 404
    assert response.json()["error"]["code"] == ("ticket_escalation_not_found")
