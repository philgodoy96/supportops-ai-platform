"""Integration coverage for ticket escalation API safety boundaries."""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from uuid import UUID, uuid4

import pytest
from httpx import AsyncClient
from sqlalchemy import select, text
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
from supportops.agent_tools.infrastructure.models import (
    AgentToolCallRecord,
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
from supportops.modules.agent_runs.infrastructure.models import (
    AgentRunRecord,
)
from supportops.modules.agent_runs.infrastructure.repository import (
    SqlAlchemyAgentRunRepository,
)
from supportops.modules.approvals.domain.models import ApprovalRequest
from supportops.modules.approvals.domain.repositories import (
    ApprovalRequestPersistenceResult,
)
from supportops.modules.approvals.infrastructure.models import (
    ApprovalRequestRecord,
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
from supportops.modules.tickets.infrastructure.escalation_models import (
    TicketEscalationRecord,
)
from supportops.modules.tickets.infrastructure.escalation_repository import (
    SqlAlchemyTicketEscalationRepository,
)
from supportops.modules.tickets.infrastructure.models import TicketRecord
from supportops.modules.tickets.infrastructure.repository import (
    SqlAlchemyTicketRepository,
)
from supportops.modules.workspaces.domain.models import Workspace
from supportops.modules.workspaces.infrastructure.repository import (
    SqlAlchemyWorkspaceRepository,
)

_WORKSPACE_ID = UUID("14000000-0000-4000-8000-000000000041")
_FOREIGN_WORKSPACE_ID = UUID("14000000-0000-4000-8000-000000000091")
_TICKET_ID = UUID("24000000-0000-4000-8000-000000000042")
_OTHER_TICKET_ID = UUID("24000000-0000-4000-8000-000000000052")
_FOREIGN_TICKET_ID = UUID("24000000-0000-4000-8000-000000000092")
_AGENT_RUN_ID = UUID("34000000-0000-4000-8000-000000000043")
_OTHER_AGENT_RUN_ID = UUID("34000000-0000-4000-8000-000000000053")
_FOREIGN_AGENT_RUN_ID = UUID("34000000-0000-4000-8000-000000000093")
_LEASE_TOKEN = UUID("44000000-0000-4000-8000-000000000044")
_OTHER_LEASE_TOKEN = UUID("44000000-0000-4000-8000-000000000054")
_FOREIGN_LEASE_TOKEN = UUID("44000000-0000-4000-8000-000000000094")
_EXECUTION_REQUEST_ID = UUID("54000000-0000-4000-8000-000000000045")
_OTHER_EXECUTION_REQUEST_ID = UUID("54000000-0000-4000-8000-000000000055")
_FOREIGN_EXECUTION_REQUEST_ID = UUID(
    "54000000-0000-4000-8000-000000000095",
)
_TOOL_CALL_ID = UUID("64000000-0000-4000-8000-000000000046")
_OTHER_TOOL_CALL_ID = UUID("64000000-0000-4000-8000-000000000056")
_FOREIGN_TOOL_CALL_ID = UUID("64000000-0000-4000-8000-000000000096")
_INVOCATION_ID = UUID("84000000-0000-4000-8000-000000000048")
_OTHER_INVOCATION_ID = UUID("84000000-0000-4000-8000-000000000058")
_FOREIGN_INVOCATION_ID = UUID("84000000-0000-4000-8000-000000000098")
_APPROVAL_REQUEST_ID = UUID("94000000-0000-4000-8000-000000000049")
_OTHER_APPROVAL_REQUEST_ID = UUID("94000000-0000-4000-8000-000000000059")
_FOREIGN_APPROVAL_REQUEST_ID = UUID(
    "94000000-0000-4000-8000-000000000099",
)
_GRANT_ID = UUID("a4000000-0000-4000-8000-00000000004a")
_OTHER_GRANT_ID = UUID("a4000000-0000-4000-8000-00000000005a")
_FOREIGN_GRANT_ID = UUID("a4000000-0000-4000-8000-00000000009a")
_ESCALATION_ID = UUID("d4000000-0000-4000-8000-00000000004d")
_OTHER_ESCALATION_ID = UUID("d4000000-0000-4000-8000-00000000005d")
_FOREIGN_ESCALATION_ID = UUID("d4000000-0000-4000-8000-00000000009d")
_DECISION_REQUEST_ID = UUID("b4000000-0000-4000-8000-00000000004b")
_OTHER_DECISION_REQUEST_ID = UUID("b4000000-0000-4000-8000-00000000005b")
_FOREIGN_DECISION_REQUEST_ID = UUID(
    "b4000000-0000-4000-8000-00000000009b",
)
_DECISION_CORRELATION_ID = UUID("c4000000-0000-4000-8000-00000000004c")
_OTHER_DECISION_CORRELATION_ID = UUID(
    "c4000000-0000-4000-8000-00000000005c",
)
_FOREIGN_DECISION_CORRELATION_ID = UUID(
    "c4000000-0000-4000-8000-00000000009c",
)

_CREATED_AT = datetime(2026, 8, 3, 20, 30, tzinfo=UTC)
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
class TicketEscalationApiSafetyFixture:
    """Seeded escalations for workspace-scoped inspection safety tests."""

    workspace_id: UUID
    ticket_id: UUID
    escalation: TicketEscalation
    foreign_workspace_id: UUID
    foreign_escalation: TicketEscalation
    other_escalation_id: UUID
    session_factory: async_sessionmaker[AsyncSession]

    async def snapshot_business_state(self) -> tuple[object, ...]:
        """Capture durable rows that inspection must leave unchanged."""

        async with self.session_factory() as session:
            escalation = (
                await session.execute(
                    select(TicketEscalationRecord).where(
                        TicketEscalationRecord.id == self.escalation.id,
                    ),
                )
            ).scalar_one()
            ticket = (
                await session.execute(
                    select(TicketRecord).where(
                        TicketRecord.id == self.ticket_id,
                    ),
                )
            ).scalar_one()
            agent_run = (
                await session.execute(
                    select(AgentRunRecord).where(
                        AgentRunRecord.id == escalation.agent_run_id,
                    ),
                )
            ).scalar_one()
            approval = (
                await session.execute(
                    select(ApprovalRequestRecord).where(
                        ApprovalRequestRecord.id == escalation.approval_request_id,
                    ),
                )
            ).scalar_one()
            tool_call = (
                await session.execute(
                    select(AgentToolCallRecord).where(
                        AgentToolCallRecord.id == escalation.agent_tool_call_id,
                    ),
                )
            ).scalar_one()
            grant_count = int(
                (
                    await session.execute(
                        text(
                            "SELECT COUNT(*) FROM "
                            "sensitive_execution_grants "
                            "WHERE workspace_id = :workspace_id",
                        ),
                        {"workspace_id": self.workspace_id},
                    )
                ).scalar_one(),
            )
            recommendation_count = int(
                (
                    await session.execute(
                        text(
                            "SELECT COUNT(*) FROM support_recommendations "
                            "WHERE workspace_id = :workspace_id",
                        ),
                        {"workspace_id": self.workspace_id},
                    )
                ).scalar_one(),
            )

            return (
                (
                    escalation.id,
                    escalation.workspace_id,
                    escalation.ticket_id,
                    escalation.agent_run_id,
                    escalation.executed_by_agent_run_attempt_id,
                    escalation.approval_request_id,
                    escalation.agent_tool_call_id,
                    escalation.target_queue,
                    escalation.reason,
                    escalation.created_at,
                ),
                (
                    ticket.id,
                    ticket.workspace_id,
                    ticket.status,
                    ticket.updated_at,
                ),
                (
                    agent_run.id,
                    agent_run.status,
                    agent_run.attempt_count,
                    agent_run.retryable_failure_count,
                    agent_run.updated_at,
                    agent_run.lease_token,
                ),
                (
                    approval.id,
                    approval.status,
                    approval.decision_actor_reference,
                    approval.decision_comment,
                    approval.updated_at,
                ),
                (
                    tool_call.id,
                    tool_call.status,
                    tool_call.executed_by_agent_run_attempt_id,
                    tool_call.safe_output,
                    tool_call.finished_at,
                ),
                grant_count,
                recommendation_count,
            )


@pytest.fixture
async def api_client(
    integration_client: AsyncClient,
) -> AsyncClient:
    """Alias the shared integration HTTP client for safety tests."""

    return integration_client


@pytest.fixture
async def ticket_escalation_api_safety_fixture(
    postgresql_session_factory: async_sessionmaker[AsyncSession],
    clean_business_tables: None,
) -> AsyncIterator[TicketEscalationApiSafetyFixture]:
    """Persist primary, sibling, and foreign escalations for safety tests."""

    del clean_business_tables

    async with postgresql_session_factory() as session:
        escalation = await _seed_escalation_graph(
            session,
            workspace_id=_WORKSPACE_ID,
            workspace_name="Escalation Safety Workspace",
            workspace_slug="escalation-safety-workspace",
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
            worker_id="escalation-safety-worker-1",
            provider_tool_call_id="escalation-safety-call-1",
            input_fingerprint="b" * 64,
            reason="Primary ticket requires escalation review.",
            ingestion_request_id=UUID(
                "81400000-0000-4000-8000-000000000048",
            ),
            correlation_id=UUID(
                "82400000-0000-4000-8000-000000000049",
            ),
            create_workspace=True,
        )
        await _seed_escalation_graph(
            session,
            workspace_id=_WORKSPACE_ID,
            workspace_name="Escalation Safety Workspace",
            workspace_slug="escalation-safety-workspace",
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
            worker_id="escalation-safety-worker-2",
            provider_tool_call_id="escalation-safety-call-2",
            input_fingerprint="e" * 64,
            reason="Sibling ticket escalation in the same workspace.",
            ingestion_request_id=UUID(
                "81400000-0000-4000-8000-000000000058",
            ),
            correlation_id=UUID(
                "82400000-0000-4000-8000-000000000059",
            ),
            create_workspace=False,
        )
        foreign_escalation = await _seed_escalation_graph(
            session,
            workspace_id=_FOREIGN_WORKSPACE_ID,
            workspace_name="Foreign Escalation Safety Workspace",
            workspace_slug="foreign-escalation-safety-workspace",
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
            worker_id="escalation-safety-worker-foreign",
            provider_tool_call_id="escalation-safety-call-foreign",
            input_fingerprint="2" * 64,
            reason="Foreign workspace escalation must stay hidden.",
            ingestion_request_id=UUID(
                "81400000-0000-4000-8000-000000000098",
            ),
            correlation_id=UUID(
                "82400000-0000-4000-8000-000000000099",
            ),
            create_workspace=True,
        )

    yield TicketEscalationApiSafetyFixture(
        workspace_id=_WORKSPACE_ID,
        ticket_id=_TICKET_ID,
        escalation=escalation,
        foreign_workspace_id=_FOREIGN_WORKSPACE_ID,
        foreign_escalation=foreign_escalation,
        other_escalation_id=_OTHER_ESCALATION_ID,
        session_factory=postgresql_session_factory,
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
        subject="Needs escalation safety inspection",
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
        claim = await SqlAlchemyAgentRunRepository(
            session,
        ).claim_next_available(
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
async def test_escalation_list_never_discloses_foreign_workspace_records(
    api_client: AsyncClient,
    ticket_escalation_api_safety_fixture: TicketEscalationApiSafetyFixture,
) -> None:
    fixture = ticket_escalation_api_safety_fixture

    response = await api_client.get(
        (f"/api/v1/workspaces/{fixture.workspace_id}/ticket-escalations"),
    )

    assert response.status_code == 200
    ids = {item["id"] for item in response.json()["items"]}
    assert str(fixture.escalation.id) in ids
    assert str(fixture.foreign_escalation.id) not in ids


@pytest.mark.integration
async def test_ticket_filter_does_not_cross_workspace_boundary(
    api_client: AsyncClient,
    ticket_escalation_api_safety_fixture: TicketEscalationApiSafetyFixture,
) -> None:
    fixture = ticket_escalation_api_safety_fixture

    response = await api_client.get(
        (f"/api/v1/workspaces/{fixture.workspace_id}/ticket-escalations"),
        params={"ticket_id": str(fixture.ticket_id)},
    )

    assert response.status_code == 200
    assert all(
        item["workspace_id"] == str(fixture.workspace_id)
        and item["ticket_id"] == str(fixture.ticket_id)
        for item in response.json()["items"]
    )
    assert str(fixture.other_escalation_id) not in {item["id"] for item in response.json()["items"]}


@pytest.mark.integration
async def test_foreign_escalation_detail_matches_missing_404(
    api_client: AsyncClient,
    ticket_escalation_api_safety_fixture: TicketEscalationApiSafetyFixture,
) -> None:
    fixture = ticket_escalation_api_safety_fixture

    foreign = await api_client.get(
        (
            f"/api/v1/workspaces/{fixture.foreign_workspace_id}/"
            f"ticket-escalations/{fixture.escalation.id}"
        ),
    )
    missing = await api_client.get(
        (f"/api/v1/workspaces/{fixture.workspace_id}/ticket-escalations/{uuid4()}"),
    )

    assert foreign.status_code == 404
    assert missing.status_code == 404
    assert foreign.json()["error"]["code"] == ("ticket_escalation_not_found")
    assert missing.json()["error"]["code"] == ("ticket_escalation_not_found")


@pytest.mark.integration
async def test_escalation_inspection_is_read_only(
    api_client: AsyncClient,
    ticket_escalation_api_safety_fixture: TicketEscalationApiSafetyFixture,
) -> None:
    fixture = ticket_escalation_api_safety_fixture
    before = await fixture.snapshot_business_state()

    list_response = await api_client.get(
        (f"/api/v1/workspaces/{fixture.workspace_id}/ticket-escalations"),
    )
    detail_response = await api_client.get(
        (f"/api/v1/workspaces/{fixture.workspace_id}/ticket-escalations/{fixture.escalation.id}"),
    )

    after = await fixture.snapshot_business_state()

    assert list_response.status_code == 200
    assert detail_response.status_code == 200
    assert after == before


@pytest.mark.integration
async def test_escalation_response_excludes_internal_execution_data(
    api_client: AsyncClient,
    ticket_escalation_api_safety_fixture: TicketEscalationApiSafetyFixture,
) -> None:
    fixture = ticket_escalation_api_safety_fixture

    response = await api_client.get(
        (f"/api/v1/workspaces/{fixture.workspace_id}/ticket-escalations/{fixture.escalation.id}"),
    )

    assert response.status_code == 200
    payload = response.json()

    forbidden = {
        "grant_id",
        "granted_input",
        "approval_actor_reference",
        "decision_comment",
        "proposed_input",
        "execution_output",
        "checkpoint",
        "lease_token",
    }
    assert forbidden.isdisjoint(payload)
    assert payload["workspace_id"] == str(fixture.workspace_id)
    assert payload["ticket_id"] == str(fixture.ticket_id)
    assert payload["agent_run_id"] == str(fixture.escalation.agent_run_id)
    assert payload["approval_request_id"] == str(
        fixture.escalation.approval_request_id,
    )
    assert payload["agent_tool_call_id"] == str(
        fixture.escalation.agent_tool_call_id,
    )
    assert payload["executed_by_agent_run_attempt_id"] == str(
        fixture.escalation.executed_by_agent_run_attempt_id,
    )
