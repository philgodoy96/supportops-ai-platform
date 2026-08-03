"""Integration tests for approval decision HTTP endpoints."""

from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from uuid import UUID, uuid4

import pytest
from httpx import AsyncClient
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from supportops.agent_tools.application.persistence import (
    AgentToolCallPersistenceResult,
    PersistAgentToolCallCommand,
)
from supportops.agent_tools.domain.audit import AgentToolCall
from supportops.agent_tools.infrastructure.repository import (
    SqlAlchemyAgentToolCallExecutionRepository,
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
from supportops.modules.agent_runs.domain.transitions import (
    AgentRunTransitionResult,
    WaitForApprovalAgentRunCommand,
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
from supportops.modules.approvals.infrastructure.repository import (
    SqlAlchemyApprovalRequestRepository,
)
from supportops.modules.ticket_classifications.domain.models import (
    LLMInvocation,
)
from supportops.modules.ticket_classifications.infrastructure.models import (
    LLMInvocationRecord,
)
from supportops.modules.tickets.domain.models import Ticket
from supportops.modules.tickets.infrastructure.repository import (
    SqlAlchemyTicketRepository,
)
from supportops.modules.workspaces.domain.models import Workspace
from supportops.modules.workspaces.infrastructure.repository import (
    SqlAlchemyWorkspaceRepository,
)

_WORKSPACE_ID = UUID("11000000-0000-4000-8000-000000000011")
_TICKET_ID = UUID("21000000-0000-4000-8000-000000000012")
_AGENT_RUN_ID = UUID("31000000-0000-4000-8000-000000000013")
_LEASE_TOKEN = UUID("41000000-0000-4000-8000-000000000014")
_EXECUTION_REQUEST_ID = UUID("51000000-0000-4000-8000-000000000015")
_TOOL_CALL_ID = UUID("61000000-0000-4000-8000-000000000016")
_INVOCATION_ID = UUID("81000000-0000-4000-8000-000000000018")
_APPROVAL_REQUEST_ID = UUID("91000000-0000-4000-8000-000000000019")

_CREATED_AT = datetime(2026, 8, 2, 22, 0, tzinfo=UTC)
_CLAIMED_AT = _CREATED_AT + timedelta(minutes=1)
_LEASE_EXPIRES_AT = _CLAIMED_AT + timedelta(seconds=45)
_TOOL_PROPOSED_AT = _CLAIMED_AT + timedelta(seconds=1)
_INVOCATION_AT = _CLAIMED_AT + timedelta(seconds=2)
_APPROVAL_AT = _CLAIMED_AT + timedelta(seconds=3)
_WAITING_AT = _APPROVAL_AT + timedelta(seconds=1)
# Decision endpoints stamp decided_at with wall-clock UTC; keep expiry far ahead.
_EXPIRES_AT = _APPROVAL_AT + timedelta(days=365)


@dataclass(slots=True)
class PendingApprovalFixture:
    """Seeded waiting-approval graph for decision API tests."""

    workspace_id: UUID
    approval_request: ApprovalRequest
    agent_run: AgentRun
    session_factory: async_sessionmaker[AsyncSession]

    async def load_agent_run(self) -> AgentRun:
        """Reload the AgentRun after a decision request."""

        async with self.session_factory() as session:
            result = await session.execute(
                select(AgentRunRecord).where(
                    AgentRunRecord.id == self.agent_run.id,
                ),
            )
            return result.scalar_one().to_domain()

    async def count_execution_grants(self) -> int:
        """Count sensitive execution grants for the seeded workspace."""

        async with self.session_factory() as session:
            result = await session.execute(
                text(
                    "SELECT COUNT(*) FROM sensitive_execution_grants "
                    "WHERE workspace_id = :workspace_id",
                ),
                {"workspace_id": self.workspace_id},
            )
            return int(result.scalar_one())

    async def count_ticket_escalations(self) -> int:
        """Count ticket escalations for the seeded workspace."""

        async with self.session_factory() as session:
            result = await session.execute(
                text(
                    "SELECT COUNT(*) FROM ticket_escalations WHERE workspace_id = :workspace_id",
                ),
                {"workspace_id": self.workspace_id},
            )
            return int(result.scalar_one())


@pytest.fixture
async def api_client(
    integration_client: AsyncClient,
) -> AsyncClient:
    """Alias the shared integration HTTP client for decision tests."""

    return integration_client


@pytest.fixture
async def pending_approval_fixture(
    postgresql_session_factory: async_sessionmaker[AsyncSession],
    clean_business_tables: None,
) -> AsyncIterator[PendingApprovalFixture]:
    """Seed one pending approval with a waiting AgentRun."""

    del clean_business_tables

    async with postgresql_session_factory() as session:
        approval, agent_run = await _seed_pending_approval(session)

    yield PendingApprovalFixture(
        workspace_id=_WORKSPACE_ID,
        approval_request=approval,
        agent_run=agent_run,
        session_factory=postgresql_session_factory,
    )


async def _seed_pending_approval(
    session: AsyncSession,
) -> tuple[ApprovalRequest, AgentRun]:
    workspace = Workspace(
        id=_WORKSPACE_ID,
        name="Decision Approval Workspace",
        slug="decision-approval-workspace",
        created_at=_CREATED_AT,
        updated_at=_CREATED_AT,
    )
    ticket = Ticket.create(
        ticket_id=_TICKET_ID,
        workspace_id=_WORKSPACE_ID,
        subject="Needs escalation approval",
        description=("The customer requested a policy-sensitive escalation."),
        external_reference=None,
        ingestion_request_id=UUID(
            "81100000-0000-4000-8000-000000000028",
        ),
        correlation_id=UUID(
            "82100000-0000-4000-8000-000000000029",
        ),
        now=_CREATED_AT,
    )
    agent_run = AgentRun.create_initial(
        agent_run_id=_AGENT_RUN_ID,
        workspace_id=_WORKSPACE_ID,
        ticket_id=_TICKET_ID,
        ingestion_request_id=ticket.ingestion_request_id,
        correlation_id=ticket.correlation_id,
        workflow_version=DETERMINISTIC_BASELINE_WORKFLOW_VERSION,
        max_retryable_failures=3,
        now=_CREATED_AT,
    )
    transaction_manager = SqlAlchemyTransactionManager(session)

    async with transaction_manager.transaction():
        await SqlAlchemyWorkspaceRepository(session).add(workspace)
        await SqlAlchemyTicketRepository(session).add(ticket)
        await SqlAlchemyAgentRunRepository(session).add(agent_run)

    async with transaction_manager.transaction():
        claim = await SqlAlchemyAgentRunRepository(
            session,
        ).claim_next_available(
            ClaimAgentRunCommand(
                worker_id="decision-approval-worker-1",
                lease_token=_LEASE_TOKEN,
                execution_request_id=_EXECUTION_REQUEST_ID,
                claimed_at=_CLAIMED_AT,
                lease_expires_at=_LEASE_EXPIRES_AT,
            ),
        )

    assert claim is not None

    tool_call = AgentToolCall.propose_for_approval(
        tool_call_id=_TOOL_CALL_ID,
        workspace_id=_WORKSPACE_ID,
        ticket_id=_TICKET_ID,
        agent_run_id=claim.agent_run.id,
        proposed_by_agent_run_attempt_id=claim.attempt.id,
        sequence=1,
        provider_tool_call_id="decision-approval-call-1",
        tool_name="escalate_ticket",
        tool_version=1,
        input_fingerprint="b" * 64,
        safe_input={
            "target_queue": "support_operations",
            "reason": "Operational review required.",
        },
        proposed_at=_TOOL_PROPOSED_AT,
    )
    invocation = LLMInvocation.create(
        invocation_id=_INVOCATION_ID,
        workspace_id=_WORKSPACE_ID,
        ticket_id=_TICKET_ID,
        agent_run_id=claim.agent_run.id,
        agent_run_attempt_id=claim.attempt.id,
        invocation_sequence=1,
        status=LLMInvocationStatus.TIMED_OUT,
        provider="mock",
        model="mock-support-v1",
        provider_request_id="mock-request-1",
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
    approval = ApprovalRequest.create_pending(
        tool_call=tool_call,
        requested_by_llm_invocation_id=_INVOCATION_ID,
        request_reason="Requires human review before escalation.",
        expires_at=_EXPIRES_AT,
        approval_request_id=_APPROVAL_REQUEST_ID,
        now=_APPROVAL_AT,
    )

    async with transaction_manager.transaction():
        session.add(LLMInvocationRecord.from_domain(invocation))
        await session.flush()
        tool_result = await SqlAlchemyAgentToolCallExecutionRepository(
            session,
        ).persist_fenced(
            PersistAgentToolCallCommand(
                workspace_id=_WORKSPACE_ID,
                ticket_id=_TICKET_ID,
                agent_run_id=claim.agent_run.id,
                agent_run_attempt_id=claim.attempt.id,
                lease_token=_LEASE_TOKEN,
                persisted_at=_APPROVAL_AT,
                tool_call=tool_call,
            ),
        )
        assert tool_result is AgentToolCallPersistenceResult.APPLIED
        approval_result = await SqlAlchemyApprovalRequestRepository(
            session,
        ).persist_pending(approval)
        assert approval_result is ApprovalRequestPersistenceResult.APPLIED
        waiting_result = await SqlAlchemyAgentRunRepository(
            session,
        ).mark_waiting_for_approval(
            WaitForApprovalAgentRunCommand(
                agent_run_id=claim.agent_run.id,
                lease_token=_LEASE_TOKEN,
                finished_at=_WAITING_AT,
            ),
        )
        assert waiting_result is AgentRunTransitionResult.APPLIED

    session.expire_all()
    loaded_run = (
        (
            await session.execute(
                select(AgentRunRecord).where(
                    AgentRunRecord.id == _AGENT_RUN_ID,
                ),
            )
        )
        .scalar_one()
        .to_domain()
    )

    return approval, loaded_run


@pytest.mark.integration
async def test_approve_requeues_waiting_agent_run_without_execution(
    api_client: AsyncClient,
    pending_approval_fixture: PendingApprovalFixture,
) -> None:
    workspace_id = pending_approval_fixture.workspace_id
    approval = pending_approval_fixture.approval_request
    agent_run = pending_approval_fixture.agent_run

    response = await api_client.post(
        (f"/api/v1/workspaces/{workspace_id}/approvals/{approval.id}/approve"),
        json={
            "actor_reference": "operator:alice",
            "decision_request_id": str(uuid4()),
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "approved"
    assert payload["idempotent"] is False

    refreshed = await pending_approval_fixture.load_agent_run()
    assert refreshed.status.value == "queued"
    assert refreshed.retryable_failure_count == (agent_run.retryable_failure_count)
    assert await pending_approval_fixture.count_execution_grants() == 0
    assert await pending_approval_fixture.count_ticket_escalations() == 0


@pytest.mark.integration
async def test_identical_approve_replay_is_idempotent(
    api_client: AsyncClient,
    pending_approval_fixture: PendingApprovalFixture,
) -> None:
    workspace_id = pending_approval_fixture.workspace_id
    approval = pending_approval_fixture.approval_request
    body = {
        "actor_reference": "operator:alice",
        "decision_request_id": str(uuid4()),
        "comment": "Approved after review.",
    }

    first = await api_client.post(
        (f"/api/v1/workspaces/{workspace_id}/approvals/{approval.id}/approve"),
        json=body,
    )
    second = await api_client.post(
        (f"/api/v1/workspaces/{workspace_id}/approvals/{approval.id}/approve"),
        json={
            **body,
            "decision_request_id": str(uuid4()),
        },
    )

    assert first.status_code == 200
    assert second.status_code == 200
    assert first.json()["idempotent"] is False
    assert second.json()["idempotent"] is True


@pytest.mark.integration
async def test_conflicting_terminal_decision_returns_409(
    api_client: AsyncClient,
    pending_approval_fixture: PendingApprovalFixture,
) -> None:
    workspace_id = pending_approval_fixture.workspace_id
    approval = pending_approval_fixture.approval_request

    approve = await api_client.post(
        (f"/api/v1/workspaces/{workspace_id}/approvals/{approval.id}/approve"),
        json={
            "actor_reference": "operator:alice",
            "decision_request_id": str(uuid4()),
        },
    )
    reject = await api_client.post(
        (f"/api/v1/workspaces/{workspace_id}/approvals/{approval.id}/reject"),
        json={
            "actor_reference": "operator:alice",
            "decision_request_id": str(uuid4()),
            "comment": "Reject after approval.",
        },
    )

    assert approve.status_code == 200
    assert reject.status_code == 409
    assert reject.json()["error"]["code"] == ("approval_decision_conflict")


@pytest.mark.integration
async def test_foreign_workspace_decision_returns_404(
    api_client: AsyncClient,
    pending_approval_fixture: PendingApprovalFixture,
) -> None:
    response = await api_client.post(
        (
            f"/api/v1/workspaces/{uuid4()}/approvals/"
            f"{pending_approval_fixture.approval_request.id}/approve"
        ),
        json={
            "actor_reference": "operator:alice",
            "decision_request_id": str(uuid4()),
        },
    )

    assert response.status_code == 404
    assert response.json()["error"]["code"] == ("approval_request_not_found")
