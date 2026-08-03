"""Integration coverage for approval API safety boundaries."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from uuid import UUID, uuid4

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient, Response
from sqlalchemy import func, select, text
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
    AgentRunAttemptRecord,
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
from supportops.modules.tickets.domain.models import Ticket
from supportops.modules.tickets.infrastructure.repository import (
    SqlAlchemyTicketRepository,
)
from supportops.modules.workspaces.domain.models import Workspace
from supportops.modules.workspaces.infrastructure.repository import (
    SqlAlchemyWorkspaceRepository,
)

_WORKSPACE_ID = UUID("13000000-0000-4000-8000-000000000031")
_FOREIGN_WORKSPACE_ID = UUID("13000000-0000-4000-8000-000000000091")
_CREATED_AT = datetime(2026, 8, 3, 20, 0, tzinfo=UTC)
_CLAIMED_AT = _CREATED_AT + timedelta(minutes=1)
_LEASE_EXPIRES_AT = _CLAIMED_AT + timedelta(seconds=45)
_TOOL_PROPOSED_AT = _CLAIMED_AT + timedelta(seconds=1)
_INVOCATION_AT = _CLAIMED_AT + timedelta(seconds=2)
_APPROVAL_AT = _CLAIMED_AT + timedelta(seconds=3)
_WAITING_AT = _APPROVAL_AT + timedelta(seconds=1)
_EXPIRES_AT = _APPROVAL_AT + timedelta(days=365)


@dataclass(slots=True)
class ApprovalApiSafetyFixture:
    """Seeded multi-workspace pending approvals for API safety tests."""

    workspace_id: UUID
    foreign_workspace_id: UUID
    approval_request: ApprovalRequest
    foreign_approval_request: ApprovalRequest
    pending_approval_request: ApprovalRequest
    concurrent_approval_request: ApprovalRequest
    execution_boundary_approval: ApprovalRequest
    session_factory: async_sessionmaker[AsyncSession]
    initial_attempt_count: int
    initial_retryable_failure_count: int

    async def run_concurrently(
        self,
        *awaitables: Awaitable[Response],
    ) -> tuple[Response, ...]:
        """Start awaitables together after a shared barrier."""

        barrier = asyncio.Barrier(len(awaitables))

        async def _run_one(awaitable: Awaitable[Response]) -> Response:
            await barrier.wait()
            return await awaitable

        return tuple(
            await asyncio.gather(
                *(_run_one(awaitable) for awaitable in awaitables),
            )
        )

    async def load_concurrent_approval(self) -> ApprovalRequest:
        """Reload the concurrent-decision ApprovalRequest."""

        return await self._load_approval(
            self.concurrent_approval_request.id,
        )

    async def load_approval(
        self,
        approval_request_id: UUID,
    ) -> ApprovalRequest:
        """Reload one ApprovalRequest by id."""

        return await self._load_approval(approval_request_id)

    async def load_agent_run(self, agent_run_id: UUID) -> AgentRun:
        """Reload one AgentRun after a decision request."""

        async with self.session_factory() as session:
            result = await session.execute(
                select(AgentRunRecord).where(
                    AgentRunRecord.id == agent_run_id,
                ),
            )
            return result.scalar_one().to_domain()

    async def count_attempts(self, agent_run_id: UUID) -> int:
        """Count AgentRunAttempt rows for one AgentRun."""

        async with self.session_factory() as session:
            result = await session.execute(
                select(func.count())
                .select_from(AgentRunAttemptRecord)
                .where(
                    AgentRunAttemptRecord.agent_run_id == agent_run_id,
                ),
            )
            return int(result.scalar_one())

    async def load_attempt_outcomes(
        self,
        agent_run_id: UUID,
    ) -> tuple[str | None, ...]:
        """Return attempt outcomes for one AgentRun."""

        async with self.session_factory() as session:
            result = await session.execute(
                select(AgentRunAttemptRecord.outcome)
                .where(
                    AgentRunAttemptRecord.agent_run_id == agent_run_id,
                )
                .order_by(AgentRunAttemptRecord.attempt_number),
            )
            return tuple(row[0] for row in result.all())

    async def count_execution_grants(
        self,
        approval_request_id: UUID,
    ) -> int:
        """Count sensitive execution grants for one approval."""

        async with self.session_factory() as session:
            result = await session.execute(
                text(
                    "SELECT COUNT(*) FROM sensitive_execution_grants "
                    "WHERE approval_request_id = :approval_request_id",
                ),
                {"approval_request_id": approval_request_id},
            )
            return int(result.scalar_one())

    async def count_ticket_escalations(
        self,
        approval_request_id: UUID,
    ) -> int:
        """Count ticket escalations for one approval."""

        async with self.session_factory() as session:
            result = await session.execute(
                text(
                    "SELECT COUNT(*) FROM ticket_escalations "
                    "WHERE approval_request_id = :approval_request_id",
                ),
                {"approval_request_id": approval_request_id},
            )
            return int(result.scalar_one())

    async def count_new_recommendations(
        self,
        agent_run_id: UUID,
    ) -> int:
        """Count support recommendations for one AgentRun."""

        async with self.session_factory() as session:
            result = await session.execute(
                text(
                    "SELECT COUNT(*) FROM support_recommendations "
                    "WHERE agent_run_id = :agent_run_id",
                ),
                {"agent_run_id": agent_run_id},
            )
            return int(result.scalar_one())

    async def _load_approval(
        self,
        approval_request_id: UUID,
    ) -> ApprovalRequest:
        async with self.session_factory() as session:
            result = await session.execute(
                select(ApprovalRequestRecord).where(
                    ApprovalRequestRecord.id == approval_request_id,
                ),
            )
            return result.scalar_one().to_domain()


@pytest.fixture
async def api_client(
    integration_client: AsyncClient,
) -> AsyncClient:
    """Alias the shared integration HTTP client for safety tests."""

    return integration_client


@pytest.fixture
async def concurrent_api_clients(
    integration_application: FastAPI,
) -> AsyncIterator[tuple[AsyncClient, AsyncClient]]:
    """Create two real ASGI clients sharing one application lifespan."""

    async with (
        integration_application.router.lifespan_context(
            integration_application,
        ),
        AsyncClient(
            transport=ASGITransport(app=integration_application),
            base_url="http://test",
        ) as first_client,
        AsyncClient(
            transport=ASGITransport(app=integration_application),
            base_url="http://test",
        ) as second_client,
    ):
        yield first_client, second_client


@pytest.fixture
async def approval_api_safety_fixture(
    postgresql_session_factory: async_sessionmaker[AsyncSession],
    clean_business_tables: None,
) -> AsyncIterator[ApprovalApiSafetyFixture]:
    """Seed own/foreign pending approvals for isolation and decision tests."""

    del clean_business_tables

    async with postgresql_session_factory() as session:
        await _ensure_workspace(
            session,
            workspace_id=_WORKSPACE_ID,
            name="Approval Safety Workspace",
            slug="approval-safety-workspace",
        )
        primary = await _seed_pending_approval_graph(
            session,
            workspace_id=_WORKSPACE_ID,
            ticket_id=UUID("23000000-0000-4000-8000-000000000032"),
            agent_run_id=UUID("33000000-0000-4000-8000-000000000033"),
            lease_token=UUID("43000000-0000-4000-8000-000000000034"),
            execution_request_id=UUID(
                "53000000-0000-4000-8000-000000000035",
            ),
            tool_call_id=UUID("63000000-0000-4000-8000-000000000036"),
            invocation_id=UUID("83000000-0000-4000-8000-000000000038"),
            approval_request_id=UUID(
                "93000000-0000-4000-8000-000000000039",
            ),
            worker_id="approval-safety-worker-1",
            provider_tool_call_id="approval-safety-call-1",
            input_fingerprint="a" * 64,
            ingestion_request_id=UUID(
                "81300000-0000-4000-8000-000000000038",
            ),
            correlation_id=UUID(
                "82300000-0000-4000-8000-000000000039",
            ),
        )
        concurrent = await _seed_pending_approval_graph(
            session,
            workspace_id=_WORKSPACE_ID,
            ticket_id=UUID("23000000-0000-4000-8000-000000000042"),
            agent_run_id=UUID("33000000-0000-4000-8000-000000000043"),
            lease_token=UUID("43000000-0000-4000-8000-000000000044"),
            execution_request_id=UUID(
                "53000000-0000-4000-8000-000000000045",
            ),
            tool_call_id=UUID("63000000-0000-4000-8000-000000000046"),
            invocation_id=UUID("83000000-0000-4000-8000-000000000048"),
            approval_request_id=UUID(
                "93000000-0000-4000-8000-000000000049",
            ),
            worker_id="approval-safety-worker-2",
            provider_tool_call_id="approval-safety-call-2",
            input_fingerprint="b" * 64,
            ingestion_request_id=UUID(
                "81300000-0000-4000-8000-000000000048",
            ),
            correlation_id=UUID(
                "82300000-0000-4000-8000-000000000049",
            ),
        )
        await _ensure_workspace(
            session,
            workspace_id=_FOREIGN_WORKSPACE_ID,
            name="Foreign Approval Safety Workspace",
            slug="foreign-approval-safety-workspace",
        )
        foreign = await _seed_pending_approval_graph(
            session,
            workspace_id=_FOREIGN_WORKSPACE_ID,
            ticket_id=UUID("23000000-0000-4000-8000-000000000092"),
            agent_run_id=UUID("33000000-0000-4000-8000-000000000093"),
            lease_token=UUID("43000000-0000-4000-8000-000000000094"),
            execution_request_id=UUID(
                "53000000-0000-4000-8000-000000000095",
            ),
            tool_call_id=UUID("63000000-0000-4000-8000-000000000096"),
            invocation_id=UUID("83000000-0000-4000-8000-000000000098"),
            approval_request_id=UUID(
                "93000000-0000-4000-8000-000000000099",
            ),
            worker_id="approval-safety-worker-foreign",
            provider_tool_call_id="approval-safety-call-foreign",
            input_fingerprint="c" * 64,
            ingestion_request_id=UUID(
                "81300000-0000-4000-8000-000000000098",
            ),
            correlation_id=UUID(
                "82300000-0000-4000-8000-000000000099",
            ),
        )

    yield ApprovalApiSafetyFixture(
        workspace_id=_WORKSPACE_ID,
        foreign_workspace_id=_FOREIGN_WORKSPACE_ID,
        approval_request=primary,
        foreign_approval_request=foreign,
        pending_approval_request=primary,
        concurrent_approval_request=concurrent,
        execution_boundary_approval=primary,
        session_factory=postgresql_session_factory,
        initial_attempt_count=1,
        initial_retryable_failure_count=0,
    )


async def _ensure_workspace(
    session: AsyncSession,
    *,
    workspace_id: UUID,
    name: str,
    slug: str,
) -> None:
    transaction_manager = SqlAlchemyTransactionManager(session)
    async with transaction_manager.transaction():
        await SqlAlchemyWorkspaceRepository(session).add(
            Workspace(
                id=workspace_id,
                name=name,
                slug=slug,
                created_at=_CREATED_AT,
                updated_at=_CREATED_AT,
            ),
        )


async def _seed_pending_approval_graph(
    session: AsyncSession,
    *,
    workspace_id: UUID,
    ticket_id: UUID,
    agent_run_id: UUID,
    lease_token: UUID,
    execution_request_id: UUID,
    tool_call_id: UUID,
    invocation_id: UUID,
    approval_request_id: UUID,
    worker_id: str,
    provider_tool_call_id: str,
    input_fingerprint: str,
    ingestion_request_id: UUID,
    correlation_id: UUID,
) -> ApprovalRequest:
    ticket = Ticket.create(
        ticket_id=ticket_id,
        workspace_id=workspace_id,
        subject="Needs approval safety coverage",
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
            "reason": "Operational review required.",
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
    approval = ApprovalRequest.create_pending(
        tool_call=tool_call,
        requested_by_llm_invocation_id=invocation_id,
        request_reason="Requires human review before escalation.",
        expires_at=_EXPIRES_AT,
        approval_request_id=approval_request_id,
        now=_APPROVAL_AT,
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
        ).persist_pending(approval)
        assert approval_result is ApprovalRequestPersistenceResult.APPLIED
        waiting_result = await SqlAlchemyAgentRunRepository(
            session,
        ).mark_waiting_for_approval(
            WaitForApprovalAgentRunCommand(
                agent_run_id=claim.agent_run.id,
                lease_token=lease_token,
                finished_at=_WAITING_AT,
            ),
        )
        assert waiting_result is AgentRunTransitionResult.APPLIED

    return approval


@pytest.mark.integration
async def test_approval_list_never_discloses_foreign_workspace_records(
    api_client: AsyncClient,
    approval_api_safety_fixture: ApprovalApiSafetyFixture,
) -> None:
    fixture = approval_api_safety_fixture

    response = await api_client.get(
        f"/api/v1/workspaces/{fixture.workspace_id}/approvals",
    )

    assert response.status_code == 200
    ids = {item["id"] for item in response.json()["items"]}
    assert str(fixture.approval_request.id) in ids
    assert str(fixture.foreign_approval_request.id) not in ids


@pytest.mark.integration
async def test_foreign_workspace_approval_detail_matches_missing_404(
    api_client: AsyncClient,
    approval_api_safety_fixture: ApprovalApiSafetyFixture,
) -> None:
    fixture = approval_api_safety_fixture

    foreign = await api_client.get(
        (
            f"/api/v1/workspaces/{fixture.foreign_workspace_id}/"
            f"approvals/{fixture.approval_request.id}"
        ),
    )
    missing = await api_client.get(
        (f"/api/v1/workspaces/{fixture.workspace_id}/approvals/{uuid4()}"),
    )

    assert foreign.status_code == 404
    assert missing.status_code == 404
    assert foreign.json()["error"]["code"] == ("approval_request_not_found")
    assert missing.json()["error"]["code"] == ("approval_request_not_found")


@pytest.mark.integration
async def test_identical_decision_replay_uses_actor_comment_semantics(
    api_client: AsyncClient,
    approval_api_safety_fixture: ApprovalApiSafetyFixture,
) -> None:
    fixture = approval_api_safety_fixture
    path = (
        f"/api/v1/workspaces/{fixture.workspace_id}/"
        f"approvals/{fixture.pending_approval_request.id}/approve"
    )

    first = await api_client.post(
        path,
        json={
            "actor_reference": "operator:alice",
            "decision_request_id": str(uuid4()),
            "comment": "Approved after review.",
        },
    )
    replay = await api_client.post(
        path,
        json={
            "actor_reference": "operator:alice",
            "decision_request_id": str(uuid4()),
            "comment": "Approved after review.",
        },
    )

    assert first.status_code == 200
    assert replay.status_code == 200
    assert first.json()["idempotent"] is False
    assert replay.json()["idempotent"] is True


@pytest.mark.integration
async def test_same_terminal_decision_with_different_actor_conflicts(
    api_client: AsyncClient,
    approval_api_safety_fixture: ApprovalApiSafetyFixture,
) -> None:
    fixture = approval_api_safety_fixture
    path = (
        f"/api/v1/workspaces/{fixture.workspace_id}/"
        f"approvals/{fixture.pending_approval_request.id}/approve"
    )

    first = await api_client.post(
        path,
        json={
            "actor_reference": "operator:alice",
            "decision_request_id": str(uuid4()),
        },
    )
    conflict = await api_client.post(
        path,
        json={
            "actor_reference": "operator:bob",
            "decision_request_id": str(uuid4()),
        },
    )

    assert first.status_code == 200
    assert conflict.status_code == 409
    assert conflict.json()["error"]["code"] == ("approval_decision_conflict")


@pytest.mark.integration
async def test_conflicting_terminal_decisions_have_one_winner(
    concurrent_api_clients: tuple[AsyncClient, AsyncClient],
    approval_api_safety_fixture: ApprovalApiSafetyFixture,
) -> None:
    fixture = approval_api_safety_fixture
    path = (
        f"/api/v1/workspaces/{fixture.workspace_id}/"
        f"approvals/{fixture.concurrent_approval_request.id}"
    )

    approve_client, reject_client = concurrent_api_clients

    approve_response, reject_response = await fixture.run_concurrently(
        approve_client.post(
            f"{path}/approve",
            json={
                "actor_reference": "operator:alice",
                "decision_request_id": str(uuid4()),
            },
        ),
        reject_client.post(
            f"{path}/reject",
            json={
                "actor_reference": "operator:bob",
                "decision_request_id": str(uuid4()),
                "comment": "Reject after independent review.",
            },
        ),
    )

    statuses = {
        approve_response.status_code,
        reject_response.status_code,
    }
    assert statuses == {200, 409}

    refreshed = await fixture.load_concurrent_approval()
    assert refreshed.status.value in {"approved", "rejected"}
    if approve_response.status_code == 200:
        assert refreshed.status.value == "approved"
        assert refreshed.decision_actor_reference == "operator:alice"
        assert reject_response.json()["error"]["code"] == ("approval_decision_conflict")
    else:
        assert refreshed.status.value == "rejected"
        assert refreshed.decision_actor_reference == "operator:bob"
        assert refreshed.decision_comment == ("Reject after independent review.")
        assert approve_response.json()["error"]["code"] == ("approval_decision_conflict")

    agent_run = await fixture.load_agent_run(
        fixture.concurrent_approval_request.agent_run_id,
    )
    assert agent_run.status.value == "queued"
    assert (
        await fixture.count_attempts(
            fixture.concurrent_approval_request.agent_run_id,
        )
        == 1
    )
    assert (
        await fixture.count_execution_grants(
            fixture.concurrent_approval_request.id,
        )
        == 0
    )
    assert (
        await fixture.count_ticket_escalations(
            fixture.concurrent_approval_request.id,
        )
        == 0
    )


@pytest.mark.integration
async def test_decision_endpoint_requeues_without_executing(
    api_client: AsyncClient,
    approval_api_safety_fixture: ApprovalApiSafetyFixture,
) -> None:
    fixture = approval_api_safety_fixture
    approval = fixture.execution_boundary_approval

    response = await api_client.post(
        (f"/api/v1/workspaces/{fixture.workspace_id}/approvals/{approval.id}/approve"),
        json={
            "actor_reference": "operator:alice",
            "decision_request_id": str(uuid4()),
        },
    )

    assert response.status_code == 200

    agent_run = await fixture.load_agent_run(approval.agent_run_id)
    assert agent_run.status.value == "queued"
    assert await fixture.count_execution_grants(approval.id) == 0
    assert await fixture.count_ticket_escalations(approval.id) == 0
    assert (
        await fixture.count_new_recommendations(
            approval.agent_run_id,
        )
        == 0
    )


@pytest.mark.integration
async def test_status_filter_preserves_workspace_isolation(
    api_client: AsyncClient,
    approval_api_safety_fixture: ApprovalApiSafetyFixture,
) -> None:
    fixture = approval_api_safety_fixture

    response = await api_client.get(
        f"/api/v1/workspaces/{fixture.workspace_id}/approvals",
        params={"status": "pending"},
    )

    assert response.status_code == 200
    ids = {item["id"] for item in response.json()["items"]}
    assert str(fixture.approval_request.id) in ids
    assert str(fixture.foreign_approval_request.id) not in ids
    assert all(item["status"] == "pending" for item in response.json()["items"])
    assert all(
        item["workspace_id"] == str(fixture.workspace_id) for item in response.json()["items"]
    )


@pytest.mark.integration
async def test_own_approval_detail_returns_200_without_internal_fields(
    api_client: AsyncClient,
    approval_api_safety_fixture: ApprovalApiSafetyFixture,
) -> None:
    fixture = approval_api_safety_fixture

    response = await api_client.get(
        (f"/api/v1/workspaces/{fixture.workspace_id}/approvals/{fixture.approval_request.id}"),
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["id"] == str(fixture.approval_request.id)
    forbidden = {
        "grant_id",
        "execution_output",
        "checkpoint",
        "lease_token",
        "raw_prompt",
        "raw_model_output",
        "provider_state",
    }
    assert forbidden.isdisjoint(payload)


@pytest.mark.integration
async def test_foreign_workspace_decision_matches_missing_404(
    api_client: AsyncClient,
    approval_api_safety_fixture: ApprovalApiSafetyFixture,
) -> None:
    fixture = approval_api_safety_fixture

    foreign = await api_client.post(
        (
            f"/api/v1/workspaces/{fixture.foreign_workspace_id}/"
            f"approvals/{fixture.pending_approval_request.id}/approve"
        ),
        json={
            "actor_reference": "operator:alice",
            "decision_request_id": str(uuid4()),
        },
    )
    missing = await api_client.post(
        (f"/api/v1/workspaces/{fixture.workspace_id}/approvals/{uuid4()}/approve"),
        json={
            "actor_reference": "operator:alice",
            "decision_request_id": str(uuid4()),
        },
    )

    assert foreign.status_code == 404
    assert missing.status_code == 404
    assert foreign.json()["error"]["code"] == ("approval_request_not_found")
    assert missing.json()["error"]["code"] == ("approval_request_not_found")
    assert foreign.json()["error"]["message"] == (missing.json()["error"]["message"])
    assert foreign.json()["error"]["request_id"] == (foreign.headers["X-Request-ID"])


@pytest.mark.integration
async def test_concurrent_equivalent_approvals_converge(
    concurrent_api_clients: tuple[AsyncClient, AsyncClient],
    approval_api_safety_fixture: ApprovalApiSafetyFixture,
) -> None:
    fixture = approval_api_safety_fixture
    path = (
        f"/api/v1/workspaces/{fixture.workspace_id}/"
        f"approvals/{fixture.concurrent_approval_request.id}/approve"
    )
    first_client, second_client = concurrent_api_clients

    first_response, second_response = await fixture.run_concurrently(
        first_client.post(
            path,
            json={
                "actor_reference": "operator:alice",
                "decision_request_id": str(uuid4()),
                "comment": "Approved after review.",
            },
        ),
        second_client.post(
            path,
            json={
                "actor_reference": "operator:alice",
                "decision_request_id": str(uuid4()),
                "comment": "Approved after review.",
            },
        ),
    )

    assert {first_response.status_code, second_response.status_code} == {200}
    assert sorted(
        [
            first_response.json()["idempotent"],
            second_response.json()["idempotent"],
        ],
    ) == [False, True]

    refreshed = await fixture.load_concurrent_approval()
    assert refreshed.status.value == "approved"
    assert refreshed.decision_actor_reference == "operator:alice"

    agent_run = await fixture.load_agent_run(
        fixture.concurrent_approval_request.agent_run_id,
    )
    assert agent_run.status.value == "queued"
    assert (
        await fixture.count_attempts(
            fixture.concurrent_approval_request.agent_run_id,
        )
        == fixture.initial_attempt_count
    )
    assert (
        await fixture.count_execution_grants(
            fixture.concurrent_approval_request.id,
        )
        == 0
    )
    assert (
        await fixture.count_ticket_escalations(
            fixture.concurrent_approval_request.id,
        )
        == 0
    )


@pytest.mark.integration
async def test_approval_malformed_cursor_returns_400(
    api_client: AsyncClient,
    approval_api_safety_fixture: ApprovalApiSafetyFixture,
) -> None:
    fixture = approval_api_safety_fixture

    response = await api_client.get(
        f"/api/v1/workspaces/{fixture.workspace_id}/approvals",
        params={"cursor": "not-a-valid-cursor"},
    )

    assert response.status_code == 400
    assert response.json() == {
        "error": {
            "code": "invalid_pagination_cursor",
            "message": "Approval pagination cursor is invalid.",
            "request_id": response.headers["X-Request-ID"],
        }
    }


@pytest.mark.integration
async def test_decision_boundary_preserves_attempt_and_retry_counters(
    api_client: AsyncClient,
    approval_api_safety_fixture: ApprovalApiSafetyFixture,
) -> None:
    fixture = approval_api_safety_fixture
    approval = fixture.execution_boundary_approval
    before = await fixture.load_agent_run(approval.agent_run_id)
    before_outcomes = await fixture.load_attempt_outcomes(
        approval.agent_run_id,
    )

    response = await api_client.post(
        (f"/api/v1/workspaces/{fixture.workspace_id}/approvals/{approval.id}/approve"),
        json={
            "actor_reference": "operator:alice",
            "decision_request_id": str(uuid4()),
        },
    )

    assert response.status_code == 200
    refreshed = await fixture.load_approval(approval.id)
    agent_run = await fixture.load_agent_run(approval.agent_run_id)
    outcomes = await fixture.load_attempt_outcomes(approval.agent_run_id)

    assert refreshed.status.value == "approved"
    assert agent_run.status.value == "queued"
    assert agent_run.lease_token is None
    assert agent_run.retryable_failure_count == (before.retryable_failure_count)
    assert agent_run.attempt_count == before.attempt_count
    assert outcomes == before_outcomes
    assert outcomes == ("awaiting_approval",)
    assert await fixture.count_attempts(approval.agent_run_id) == 1
    assert await fixture.count_execution_grants(approval.id) == 0
    assert await fixture.count_ticket_escalations(approval.id) == 0
    assert await fixture.count_new_recommendations(approval.agent_run_id) == 0
