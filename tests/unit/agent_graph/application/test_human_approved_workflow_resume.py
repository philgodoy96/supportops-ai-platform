"""Unit tests for end-to-end human-approved workflow resume."""

from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock
from uuid import UUID, uuid4

import pytest

from supportops.agent_graph.application.approval_decision_handling import (
    ApprovalDecisionResumePayload,
)
from supportops.agent_graph.application.human_approved_nodes import (
    HumanApprovedSupportWorkflowNodes,
)
from supportops.agent_graph.application.human_approved_recommendation import (
    HumanApprovedRecommendationOutcome,
)
from supportops.agent_graph.application.human_approved_workflow import (
    HumanApprovedSupportWorkflowExecutor,
)
from supportops.agent_graph.domain.human_approved_identity import (
    derive_human_approved_support_graph_identity,
)
from supportops.agent_graph.domain.human_approved_state import (
    HUMAN_APPROVED_SUPPORT_WORKFLOW_VERSION,
    HumanApprovalCheckpointStatus,
    HumanApprovedSensitiveExecutionOutput,
    HumanApprovedSupportGraphState,
    create_initial_human_approved_support_state,
    validate_human_approved_support_state,
)
from supportops.agent_graph.domain.resume_planning import (
    ApprovalResumeDecisionStatus,
    InitialGraphExecution,
    ResumeGraphExecution,
)
from supportops.agent_tools.domain.audit import AgentToolCall
from supportops.modules.agent_runs.application.execution import (
    AgentRunExecutionContext,
    CompletedExecution,
    PausedForApproval,
    TerminalAgentRunExecutionError,
)
from supportops.modules.agent_runs.domain.models import (
    AgentRun,
    AgentRunAttempt,
    AgentRunStatus,
)
from supportops.modules.approvals.domain.models import ApprovalRequest
from supportops.modules.support_recommendations.domain.models import (
    SupportRecommendation,
    SupportRecommendationAction,
)
from supportops.modules.tickets.domain.models import Ticket

_NOW = datetime(2026, 8, 3, 19, 30, tzinfo=UTC)
_WORKSPACE_ID = UUID("032c8c87-57cc-4d14-bfbd-04968b4e8cd4")
_TICKET_ID = UUID("38bb60fe-d2ea-4615-b499-91aa45069019")
_AGENT_RUN_ID = UUID("69184ef1-4d71-452e-8070-0b784c29368e")
_ATTEMPT_ID = UUID("a1b2c3d4-e5f6-7890-abcd-ef1234567890")
_LEASE_TOKEN = UUID("dd0ae456-3467-41db-93d1-a908f40e8365")
_CLASSIFICATION_ID = UUID("11111111-1111-1111-1111-111111111111")
_DECISION_INVOCATION_ID = UUID("22222222-2222-2222-2222-222222222222")
_APPROVAL_REQUEST_ID = UUID("33333333-3333-3333-3333-333333333333")
_AGENT_TOOL_CALL_ID = UUID("44444444-4444-4444-4444-444444444444")
_RECOMMENDATION_ID = UUID("55555555-5555-5555-5555-555555555555")
_RECOMMENDATION_INVOCATION_ID = UUID("66666666-6666-6666-6666-666666666666")


class _NoOpTransactionManager:
    @asynccontextmanager
    async def transaction(self) -> AsyncIterator[None]:
        yield


class _InterruptResult:
    def __init__(self, value: Mapping[str, object]) -> None:
        self.interrupts = (SimpleNamespace(value=value),)


class _StubCheckpointSnapshot:
    def __init__(
        self,
        values: Mapping[str, object],
        interrupts: tuple[object, ...] = (),
    ) -> None:
        self.values = values
        self.interrupts = interrupts


class _PauseGraph:
    def __init__(self, interrupt_value: Mapping[str, object]) -> None:
        self._interrupt_value = interrupt_value
        self.ainvoke_calls = 0

    async def aget_state(self, config: Mapping[str, object]) -> _StubCheckpointSnapshot:
        del config
        return _StubCheckpointSnapshot({})

    async def ainvoke(
        self,
        input: object,
        config: Mapping[str, object],
        *,
        context: AgentRunExecutionContext,
        version: str = "v2",
    ) -> object:
        del input, config, context, version
        self.ainvoke_calls += 1
        return _InterruptResult(self._interrupt_value)


class _ResumeGraph:
    def __init__(
        self,
        *,
        checkpoint_values: Mapping[str, object],
        nodes: HumanApprovedSupportWorkflowNodes,
        approval_request: ApprovalRequest,
        decision_status: ApprovalResumeDecisionStatus,
    ) -> None:
        self._checkpoint_values = dict(checkpoint_values)
        self._nodes = nodes
        self._approval_request = approval_request
        self._decision_status = decision_status
        self.ainvoke_calls = 0

    async def aget_state(self, config: Mapping[str, object]) -> _StubCheckpointSnapshot:
        del config
        return _StubCheckpointSnapshot(
            self._checkpoint_values,
            interrupts=(
                SimpleNamespace(
                    value={
                        "approval_request_id": str(_APPROVAL_REQUEST_ID),
                        "agent_tool_call_id": str(_AGENT_TOOL_CALL_ID),
                        "agent_run_id": str(_AGENT_RUN_ID),
                        "ticket_id": str(_TICKET_ID),
                        "tool_name": "escalate_ticket",
                        "tool_version": 1,
                        "proposed_input": {
                            "target_queue": "support_operations",
                            "reason": "Operational review required.",
                        },
                        "request_reason": "Operational review required.",
                        "expires_at": (_NOW + timedelta(days=1)).isoformat(),
                    },
                ),
            ),
        )

    async def ainvoke(
        self,
        input: object,
        config: Mapping[str, object],
        *,
        context: AgentRunExecutionContext,
        version: str = "v2",
    ) -> object:
        del config, version
        self.ainvoke_calls += 1
        resume_value = getattr(input, "resume", None)
        assert resume_value == {
            "approval_request_id": str(_APPROVAL_REQUEST_ID),
            "agent_tool_call_id": str(_AGENT_TOOL_CALL_ID),
            "decision_status": self._decision_status.value,
        }

        runtime = SimpleNamespace(context=context)
        state = cast(
            HumanApprovedSupportGraphState,
            dict(self._checkpoint_values),
        )
        payload = ApprovalDecisionResumePayload(
            approval_request_id=_APPROVAL_REQUEST_ID,
            agent_tool_call_id=_AGENT_TOOL_CALL_ID,
            decision_status=self._decision_status,
        )
        state = cast(
            HumanApprovedSupportGraphState,
            {
                **state,
                "approval_resume_payload": payload.to_json_value(),
                "graph_step_count": int(state["graph_step_count"]) + 1,
            },
        )
        state = await self._nodes.handle_approval_decision(
            state,
            cast(Any, runtime),
        )
        snapshot = validate_human_approved_support_state(state)
        if snapshot.approval_status is HumanApprovalCheckpointStatus.APPROVED:
            state = await self._nodes.execute_sensitive_tool(
                state,
                cast(Any, runtime),
            )
        state = await self._nodes.draft_grounded_recommendation(
            state,
            cast(Any, runtime),
        )
        state = await self._nodes.validate_recommendation(state)
        state = await self._nodes.persist_recommendation(state)
        return state


class _FailingResumeGraph(_ResumeGraph):
    async def ainvoke(
        self,
        input: object,
        config: Mapping[str, object],
        *,
        context: AgentRunExecutionContext,
        version: str = "v2",
    ) -> object:
        del input, config, version
        runtime = SimpleNamespace(context=context)
        step_count = self._checkpoint_values["graph_step_count"]
        assert isinstance(step_count, int)
        state = cast(
            HumanApprovedSupportGraphState,
            {
                **self._checkpoint_values,
                "approval_resume_payload": ApprovalDecisionResumePayload(
                    approval_request_id=_APPROVAL_REQUEST_ID,
                    agent_tool_call_id=_AGENT_TOOL_CALL_ID,
                    decision_status=ApprovalResumeDecisionStatus.APPROVED,
                ).to_json_value(),
                "graph_step_count": step_count + 1,
            },
        )
        return await self._nodes.handle_approval_decision(
            state,
            cast(Any, runtime),
        )


class _Planner:
    def __init__(self, plan: object) -> None:
        self._plan = plan

    async def plan(self, **kwargs: object) -> object:
        del kwargs
        return self._plan


def _execution_context() -> AgentRunExecutionContext:
    ticket = Ticket.create(
        ticket_id=_TICKET_ID,
        workspace_id=_WORKSPACE_ID,
        subject="Need escalation review",
        description="Operational review is required.",
        external_reference=None,
        ingestion_request_id=UUID("90000000-0000-4000-8000-000000000009"),
        correlation_id=UUID("a0000000-0000-4000-8000-000000000010"),
        now=_NOW - timedelta(minutes=1),
    )
    initial_run = AgentRun.create_initial(
        agent_run_id=_AGENT_RUN_ID,
        workspace_id=_WORKSPACE_ID,
        ticket_id=_TICKET_ID,
        ingestion_request_id=ticket.ingestion_request_id,
        correlation_id=ticket.correlation_id,
        workflow_version=HUMAN_APPROVED_SUPPORT_WORKFLOW_VERSION,
        max_retryable_failures=3,
        now=_NOW - timedelta(minutes=1),
    )
    running_run = replace(
        initial_run,
        status=AgentRunStatus.RUNNING,
        attempt_count=1,
        lease_owner="worker-1",
        lease_token=_LEASE_TOKEN,
        lease_expires_at=_NOW + timedelta(minutes=5),
        first_started_at=_NOW,
        updated_at=_NOW,
    )
    attempt = AgentRunAttempt.start(
        attempt_id=_ATTEMPT_ID,
        agent_run_id=_AGENT_RUN_ID,
        attempt_number=1,
        worker_id="worker-1",
        lease_token=_LEASE_TOKEN,
        execution_request_id=uuid4(),
        now=_NOW,
    )
    return AgentRunExecutionContext(
        agent_run=running_run,
        attempt=attempt,
        ticket=ticket,
    )


def _pending_checkpoint() -> dict[str, object]:
    state = create_initial_human_approved_support_state(
        workspace_id=_WORKSPACE_ID,
        ticket_id=_TICKET_ID,
        agent_run_id=_AGENT_RUN_ID,
    )
    state.update(
        {
            "run_context_loaded": True,
            "classification_id": str(_CLASSIFICATION_ID),
            "classification_category": "product_bug",
            "classification_intent": "report_problem",
            "classification_urgency": "high",
            "classification_sentiment": "negative",
            "classification_requires_human_review": True,
            "classification_summary": "Operational review required.",
            "graph_step_count": 5,
            "decision_turn_count": 1,
            "tool_call_count": 1,
            "decision_kind": "sensitive_tool",
            "decision_invocation_id": str(_DECISION_INVOCATION_ID),
            "decision_summary": "Operational review required.",
            "proposed_tool_provider_call_id": "call-1",
            "proposed_tool_name": "escalate_ticket",
            "proposed_tool_version": 1,
            "proposed_tool_input": {
                "target_queue": "support_operations",
                "reason": "Operational review required.",
            },
            "proposed_tool_fingerprint": "a" * 64,
            "approval_request_reason": "Operational review required.",
            "agent_tool_call_id": str(_AGENT_TOOL_CALL_ID),
            "approval_request_id": str(_APPROVAL_REQUEST_ID),
            "approval_status": "pending",
            "approval_expires_at": (_NOW + timedelta(days=1)).isoformat(),
        },
    )
    return cast(dict[str, object], state)


def _tool_call_and_pending() -> tuple[AgentToolCall, ApprovalRequest]:
    tool_call = AgentToolCall.propose_for_approval(
        workspace_id=_WORKSPACE_ID,
        ticket_id=_TICKET_ID,
        agent_run_id=_AGENT_RUN_ID,
        proposed_by_agent_run_attempt_id=_ATTEMPT_ID,
        sequence=1,
        provider_tool_call_id="call-1",
        tool_name="escalate_ticket",
        tool_version=1,
        input_fingerprint="a" * 64,
        safe_input={
            "target_queue": "support_operations",
            "reason": "Operational review required.",
        },
        proposed_at=_NOW,
        tool_call_id=_AGENT_TOOL_CALL_ID,
    )
    pending = ApprovalRequest.create_pending(
        tool_call=tool_call,
        requested_by_llm_invocation_id=_DECISION_INVOCATION_ID,
        request_reason="Operational review required.",
        expires_at=_NOW + timedelta(days=1),
        now=_NOW,
        approval_request_id=_APPROVAL_REQUEST_ID,
    )
    return tool_call, pending


def _recommendation() -> SupportRecommendation:
    return SupportRecommendation.create(
        recommendation_id=_RECOMMENDATION_ID,
        workspace_id=_WORKSPACE_ID,
        ticket_id=_TICKET_ID,
        agent_run_id=_AGENT_RUN_ID,
        classification_id=_CLASSIFICATION_ID,
        accepted_llm_invocation_id=_RECOMMENDATION_INVOCATION_ID,
        recommended_action=SupportRecommendationAction.RESPOND,
        response_text="Grounded recommendation after approval decision.",
        requires_human_review=False,
        decision_summary="Approval-aware recommendation ready.",
        prompt_id="human-approved-support-recommendation",
        prompt_version=1,
        prompt_content_hash="b" * 64,
        provider="mock",
        model="mock-ticket-classifier-v1",
        now=_NOW,
    )


def _nodes(
    *,
    approval_request: ApprovalRequest,
    sensitive_execution_executor: AsyncMock,
) -> HumanApprovedSupportWorkflowNodes:
    recommendation = _recommendation()
    recommendation_executor = SimpleNamespace(
        execute=AsyncMock(
            return_value=HumanApprovedRecommendationOutcome(
                invocation_id=_RECOMMENDATION_INVOCATION_ID,
                recommendation=recommendation,
            ),
        ),
    )
    sensitive_output = SimpleNamespace(
        model_dump=lambda mode: {
            "escalation_id": str(uuid4()),
            "ticket_id": str(_TICKET_ID),
            "target_queue": "support_operations",
            "status": "escalated",
        },
    )
    sensitive_execution_executor.execute = AsyncMock(
        return_value=SimpleNamespace(output=sensitive_output),
    )
    sensitive_tool_execution = SimpleNamespace(
        execute=AsyncMock(
            side_effect=lambda state, context: (
                validate_human_approved_support_state(state)
                .model_copy(
                    update={
                        "approval_status": HumanApprovalCheckpointStatus.APPROVED,
                        "sensitive_execution_output": (
                            HumanApprovedSensitiveExecutionOutput(
                                escalation_id=uuid4(),
                                ticket_id=_TICKET_ID,
                                target_queue="support_operations",
                                status="escalated",
                            )
                        ),
                    },
                )
                .to_graph_state()
            ),
        ),
    )
    # Keep the shared mock in sync with node calls.
    original_execute = sensitive_tool_execution.execute

    async def _tracked_execute(
        state: HumanApprovedSupportGraphState,
        context: AgentRunExecutionContext,
    ) -> HumanApprovedSupportGraphState:
        await sensitive_execution_executor.execute(
            context=context,
            approval_request_id=_APPROVAL_REQUEST_ID,
            agent_tool_call_id=_AGENT_TOOL_CALL_ID,
        )
        return cast(
            HumanApprovedSupportGraphState,
            await original_execute(state, context),
        )

    sensitive_tool_execution.execute = AsyncMock(side_effect=_tracked_execute)

    return HumanApprovedSupportWorkflowNodes(
        transaction_manager=_NoOpTransactionManager(),
        classification_repository=cast(Any, SimpleNamespace()),
        classification_executor=cast(Any, SimpleNamespace()),
        decision_executor=cast(Any, SimpleNamespace()),
        sensitive_tool_registry=cast(Any, SimpleNamespace()),
        sensitive_proposal_service=cast(Any, SimpleNamespace()),
        sensitive_tool_execution=cast(Any, sensitive_tool_execution),
        approval_request_repository=cast(
            Any,
            SimpleNamespace(
                get_by_id=AsyncMock(return_value=approval_request),
            ),
        ),
        recommendation_executor=cast(Any, recommendation_executor),
    )


@pytest.fixture
def sensitive_execution_executor() -> AsyncMock:
    return AsyncMock()


@pytest.fixture
def initial_human_approved_execution_context() -> AgentRunExecutionContext:
    return _execution_context()


@pytest.fixture
def approved_human_approved_execution_context() -> AgentRunExecutionContext:
    return _execution_context()


@pytest.fixture
def rejected_human_approved_execution_context() -> AgentRunExecutionContext:
    return _execution_context()


@pytest.fixture
def expired_human_approved_execution_context() -> AgentRunExecutionContext:
    return _execution_context()


@pytest.fixture
def pending_human_approved_execution_context() -> AgentRunExecutionContext:
    return _execution_context()


@pytest.fixture
def human_approved_workflow_executor() -> HumanApprovedSupportWorkflowExecutor:
    interrupt_value = {
        "approval_request_id": str(_APPROVAL_REQUEST_ID),
        "agent_tool_call_id": str(_AGENT_TOOL_CALL_ID),
        "agent_run_id": str(_AGENT_RUN_ID),
        "ticket_id": str(_TICKET_ID),
        "tool_name": "escalate_ticket",
        "tool_version": 1,
        "proposed_input": {
            "target_queue": "support_operations",
            "reason": "Operational review required.",
        },
        "request_reason": "Operational review required.",
        "expires_at": (_NOW + timedelta(days=1)).isoformat(),
    }
    return HumanApprovedSupportWorkflowExecutor(
        graph=cast(Any, _PauseGraph(interrupt_value)),
        resume_planner=cast(Any, _Planner(InitialGraphExecution())),
    )


@pytest.fixture
def resumed_human_approved_workflow_executor(
    sensitive_execution_executor: AsyncMock,
) -> HumanApprovedSupportWorkflowExecutor:
    _tool_call, pending = _tool_call_and_pending()
    approved = pending.approve(
        actor_reference="operator:alice",
        comment=None,
        request_id=uuid4(),
        correlation_id=uuid4(),
        decided_at=_NOW + timedelta(minutes=5),
    )
    nodes = _nodes(
        approval_request=approved,
        sensitive_execution_executor=sensitive_execution_executor,
    )
    plan = ResumeGraphExecution(
        approval_request_id=_APPROVAL_REQUEST_ID,
        agent_tool_call_id=_AGENT_TOOL_CALL_ID,
        decision_status=ApprovalResumeDecisionStatus.APPROVED,
    )
    return HumanApprovedSupportWorkflowExecutor(
        graph=cast(
            Any,
            _ResumeGraph(
                checkpoint_values=_pending_checkpoint(),
                nodes=nodes,
                approval_request=approved,
                decision_status=ApprovalResumeDecisionStatus.APPROVED,
            ),
        ),
        resume_planner=cast(Any, _Planner(plan)),
    )


@pytest.fixture
def rejected_human_approved_workflow_executor(
    sensitive_execution_executor: AsyncMock,
) -> HumanApprovedSupportWorkflowExecutor:
    _tool_call, pending = _tool_call_and_pending()
    rejected = pending.reject(
        actor_reference="operator:alice",
        comment="Do not escalate.",
        request_id=uuid4(),
        correlation_id=uuid4(),
        decided_at=_NOW + timedelta(minutes=5),
    )
    nodes = _nodes(
        approval_request=rejected,
        sensitive_execution_executor=sensitive_execution_executor,
    )
    plan = ResumeGraphExecution(
        approval_request_id=_APPROVAL_REQUEST_ID,
        agent_tool_call_id=_AGENT_TOOL_CALL_ID,
        decision_status=ApprovalResumeDecisionStatus.REJECTED,
    )
    return HumanApprovedSupportWorkflowExecutor(
        graph=cast(
            Any,
            _ResumeGraph(
                checkpoint_values=_pending_checkpoint(),
                nodes=nodes,
                approval_request=rejected,
                decision_status=ApprovalResumeDecisionStatus.REJECTED,
            ),
        ),
        resume_planner=cast(Any, _Planner(plan)),
    )


@pytest.fixture
def expired_human_approved_workflow_executor(
    sensitive_execution_executor: AsyncMock,
) -> HumanApprovedSupportWorkflowExecutor:
    _tool_call, pending = _tool_call_and_pending()
    expired = pending.expire(
        decided_at=_NOW + timedelta(days=1),
    )
    nodes = _nodes(
        approval_request=expired,
        sensitive_execution_executor=sensitive_execution_executor,
    )
    plan = ResumeGraphExecution(
        approval_request_id=_APPROVAL_REQUEST_ID,
        agent_tool_call_id=_AGENT_TOOL_CALL_ID,
        decision_status=ApprovalResumeDecisionStatus.EXPIRED,
    )
    return HumanApprovedSupportWorkflowExecutor(
        graph=cast(
            Any,
            _ResumeGraph(
                checkpoint_values=_pending_checkpoint(),
                nodes=nodes,
                approval_request=expired,
                decision_status=ApprovalResumeDecisionStatus.EXPIRED,
            ),
        ),
        resume_planner=cast(Any, _Planner(plan)),
    )


@pytest.fixture
def pending_human_approved_workflow_executor(
    sensitive_execution_executor: AsyncMock,
) -> HumanApprovedSupportWorkflowExecutor:
    _tool_call, pending = _tool_call_and_pending()
    nodes = _nodes(
        approval_request=pending,
        sensitive_execution_executor=sensitive_execution_executor,
    )
    plan = ResumeGraphExecution(
        approval_request_id=_APPROVAL_REQUEST_ID,
        agent_tool_call_id=_AGENT_TOOL_CALL_ID,
        decision_status=ApprovalResumeDecisionStatus.APPROVED,
    )
    return HumanApprovedSupportWorkflowExecutor(
        graph=cast(
            Any,
            _FailingResumeGraph(
                checkpoint_values=_pending_checkpoint(),
                nodes=nodes,
                approval_request=pending,
                decision_status=ApprovalResumeDecisionStatus.APPROVED,
            ),
        ),
        resume_planner=cast(Any, _Planner(plan)),
    )


@pytest.mark.asyncio
async def test_initial_execution_pauses_for_approval(
    human_approved_workflow_executor: HumanApprovedSupportWorkflowExecutor,
    initial_human_approved_execution_context: AgentRunExecutionContext,
) -> None:
    result = await human_approved_workflow_executor.execute(
        initial_human_approved_execution_context,
    )

    assert isinstance(result, PausedForApproval)
    assert result.approval_request_id == _APPROVAL_REQUEST_ID
    identity = derive_human_approved_support_graph_identity(_AGENT_RUN_ID)
    assert result.graph_thread_id == identity.thread_id


@pytest.mark.asyncio
async def test_approved_resume_completes_after_sensitive_execution(
    resumed_human_approved_workflow_executor: HumanApprovedSupportWorkflowExecutor,
    approved_human_approved_execution_context: AgentRunExecutionContext,
) -> None:
    result = await resumed_human_approved_workflow_executor.execute(
        approved_human_approved_execution_context,
    )

    assert isinstance(result, CompletedExecution)


@pytest.mark.asyncio
async def test_rejected_resume_completes_without_sensitive_execution(
    rejected_human_approved_workflow_executor: HumanApprovedSupportWorkflowExecutor,
    rejected_human_approved_execution_context: AgentRunExecutionContext,
    sensitive_execution_executor: AsyncMock,
) -> None:
    result = await rejected_human_approved_workflow_executor.execute(
        rejected_human_approved_execution_context,
    )

    assert isinstance(result, CompletedExecution)
    sensitive_execution_executor.execute.assert_not_awaited()


@pytest.mark.asyncio
async def test_expired_resume_completes_without_sensitive_execution(
    expired_human_approved_workflow_executor: HumanApprovedSupportWorkflowExecutor,
    expired_human_approved_execution_context: AgentRunExecutionContext,
    sensitive_execution_executor: AsyncMock,
) -> None:
    result = await expired_human_approved_workflow_executor.execute(
        expired_human_approved_execution_context,
    )

    assert isinstance(result, CompletedExecution)
    sensitive_execution_executor.execute.assert_not_awaited()


@pytest.mark.asyncio
async def test_pending_resume_fails_closed(
    pending_human_approved_workflow_executor: HumanApprovedSupportWorkflowExecutor,
    pending_human_approved_execution_context: AgentRunExecutionContext,
) -> None:
    with pytest.raises(TerminalAgentRunExecutionError):
        await pending_human_approved_workflow_executor.execute(
            pending_human_approved_execution_context,
        )


@pytest.mark.asyncio
async def test_await_human_approval_projects_resume_payload(
    monkeypatch: pytest.MonkeyPatch,
    sensitive_execution_executor: AsyncMock,
) -> None:
    _tool_call, pending = _tool_call_and_pending()
    nodes = _nodes(
        approval_request=pending,
        sensitive_execution_executor=sensitive_execution_executor,
    )
    resume_payload = ApprovalDecisionResumePayload(
        approval_request_id=_APPROVAL_REQUEST_ID,
        agent_tool_call_id=_AGENT_TOOL_CALL_ID,
        decision_status=ApprovalResumeDecisionStatus.APPROVED,
    )

    def _fake_interrupt(payload: object) -> object:
        del payload
        return resume_payload.to_json_value()

    monkeypatch.setattr(
        "supportops.agent_graph.application.human_approved_nodes.interrupt_for_approval",
        _fake_interrupt,
    )
    state = cast(HumanApprovedSupportGraphState, _pending_checkpoint())

    result = await nodes.await_human_approval(state)
    snapshot = validate_human_approved_support_state(result)

    assert snapshot.approval_status is HumanApprovalCheckpointStatus.PENDING
    assert snapshot.approval_resume_payload is not None
    assert snapshot.approval_resume_payload.decision_status is (
        ApprovalResumeDecisionStatus.APPROVED
    )


@pytest.mark.asyncio
async def test_await_human_approval_rejects_malformed_resume_payload(
    monkeypatch: pytest.MonkeyPatch,
    sensitive_execution_executor: AsyncMock,
) -> None:
    _tool_call, pending = _tool_call_and_pending()
    nodes = _nodes(
        approval_request=pending,
        sensitive_execution_executor=sensitive_execution_executor,
    )

    monkeypatch.setattr(
        "supportops.agent_graph.application.human_approved_nodes.interrupt_for_approval",
        lambda payload: {"unexpected": True},
    )
    state = cast(HumanApprovedSupportGraphState, _pending_checkpoint())

    with pytest.raises(TerminalAgentRunExecutionError) as captured:
        await nodes.await_human_approval(state)

    assert captured.value.error_code == "approval_resume_payload_invalid"
