"""Unit tests for the controlled support graph executor."""

from collections.abc import Mapping
from contextlib import AbstractContextManager
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace, TracebackType
from typing import Any, Literal, cast
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID

import pytest
from langgraph.checkpoint.memory import MemorySaver
from langgraph.errors import GraphRecursionError
from langgraph.types import Command

from supportops.agent_graph.application.human_approved_workflow import (
    HumanApprovedSupportWorkflowExecutor,
)
from supportops.agent_graph.application.workflow import (
    CONTROLLED_SUPPORT_LANGGRAPH_RECURSION_LIMIT,
    ControlledSupportWorkflowExecutor,
    ControlledSupportWorkflowNodes,
    compile_controlled_support_graph,
)
from supportops.agent_graph.domain.human_approved_identity import (
    derive_human_approved_support_graph_identity,
)
from supportops.agent_graph.domain.human_approved_state import (
    HUMAN_APPROVED_SUPPORT_WORKFLOW_VERSION,
    create_initial_human_approved_support_state,
)
from supportops.agent_graph.domain.identity import (
    derive_controlled_support_graph_identity,
)
from supportops.agent_graph.domain.resume_planning import (
    ApprovalResumeDecisionStatus,
    CompletedGraphExecution,
    ContinueGraphExecution,
    IncompatibleGraphState,
    InitialGraphExecution,
    ResumeGraphExecution,
)
from supportops.agent_graph.domain.state import (
    CONTROLLED_SUPPORT_GRAPH_VERSION,
    CONTROLLED_SUPPORT_STATE_SCHEMA_VERSION,
    CONTROLLED_SUPPORT_WORKFLOW_NAME,
    CONTROLLED_SUPPORT_WORKFLOW_VERSION,
    ControlledSupportGraphState,
    ControlledSupportGraphStateSnapshot,
    SupportAnalysisCompletionSnapshot,
    create_initial_controlled_support_state,
)
from supportops.ai.schemas.ticket_classification import (
    TicketCategory,
    TicketIntent,
    TicketSentiment,
    TicketUrgency,
)
from supportops.modules.agent_runs.application.execution import (
    AgentRunExecutionContext,
    CompletedExecution,
    RetryableAgentRunExecutionError,
    TerminalAgentRunExecutionError,
)
from supportops.modules.agent_runs.domain.models import (
    AgentRun,
    AgentRunAttempt,
    AgentRunStatus,
)
from supportops.modules.tickets.domain.models import Ticket
from supportops.observability.context import (
    ActiveObservationContext,
    current_observation_context,
    observation_context_scope,
)
from supportops.observability.models import (
    ObservabilityProvider,
    ObservationAttributes,
    ObservationStatus,
    ObservationType,
    ObservationUpdate,
    TraceAttributes,
)

_WORKSPACE_ID = UUID("10000000-0000-4000-8000-000000000001")
_TICKET_ID = UUID("20000000-0000-4000-8000-000000000002")
_AGENT_RUN_ID = UUID("30000000-0000-4000-8000-000000000003")
_ATTEMPT_ID = UUID("40000000-0000-4000-8000-000000000004")
_LEASE_TOKEN = UUID("50000000-0000-4000-8000-000000000005")
_CLASSIFICATION_ID = UUID("60000000-0000-4000-8000-000000000006")
_RECOMMENDATION_INVOCATION_ID = UUID("70000000-0000-4000-8000-000000000007")
_RECOMMENDATION_ID = UUID("80000000-0000-4000-8000-000000000008")

_NOW = datetime(
    2026,
    8,
    2,
    18,
    0,
    tzinfo=UTC,
)


@dataclass(frozen=True, slots=True)
class StubCheckpointSnapshot:
    """Expose one configured checkpoint value mapping."""

    values: Mapping[str, object]


class RecordingCompiledGraph:
    """Record graph resume and invocation behavior."""

    def __init__(
        self,
        *,
        checkpoint_values: Mapping[str, object],
        result: Mapping[str, object] | None,
        recursion_failure: bool = False,
    ) -> None:
        self.checkpoint_values = checkpoint_values
        self.result = result
        self.recursion_failure = recursion_failure
        self.state_configs: list[Mapping[str, object]] = []
        self.inputs: list[ControlledSupportGraphState | None] = []
        self.invoke_configs: list[Mapping[str, object]] = []
        self.contexts: list[AgentRunExecutionContext] = []

    async def aget_state(
        self,
        config: Mapping[str, object],
    ) -> StubCheckpointSnapshot:
        self.state_configs.append(config)

        return StubCheckpointSnapshot(values=self.checkpoint_values)

    async def ainvoke(
        self,
        input: ControlledSupportGraphState | None,
        config: Mapping[str, object],
        *,
        context: AgentRunExecutionContext,
    ) -> Mapping[str, object] | None:
        self.inputs.append(input)
        self.invoke_configs.append(config)
        self.contexts.append(context)

        if self.recursion_failure:
            raise GraphRecursionError("recursion limit exceeded")

        return self.result


def _context() -> AgentRunExecutionContext:
    ticket = Ticket.create(
        ticket_id=_TICKET_ID,
        workspace_id=_WORKSPACE_ID,
        subject="Unable to reset account access",
        description=("The customer cannot complete the documented account recovery procedure."),
        external_reference=None,
        ingestion_request_id=UUID("90000000-0000-4000-8000-000000000009"),
        correlation_id=UUID("a0000000-0000-4000-8000-000000000010"),
        now=_NOW - timedelta(minutes=1),
    )
    initial_run = AgentRun.create_initial(
        agent_run_id=_AGENT_RUN_ID,
        workspace_id=_WORKSPACE_ID,
        ticket_id=_TICKET_ID,
        ingestion_request_id=(ticket.ingestion_request_id),
        correlation_id=ticket.correlation_id,
        workflow_version=(CONTROLLED_SUPPORT_WORKFLOW_VERSION),
        max_retryable_failures=3,
        now=_NOW - timedelta(minutes=1),
    )
    running_run = replace(
        initial_run,
        status=AgentRunStatus.RUNNING,
        attempt_count=1,
        lease_owner="worker-a",
        lease_token=_LEASE_TOKEN,
        lease_expires_at=_NOW + timedelta(minutes=5),
        first_started_at=_NOW,
        updated_at=_NOW,
    )
    attempt = AgentRunAttempt.start(
        attempt_id=_ATTEMPT_ID,
        agent_run_id=_AGENT_RUN_ID,
        attempt_number=1,
        worker_id="worker-a",
        lease_token=_LEASE_TOKEN,
        execution_request_id=UUID("b0000000-0000-4000-8000-000000000011"),
        now=_NOW,
    )

    return AgentRunExecutionContext(
        agent_run=running_run,
        attempt=attempt,
        ticket=ticket,
    )


def _completed_state(
    *,
    workspace_id: UUID = _WORKSPACE_ID,
) -> ControlledSupportGraphState:
    snapshot = ControlledSupportGraphStateSnapshot(
        state_schema_version=(CONTROLLED_SUPPORT_STATE_SCHEMA_VERSION),
        workflow_name=CONTROLLED_SUPPORT_WORKFLOW_NAME,
        workflow_version=(CONTROLLED_SUPPORT_WORKFLOW_VERSION),
        graph_version=CONTROLLED_SUPPORT_GRAPH_VERSION,
        workspace_id=workspace_id,
        ticket_id=_TICKET_ID,
        agent_run_id=_AGENT_RUN_ID,
        classification_id=_CLASSIFICATION_ID,
        classification_category=(TicketCategory.ACCOUNT_ACCESS),
        classification_intent=(TicketIntent.REQUEST_ACCESS),
        classification_urgency=TicketUrgency.NORMAL,
        classification_sentiment=(TicketSentiment.NEUTRAL),
        classification_requires_human_review=False,
        classification_summary=("The customer needs documented account recovery guidance."),
        graph_step_count=3,
        decision_turn_count=1,
        tool_call_count=0,
        seen_tool_call_fingerprints=(),
        tool_call_ids=(),
        retrieval_query_ids=(),
        retrieved_chunk_ids=(),
        service_status_tool_call_ids=(),
        analysis_completion=(
            SupportAnalysisCompletionSnapshot(
                recommended_action="respond",
                evidence_sufficient=True,
                requires_human_review=False,
                decision_summary=("Available evidence supports a direct response."),
            )
        ),
        recommendation_invocation_id=(_RECOMMENDATION_INVOCATION_ID),
        recommendation_id=_RECOMMENDATION_ID,
        current_error_code=None,
    )

    return snapshot.to_graph_state()


async def test_starts_new_graph_with_exact_checkpoint_identity() -> None:
    graph = RecordingCompiledGraph(
        checkpoint_values={},
        result=_completed_state(),
    )
    context = _context()
    executor = ControlledSupportWorkflowExecutor(graph=graph)

    result = await executor.execute(context)

    assert result == CompletedExecution()
    assert len(graph.inputs) == 1
    initial_input = graph.inputs[0]

    assert initial_input is not None
    assert initial_input["workspace_id"] == str(_WORKSPACE_ID)
    assert initial_input["ticket_id"] == str(_TICKET_ID)
    assert initial_input["agent_run_id"] == str(_AGENT_RUN_ID)

    identity = derive_controlled_support_graph_identity(_AGENT_RUN_ID)
    config = graph.invoke_configs[0]
    configurable = config["configurable"]

    assert isinstance(configurable, dict)
    assert configurable["thread_id"] == identity.thread_id
    assert configurable["checkpoint_ns"] == (identity.checkpoint_namespace)
    assert config["recursion_limit"] == (CONTROLLED_SUPPORT_LANGGRAPH_RECURSION_LIMIT)
    assert graph.contexts == [context]


async def test_resumes_existing_checkpoint_with_none_input() -> None:
    completed = _completed_state()
    graph = RecordingCompiledGraph(
        checkpoint_values=completed,
        result=completed,
    )
    executor = ControlledSupportWorkflowExecutor(graph=graph)

    result = await executor.execute(_context())

    assert result == CompletedExecution()
    assert graph.inputs == [None]


async def test_rejects_checkpoint_ownership_mismatch() -> None:
    graph = RecordingCompiledGraph(
        checkpoint_values=_completed_state(
            workspace_id=UUID("c0000000-0000-4000-8000-000000000012")
        ),
        result=None,
    )
    executor = ControlledSupportWorkflowExecutor(graph=graph)

    with pytest.raises(
        TerminalAgentRunExecutionError,
    ) as captured:
        await executor.execute(_context())

    assert captured.value.error_code == ("graph_state_ownership_mismatch")
    assert graph.inputs == []


async def test_rejects_graph_completion_without_recommendation() -> None:
    incomplete = create_initial_controlled_support_state(
        workspace_id=_WORKSPACE_ID,
        ticket_id=_TICKET_ID,
        agent_run_id=_AGENT_RUN_ID,
    )
    graph = RecordingCompiledGraph(
        checkpoint_values={},
        result=incomplete,
    )
    executor = ControlledSupportWorkflowExecutor(graph=graph)

    with pytest.raises(
        TerminalAgentRunExecutionError,
    ) as captured:
        await executor.execute(_context())

    assert captured.value.error_code == ("controlled_workflow_incomplete")


async def test_normalizes_graph_recursion_failure() -> None:
    graph = RecordingCompiledGraph(
        checkpoint_values={},
        result=None,
        recursion_failure=True,
    )
    executor = ControlledSupportWorkflowExecutor(graph=graph)

    with pytest.raises(
        TerminalAgentRunExecutionError,
    ) as captured:
        await executor.execute(_context())

    assert captured.value.error_code == ("graph_recursion_limit_exceeded")


async def test_rejects_unsupported_workflow_version() -> None:
    context = _context()
    unsupported = replace(
        context,
        agent_run=replace(
            context.agent_run,
            workflow_version="unsupported-v1",
        ),
    )
    graph = RecordingCompiledGraph(
        checkpoint_values={},
        result=_completed_state(),
    )
    executor = ControlledSupportWorkflowExecutor(graph=graph)

    with pytest.raises(
        TerminalAgentRunExecutionError,
    ) as captured:
        await executor.execute(unsupported)

    assert captured.value.error_code == ("unsupported_workflow_version")
    assert graph.state_configs == []


@dataclass(frozen=True, slots=True)
class HumanApprovedStubCheckpointSnapshot:
    """Expose checkpoint values and active interrupts."""

    values: Mapping[str, object]
    interrupts: tuple[object, ...] = ()


class HumanApprovedRecordingCompiledGraph:
    """Record human-approved graph resume and invocation behavior."""

    def __init__(
        self,
        *,
        checkpoint_values: Mapping[str, object],
        result: object,
        checkpoint_interrupts: tuple[object, ...] = (),
        recursion_failure: bool = False,
    ) -> None:
        self.checkpoint_values = checkpoint_values
        self.checkpoint_interrupts = checkpoint_interrupts
        self.result = result
        self.recursion_failure = recursion_failure
        self.state_configs: list[Mapping[str, object]] = []
        self.inputs: list[object] = []
        self.invoke_configs: list[Mapping[str, object]] = []
        self.contexts: list[AgentRunExecutionContext] = []

    async def aget_state(
        self,
        config: Mapping[str, object],
    ) -> HumanApprovedStubCheckpointSnapshot:
        self.state_configs.append(config)
        return HumanApprovedStubCheckpointSnapshot(
            values=self.checkpoint_values,
            interrupts=self.checkpoint_interrupts,
        )

    async def ainvoke(
        self,
        input: object,
        config: Mapping[str, object],
        *,
        context: AgentRunExecutionContext,
        version: str = "v2",
    ) -> object:
        del version
        self.inputs.append(input)
        self.invoke_configs.append(config)
        self.contexts.append(context)
        if self.recursion_failure:
            raise GraphRecursionError("recursion limit exceeded")
        return self.result


class RecordingResumePlanner:
    """Return one configured human-approved execution plan."""

    def __init__(self, planned: object) -> None:
        self.planned = planned
        self.calls: list[dict[str, object]] = []

    async def plan(
        self,
        *,
        context: object,
        checkpoint_values: Mapping[str, object],
        checkpoint_interrupts: tuple[object, ...],
    ) -> object:
        self.calls.append(
            {
                "context": context,
                "checkpoint_values": checkpoint_values,
                "checkpoint_interrupts": checkpoint_interrupts,
            }
        )
        return self.planned


def _human_approved_context() -> AgentRunExecutionContext:
    context = _context()
    return replace(
        context,
        agent_run=replace(
            context.agent_run,
            workflow_version=HUMAN_APPROVED_SUPPORT_WORKFLOW_VERSION,
        ),
    )


def _human_approved_completed_state() -> Mapping[str, object]:
    state = create_initial_human_approved_support_state(
        workspace_id=_WORKSPACE_ID,
        ticket_id=_TICKET_ID,
        agent_run_id=_AGENT_RUN_ID,
    )
    state.update(
        {
            "run_context_loaded": True,
            "decision_kind": "terminal",
            "decision_invocation_id": str(_RECOMMENDATION_INVOCATION_ID),
            "decision_summary": "Respond with available evidence.",
            "analysis_recommended_action": "respond",
            "analysis_evidence_sufficient": True,
            "analysis_requires_human_review": False,
            "recommendation_invocation_id": str(_RECOMMENDATION_INVOCATION_ID),
            "recommendation_id": str(_RECOMMENDATION_ID),
        },
    )
    return state


async def test_human_approved_initial_plan_invokes_with_state() -> None:
    graph = HumanApprovedRecordingCompiledGraph(
        checkpoint_values={},
        result=_human_approved_completed_state(),
    )
    planner = RecordingResumePlanner(InitialGraphExecution())
    executor = HumanApprovedSupportWorkflowExecutor(
        graph=graph,
        resume_planner=planner,  # type: ignore[arg-type]
    )

    result = await executor.execute(_human_approved_context())

    assert result == CompletedExecution()
    assert len(graph.inputs) == 1
    initial_input = graph.inputs[0]
    assert isinstance(initial_input, dict)
    assert initial_input["workspace_id"] == str(_WORKSPACE_ID)
    assert initial_input["ticket_id"] == str(_TICKET_ID)
    assert initial_input["agent_run_id"] == str(_AGENT_RUN_ID)
    identity = derive_human_approved_support_graph_identity(_AGENT_RUN_ID)
    configurable = graph.invoke_configs[0]["configurable"]
    assert isinstance(configurable, dict)
    assert configurable["thread_id"] == identity.thread_id
    assert configurable["checkpoint_ns"] == (identity.checkpoint_namespace)


async def test_human_approved_continue_plan_invokes_with_none() -> None:
    completed = _human_approved_completed_state()
    graph = HumanApprovedRecordingCompiledGraph(
        checkpoint_values={
            **create_initial_human_approved_support_state(
                workspace_id=_WORKSPACE_ID,
                ticket_id=_TICKET_ID,
                agent_run_id=_AGENT_RUN_ID,
            ),
            "run_context_loaded": True,
        },
        result=completed,
    )
    planner = RecordingResumePlanner(ContinueGraphExecution())
    executor = HumanApprovedSupportWorkflowExecutor(
        graph=graph,
        resume_planner=planner,  # type: ignore[arg-type]
    )

    result = await executor.execute(_human_approved_context())

    assert result == CompletedExecution()
    assert graph.inputs == [None]


async def test_human_approved_resume_plan_invokes_with_command() -> None:
    approval_request_id = UUID("d0000000-0000-4000-8000-000000000013")
    agent_tool_call_id = UUID("e0000000-0000-4000-8000-000000000014")
    completed = _human_approved_completed_state()
    graph = HumanApprovedRecordingCompiledGraph(
        checkpoint_values=create_initial_human_approved_support_state(
            workspace_id=_WORKSPACE_ID,
            ticket_id=_TICKET_ID,
            agent_run_id=_AGENT_RUN_ID,
        ),
        result=completed,
    )
    plan = ResumeGraphExecution(
        approval_request_id=approval_request_id,
        agent_tool_call_id=agent_tool_call_id,
        decision_status=ApprovalResumeDecisionStatus.APPROVED,
    )
    planner = RecordingResumePlanner(plan)
    executor = HumanApprovedSupportWorkflowExecutor(
        graph=graph,
        resume_planner=planner,  # type: ignore[arg-type]
    )

    result = await executor.execute(_human_approved_context())

    assert result == CompletedExecution()
    assert len(graph.inputs) == 1
    command = graph.inputs[0]
    assert isinstance(command, Command)
    assert command.resume == {
        "approval_request_id": str(approval_request_id),
        "agent_tool_call_id": str(agent_tool_call_id),
        "decision_status": "approved",
    }
    identity = derive_human_approved_support_graph_identity(_AGENT_RUN_ID)
    configurable = graph.invoke_configs[0]["configurable"]
    assert isinstance(configurable, dict)
    assert configurable["thread_id"] == identity.thread_id
    assert configurable["checkpoint_ns"] == (identity.checkpoint_namespace)
    assert graph.state_configs[0]["configurable"] == configurable


async def test_human_approved_completed_plan_skips_ainvoke() -> None:
    graph = HumanApprovedRecordingCompiledGraph(
        checkpoint_values=_human_approved_completed_state(),
        result=_human_approved_completed_state(),
    )
    planner = RecordingResumePlanner(CompletedGraphExecution())
    executor = HumanApprovedSupportWorkflowExecutor(
        graph=graph,
        resume_planner=planner,  # type: ignore[arg-type]
    )

    result = await executor.execute(_human_approved_context())

    assert result == CompletedExecution()
    assert graph.inputs == []
    assert graph.invoke_configs == []
    assert len(graph.state_configs) == 1


async def test_human_approved_incompatible_plan_raises_without_ainvoke() -> None:
    graph = HumanApprovedRecordingCompiledGraph(
        checkpoint_values={},
        result=_human_approved_completed_state(),
    )
    planner = RecordingResumePlanner(
        IncompatibleGraphState(
            error_code="approval_request_still_pending",
            error_summary=("A pending approval request cannot resume execution."),
        ),
    )
    executor = HumanApprovedSupportWorkflowExecutor(
        graph=graph,
        resume_planner=planner,  # type: ignore[arg-type]
    )

    with pytest.raises(TerminalAgentRunExecutionError) as captured:
        await executor.execute(_human_approved_context())

    assert captured.value.error_code == ("approval_request_still_pending")
    assert graph.inputs == []
    assert graph.invoke_configs == []


async def test_human_approved_pending_approval_fails_closed() -> None:
    graph = HumanApprovedRecordingCompiledGraph(
        checkpoint_values={},
        result=_human_approved_completed_state(),
    )
    planner = RecordingResumePlanner(
        IncompatibleGraphState(
            error_code="approval_request_still_pending",
            error_summary=("A pending approval request cannot resume execution."),
        ),
    )
    executor = HumanApprovedSupportWorkflowExecutor(
        graph=graph,
        resume_planner=planner,  # type: ignore[arg-type]
    )

    with pytest.raises(TerminalAgentRunExecutionError) as captured:
        await executor.execute(_human_approved_context())

    assert captured.value.error_code == ("approval_request_still_pending")
    assert graph.inputs == []


# ---------------------------------------------------------------------------
# Controlled-support workflow / node observability (Commit 2 / PR C)
# ---------------------------------------------------------------------------


@dataclass
class _RecordingObservationScope:
    attributes: ObservationAttributes
    updates: list[ObservationUpdate] = field(default_factory=list)
    closed: bool = False
    update_error: Exception | None = None
    observation_id: str | None = "obs-1"
    parent_name: str | None = None

    def update(self, update: ObservationUpdate) -> None:
        if self.update_error is not None:
            raise self.update_error
        self.updates.append(update)

    def start_observation(
        self,
        attributes: ObservationAttributes,
    ) -> AbstractContextManager["_RecordingObservationScope"]:
        del attributes
        raise AssertionError("nested start_observation via scope is unused")

    def record_event(self, event: object) -> None:
        del event


class _RecordingObservationManager(AbstractContextManager[_RecordingObservationScope]):
    def __init__(
        self,
        scope: _RecordingObservationScope,
        *,
        client: "_RecordingObservabilityClient",
        exit_error: Exception | None = None,
        install_context: bool = False,
    ) -> None:
        self._scope = scope
        self._client = client
        self._exit_error = exit_error
        self._install_context = install_context
        self._context: Any = None
        self.exit_args: (
            tuple[
                type[BaseException] | None,
                BaseException | None,
                TracebackType | None,
            ]
            | None
        ) = None

    def __enter__(self) -> _RecordingObservationScope:
        if self._install_context:
            self._context = observation_context_scope(
                ActiveObservationContext(
                    name=self._scope.attributes.name,
                    observation_id=self._scope.observation_id,
                )
            )
            self._context.__enter__()
        self._client.active_stack.append(self._scope.attributes.name)
        self._client.lifecycle.append(("enter", self._scope.attributes.name))
        return self._scope

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> Literal[False]:
        self.exit_args = (exc_type, exc, traceback)
        self._client.lifecycle.append(("exit", self._scope.attributes.name))
        if self._client.active_stack:
            self._client.active_stack.pop()
        if self._context is not None:
            self._context.__exit__(None, None, None)
        self._scope.closed = True
        if self._exit_error is not None:
            raise self._exit_error
        return False


class _RecordingObservabilityClient:
    """Capture workflow/node observations without a provider backend."""

    def __init__(
        self,
        *,
        start_error: Exception | None = None,
        update_error: Exception | None = None,
        exit_error: Exception | None = None,
        install_context: bool = False,
    ) -> None:
        self.started_attributes: list[ObservationAttributes] = []
        self.scopes: list[_RecordingObservationScope] = []
        self.managers: list[_RecordingObservationManager] = []
        self.lifecycle: list[tuple[str, str]] = []
        self.active_stack: list[str] = []
        self.parent_at_start: list[str | None] = []
        self.start_error = start_error
        self.update_error = update_error
        self.exit_error = exit_error
        self.install_context = install_context
        self.enabled = True
        self.provider = ObservabilityProvider.NOOP
        self.shutdown_calls = 0

    def start_trace(self, attributes: TraceAttributes) -> AbstractContextManager[Any]:
        del attributes
        raise AssertionError("controlled workflow must not start traces")

    def start_observation(
        self,
        attributes: ObservationAttributes,
    ) -> AbstractContextManager[_RecordingObservationScope]:
        if self.start_error is not None:
            raise self.start_error

        parent_context = current_observation_context()
        parent_name = (
            parent_context.name
            if parent_context is not None
            else (self.active_stack[-1] if self.active_stack else None)
        )
        self.parent_at_start.append(parent_name)
        self.started_attributes.append(attributes)
        scope = _RecordingObservationScope(
            attributes=attributes,
            update_error=self.update_error,
            parent_name=parent_name,
        )
        manager = _RecordingObservationManager(
            scope,
            client=self,
            exit_error=self.exit_error,
            install_context=self.install_context,
        )
        self.scopes.append(scope)
        self.managers.append(manager)
        return manager

    def record_event(self, event: object) -> None:
        del event

    def record_trace_event(self, *, identity: object, event: object) -> None:
        del identity, event

    def flush(self) -> None:
        return None

    def shutdown(self) -> None:
        self.shutdown_calls += 1


_FORBIDDEN_METADATA_KEYS = frozenset(
    {
        "graph_state",
        "ticket_subject",
        "ticket_description",
        "conversation_content",
        "classification_text",
        "classification_summary",
        "tool_arguments",
        "tool_output",
        "recommendation_text",
        "response_text",
        "evidence_content",
        "checkpoint_payload",
        "decision_summary",
        "prompt_content",
        "model_output",
        "proposed_input",
        "approval_comment",
        "approver_identity",
        "escalation_reason",
        "citation_text",
        "document_content",
        "chunk_content",
        "embedding_vectors",
        "lease_token",
        "execution_grant",
        "authorization_headers",
        "credentials",
        "traceback",
        "user_id",
    }
)
_WORKFLOW_ALLOWED_METADATA_KEYS = frozenset(
    {
        "agent_run_id",
        "agent_run_attempt_id",
        "execution_request_id",
        "workspace_id",
        "ticket_id",
        "workflow_name",
        "workflow_version",
        "trigger_key",
        "correlation_id",
        "graph_thread_id",
        "invocation_mode",
        "workflow_outcome",
        "error_code",
        "latency_ms",
    }
)
_NODE_ALLOWED_METADATA_KEYS = frozenset(
    {
        "node_name",
        "agent_run_id",
        "agent_run_attempt_id",
        "execution_request_id",
        "workspace_id",
        "ticket_id",
        "workflow_name",
        "workflow_version",
        "correlation_id",
        "classification_present",
        "tool_decision_mode",
        "tool_execution_count",
        "evidence_count",
        "recommendation_created",
        "error_code",
        "latency_ms",
    }
)


def _assert_safe_metadata(
    metadata: Mapping[str, object],
    *,
    allowed: frozenset[str],
) -> None:
    assert _FORBIDDEN_METADATA_KEYS.isdisjoint(metadata)
    assert set(metadata).issubset(allowed)


def _workflow_scopes(
    client: _RecordingObservabilityClient,
) -> list[_RecordingObservationScope]:
    return [
        scope
        for scope in client.scopes
        if scope.attributes.name == "workflow.controlled-support-v1"
    ]


def _node_scopes(
    client: _RecordingObservabilityClient,
) -> list[_RecordingObservationScope]:
    return [scope for scope in client.scopes if scope.attributes.name.startswith("graph-node.")]


def _failing_graph(*, error: BaseException) -> RecordingCompiledGraph:
    graph = RecordingCompiledGraph(
        checkpoint_values={},
        result=_completed_state(),
    )

    async def failing_ainvoke(
        input: ControlledSupportGraphState | None,
        config: Mapping[str, object],
        *,
        context: AgentRunExecutionContext,
    ) -> Mapping[str, object] | None:
        graph.inputs.append(input)
        graph.invoke_configs.append(config)
        graph.contexts.append(context)
        raise error

    graph.ainvoke = failing_ainvoke  # type: ignore[method-assign]
    return graph


def _stub_nodes(
    *,
    observability_client: _RecordingObservabilityClient,
) -> ControlledSupportWorkflowNodes:
    return ControlledSupportWorkflowNodes(
        transaction_manager=cast(Any, MagicMock()),
        classification_repository=cast(Any, MagicMock()),
        classification_executor=cast(Any, MagicMock()),
        observation_assembler=cast(Any, MagicMock()),
        decision_executor=cast(Any, MagicMock()),
        tool_executor=cast(Any, MagicMock()),
        recommendation_executor=cast(Any, MagicMock()),
        observability_client=cast(Any, observability_client),
    )


def _runtime() -> Any:
    return SimpleNamespace(context=_context())


def _initial_graph_state() -> ControlledSupportGraphState:
    return create_initial_controlled_support_state(
        workspace_id=_WORKSPACE_ID,
        ticket_id=_TICKET_ID,
        agent_run_id=_AGENT_RUN_ID,
    )


def _error_graph_state(*, error_code: str) -> ControlledSupportGraphState:
    state = dict(_initial_graph_state())
    state["current_error_code"] = error_code
    return cast(ControlledSupportGraphState, state)


async def test_controlled_graph_invocation_creates_one_chain_observation() -> None:
    observability = _RecordingObservabilityClient()
    executor = ControlledSupportWorkflowExecutor(
        graph=RecordingCompiledGraph(
            checkpoint_values={},
            result=_completed_state(),
        ),
        observability_client=cast(Any, observability),
    )

    result = await executor.execute(_context())

    assert result == CompletedExecution()
    workflows = _workflow_scopes(observability)
    assert len(workflows) == 1
    assert workflows[0].attributes.observation_type is ObservationType.CHAIN
    assert workflows[0].attributes.name == "workflow.controlled-support-v1"
    assert workflows[0].closed is True


async def test_graph_compilation_creates_no_observation() -> None:
    observability = _RecordingObservabilityClient()
    compile_controlled_support_graph(
        nodes=_stub_nodes(observability_client=observability),
        checkpointer=MemorySaver(),
    )
    assert observability.started_attributes == []


async def test_executor_construction_creates_no_observation() -> None:
    observability = _RecordingObservabilityClient()
    ControlledSupportWorkflowExecutor(
        graph=RecordingCompiledGraph(
            checkpoint_values={},
            result=_completed_state(),
        ),
        observability_client=cast(Any, observability),
    )
    assert observability.started_attributes == []


async def test_successful_invocation_updates_workflow_status_ok() -> None:
    observability = _RecordingObservabilityClient()
    await ControlledSupportWorkflowExecutor(
        graph=RecordingCompiledGraph(
            checkpoint_values={},
            result=_completed_state(),
        ),
        observability_client=cast(Any, observability),
    ).execute(_context())

    update = _workflow_scopes(observability)[0].updates[0]
    assert update.status is ObservationStatus.OK
    assert update.metadata["workflow_outcome"] == "completed"


async def test_retryable_exception_updates_workflow_error() -> None:
    observability = _RecordingObservabilityClient()
    original = RetryableAgentRunExecutionError(
        error_code="tool_timeout",
        error_summary="Tool timed out.",
    )
    with pytest.raises(RetryableAgentRunExecutionError) as captured:
        await ControlledSupportWorkflowExecutor(
            graph=_failing_graph(error=original),
            observability_client=cast(Any, observability),
        ).execute(_context())

    assert captured.value is original
    update = _workflow_scopes(observability)[0].updates[0]
    assert update.status is ObservationStatus.ERROR
    assert update.metadata["workflow_outcome"] == "retryable_failure"
    assert update.error_code == "tool_timeout"


async def test_terminal_exception_updates_workflow_error() -> None:
    observability = _RecordingObservabilityClient()
    original = TerminalAgentRunExecutionError(
        error_code="controlled_workflow_state_invalid",
        error_summary="Invalid state.",
    )
    with pytest.raises(TerminalAgentRunExecutionError) as captured:
        await ControlledSupportWorkflowExecutor(
            graph=_failing_graph(error=original),
            observability_client=cast(Any, observability),
        ).execute(_context())

    assert captured.value is original
    update = _workflow_scopes(observability)[0].updates[0]
    assert update.status is ObservationStatus.ERROR
    assert update.metadata["workflow_outcome"] == "terminal_failure"
    assert update.error_code == "controlled_workflow_state_invalid"


async def test_unexpected_exception_updates_workflow_with_safe_code() -> None:
    observability = _RecordingObservabilityClient()
    original = RuntimeError("boom")
    with pytest.raises(RuntimeError) as captured:
        await ControlledSupportWorkflowExecutor(
            graph=_failing_graph(error=original),
            observability_client=cast(Any, observability),
        ).execute(_context())

    assert captured.value is original
    update = _workflow_scopes(observability)[0].updates[0]
    assert update.status is ObservationStatus.ERROR
    assert update.metadata["workflow_outcome"] == "unexpected_failure"
    assert update.error_code == "controlled_support_unexpected_failure"


async def test_workflow_observation_closes_after_graph_invocation() -> None:
    observability = _RecordingObservabilityClient()
    events: list[str] = []

    class OrderedGraph(RecordingCompiledGraph):
        async def ainvoke(
            self,
            input: ControlledSupportGraphState | None,
            config: Mapping[str, object],
            *,
            context: AgentRunExecutionContext,
        ) -> Mapping[str, object] | None:
            events.append("ainvoke")
            return await super().ainvoke(input, config, context=context)

    original_start = observability.start_observation

    def tracking_start(
        attributes: ObservationAttributes,
    ) -> AbstractContextManager[_RecordingObservationScope]:
        manager = original_start(attributes)
        original_enter = manager.__enter__
        original_exit = manager.__exit__

        def enter() -> _RecordingObservationScope:
            events.append(f"enter:{attributes.name}")
            return original_enter()

        def exit_fn(
            exc_type: type[BaseException] | None,
            exc: BaseException | None,
            traceback: TracebackType | None,
        ) -> Literal[False]:
            events.append(f"exit:{attributes.name}")
            original_exit(exc_type, exc, traceback)
            return False

        manager.__enter__ = enter  # type: ignore[method-assign]
        manager.__exit__ = exit_fn  # type: ignore[method-assign]
        return manager

    observability.start_observation = tracking_start  # type: ignore[method-assign]
    await ControlledSupportWorkflowExecutor(
        graph=OrderedGraph(
            checkpoint_values={},
            result=_completed_state(),
        ),
        observability_client=cast(Any, observability),
    ).execute(_context())

    assert events == [
        "enter:workflow.controlled-support-v1",
        "ainvoke",
        "exit:workflow.controlled-support-v1",
    ]


@pytest.mark.parametrize(
    ("start_error", "update_error", "exit_error"),
    [
        (RuntimeError("start failed"), None, None),
        (None, RuntimeError("update failed"), None),
        (None, None, RuntimeError("exit failed")),
    ],
)
async def test_workflow_observability_failures_preserve_graph_result(
    start_error: Exception | None,
    update_error: Exception | None,
    exit_error: Exception | None,
) -> None:
    observability = _RecordingObservabilityClient(
        start_error=start_error,
        update_error=update_error,
        exit_error=exit_error,
    )
    result = await ControlledSupportWorkflowExecutor(
        graph=RecordingCompiledGraph(
            checkpoint_values={},
            result=_completed_state(),
        ),
        observability_client=cast(Any, observability),
    ).execute(_context())
    assert result == CompletedExecution()


async def test_workflow_observability_failures_preserve_exception_identity() -> None:
    observability = _RecordingObservabilityClient(
        update_error=RuntimeError("update failed"),
        exit_error=RuntimeError("exit failed"),
    )
    original = TerminalAgentRunExecutionError(
        error_code="controlled_workflow_state_invalid",
        error_summary="Invalid state.",
    )
    with pytest.raises(TerminalAgentRunExecutionError) as captured:
        await ControlledSupportWorkflowExecutor(
            graph=_failing_graph(error=original),
            observability_client=cast(Any, observability),
        ).execute(_context())
    assert captured.value is original


async def test_workflow_metadata_contains_only_safe_fields() -> None:
    observability = _RecordingObservabilityClient()
    context = _context()
    await ControlledSupportWorkflowExecutor(
        graph=RecordingCompiledGraph(
            checkpoint_values={},
            result=_completed_state(),
        ),
        observability_client=cast(Any, observability),
    ).execute(context)

    workflow = _workflow_scopes(observability)[0]
    attributes = workflow.attributes
    assert attributes.input_data is None
    assert attributes.input_paths == frozenset()
    assert attributes.output_paths == frozenset()
    _assert_safe_metadata(
        attributes.metadata,
        allowed=_WORKFLOW_ALLOWED_METADATA_KEYS,
    )
    identity = derive_controlled_support_graph_identity(_AGENT_RUN_ID)
    assert attributes.metadata == {
        "agent_run_id": str(_AGENT_RUN_ID),
        "agent_run_attempt_id": str(_ATTEMPT_ID),
        "execution_request_id": str(context.attempt.execution_request_id),
        "workspace_id": str(_WORKSPACE_ID),
        "ticket_id": str(_TICKET_ID),
        "workflow_name": CONTROLLED_SUPPORT_WORKFLOW_NAME,
        "workflow_version": CONTROLLED_SUPPORT_WORKFLOW_VERSION,
        "trigger_key": context.agent_run.trigger_key,
        "correlation_id": str(context.agent_run.correlation_id),
        "graph_thread_id": identity.thread_id,
        "invocation_mode": "initial",
    }
    for update in workflow.updates:
        _assert_safe_metadata(
            update.metadata,
            allowed=_WORKFLOW_ALLOWED_METADATA_KEYS,
        )
        assert update.output_data is None


async def test_each_controlled_node_creates_one_span_observation() -> None:
    observability = _RecordingObservabilityClient()
    nodes = _stub_nodes(observability_client=observability)
    state = _initial_graph_state()
    classified = dict(state)
    classified["classification_id"] = str(_CLASSIFICATION_ID)
    runtime = _runtime()

    with patch.object(
        ControlledSupportWorkflowNodes,
        "_ensure_classification",
        new=AsyncMock(return_value=classified),
    ):
        await nodes.ensure_classification(state, runtime)

    with patch.object(
        ControlledSupportWorkflowNodes,
        "_decide_and_execute",
        new=AsyncMock(return_value=classified),
    ):
        await nodes.decide_and_execute(state, runtime)

    with patch.object(
        ControlledSupportWorkflowNodes,
        "_draft_recommendation",
        new=AsyncMock(
            return_value={
                **classified,
                "recommendation_id": str(_RECOMMENDATION_ID),
            }
        ),
    ):
        await nodes.draft_recommendation(state, runtime)

    with pytest.raises(TerminalAgentRunExecutionError):
        await nodes.fail_workflow(
            _error_graph_state(error_code="controlled_workflow_state_invalid"),
        )

    names = [scope.attributes.name for scope in _node_scopes(observability)]
    assert names == [
        "graph-node.ensure_classification",
        "graph-node.decide_and_execute",
        "graph-node.draft_recommendation",
        "graph-node.fail_workflow",
    ]
    assert all(
        scope.attributes.observation_type is ObservationType.SPAN
        for scope in _node_scopes(observability)
    )
    assert all(scope.closed for scope in _node_scopes(observability))


async def test_routing_helper_creates_no_node_observation() -> None:
    observability = _RecordingObservabilityClient()
    nodes = _stub_nodes(observability_client=observability)

    route = nodes.route(_initial_graph_state())

    assert route == "ensure_classification"
    assert _node_scopes(observability) == []


async def test_successful_node_status_is_ok() -> None:
    observability = _RecordingObservabilityClient()
    nodes = _stub_nodes(observability_client=observability)
    result_state = {
        **_initial_graph_state(),
        "classification_id": str(_CLASSIFICATION_ID),
    }

    with patch.object(
        ControlledSupportWorkflowNodes,
        "_ensure_classification",
        new=AsyncMock(return_value=result_state),
    ):
        await nodes.ensure_classification(_initial_graph_state(), _runtime())

    scope = _node_scopes(observability)[0]
    assert scope.updates[0].status is ObservationStatus.OK
    assert scope.updates[0].metadata["classification_present"] is True
    _assert_safe_metadata(
        scope.attributes.metadata,
        allowed=_NODE_ALLOWED_METADATA_KEYS,
    )
    _assert_safe_metadata(
        scope.updates[0].metadata,
        allowed=_NODE_ALLOWED_METADATA_KEYS,
    )


async def test_typed_node_failure_status_is_error() -> None:
    observability = _RecordingObservabilityClient()
    nodes = _stub_nodes(observability_client=observability)
    original = RetryableAgentRunExecutionError(
        error_code="tool_timeout",
        error_summary="Tool timed out.",
    )

    with (
        patch.object(
            ControlledSupportWorkflowNodes,
            "_decide_and_execute",
            new=AsyncMock(side_effect=original),
        ),
        pytest.raises(RetryableAgentRunExecutionError) as captured,
    ):
        await nodes.decide_and_execute(_initial_graph_state(), _runtime())

    assert captured.value is original
    update = _node_scopes(observability)[0].updates[0]
    assert update.status is ObservationStatus.ERROR
    assert update.error_code == "tool_timeout"


async def test_fail_workflow_preserves_original_typed_exception() -> None:
    observability = _RecordingObservabilityClient()
    nodes = _stub_nodes(observability_client=observability)

    with pytest.raises(RetryableAgentRunExecutionError) as captured:
        await nodes.fail_workflow(_error_graph_state(error_code="tool_timeout"))

    assert captured.value.error_code == "tool_timeout"
    scope = _node_scopes(observability)[0]
    assert scope.attributes.name == "graph-node.fail_workflow"
    assert scope.updates[0].status is ObservationStatus.ERROR
    assert scope.updates[0].error_code == "tool_timeout"
    assert scope.closed is True


async def test_node_observation_exports_no_content_fields() -> None:
    observability = _RecordingObservabilityClient()
    nodes = _stub_nodes(observability_client=observability)
    result_state = {
        **_initial_graph_state(),
        "recommendation_id": str(_RECOMMENDATION_ID),
        "retrieved_chunk_ids": [str(_CLASSIFICATION_ID)],
        "classification_summary": "should not export",
    }

    with patch.object(
        ControlledSupportWorkflowNodes,
        "_draft_recommendation",
        new=AsyncMock(return_value=result_state),
    ):
        await nodes.draft_recommendation(_initial_graph_state(), _runtime())

    scope = _node_scopes(observability)[0]
    assert scope.attributes.input_data is None
    assert scope.attributes.input_paths == frozenset()
    assert scope.attributes.output_paths == frozenset()
    _assert_safe_metadata(
        scope.attributes.metadata,
        allowed=_NODE_ALLOWED_METADATA_KEYS,
    )
    for update in scope.updates:
        _assert_safe_metadata(
            update.metadata,
            allowed=_NODE_ALLOWED_METADATA_KEYS,
        )
        assert update.output_data is None
        assert "classification_summary" not in update.metadata
        assert "graph_state" not in update.metadata


@pytest.mark.parametrize(
    ("start_error", "update_error", "exit_error"),
    [
        (RuntimeError("start failed"), None, None),
        (None, RuntimeError("update failed"), None),
        (None, None, RuntimeError("exit failed")),
    ],
)
async def test_node_observability_failures_fail_open(
    start_error: Exception | None,
    update_error: Exception | None,
    exit_error: Exception | None,
) -> None:
    observability = _RecordingObservabilityClient(
        start_error=start_error,
        update_error=update_error,
        exit_error=exit_error,
    )
    nodes = _stub_nodes(observability_client=observability)
    result_state = {
        **_initial_graph_state(),
        "classification_id": str(_CLASSIFICATION_ID),
    }

    with patch.object(
        ControlledSupportWorkflowNodes,
        "_ensure_classification",
        new=AsyncMock(return_value=result_state),
    ):
        result = await nodes.ensure_classification(
            _initial_graph_state(),
            _runtime(),
        )

    assert result == result_state


async def test_node_contextvars_restore_after_success_and_failure() -> None:
    observability = _RecordingObservabilityClient(install_context=True)
    nodes = _stub_nodes(observability_client=observability)
    assert current_observation_context() is None

    with patch.object(
        ControlledSupportWorkflowNodes,
        "_ensure_classification",
        new=AsyncMock(
            return_value={
                **_initial_graph_state(),
                "classification_id": str(_CLASSIFICATION_ID),
            }
        ),
    ):
        await nodes.ensure_classification(_initial_graph_state(), _runtime())

    assert current_observation_context() is None

    with (
        patch.object(
            ControlledSupportWorkflowNodes,
            "_decide_and_execute",
            new=AsyncMock(
                side_effect=TerminalAgentRunExecutionError(
                    error_code="controlled_workflow_state_invalid",
                    error_summary="Invalid.",
                )
            ),
        ),
        pytest.raises(TerminalAgentRunExecutionError),
    ):
        await nodes.decide_and_execute(_initial_graph_state(), _runtime())

    assert current_observation_context() is None


async def test_observation_hierarchy_and_no_duplicate_provider_instrumentation() -> None:
    observability = _RecordingObservabilityClient(install_context=True)
    nodes = _stub_nodes(observability_client=observability)
    provider_parents: list[str | None] = []

    async def execute_with_provider_nesting(
        *args: object,
        **kwargs: object,
    ) -> ControlledSupportGraphState:
        del args, kwargs
        active = current_observation_context()
        assert active is not None
        assert active.name == "graph-node.ensure_classification"

        with observability.start_observation(
            ObservationAttributes(
                name="llm.generate",
                observation_type=ObservationType.GENERATION,
            )
        ):
            nested = current_observation_context()
            provider_parents.append(nested.name if nested is not None else None)

        with observability.start_observation(
            ObservationAttributes(
                name="llm.tool_decision",
                observation_type=ObservationType.GENERATION,
            )
        ):
            pass

        with (
            observability.start_observation(
                ObservationAttributes(
                    name="knowledge.search",
                    observation_type=ObservationType.RETRIEVER,
                )
            ),
            observability.start_observation(
                ObservationAttributes(
                    name="embedding.request",
                    observation_type=ObservationType.EMBEDDING,
                )
            ),
        ):
            pass

        return {
            **_initial_graph_state(),
            "classification_id": str(_CLASSIFICATION_ID),
        }

    class NestedGraph(RecordingCompiledGraph):
        async def ainvoke(
            self,
            input: ControlledSupportGraphState | None,
            config: Mapping[str, object],
            *,
            context: AgentRunExecutionContext,
        ) -> Mapping[str, object] | None:
            self.inputs.append(input)
            self.invoke_configs.append(config)
            self.contexts.append(context)
            active = current_observation_context()
            assert active is not None
            assert active.name == "workflow.controlled-support-v1"

            with patch.object(
                ControlledSupportWorkflowNodes,
                "_ensure_classification",
                new=execute_with_provider_nesting,
            ):
                await nodes.ensure_classification(
                    _initial_graph_state(),
                    cast(Any, SimpleNamespace(context=context)),
                )

            return _completed_state()

    with observation_context_scope(
        ActiveObservationContext(name="worker-attempt", observation_id="attempt-1")
    ):
        await ControlledSupportWorkflowExecutor(
            graph=NestedGraph(
                checkpoint_values={},
                result=_completed_state(),
            ),
            observability_client=cast(Any, observability),
        ).execute(_context())

    names = [attributes.name for attributes in observability.started_attributes]
    assert names.count("workflow.controlled-support-v1") == 1
    assert names.count("graph-node.ensure_classification") == 1
    assert names.count("llm.generate") == 1
    assert names.count("llm.tool_decision") == 1
    assert names.count("knowledge.search") == 1
    assert names.count("embedding.request") == 1

    workflow_index = names.index("workflow.controlled-support-v1")
    node_index = names.index("graph-node.ensure_classification")
    generate_index = names.index("llm.generate")
    search_index = names.index("knowledge.search")
    embedding_index = names.index("embedding.request")

    assert observability.parent_at_start[workflow_index] == "worker-attempt"
    assert observability.parent_at_start[node_index] == "workflow.controlled-support-v1"
    assert observability.parent_at_start[generate_index] == "graph-node.ensure_classification"
    assert observability.parent_at_start[search_index] == "graph-node.ensure_classification"
    assert observability.parent_at_start[embedding_index] == "knowledge.search"
    assert provider_parents == ["llm.generate"]
    assert current_observation_context() is None
