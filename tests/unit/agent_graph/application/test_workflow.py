"""Unit tests for the controlled support graph executor."""

from collections.abc import Mapping
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from uuid import UUID

import pytest
from langgraph.errors import GraphRecursionError

from supportops.agent_graph.application.workflow import (
    CONTROLLED_SUPPORT_LANGGRAPH_RECURSION_LIMIT,
    ControlledSupportWorkflowExecutor,
)
from supportops.agent_graph.domain.identity import (
    derive_controlled_support_graph_identity,
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
    TerminalAgentRunExecutionError,
)
from supportops.modules.agent_runs.domain.models import (
    AgentRun,
    AgentRunAttempt,
    AgentRunStatus,
)
from supportops.modules.tickets.domain.models import Ticket

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
