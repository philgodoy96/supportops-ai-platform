"""LangGraph composition for the controlled support workflow."""

from collections.abc import Hashable, Mapping
from dataclasses import dataclass
from typing import Any, Never, Protocol, cast

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.errors import GraphRecursionError
from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime

from supportops.agent_graph.application.decision_execution import (
    ControlledSupportDecisionExecutor,
)
from supportops.agent_graph.application.recommendation_execution import (
    ControlledSupportRecommendationExecutor,
)
from supportops.agent_graph.application.tool_execution import (
    ControlledToolDecisionExecutor,
)
from supportops.agent_graph.application.tool_observations import (
    ControlledToolObservationAssembler,
)
from supportops.agent_graph.application.transitions import (
    advance_graph_step,
    attach_classification,
)
from supportops.agent_graph.domain.identity import (
    derive_controlled_support_graph_identity,
)
from supportops.agent_graph.domain.routing import (
    ControlledSupportGraphRoute,
    select_controlled_support_route,
)
from supportops.agent_graph.domain.state import (
    CONTROLLED_SUPPORT_WORKFLOW_NAME,
    CONTROLLED_SUPPORT_WORKFLOW_VERSION,
    ControlledSupportGraphState,
    ControlledSupportGraphStateSnapshot,
    GraphStateIncompatibleError,
    create_initial_controlled_support_state,
    validate_controlled_support_state,
)
from supportops.agent_graph.infrastructure.checkpoints import (
    GraphCheckpointError,
)
from supportops.agent_tools.application.execution import (
    ToolExecutionContext,
)
from supportops.ai.gateway.tool_decisions import (
    LLMExecutableToolCallDecision,
    LLMTerminalControlDecision,
)
from supportops.core.transactions import TransactionManager
from supportops.modules.agent_runs.application.execution import (
    AgentRunExecutionContext,
    CompletedExecution,
    RetryableAgentRunExecutionError,
    TerminalAgentRunExecutionError,
)
from supportops.modules.agent_runs.domain.models import (
    INITIAL_TICKET_PROCESSING_TRIGGER_KEY,
)
from supportops.modules.ticket_classifications.application.executor import (
    TicketClassificationExecutor,
)
from supportops.modules.ticket_classifications.domain.models import (
    TicketClassification,
)
from supportops.modules.ticket_classifications.domain.repositories import (
    TicketClassificationRepository,
)

CONTROLLED_SUPPORT_LANGGRAPH_RECURSION_LIMIT = 20

_ENSURE_CLASSIFICATION_NODE = "ensure_classification"
_DECIDE_AND_EXECUTE_NODE = "decide_and_execute"
_DRAFT_RECOMMENDATION_NODE = "draft_recommendation"
_FAIL_WORKFLOW_NODE = "fail_workflow"

_RETRYABLE_GRAPH_ERROR_CODES = frozenset(
    {
        "tool_timeout",
        "tool_dependency_unavailable",
        "tool_execution_failed",
        "graph_checkpoint_unavailable",
    }
)


class GraphCheckpointSnapshot(Protocol):
    """Minimum checkpoint state required by the workflow executor."""

    @property
    def values(self) -> Mapping[str, object]:
        """Return the latest checkpointed graph values."""

        ...


class ControlledSupportCompiledGraph(Protocol):
    """Compiled graph operations used by the AgentRun executor."""

    async def aget_state(
        self,
        config: Mapping[str, object],
    ) -> GraphCheckpointSnapshot:
        """Return the latest state for one checkpoint thread."""

        ...

    async def ainvoke(
        self,
        input: ControlledSupportGraphState | None,
        config: Mapping[str, object],
        *,
        context: AgentRunExecutionContext,
    ) -> Mapping[str, object] | None:
        """Execute or resume one controlled workflow thread."""

        ...


@dataclass(frozen=True, slots=True)
class ControlledSupportWorkflowNodes:
    """Application-owned nodes composed by the controlled graph."""

    transaction_manager: TransactionManager
    classification_repository: TicketClassificationRepository
    classification_executor: TicketClassificationExecutor
    observation_assembler: ControlledToolObservationAssembler
    decision_executor: ControlledSupportDecisionExecutor
    tool_executor: ControlledToolDecisionExecutor
    recommendation_executor: ControlledSupportRecommendationExecutor

    async def ensure_classification(
        self,
        state: ControlledSupportGraphState,
        runtime: Runtime[AgentRunExecutionContext],
    ) -> ControlledSupportGraphState:
        """Load or durably produce the ticket classification."""

        snapshot = advance_graph_step(validate_controlled_support_state(state))
        classification = await self._load_classification(snapshot)

        if classification is None:
            await self.classification_executor.execute(runtime.context)
            classification = await self._load_classification(snapshot)

        if classification is None:
            raise RuntimeError(
                "Classification execution completed without a durable classification."
            )

        return attach_classification(
            snapshot,
            classification,
        ).to_graph_state()

    async def decide_and_execute(
        self,
        state: ControlledSupportGraphState,
        runtime: Runtime[AgentRunExecutionContext],
    ) -> ControlledSupportGraphState:
        """Recover a tool outcome or execute one complete decision turn."""

        snapshot = advance_graph_step(validate_controlled_support_state(state))
        recovered = await self.tool_executor.recover_next_persisted_outcome(
            state=snapshot,
            context=runtime.context,
        )

        if recovered is not None:
            return recovered.state.to_graph_state()

        observation_bundle = await self.observation_assembler.assemble(
            state=snapshot,
            context=_tool_execution_context(runtime.context),
        )
        decision_outcome = await self.decision_executor.execute(
            state=snapshot,
            context=runtime.context,
            tool_observations=(observation_bundle.to_prompt_observations()),
        )
        decision = decision_outcome.decision

        if isinstance(
            decision,
            LLMTerminalControlDecision,
        ):
            return decision_outcome.state.to_graph_state()

        if not isinstance(
            decision,
            LLMExecutableToolCallDecision,
        ):
            raise RuntimeError(
                "The controlled decision executor returned an unsupported decision type."
            )

        tool_outcome = await self.tool_executor.execute(
            state=decision_outcome.state,
            context=runtime.context,
            decision=decision,
        )

        return tool_outcome.state.to_graph_state()

    async def draft_recommendation(
        self,
        state: ControlledSupportGraphState,
        runtime: Runtime[AgentRunExecutionContext],
    ) -> ControlledSupportGraphState:
        """Draft or recover the final persisted recommendation."""

        snapshot = advance_graph_step(validate_controlled_support_state(state))
        outcome = await self.recommendation_executor.execute(
            state=snapshot,
            context=runtime.context,
        )

        return outcome.state.to_graph_state()

    async def fail_workflow(
        self,
        state: ControlledSupportGraphState,
    ) -> Never:
        """Translate a fail-closed graph route into an AgentRun error."""

        snapshot = validate_controlled_support_state(state)
        route_decision = select_controlled_support_route(snapshot)
        error_code = (
            snapshot.current_error_code
            or route_decision.error_code
            or "controlled_workflow_state_invalid"
        )

        if error_code in _RETRYABLE_GRAPH_ERROR_CODES:
            raise RetryableAgentRunExecutionError(
                error_code=error_code,
                error_summary=(
                    "The controlled support workflow stopped "
                    "after a retryable dependency or tool failure."
                ),
            )

        raise TerminalAgentRunExecutionError(
            error_code=error_code,
            error_summary=(
                "The controlled support workflow stopped after a non-retryable graph failure."
            ),
        )

    def route(
        self,
        state: ControlledSupportGraphState,
    ) -> str:
        """Return the deterministic route selected from graph state."""

        snapshot = validate_controlled_support_state(state)

        return select_controlled_support_route(snapshot).route.value

    async def _load_classification(
        self,
        state: ControlledSupportGraphStateSnapshot,
    ) -> TicketClassification | None:
        async with self.transaction_manager.transaction():
            return await self.classification_repository.get_by_agent_run_id(
                workspace_id=state.workspace_id,
                agent_run_id=state.agent_run_id,
            )


def compile_controlled_support_graph(
    *,
    nodes: ControlledSupportWorkflowNodes,
    checkpointer: BaseCheckpointSaver[Any],
) -> ControlledSupportCompiledGraph:
    """Compile the controlled graph with durable checkpointing."""

    builder = StateGraph(
        ControlledSupportGraphState,
        context_schema=AgentRunExecutionContext,
    )
    builder.add_node(
        _ENSURE_CLASSIFICATION_NODE,
        nodes.ensure_classification,
    )
    builder.add_node(
        _DECIDE_AND_EXECUTE_NODE,
        nodes.decide_and_execute,
    )
    builder.add_node(
        _DRAFT_RECOMMENDATION_NODE,
        nodes.draft_recommendation,
    )
    builder.add_node(
        _FAIL_WORKFLOW_NODE,
        nodes.fail_workflow,
    )

    route_targets: dict[Hashable, str] = {
        ControlledSupportGraphRoute.ENSURE_CLASSIFICATION.value: (_ENSURE_CLASSIFICATION_NODE),
        ControlledSupportGraphRoute.DECIDE_NEXT_ACTION.value: (_DECIDE_AND_EXECUTE_NODE),
        ControlledSupportGraphRoute.DRAFT_RECOMMENDATION.value: (_DRAFT_RECOMMENDATION_NODE),
        ControlledSupportGraphRoute.PERSIST_RECOMMENDATION.value: (_DRAFT_RECOMMENDATION_NODE),
        ControlledSupportGraphRoute.COMPLETE_WORKFLOW.value: END,
        ControlledSupportGraphRoute.FAIL_WORKFLOW.value: _FAIL_WORKFLOW_NODE,
    }

    builder.add_conditional_edges(
        START,
        nodes.route,
        route_targets,
    )

    for source_node in (
        _ENSURE_CLASSIFICATION_NODE,
        _DECIDE_AND_EXECUTE_NODE,
        _DRAFT_RECOMMENDATION_NODE,
    ):
        builder.add_conditional_edges(
            source_node,
            nodes.route,
            route_targets,
        )

    builder.add_edge(
        _FAIL_WORKFLOW_NODE,
        END,
    )

    return cast(
        ControlledSupportCompiledGraph,
        builder.compile(
            checkpointer=checkpointer,
        ),
    )


class ControlledSupportWorkflowExecutor:
    """Execute one checkpointed graph inside an outer AgentRun."""

    def __init__(
        self,
        *,
        graph: ControlledSupportCompiledGraph,
    ) -> None:
        self._graph = graph

    async def execute(
        self,
        context: AgentRunExecutionContext,
    ) -> CompletedExecution:
        """Execute or resume the controlled graph to recommendation persistence."""

        _validate_supported_workflow(context)
        identity = derive_controlled_support_graph_identity(context.agent_run.id)
        config: dict[str, object] = {
            "configurable": {
                "thread_id": identity.thread_id,
                "checkpoint_ns": (identity.checkpoint_namespace),
            },
            "recursion_limit": (CONTROLLED_SUPPORT_LANGGRAPH_RECURSION_LIMIT),
        }

        try:
            checkpoint = await self._graph.aget_state(config)
            graph_input: ControlledSupportGraphState | None

            if checkpoint.values:
                recovered_state = validate_controlled_support_state(checkpoint.values)
                _validate_state_ownership(
                    state=recovered_state,
                    context=context,
                )
                graph_input = None
            else:
                graph_input = create_initial_controlled_support_state(
                    workspace_id=(context.agent_run.workspace_id),
                    ticket_id=context.ticket.id,
                    agent_run_id=context.agent_run.id,
                )

            result = await self._graph.ainvoke(
                graph_input,
                config,
                context=context,
            )

            if result is None:
                result = (await self._graph.aget_state(config)).values
        except GraphRecursionError as exc:
            raise TerminalAgentRunExecutionError(
                error_code=("graph_recursion_limit_exceeded"),
                error_summary=(
                    "The controlled support graph exceeded its configured recursion limit."
                ),
            ) from exc
        except GraphStateIncompatibleError as exc:
            raise TerminalAgentRunExecutionError(
                error_code=exc.error_code,
                error_summary=("The checkpointed controlled support state is incompatible."),
            ) from exc
        except GraphCheckpointError as exc:
            _raise_checkpoint_error(exc)

        final_state = validate_controlled_support_state(result)
        _validate_state_ownership(
            state=final_state,
            context=context,
        )

        if final_state.current_error_code is not None:
            raise TerminalAgentRunExecutionError(
                error_code=final_state.current_error_code,
                error_summary=("The controlled support graph completed with an unresolved error."),
            )

        if final_state.recommendation_id is None:
            raise TerminalAgentRunExecutionError(
                error_code="controlled_workflow_incomplete",
                error_summary=(
                    "The controlled support graph completed without a persisted recommendation."
                ),
            )

        return CompletedExecution()


def _tool_execution_context(
    context: AgentRunExecutionContext,
) -> ToolExecutionContext:
    return ToolExecutionContext(
        workspace_id=context.agent_run.workspace_id,
        ticket_id=context.ticket.id,
        agent_run_id=context.agent_run.id,
        agent_run_attempt_id=context.attempt.id,
    )


def _validate_supported_workflow(
    context: AgentRunExecutionContext,
) -> None:
    run = context.agent_run

    if run.workflow_name != CONTROLLED_SUPPORT_WORKFLOW_NAME:
        raise TerminalAgentRunExecutionError(
            error_code="unsupported_workflow",
            error_summary=(
                "The AgentRun workflow is not supported by the controlled support executor."
            ),
        )

    if run.workflow_version != (CONTROLLED_SUPPORT_WORKFLOW_VERSION):
        raise TerminalAgentRunExecutionError(
            error_code="unsupported_workflow_version",
            error_summary=(
                "The AgentRun workflow version is not supported by the controlled support executor."
            ),
        )

    if run.trigger_key != (INITIAL_TICKET_PROCESSING_TRIGGER_KEY):
        raise TerminalAgentRunExecutionError(
            error_code="unsupported_trigger",
            error_summary=(
                "The AgentRun trigger is not supported by the controlled support executor."
            ),
        )


def _validate_state_ownership(
    *,
    state: ControlledSupportGraphStateSnapshot,
    context: AgentRunExecutionContext,
) -> None:
    ownership_values = (
        (
            state.workspace_id,
            context.agent_run.workspace_id,
        ),
        (
            state.ticket_id,
            context.ticket.id,
        ),
        (
            state.agent_run_id,
            context.agent_run.id,
        ),
    )

    if any(actual != expected for actual, expected in ownership_values):
        raise TerminalAgentRunExecutionError(
            error_code="graph_state_ownership_mismatch",
            error_summary=("The checkpointed graph state does not belong to the claimed AgentRun."),
        )


def _raise_checkpoint_error(
    error: GraphCheckpointError,
) -> Never:
    if error.retryable:
        raise RetryableAgentRunExecutionError(
            error_code=error.error_code,
            error_summary=(
                "The controlled support checkpoint infrastructure is temporarily unavailable."
            ),
        ) from error

    raise TerminalAgentRunExecutionError(
        error_code=error.error_code,
        error_summary=("The controlled support checkpoint runtime cannot continue."),
    ) from error
