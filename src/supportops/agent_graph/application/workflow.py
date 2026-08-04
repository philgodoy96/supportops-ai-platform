"""LangGraph composition for the controlled support workflow."""

from collections.abc import Awaitable, Callable, Hashable, Mapping
from contextlib import AbstractContextManager, suppress
from dataclasses import dataclass, field
from time import monotonic
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
    ControlledSupportGraphIdentity,
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
from supportops.observability.contracts import (
    ObservabilityClient,
    ObservationScope,
)
from supportops.observability.models import (
    FieldPaths,
    JsonValue,
    ObservationAttributes,
    ObservationStatus,
    ObservationType,
    ObservationUpdate,
)
from supportops.observability.noop import NoOpObservabilityClient

CONTROLLED_SUPPORT_LANGGRAPH_RECURSION_LIMIT = 20

_ENSURE_CLASSIFICATION_NODE = "ensure_classification"
_DECIDE_AND_EXECUTE_NODE = "decide_and_execute"
_DRAFT_RECOMMENDATION_NODE = "draft_recommendation"
_FAIL_WORKFLOW_NODE = "fail_workflow"

_WORKFLOW_OBSERVATION_NAME = "workflow.controlled-support-v1"
_ENSURE_CLASSIFICATION_OBSERVATION_NAME = "graph-node.ensure_classification"
_DECIDE_AND_EXECUTE_OBSERVATION_NAME = "graph-node.decide_and_execute"
_DRAFT_RECOMMENDATION_OBSERVATION_NAME = "graph-node.draft_recommendation"
_FAIL_WORKFLOW_OBSERVATION_NAME = "graph-node.fail_workflow"

_UNEXPECTED_WORKFLOW_ERROR_CODE = "controlled_support_unexpected_failure"
_UNEXPECTED_NODE_ERROR_CODE = "controlled_support_node_unexpected_failure"

_INVOCATION_MODE_INITIAL = "initial"
_INVOCATION_MODE_CONTINUE = "continue"

_RETRYABLE_GRAPH_ERROR_CODES = frozenset(
    {
        "tool_timeout",
        "tool_dependency_unavailable",
        "tool_execution_failed",
        "graph_checkpoint_unavailable",
    }
)

_WORKFLOW_METADATA_PATHS: FieldPaths = frozenset(
    {
        ("agent_run_id",),
        ("agent_run_attempt_id",),
        ("execution_request_id",),
        ("workspace_id",),
        ("ticket_id",),
        ("workflow_name",),
        ("workflow_version",),
        ("trigger_key",),
        ("correlation_id",),
        ("graph_thread_id",),
        ("invocation_mode",),
        ("workflow_outcome",),
        ("error_code",),
        ("latency_ms",),
    }
)

_NODE_METADATA_PATHS: FieldPaths = frozenset(
    {
        ("node_name",),
        ("agent_run_id",),
        ("agent_run_attempt_id",),
        ("execution_request_id",),
        ("workspace_id",),
        ("ticket_id",),
        ("workflow_name",),
        ("workflow_version",),
        ("correlation_id",),
        ("classification_present",),
        ("tool_decision_mode",),
        ("tool_execution_count",),
        ("evidence_count",),
        ("recommendation_created",),
        ("error_code",),
        ("latency_ms",),
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
    observability_client: ObservabilityClient = field(
        default_factory=NoOpObservabilityClient,
    )

    async def ensure_classification(
        self,
        state: ControlledSupportGraphState,
        runtime: Runtime[AgentRunExecutionContext],
    ) -> ControlledSupportGraphState:
        """Load or durably produce the ticket classification."""

        return await self._observe_node(
            node_name=_ENSURE_CLASSIFICATION_NODE,
            observation_name=_ENSURE_CLASSIFICATION_OBSERVATION_NAME,
            state=state,
            context=runtime.context,
            execute=lambda: self._ensure_classification(
                state=state,
                runtime=runtime,
            ),
            success_metadata=lambda result: {
                "classification_present": (result.get("classification_id") is not None),
            },
        )

    async def decide_and_execute(
        self,
        state: ControlledSupportGraphState,
        runtime: Runtime[AgentRunExecutionContext],
    ) -> ControlledSupportGraphState:
        """Recover a tool outcome or execute one complete decision turn."""

        decision_mode_holder: dict[str, str | int] = {
            "tool_decision_mode": "unknown",
            "tool_execution_count": 0,
        }

        async def execute() -> ControlledSupportGraphState:
            return await self._decide_and_execute(
                state=state,
                runtime=runtime,
                decision_mode_holder=decision_mode_holder,
            )

        return await self._observe_node(
            node_name=_DECIDE_AND_EXECUTE_NODE,
            observation_name=_DECIDE_AND_EXECUTE_OBSERVATION_NAME,
            state=state,
            context=runtime.context,
            execute=execute,
            success_metadata=lambda result: {
                "tool_decision_mode": (decision_mode_holder["tool_decision_mode"]),
                "tool_execution_count": (decision_mode_holder["tool_execution_count"]),
                "evidence_count": len(
                    cast(list[object], result.get("retrieved_chunk_ids") or ()),
                ),
            },
        )

    async def draft_recommendation(
        self,
        state: ControlledSupportGraphState,
        runtime: Runtime[AgentRunExecutionContext],
    ) -> ControlledSupportGraphState:
        """Draft or recover the final persisted recommendation."""

        return await self._observe_node(
            node_name=_DRAFT_RECOMMENDATION_NODE,
            observation_name=_DRAFT_RECOMMENDATION_OBSERVATION_NAME,
            state=state,
            context=runtime.context,
            execute=lambda: self._draft_recommendation(
                state=state,
                runtime=runtime,
            ),
            success_metadata=lambda result: {
                "recommendation_created": (result.get("recommendation_id") is not None),
            },
        )

    async def fail_workflow(
        self,
        state: ControlledSupportGraphState,
    ) -> Never:
        """Translate a fail-closed graph route into an AgentRun error."""

        observation = _FailOpenObservation(
            client=self.observability_client,
            attributes=_build_node_attributes(
                node_name=_FAIL_WORKFLOW_NODE,
                observation_name=_FAIL_WORKFLOW_OBSERVATION_NAME,
                state=state,
                context=None,
            ),
        )
        started_at = monotonic()
        observation.start()

        try:
            self._fail_workflow(state)
        except (
            RetryableAgentRunExecutionError,
            TerminalAgentRunExecutionError,
        ) as error:
            observation.complete(
                ObservationUpdate(
                    status=ObservationStatus.ERROR,
                    metadata={
                        "error_code": error.error_code,
                        "latency_ms": _elapsed_ms(started_at),
                    },
                    error_code=error.error_code,
                )
            )
            raise
        except Exception:
            observation.complete(
                ObservationUpdate(
                    status=ObservationStatus.ERROR,
                    metadata={"latency_ms": _elapsed_ms(started_at)},
                    error_code=_UNEXPECTED_NODE_ERROR_CODE,
                )
            )
            raise
        finally:
            observation.close()

        raise RuntimeError("fail_workflow must raise a typed workflow exception.")

    def route(
        self,
        state: ControlledSupportGraphState,
    ) -> str:
        """Return the deterministic route selected from graph state."""

        snapshot = validate_controlled_support_state(state)

        return select_controlled_support_route(snapshot).route.value

    async def _ensure_classification(
        self,
        *,
        state: ControlledSupportGraphState,
        runtime: Runtime[AgentRunExecutionContext],
    ) -> ControlledSupportGraphState:
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

    async def _decide_and_execute(
        self,
        *,
        state: ControlledSupportGraphState,
        runtime: Runtime[AgentRunExecutionContext],
        decision_mode_holder: dict[str, str | int],
    ) -> ControlledSupportGraphState:
        snapshot = advance_graph_step(validate_controlled_support_state(state))
        recovered = await self.tool_executor.recover_next_persisted_outcome(
            state=snapshot,
            context=runtime.context,
        )

        if recovered is not None:
            decision_mode_holder["tool_decision_mode"] = "recovered"
            decision_mode_holder["tool_execution_count"] = 0
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
            decision_mode_holder["tool_decision_mode"] = "terminal"
            decision_mode_holder["tool_execution_count"] = 0
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
        decision_mode_holder["tool_decision_mode"] = "tool_call"
        decision_mode_holder["tool_execution_count"] = 1

        return tool_outcome.state.to_graph_state()

    async def _draft_recommendation(
        self,
        *,
        state: ControlledSupportGraphState,
        runtime: Runtime[AgentRunExecutionContext],
    ) -> ControlledSupportGraphState:
        snapshot = advance_graph_step(validate_controlled_support_state(state))
        outcome = await self.recommendation_executor.execute(
            state=snapshot,
            context=runtime.context,
        )

        return outcome.state.to_graph_state()

    def _fail_workflow(
        self,
        state: ControlledSupportGraphState,
    ) -> Never:
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

    async def _observe_node(
        self,
        *,
        node_name: str,
        observation_name: str,
        state: ControlledSupportGraphState,
        context: AgentRunExecutionContext,
        execute: Callable[[], Awaitable[ControlledSupportGraphState]],
        success_metadata: Callable[
            [ControlledSupportGraphState],
            Mapping[str, JsonValue],
        ],
    ) -> ControlledSupportGraphState:
        observation = _FailOpenObservation(
            client=self.observability_client,
            attributes=_build_node_attributes(
                node_name=node_name,
                observation_name=observation_name,
                state=state,
                context=context,
            ),
        )
        started_at = monotonic()
        observation.start()

        try:
            result = await execute()
        except (
            RetryableAgentRunExecutionError,
            TerminalAgentRunExecutionError,
        ) as error:
            observation.complete(
                ObservationUpdate(
                    status=ObservationStatus.ERROR,
                    metadata={
                        "error_code": error.error_code,
                        "latency_ms": _elapsed_ms(started_at),
                    },
                    error_code=error.error_code,
                )
            )
            raise
        except Exception:
            observation.complete(
                ObservationUpdate(
                    status=ObservationStatus.ERROR,
                    metadata={"latency_ms": _elapsed_ms(started_at)},
                    error_code=_UNEXPECTED_NODE_ERROR_CODE,
                )
            )
            raise
        else:
            metadata: dict[str, JsonValue] = {
                "latency_ms": _elapsed_ms(started_at),
            }
            with suppress(Exception):
                metadata.update(dict(success_metadata(result)))
            observation.complete(
                ObservationUpdate(
                    status=ObservationStatus.OK,
                    metadata=metadata,
                )
            )
            return result
        finally:
            observation.close()

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
        observability_client: ObservabilityClient | None = None,
    ) -> None:
        self._graph = graph
        self._observability_client = observability_client or NoOpObservabilityClient()

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
            invocation_mode: str

            if checkpoint.values:
                recovered_state = validate_controlled_support_state(checkpoint.values)
                _validate_state_ownership(
                    state=recovered_state,
                    context=context,
                )
                graph_input = None
                invocation_mode = _INVOCATION_MODE_CONTINUE
            else:
                graph_input = create_initial_controlled_support_state(
                    workspace_id=(context.agent_run.workspace_id),
                    ticket_id=context.ticket.id,
                    agent_run_id=context.agent_run.id,
                )
                invocation_mode = _INVOCATION_MODE_INITIAL

            result = await self._ainvoke_with_observation(
                graph_input=graph_input,
                config=config,
                context=context,
                identity=identity,
                invocation_mode=invocation_mode,
            )

            if result is None:
                result = (await self._graph.aget_state(config)).values
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

    async def _ainvoke_with_observation(
        self,
        *,
        graph_input: ControlledSupportGraphState | None,
        config: Mapping[str, object],
        context: AgentRunExecutionContext,
        identity: ControlledSupportGraphIdentity,
        invocation_mode: str,
    ) -> Mapping[str, object] | None:
        observation = _FailOpenObservation(
            client=self._observability_client,
            attributes=_build_workflow_attributes(
                context=context,
                identity=identity,
                invocation_mode=invocation_mode,
            ),
        )
        started_at = monotonic()
        observation.start()

        try:
            result = await self._graph.ainvoke(
                graph_input,
                config,
                context=context,
            )
        except GraphRecursionError as exc:
            observation.complete(
                ObservationUpdate(
                    status=ObservationStatus.ERROR,
                    metadata={
                        "workflow_outcome": "terminal_failure",
                        "error_code": "graph_recursion_limit_exceeded",
                        "latency_ms": _elapsed_ms(started_at),
                    },
                    error_code="graph_recursion_limit_exceeded",
                )
            )
            raise TerminalAgentRunExecutionError(
                error_code=("graph_recursion_limit_exceeded"),
                error_summary=(
                    "The controlled support graph exceeded its configured recursion limit."
                ),
            ) from exc
        except GraphStateIncompatibleError as exc:
            observation.complete(
                ObservationUpdate(
                    status=ObservationStatus.ERROR,
                    metadata={
                        "workflow_outcome": "terminal_failure",
                        "error_code": exc.error_code,
                        "latency_ms": _elapsed_ms(started_at),
                    },
                    error_code=exc.error_code,
                )
            )
            raise
        except GraphCheckpointError as exc:
            error_code = exc.error_code
            workflow_outcome = "retryable_failure" if exc.retryable else "terminal_failure"
            observation.complete(
                ObservationUpdate(
                    status=ObservationStatus.ERROR,
                    metadata={
                        "workflow_outcome": workflow_outcome,
                        "error_code": error_code,
                        "latency_ms": _elapsed_ms(started_at),
                    },
                    error_code=error_code,
                )
            )
            raise
        except RetryableAgentRunExecutionError as error:
            observation.complete(
                ObservationUpdate(
                    status=ObservationStatus.ERROR,
                    metadata={
                        "workflow_outcome": "retryable_failure",
                        "error_code": error.error_code,
                        "latency_ms": _elapsed_ms(started_at),
                    },
                    error_code=error.error_code,
                )
            )
            raise
        except TerminalAgentRunExecutionError as error:
            observation.complete(
                ObservationUpdate(
                    status=ObservationStatus.ERROR,
                    metadata={
                        "workflow_outcome": "terminal_failure",
                        "error_code": error.error_code,
                        "latency_ms": _elapsed_ms(started_at),
                    },
                    error_code=error.error_code,
                )
            )
            raise
        except Exception:
            observation.complete(
                ObservationUpdate(
                    status=ObservationStatus.ERROR,
                    metadata={
                        "workflow_outcome": "unexpected_failure",
                        "error_code": _UNEXPECTED_WORKFLOW_ERROR_CODE,
                        "latency_ms": _elapsed_ms(started_at),
                    },
                    error_code=_UNEXPECTED_WORKFLOW_ERROR_CODE,
                )
            )
            raise
        else:
            observation.complete(
                ObservationUpdate(
                    status=ObservationStatus.OK,
                    metadata={
                        "workflow_outcome": "completed",
                        "latency_ms": _elapsed_ms(started_at),
                    },
                )
            )
            return result
        finally:
            observation.close()


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


class _FailOpenObservation:
    """Workflow-owned fail-open boundary around one observation."""

    def __init__(
        self,
        *,
        client: ObservabilityClient,
        attributes: ObservationAttributes | None,
    ) -> None:
        self._client = client
        self._attributes = attributes
        self._manager: AbstractContextManager[ObservationScope] | None = None
        self._scope: ObservationScope | None = None
        self._completed = False

    def start(self) -> None:
        if self._attributes is None:
            return

        try:
            self._manager = self._client.start_observation(
                self._attributes,
            )
            self._scope = self._manager.__enter__()
        except Exception:
            self._manager = None
            self._scope = None

    def complete(self, update: ObservationUpdate) -> None:
        if self._scope is None or self._completed:
            return

        try:
            self._scope.update(update)
            self._completed = True
        except Exception:
            return

    def close(self) -> None:
        if self._manager is None:
            return

        try:
            self._manager.__exit__(None, None, None)
        except Exception:
            return
        finally:
            self._manager = None
            self._scope = None


def _build_workflow_attributes(
    *,
    context: AgentRunExecutionContext,
    identity: ControlledSupportGraphIdentity,
    invocation_mode: str,
) -> ObservationAttributes | None:
    try:
        run = context.agent_run
        attempt = context.attempt
        metadata: dict[str, JsonValue] = {
            "agent_run_id": str(run.id),
            "agent_run_attempt_id": str(attempt.id),
            "execution_request_id": str(attempt.execution_request_id),
            "workspace_id": str(run.workspace_id),
            "ticket_id": str(run.ticket_id),
            "workflow_name": run.workflow_name,
            "workflow_version": run.workflow_version,
            "trigger_key": run.trigger_key,
            "correlation_id": str(run.correlation_id),
            "graph_thread_id": identity.thread_id,
            "invocation_mode": invocation_mode,
        }
        return ObservationAttributes(
            name=_WORKFLOW_OBSERVATION_NAME,
            observation_type=ObservationType.CHAIN,
            metadata=metadata,
            metadata_paths=_WORKFLOW_METADATA_PATHS,
            input_data=None,
            input_paths=frozenset(),
            output_paths=frozenset(),
        )
    except Exception:
        return None


def _build_node_attributes(
    *,
    node_name: str,
    observation_name: str,
    state: ControlledSupportGraphState,
    context: AgentRunExecutionContext | None,
) -> ObservationAttributes | None:
    try:
        metadata: dict[str, JsonValue] = {
            "node_name": node_name,
            "agent_run_id": str(state["agent_run_id"]),
            "workspace_id": str(state["workspace_id"]),
            "ticket_id": str(state["ticket_id"]),
            "workflow_name": str(state["workflow_name"]),
            "workflow_version": str(state["workflow_version"]),
        }
        if context is not None:
            metadata["agent_run_attempt_id"] = str(context.attempt.id)
            metadata["execution_request_id"] = str(
                context.attempt.execution_request_id,
            )
            metadata["correlation_id"] = str(
                context.agent_run.correlation_id,
            )

        return ObservationAttributes(
            name=observation_name,
            observation_type=ObservationType.SPAN,
            metadata=metadata,
            metadata_paths=_NODE_METADATA_PATHS,
            input_data=None,
            input_paths=frozenset(),
            output_paths=frozenset(),
        )
    except Exception:
        return None


def _elapsed_ms(started_at: float) -> int:
    return max(0, int((monotonic() - started_at) * 1000))
