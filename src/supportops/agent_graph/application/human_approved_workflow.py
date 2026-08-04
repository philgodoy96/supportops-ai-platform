"""LangGraph composition for the human-approved support workflow."""

from collections.abc import Hashable, Mapping
from contextlib import AbstractContextManager, suppress
from time import monotonic
from typing import Any, Never, Protocol, cast
from uuid import UUID

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.errors import GraphRecursionError
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from supportops.agent_graph.application.approval_interrupt import (
    ApprovalInterruptPayload,
)
from supportops.agent_graph.application.human_approved_nodes import (
    HumanApprovedSupportWorkflowNodes,
)
from supportops.agent_graph.application.resume_planning import (
    HumanApprovedGraphResumePlanner,
    HumanApprovedResumePlanningContext,
    build_approval_resume_value,
    normalize_checkpoint_interrupts,
)
from supportops.agent_graph.domain.human_approved_identity import (
    HumanApprovedSupportGraphIdentity,
    derive_human_approved_support_graph_identity,
)
from supportops.agent_graph.domain.human_approved_routing import (
    HumanApprovedSupportGraphRoute,
)
from supportops.agent_graph.domain.human_approved_state import (
    HUMAN_APPROVED_SUPPORT_WORKFLOW_NAME,
    HUMAN_APPROVED_SUPPORT_WORKFLOW_VERSION,
    HumanApprovedGraphStateIncompatibleError,
    HumanApprovedSupportGraphState,
    HumanApprovedSupportGraphStateSnapshot,
    create_initial_human_approved_support_state,
    validate_human_approved_support_state,
)
from supportops.agent_graph.domain.resume_planning import (
    CompletedGraphExecution,
    ContinueGraphExecution,
    IncompatibleGraphState,
    InitialGraphExecution,
    ResumeGraphExecution,
)
from supportops.agent_graph.infrastructure.checkpoints import (
    GraphCheckpointError,
)
from supportops.modules.agent_runs.application.execution import (
    AgentRunExecutionContext,
    AgentRunExecutionResult,
    CompletedExecution,
    PausedForApproval,
    RetryableAgentRunExecutionError,
    TerminalAgentRunExecutionError,
)
from supportops.modules.agent_runs.domain.models import (
    INITIAL_TICKET_PROCESSING_TRIGGER_KEY,
)
from supportops.observability.contracts import (
    ObservabilityClient,
    ObservationScope,
)
from supportops.observability.identity import agent_run_trace_identity
from supportops.observability.models import (
    EventObservation,
    FieldPaths,
    JsonValue,
    ObservationAttributes,
    ObservationStatus,
    ObservationType,
    ObservationUpdate,
)
from supportops.observability.noop import NoOpObservabilityClient

HUMAN_APPROVED_SUPPORT_LANGGRAPH_RECURSION_LIMIT = 32

_LOAD_RUN_CONTEXT_NODE = "load_run_context"
_ENSURE_CLASSIFICATION_NODE = "ensure_classification"
_DECIDE_NEXT_ACTION_NODE = "decide_next_action"
_EXECUTE_READ_ONLY_TOOL_NODE = "execute_read_only_tool"
_PREPARE_SENSITIVE_ACTION_NODE = "prepare_sensitive_action"
_AWAIT_HUMAN_APPROVAL_NODE = "await_human_approval"
_HANDLE_APPROVAL_DECISION_NODE = "handle_approval_decision"
_EXECUTE_SENSITIVE_TOOL_NODE = "execute_sensitive_tool"
_DRAFT_GROUNDED_RECOMMENDATION_NODE = "draft_grounded_recommendation"
_VALIDATE_RECOMMENDATION_NODE = "validate_recommendation"
_PERSIST_RECOMMENDATION_NODE = "persist_recommendation"
_FAIL_WORKFLOW_NODE = "fail_workflow"

_WORKFLOW_OBSERVATION_NAME = "workflow.human-approved-support-v1"
_UNEXPECTED_WORKFLOW_ERROR_CODE = "human_approved_support_unexpected_failure"

_INVOCATION_MODE_INITIAL = "initial"
_INVOCATION_MODE_CONTINUE = "continue"
_INVOCATION_MODE_RESUME = "resume"

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
        ("approval_request_id",),
        ("workflow_outcome",),
        ("error_code",),
        ("latency_ms",),
    }
)

_WORKFLOW_RESUMED_EVENT_METADATA_PATHS: FieldPaths = frozenset(
    {
        ("approval_request_id",),
        ("agent_run_id",),
        ("agent_run_attempt_id",),
        ("workspace_id",),
        ("ticket_id",),
        ("execution_request_id",),
        ("graph_thread_id",),
    }
)


class HumanApprovedCheckpointSnapshot(Protocol):
    """Minimum checkpoint state required by the executor."""

    @property
    def values(self) -> Mapping[str, object]:
        """Return the latest checkpointed values."""

        ...

    @property
    def interrupts(self) -> tuple[object, ...]:
        """Return active LangGraph interrupts for the thread."""

        ...


class HumanApprovedCompiledGraph(Protocol):
    """Compiled graph operations consumed by the executor."""

    async def aget_state(
        self,
        config: Mapping[str, object],
    ) -> HumanApprovedCheckpointSnapshot:
        """Return the latest thread state."""

        ...

    async def ainvoke(
        self,
        input: HumanApprovedSupportGraphState | Command[Any] | None,
        config: Mapping[str, object],
        *,
        context: AgentRunExecutionContext,
        version: str = "v2",
    ) -> object:
        """Execute until completion or interruption."""

        ...


def compile_human_approved_support_graph(
    *,
    nodes: HumanApprovedSupportWorkflowNodes,
    checkpointer: BaseCheckpointSaver[Any],
) -> HumanApprovedCompiledGraph:
    """Compile the new versioned graph without altering history."""

    builder = StateGraph(
        HumanApprovedSupportGraphState,
        context_schema=AgentRunExecutionContext,
    )
    node_bindings = {
        _LOAD_RUN_CONTEXT_NODE: nodes.load_run_context,
        _ENSURE_CLASSIFICATION_NODE: nodes.ensure_classification,
        _DECIDE_NEXT_ACTION_NODE: nodes.decide_next_action,
        _EXECUTE_READ_ONLY_TOOL_NODE: (nodes.execute_read_only_tool),
        _PREPARE_SENSITIVE_ACTION_NODE: (nodes.prepare_sensitive_action),
        _AWAIT_HUMAN_APPROVAL_NODE: (nodes.await_human_approval),
        _HANDLE_APPROVAL_DECISION_NODE: (nodes.handle_approval_decision),
        _EXECUTE_SENSITIVE_TOOL_NODE: (nodes.execute_sensitive_tool),
        _DRAFT_GROUNDED_RECOMMENDATION_NODE: (nodes.draft_grounded_recommendation),
        _VALIDATE_RECOMMENDATION_NODE: (nodes.validate_recommendation),
        _PERSIST_RECOMMENDATION_NODE: (nodes.persist_recommendation),
        _FAIL_WORKFLOW_NODE: nodes.fail_workflow,
    }
    for name, handler in node_bindings.items():
        builder.add_node(name, cast(Any, handler))

    route_targets: dict[Hashable, str] = {
        HumanApprovedSupportGraphRoute.LOAD_RUN_CONTEXT.value: (_LOAD_RUN_CONTEXT_NODE),
        HumanApprovedSupportGraphRoute.ENSURE_CLASSIFICATION.value: (_ENSURE_CLASSIFICATION_NODE),
        HumanApprovedSupportGraphRoute.DECIDE_NEXT_ACTION.value: (_DECIDE_NEXT_ACTION_NODE),
        HumanApprovedSupportGraphRoute.EXECUTE_READ_ONLY_TOOL.value: (_EXECUTE_READ_ONLY_TOOL_NODE),
        HumanApprovedSupportGraphRoute.PREPARE_SENSITIVE_ACTION.value: (
            _PREPARE_SENSITIVE_ACTION_NODE
        ),
        HumanApprovedSupportGraphRoute.AWAIT_HUMAN_APPROVAL.value: (_AWAIT_HUMAN_APPROVAL_NODE),
        HumanApprovedSupportGraphRoute.HANDLE_APPROVAL_DECISION.value: (
            _HANDLE_APPROVAL_DECISION_NODE
        ),
        HumanApprovedSupportGraphRoute.EXECUTE_SENSITIVE_TOOL.value: (_EXECUTE_SENSITIVE_TOOL_NODE),
        HumanApprovedSupportGraphRoute.DRAFT_GROUNDED_RECOMMENDATION.value: (
            _DRAFT_GROUNDED_RECOMMENDATION_NODE
        ),
        HumanApprovedSupportGraphRoute.VALIDATE_RECOMMENDATION.value: (
            _VALIDATE_RECOMMENDATION_NODE
        ),
        HumanApprovedSupportGraphRoute.PERSIST_RECOMMENDATION.value: (_PERSIST_RECOMMENDATION_NODE),
        HumanApprovedSupportGraphRoute.COMPLETE_WORKFLOW.value: END,
        HumanApprovedSupportGraphRoute.FAIL_WORKFLOW.value: (_FAIL_WORKFLOW_NODE),
    }
    builder.add_conditional_edges(
        START,
        nodes.route,
        route_targets,
    )
    for source_node in node_bindings:
        if source_node == _FAIL_WORKFLOW_NODE:
            continue
        builder.add_conditional_edges(
            source_node,
            nodes.route,
            route_targets,
        )
    builder.add_edge(_FAIL_WORKFLOW_NODE, END)

    return cast(
        HumanApprovedCompiledGraph,
        builder.compile(checkpointer=checkpointer),
    )


class HumanApprovedSupportWorkflowExecutor:
    """Execute the first phase of the approval-aware graph."""

    def __init__(
        self,
        *,
        graph: HumanApprovedCompiledGraph,
        resume_planner: HumanApprovedGraphResumePlanner,
        observability_client: ObservabilityClient | None = None,
    ) -> None:
        self._graph = graph
        self._resume_planner = resume_planner
        self._observability_client = observability_client or NoOpObservabilityClient()

    async def execute(
        self,
        context: AgentRunExecutionContext,
    ) -> AgentRunExecutionResult:
        """Run until durable interruption or final completion."""

        _validate_supported_workflow(context)
        identity = derive_human_approved_support_graph_identity(
            context.agent_run.id,
        )
        config: dict[str, object] = {
            "configurable": {
                "thread_id": identity.thread_id,
                "checkpoint_ns": identity.checkpoint_namespace,
            },
            "recursion_limit": (HUMAN_APPROVED_SUPPORT_LANGGRAPH_RECURSION_LIMIT),
        }

        try:
            checkpoint = await self._graph.aget_state(config)
            plan = await self._resume_planner.plan(
                context=HumanApprovedResumePlanningContext(
                    workspace_id=context.agent_run.workspace_id,
                    ticket_id=context.ticket.id,
                    agent_run_id=context.agent_run.id,
                ),
                checkpoint_values=dict(checkpoint.values),
                checkpoint_interrupts=normalize_checkpoint_interrupts(
                    checkpoint,
                ),
            )
            if isinstance(plan, CompletedGraphExecution):
                return CompletedExecution()
            if isinstance(plan, IncompatibleGraphState):
                raise TerminalAgentRunExecutionError(
                    error_code=plan.error_code,
                    error_summary=plan.error_summary,
                )

            graph_input: HumanApprovedSupportGraphState | Command[Any] | None
            invocation_mode: str
            approval_request_id: UUID | None = None
            if isinstance(plan, InitialGraphExecution):
                graph_input = create_initial_human_approved_support_state(
                    workspace_id=(context.agent_run.workspace_id),
                    ticket_id=context.ticket.id,
                    agent_run_id=context.agent_run.id,
                )
                invocation_mode = _INVOCATION_MODE_INITIAL
            elif isinstance(plan, ContinueGraphExecution):
                graph_input = None
                invocation_mode = _INVOCATION_MODE_CONTINUE
            elif isinstance(plan, ResumeGraphExecution):
                graph_input = Command[Any](
                    resume=build_approval_resume_value(plan),
                )
                invocation_mode = _INVOCATION_MODE_RESUME
                approval_request_id = plan.approval_request_id
                _safe_record_workflow_resumed(
                    client=self._observability_client,
                    context=context,
                    identity=identity,
                    approval_request_id=plan.approval_request_id,
                )
            else:
                raise TypeError(
                    f"Unsupported human-approved execution plan: {type(plan)!r}.",
                )

            result = await self._ainvoke_with_observation(
                graph_input=graph_input,
                config=config,
                context=context,
                identity=identity,
                invocation_mode=invocation_mode,
                approval_request_id=approval_request_id,
            )
        except HumanApprovedGraphStateIncompatibleError as exc:
            raise TerminalAgentRunExecutionError(
                error_code=exc.error_code,
                error_summary=("The checkpointed human-approved graph state is incompatible."),
            ) from exc
        except GraphCheckpointError as exc:
            _raise_checkpoint_error(exc)

        interrupt_payload = _extract_single_interrupt_payload(
            result,
        )
        if interrupt_payload is not None:
            return PausedForApproval(
                approval_request_id=(interrupt_payload.approval_request_id),
                graph_thread_id=identity.thread_id,
            )

        output = _extract_output(result)
        final_state = validate_human_approved_support_state(
            output,
        )
        _validate_state_ownership(
            state=final_state,
            context=context,
        )
        if final_state.current_error_code is not None:
            raise TerminalAgentRunExecutionError(
                error_code=final_state.current_error_code,
                error_summary=("The human-approved graph completed with an unresolved error."),
            )
        if final_state.recommendation_id is None:
            raise TerminalAgentRunExecutionError(
                error_code="human_approved_workflow_incomplete",
                error_summary=(
                    "The human-approved graph completed without "
                    "a persisted recommendation or approval pause."
                ),
            )
        return CompletedExecution()

    async def _ainvoke_with_observation(
        self,
        *,
        graph_input: HumanApprovedSupportGraphState | Command[Any] | None,
        config: Mapping[str, object],
        context: AgentRunExecutionContext,
        identity: HumanApprovedSupportGraphIdentity,
        invocation_mode: str,
        approval_request_id: UUID | None,
    ) -> object:
        observation = _FailOpenObservation(
            client=self._observability_client,
            attributes=_build_workflow_attributes(
                context=context,
                identity=identity,
                invocation_mode=invocation_mode,
                approval_request_id=approval_request_id,
            ),
        )
        started_at = monotonic()
        observation.start()

        try:
            result = await self._graph.ainvoke(
                graph_input,
                config,
                context=context,
                version="v2",
            )
        except GraphRecursionError as exc:
            observation.complete(
                ObservationUpdate(
                    status=ObservationStatus.ERROR,
                    metadata={
                        "workflow_outcome": "terminal_failure",
                        "error_code": ("human_approved_graph_recursion_limit_exceeded"),
                        "latency_ms": _elapsed_ms(started_at),
                    },
                    error_code=("human_approved_graph_recursion_limit_exceeded"),
                )
            )
            raise TerminalAgentRunExecutionError(
                error_code=("human_approved_graph_recursion_limit_exceeded"),
                error_summary=(
                    "The human-approved support graph exceeded its configured recursion limit."
                ),
            ) from exc
        except HumanApprovedGraphStateIncompatibleError as exc:
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
            workflow_outcome = "retryable_failure" if exc.retryable else "terminal_failure"
            observation.complete(
                ObservationUpdate(
                    status=ObservationStatus.ERROR,
                    metadata={
                        "workflow_outcome": workflow_outcome,
                        "error_code": exc.error_code,
                        "latency_ms": _elapsed_ms(started_at),
                    },
                    error_code=exc.error_code,
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
            interrupt_payload = _extract_single_interrupt_payload(result)
            if interrupt_payload is not None:
                observation.complete(
                    ObservationUpdate(
                        status=ObservationStatus.OK,
                        metadata={
                            "workflow_outcome": "awaiting_approval",
                            "approval_request_id": str(
                                interrupt_payload.approval_request_id,
                            ),
                            "latency_ms": _elapsed_ms(started_at),
                        },
                    )
                )
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


def _extract_single_interrupt_payload(
    result: object,
) -> ApprovalInterruptPayload | None:
    interrupts = getattr(result, "interrupts", ())
    if not interrupts:
        return None
    if len(interrupts) != 1:
        raise TerminalAgentRunExecutionError(
            error_code="approval_interrupt_count_invalid",
            error_summary=("The human-approved graph returned an invalid number of interrupts."),
        )
    value = getattr(interrupts[0], "value", None)
    if not isinstance(value, Mapping):
        raise TerminalAgentRunExecutionError(
            error_code="approval_interrupt_payload_invalid",
            error_summary=("The approval interrupt payload is invalid."),
        )
    try:
        return ApprovalInterruptPayload.model_validate(
            dict(value),
        )
    except ValueError as exc:
        raise TerminalAgentRunExecutionError(
            error_code="approval_interrupt_payload_invalid",
            error_summary=("The approval interrupt payload is invalid."),
        ) from exc


def _extract_output(
    result: object,
) -> Mapping[str, object]:
    output = getattr(result, "value", result)
    if not isinstance(output, Mapping):
        raise TerminalAgentRunExecutionError(
            error_code="human_approved_graph_output_invalid",
            error_summary=("The human-approved graph returned an invalid output."),
        )
    return output


def _validate_supported_workflow(
    context: AgentRunExecutionContext,
) -> None:
    run = context.agent_run
    if run.workflow_name != HUMAN_APPROVED_SUPPORT_WORKFLOW_NAME:
        raise TerminalAgentRunExecutionError(
            error_code="unsupported_workflow",
            error_summary=(
                "The AgentRun workflow is not supported by the human-approved executor."
            ),
        )
    if run.workflow_version != HUMAN_APPROVED_SUPPORT_WORKFLOW_VERSION:
        raise TerminalAgentRunExecutionError(
            error_code="unsupported_workflow_version",
            error_summary=(
                "The AgentRun workflow version is not supported by the human-approved executor."
            ),
        )
    if run.trigger_key != INITIAL_TICKET_PROCESSING_TRIGGER_KEY:
        raise TerminalAgentRunExecutionError(
            error_code="unsupported_trigger",
            error_summary=("The AgentRun trigger is not supported by the human-approved executor."),
        )


def _validate_state_ownership(
    *,
    state: HumanApprovedSupportGraphStateSnapshot,
    context: AgentRunExecutionContext,
) -> None:
    if (
        state.workspace_id != context.agent_run.workspace_id
        or state.ticket_id != context.ticket.id
        or state.agent_run_id != context.agent_run.id
    ):
        raise TerminalAgentRunExecutionError(
            error_code="human_approved_state_ownership_mismatch",
            error_summary=("The checkpointed graph state does not belong to the claimed AgentRun."),
        )


def _raise_checkpoint_error(
    error: GraphCheckpointError,
) -> Never:
    if error.retryable:
        raise RetryableAgentRunExecutionError(
            error_code=error.error_code,
            error_summary=(
                "The approval-aware checkpoint infrastructure is temporarily unavailable."
            ),
        ) from error
    raise TerminalAgentRunExecutionError(
        error_code=error.error_code,
        error_summary=("The approval-aware checkpoint runtime cannot continue."),
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
    identity: HumanApprovedSupportGraphIdentity,
    invocation_mode: str,
    approval_request_id: UUID | None,
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
        if approval_request_id is not None:
            metadata["approval_request_id"] = str(approval_request_id)
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


def _safe_record_workflow_resumed(
    *,
    client: ObservabilityClient,
    context: AgentRunExecutionContext,
    identity: HumanApprovedSupportGraphIdentity,
    approval_request_id: UUID,
) -> None:
    with suppress(Exception):
        run = context.agent_run
        attempt = context.attempt
        client.record_trace_event(
            identity=agent_run_trace_identity(
                agent_run_id=run.id,
                ticket_id=run.ticket_id,
            ),
            event=EventObservation(
                name="workflow.resumed",
                status=ObservationStatus.OK,
                metadata={
                    "approval_request_id": str(approval_request_id),
                    "agent_run_id": str(run.id),
                    "agent_run_attempt_id": str(attempt.id),
                    "workspace_id": str(run.workspace_id),
                    "ticket_id": str(run.ticket_id),
                    "execution_request_id": str(
                        attempt.execution_request_id,
                    ),
                    "graph_thread_id": identity.thread_id,
                },
                metadata_paths=_WORKFLOW_RESUMED_EVENT_METADATA_PATHS,
            ),
        )


def _elapsed_ms(started_at: float) -> int:
    return max(0, int((monotonic() - started_at) * 1000))
