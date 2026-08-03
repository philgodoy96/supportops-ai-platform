"""LangGraph composition for the human-approved support workflow."""

from collections.abc import Hashable, Mapping
from typing import Any, Never, Protocol, cast

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
    ) -> None:
        self._graph = graph
        self._resume_planner = resume_planner

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
            if isinstance(plan, InitialGraphExecution):
                graph_input = create_initial_human_approved_support_state(
                    workspace_id=(context.agent_run.workspace_id),
                    ticket_id=context.ticket.id,
                    agent_run_id=context.agent_run.id,
                )
            elif isinstance(plan, ContinueGraphExecution):
                graph_input = None
            elif isinstance(plan, ResumeGraphExecution):
                graph_input = Command[Any](
                    resume=build_approval_resume_value(plan),
                )
            else:
                raise TypeError(
                    f"Unsupported human-approved execution plan: {type(plan)!r}.",
                )

            result = await self._graph.ainvoke(
                graph_input,
                config,
                context=context,
                version="v2",
            )
        except GraphRecursionError as exc:
            raise TerminalAgentRunExecutionError(
                error_code=("human_approved_graph_recursion_limit_exceeded"),
                error_summary=(
                    "The human-approved support graph exceeded its configured recursion limit."
                ),
            ) from exc
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
