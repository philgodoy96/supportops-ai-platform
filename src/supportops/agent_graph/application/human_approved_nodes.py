"""Application-owned nodes for the human-approved support graph."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Never, Protocol
from uuid import UUID

from langgraph.runtime import Runtime
from pydantic import JsonValue

from supportops.agent_graph.application.approval_interrupt import (
    ApprovalInterruptPayload,
    interrupt_for_approval,
)
from supportops.agent_graph.application.sensitive_proposal import (
    SensitiveProposalCommand,
    SensitiveProposalService,
)
from supportops.agent_graph.domain.human_approved_routing import (
    select_human_approved_support_route,
)
from supportops.agent_graph.domain.human_approved_state import (
    HUMAN_APPROVED_GRAPH_STATE_MAX_DECISION_TURNS,
    HumanApprovalCheckpointStatus,
    HumanApprovedDecisionKind,
    HumanApprovedSupportGraphState,
    HumanApprovedSupportGraphStateSnapshot,
    validate_human_approved_support_state,
)
from supportops.agent_tools.application.sensitive_bindings import (
    SensitiveToolRegistry,
)
from supportops.ai.gateway.tool_decisions import (
    LLMExecutableToolCallDecision,
    LLMTerminalControlDecision,
)
from supportops.ai.schemas.human_approved_support_decision import (
    CompleteHumanApprovedSupportAnalysisInput,
)
from supportops.core.transactions import TransactionManager
from supportops.modules.agent_runs.application.execution import (
    AgentRunExecutionContext,
    TerminalAgentRunExecutionError,
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


@dataclass(frozen=True, slots=True)
class HumanApprovedDecisionExecutionOutcome:
    """Durable model decision prepared for graph routing."""

    decision: LLMExecutableToolCallDecision | LLMTerminalControlDecision
    accepted_invocation_id: UUID


class HumanApprovedDecisionExecutor(Protocol):
    """Execute and persist one human-approved decision turn."""

    async def execute(
        self,
        *,
        state: HumanApprovedSupportGraphStateSnapshot,
        context: AgentRunExecutionContext,
        tool_observations: tuple[
            Mapping[str, JsonValue],
            ...,
        ],
    ) -> HumanApprovedDecisionExecutionOutcome:
        """Return one accepted durable decision."""

        ...


@dataclass(frozen=True, slots=True)
class HumanApprovedSupportWorkflowNodes:
    """Nodes composed exclusively by human-approved-support-v1."""

    transaction_manager: TransactionManager
    classification_repository: TicketClassificationRepository
    classification_executor: TicketClassificationExecutor
    decision_executor: HumanApprovedDecisionExecutor
    sensitive_tool_registry: SensitiveToolRegistry
    sensitive_proposal_service: SensitiveProposalService

    async def load_run_context(
        self,
        state: HumanApprovedSupportGraphState,
        runtime: Runtime[AgentRunExecutionContext],
    ) -> HumanApprovedSupportGraphState:
        """Validate graph ownership before any application work."""

        snapshot = _advance_graph_step(
            validate_human_approved_support_state(state),
        )
        _validate_context_ownership(
            state=snapshot,
            context=runtime.context,
        )
        return snapshot.model_copy(
            update={"run_context_loaded": True},
        ).to_graph_state()

    async def ensure_classification(
        self,
        state: HumanApprovedSupportGraphState,
        runtime: Runtime[AgentRunExecutionContext],
    ) -> HumanApprovedSupportGraphState:
        """Load or durably produce the ticket classification."""

        snapshot = _advance_graph_step(
            validate_human_approved_support_state(state),
        )
        _validate_context_ownership(
            state=snapshot,
            context=runtime.context,
        )

        classification = await self._load_classification(snapshot)
        if classification is None:
            await self.classification_executor.execute(
                runtime.context,
            )
            classification = await self._load_classification(
                snapshot,
            )
        if classification is None:
            raise RuntimeError(
                "Classification execution completed without a durable classification.",
            )

        return _attach_classification(
            snapshot,
            classification,
        ).to_graph_state()

    async def decide_next_action(
        self,
        state: HumanApprovedSupportGraphState,
        runtime: Runtime[AgentRunExecutionContext],
    ) -> HumanApprovedSupportGraphState:
        """Persist one terminal or sensitive proposal decision."""

        snapshot = _advance_graph_step(
            validate_human_approved_support_state(state),
        )
        _validate_context_ownership(
            state=snapshot,
            context=runtime.context,
        )
        if snapshot.decision_turn_count >= HUMAN_APPROVED_GRAPH_STATE_MAX_DECISION_TURNS:
            return snapshot.model_copy(
                update={
                    "current_error_code": ("human_approved_decision_limit_exceeded"),
                },
            ).to_graph_state()

        outcome = await self.decision_executor.execute(
            state=snapshot,
            context=runtime.context,
            tool_observations=(),
        )
        decision_turn_count = snapshot.decision_turn_count + 1
        decision = outcome.decision

        if isinstance(decision, LLMTerminalControlDecision):
            output = _require_terminal_output(decision)
            return snapshot.model_copy(
                update={
                    "decision_turn_count": decision_turn_count,
                    "decision_kind": (HumanApprovedDecisionKind.TERMINAL),
                    "decision_invocation_id": (outcome.accepted_invocation_id),
                    "decision_summary": (output.decision_summary),
                    "analysis_recommended_action": (output.recommended_action),
                    "analysis_evidence_sufficient": (output.evidence_sufficient),
                    "analysis_requires_human_review": (output.requires_human_review),
                },
            ).to_graph_state()

        if not isinstance(
            decision,
            LLMExecutableToolCallDecision,
        ):
            raise RuntimeError(
                "Human-approved decision executor returned an unsupported decision.",
            )

        binding = self.sensitive_tool_registry.lookup(
            name=decision.tool_name,
            version=decision.tool_version,
        )
        safe_input = dict(
            binding.safe_input_projector(decision.arguments),
        )
        request_reason = binding.approval_reason_projector(
            decision.arguments,
        )
        from supportops.agent_tools.domain.fingerprints import (
            create_tool_call_fingerprint,
        )

        fingerprint = create_tool_call_fingerprint(
            definition=binding.definition,
            arguments=decision.arguments,
        )
        return snapshot.model_copy(
            update={
                "decision_turn_count": decision_turn_count,
                "decision_kind": (HumanApprovedDecisionKind.SENSITIVE_TOOL),
                "decision_invocation_id": (outcome.accepted_invocation_id),
                "decision_summary": request_reason,
                "proposed_tool_provider_call_id": (decision.provider_tool_call_id),
                "proposed_tool_name": decision.tool_name,
                "proposed_tool_version": decision.tool_version,
                "proposed_tool_input": safe_input,
                "proposed_tool_fingerprint": fingerprint,
                "approval_request_reason": request_reason,
            },
        ).to_graph_state()

    async def execute_read_only_tool(
        self,
        state: HumanApprovedSupportGraphState,
    ) -> Never:
        """Keep the initial workflow surface sensitive-only."""

        raise TerminalAgentRunExecutionError(
            error_code="human_approved_read_only_path_unavailable",
            error_summary=(
                "The initial human-approved workflow does not execute read-only decisions."
            ),
        )

    async def prepare_sensitive_action(
        self,
        state: HumanApprovedSupportGraphState,
        runtime: Runtime[AgentRunExecutionContext],
    ) -> HumanApprovedSupportGraphState:
        """Persist or reuse the proposal and ApprovalRequest."""

        snapshot = _advance_graph_step(
            validate_human_approved_support_state(state),
        )
        _validate_context_ownership(
            state=snapshot,
            context=runtime.context,
        )
        command = _proposal_command_from_state(
            snapshot=snapshot,
            registry=self.sensitive_tool_registry,
        )
        outcome = await self.sensitive_proposal_service.execute(
            context=runtime.context,
            command=command,
        )
        return snapshot.model_copy(
            update={
                "tool_call_count": (snapshot.tool_call_count + 1),
                "agent_tool_call_id": outcome.tool_call.id,
                "approval_request_id": (outcome.approval_request.id),
                "approval_status": (HumanApprovalCheckpointStatus.PENDING),
                "approval_expires_at": (outcome.approval_request.expires_at.isoformat()),
            },
        ).to_graph_state()

    async def await_human_approval(
        self,
        state: HumanApprovedSupportGraphState,
    ) -> Never:
        """Interrupt only after durable records are checkpointed."""

        snapshot = _advance_graph_step(
            validate_human_approved_support_state(state),
        )
        if (
            snapshot.approval_request_id is None
            or snapshot.agent_tool_call_id is None
            or snapshot.approval_status is not HumanApprovalCheckpointStatus.PENDING
        ):
            raise TerminalAgentRunExecutionError(
                error_code="approval_interrupt_state_invalid",
                error_summary=("The graph cannot interrupt without a durable pending approval."),
            )

        # The payload is reconstructed from checkpoint-safe values.
        payload = ApprovalInterruptPayload(
            approval_request_id=snapshot.approval_request_id,
            agent_tool_call_id=snapshot.agent_tool_call_id,
            agent_run_id=snapshot.agent_run_id,
            ticket_id=snapshot.ticket_id,
            tool_name=_require_text(
                snapshot.proposed_tool_name,
                "proposed_tool_name",
            ),
            tool_version=_require_int(
                snapshot.proposed_tool_version,
                "proposed_tool_version",
            ),
            proposed_input=dict(
                snapshot.proposed_tool_input or {},
            ),
            request_reason=_require_text(
                snapshot.approval_request_reason,
                "approval_request_reason",
            ),
            expires_at=_require_text(
                snapshot.approval_expires_at,
                "approval_expires_at",
            ),
        )
        interrupt_for_approval(payload)

        raise TerminalAgentRunExecutionError(
            error_code="approval_resume_not_implemented",
            error_summary=(
                "Approval resume is intentionally deferred to the next implementation commit."
            ),
        )

    async def handle_approval_decision(
        self,
        state: HumanApprovedSupportGraphState,
    ) -> Never:
        """Reserve the approval decision branch for resume work."""

        raise TerminalAgentRunExecutionError(
            error_code="approval_decision_handling_not_implemented",
            error_summary=("Approval decision handling is not available yet."),
        )

    async def execute_sensitive_tool(
        self,
        state: HumanApprovedSupportGraphState,
    ) -> Never:
        """Prevent sensitive execution before persisted grants exist."""

        raise TerminalAgentRunExecutionError(
            error_code="sensitive_execution_not_implemented",
            error_summary=("Sensitive tool execution requires a persisted execution grant."),
        )

    async def draft_grounded_recommendation(
        self,
        state: HumanApprovedSupportGraphState,
    ) -> Never:
        """Reserve recommendation drafting for the resume commit."""

        raise TerminalAgentRunExecutionError(
            error_code=("human_approved_recommendation_not_implemented"),
            error_summary=("Human-approved recommendation drafting is not available yet."),
        )

    async def validate_recommendation(
        self,
        state: HumanApprovedSupportGraphState,
    ) -> Never:
        """Reserve recommendation validation for the resume commit."""

        raise TerminalAgentRunExecutionError(
            error_code=("human_approved_recommendation_not_implemented"),
            error_summary=("Human-approved recommendation validation is not available yet."),
        )

    async def persist_recommendation(
        self,
        state: HumanApprovedSupportGraphState,
    ) -> Never:
        """Reserve recommendation persistence for the resume commit."""

        raise TerminalAgentRunExecutionError(
            error_code=("human_approved_recommendation_not_implemented"),
            error_summary=("Human-approved recommendation persistence is not available yet."),
        )

    async def fail_workflow(
        self,
        state: HumanApprovedSupportGraphState,
    ) -> Never:
        """Translate graph failure state into a terminal run error."""

        snapshot = validate_human_approved_support_state(state)
        error_code = snapshot.current_error_code or "human_approved_workflow_state_invalid"
        raise TerminalAgentRunExecutionError(
            error_code=error_code,
            error_summary=(
                "The human-approved support workflow stopped after a fail-closed state transition."
            ),
        )

    def route(
        self,
        state: HumanApprovedSupportGraphState,
    ) -> str:
        """Return the deterministic route for validated state."""

        snapshot = validate_human_approved_support_state(state)
        return select_human_approved_support_route(
            snapshot,
        ).route.value

    async def _load_classification(
        self,
        state: HumanApprovedSupportGraphStateSnapshot,
    ) -> TicketClassification | None:
        async with self.transaction_manager.transaction():
            return await self.classification_repository.get_by_agent_run_id(
                workspace_id=state.workspace_id,
                agent_run_id=state.agent_run_id,
            )


def _advance_graph_step(
    state: HumanApprovedSupportGraphStateSnapshot,
) -> HumanApprovedSupportGraphStateSnapshot:
    next_step = state.graph_step_count + 1
    return state.model_copy(
        update={"graph_step_count": next_step},
    )


def _attach_classification(
    state: HumanApprovedSupportGraphStateSnapshot,
    classification: TicketClassification,
) -> HumanApprovedSupportGraphStateSnapshot:
    return state.model_copy(
        update={
            "classification_id": classification.id,
            "classification_category": (classification.category),
            "classification_intent": classification.intent,
            "classification_urgency": classification.urgency,
            "classification_sentiment": (classification.sentiment),
            "classification_requires_human_review": (classification.requires_human_review),
            "classification_summary": classification.summary,
        },
    )


def _proposal_command_from_state(
    *,
    snapshot: HumanApprovedSupportGraphStateSnapshot,
    registry: SensitiveToolRegistry,
) -> SensitiveProposalCommand:
    tool_name = _require_text(
        snapshot.proposed_tool_name,
        "proposed_tool_name",
    )
    tool_version = _require_int(
        snapshot.proposed_tool_version,
        "proposed_tool_version",
    )
    provider_tool_call_id = _require_text(
        snapshot.proposed_tool_provider_call_id,
        "proposed_tool_provider_call_id",
    )
    invocation_id = snapshot.decision_invocation_id
    if invocation_id is None:
        raise ValueError(
            "Sensitive proposals require decision_invocation_id.",
        )
    binding = registry.lookup(
        name=tool_name,
        version=tool_version,
    )
    arguments = binding.definition.input_schema.model_validate(
        dict(snapshot.proposed_tool_input or {}),
    )
    return SensitiveProposalCommand(
        provider_tool_call_id=provider_tool_call_id,
        tool_name=tool_name,
        tool_version=tool_version,
        arguments=arguments,
        requested_by_llm_invocation_id=invocation_id,
        sequence=snapshot.tool_call_count + 1,
    )


def _require_terminal_output(
    decision: LLMTerminalControlDecision,
) -> CompleteHumanApprovedSupportAnalysisInput:
    if not isinstance(
        decision.output,
        CompleteHumanApprovedSupportAnalysisInput,
    ):
        raise RuntimeError(
            "Human-approved terminal decision returned an unexpected schema.",
        )
    return decision.output


def _validate_context_ownership(
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
            error_summary=("The checkpointed state does not belong to the claimed AgentRun."),
        )


def _require_text(
    value: str | None,
    field_name: str,
) -> str:
    if value is None:
        raise ValueError(f"{field_name} is required.")
    return value


def _require_int(
    value: int | None,
    field_name: str,
) -> int:
    if value is None:
        raise ValueError(f"{field_name} is required.")
    return value
