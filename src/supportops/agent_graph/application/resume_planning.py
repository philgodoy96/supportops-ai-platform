"""Approval-aware planning for initial, continued, and resumed graphs."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol
from uuid import UUID

from pydantic import JsonValue

from supportops.agent_graph.application.approval_interrupt import (
    ApprovalInterruptPayload,
)
from supportops.agent_graph.domain.human_approved_state import (
    HumanApprovalCheckpointStatus,
    HumanApprovedGraphStateIncompatibleError,
    HumanApprovedSupportGraphStateSnapshot,
    validate_human_approved_support_state,
)
from supportops.agent_graph.domain.resume_planning import (
    ApprovalResumeDecisionStatus,
    CompletedGraphExecution,
    ContinueGraphExecution,
    HumanApprovedGraphExecutionPlan,
    IncompatibleGraphState,
    InitialGraphExecution,
    ResumeGraphExecution,
)
from supportops.agent_tools.application.queries import (
    SensitiveAgentToolCallLookup,
)
from supportops.agent_tools.domain.audit import (
    AgentToolCall,
    AgentToolCallStatus,
)
from supportops.modules.approvals.domain.models import (
    ApprovalRequest,
    ApprovalRequestStatus,
)
from supportops.modules.approvals.domain.repositories import (
    ApprovalRequestRepository,
)


class ApprovalResumeToolCallQueryRepository(Protocol):
    """Workspace-scoped tool-call query boundary."""

    async def get_sensitive_by_identity(
        self,
        query: SensitiveAgentToolCallLookup,
    ) -> AgentToolCall | None:
        """Return one sensitive tool call without cross-workspace disclosure."""

        ...


def normalize_checkpoint_interrupts(
    snapshot: object,
) -> tuple[object, ...]:
    """Normalize LangGraph StateSnapshot interrupts for planning."""

    interrupts = getattr(snapshot, "interrupts", ())
    if interrupts is None:
        return ()
    return tuple(interrupts)


@dataclass(frozen=True, slots=True)
class HumanApprovedResumePlanningContext:
    """Runtime ownership supplied by the claimed AgentRun."""

    workspace_id: UUID
    ticket_id: UUID
    agent_run_id: UUID

    def __post_init__(self) -> None:
        for field_name in (
            "workspace_id",
            "ticket_id",
            "agent_run_id",
        ):
            if not isinstance(getattr(self, field_name), UUID):
                raise TypeError(f"{field_name} must be a UUID.")


class HumanApprovedGraphResumePlanner:
    """Resolve graph execution from checkpoint and PostgreSQL state."""

    def __init__(
        self,
        *,
        approval_request_repository: ApprovalRequestRepository,
        tool_call_query_repository: (ApprovalResumeToolCallQueryRepository),
    ) -> None:
        self._approval_request_repository = approval_request_repository
        self._tool_call_query_repository = tool_call_query_repository

    async def plan(
        self,
        *,
        context: HumanApprovedResumePlanningContext,
        checkpoint_values: Mapping[str, object],
        checkpoint_interrupts: tuple[object, ...],
    ) -> HumanApprovedGraphExecutionPlan:
        """Return one fail-closed execution plan."""

        if not checkpoint_values:
            if checkpoint_interrupts:
                return _incompatible(
                    "human_approved_checkpoint_interrupt_without_state",
                    "The checkpoint contains an interrupt without compatible graph state.",
                )
            return InitialGraphExecution()

        try:
            state = validate_human_approved_support_state(
                checkpoint_values,
            )
        except HumanApprovedGraphStateIncompatibleError:
            return _incompatible(
                "human_approved_graph_state_incompatible",
                "The checkpointed graph state is incompatible.",
            )

        if (
            state.workspace_id != context.workspace_id
            or state.ticket_id != context.ticket_id
            or state.agent_run_id != context.agent_run_id
        ):
            return _incompatible(
                "human_approved_state_ownership_mismatch",
                "The checkpoint does not belong to the claimed AgentRun.",
            )

        if state.recommendation_id is not None:
            if checkpoint_interrupts:
                return _incompatible(
                    "completed_graph_contains_interrupt",
                    "A completed graph cannot retain an active interrupt.",
                )
            return CompletedGraphExecution()

        if not checkpoint_interrupts:
            if state.approval_request_id is not None:
                return _incompatible(
                    "approval_state_without_interrupt",
                    "Checkpointed approval state requires one active interrupt.",
                )
            return ContinueGraphExecution()

        if len(checkpoint_interrupts) != 1:
            return _incompatible(
                "approval_interrupt_count_invalid",
                "The graph must contain exactly one approval interrupt.",
            )

        payload = _parse_interrupt(checkpoint_interrupts[0])
        if isinstance(payload, IncompatibleGraphState):
            return payload

        checkpoint_error = _validate_checkpoint_payload(
            state=state,
            payload=payload,
        )
        if checkpoint_error is not None:
            return checkpoint_error

        approval = await self._approval_request_repository.get_by_id(
            workspace_id=context.workspace_id,
            approval_request_id=payload.approval_request_id,
        )
        if approval is None:
            return _incompatible(
                "approval_request_not_found_for_resume",
                "The interrupted graph references a missing approval request.",
            )

        tool_call = await self._tool_call_query_repository.get_sensitive_by_identity(
            SensitiveAgentToolCallLookup(
                workspace_id=context.workspace_id,
                ticket_id=context.ticket_id,
                agent_run_id=context.agent_run_id,
                tool_name=approval.tool_name,
                tool_version=approval.tool_version,
                input_fingerprint=approval.input_fingerprint,
            ),
        )
        if tool_call is None:
            return _incompatible(
                "agent_tool_call_not_found_for_resume",
                "The interrupted graph references a missing sensitive tool call.",
            )

        durable_error = _validate_durable_state(
            context=context,
            state=state,
            payload=payload,
            approval=approval,
            tool_call=tool_call,
        )
        if durable_error is not None:
            return durable_error

        if approval.status is ApprovalRequestStatus.PENDING:
            return _incompatible(
                "approval_request_still_pending",
                "A pending approval request cannot resume execution.",
            )

        status_map = {
            ApprovalRequestStatus.APPROVED: (ApprovalResumeDecisionStatus.APPROVED),
            ApprovalRequestStatus.REJECTED: (ApprovalResumeDecisionStatus.REJECTED),
            ApprovalRequestStatus.EXPIRED: (ApprovalResumeDecisionStatus.EXPIRED),
        }
        decision_status = status_map.get(approval.status)
        if decision_status is None:
            return _incompatible(
                "approval_status_not_resumable",
                "The approval status cannot resume this workflow.",
            )

        lifecycle_error = _validate_tool_call_lifecycle(
            approval_status=approval.status,
            tool_call_status=tool_call.status,
        )
        if lifecycle_error is not None:
            return lifecycle_error

        return ResumeGraphExecution(
            approval_request_id=approval.id,
            agent_tool_call_id=approval.agent_tool_call_id,
            decision_status=decision_status,
        )


def build_approval_resume_value(
    plan: ResumeGraphExecution,
) -> dict[str, JsonValue]:
    """Build the exact value passed to Command(resume=...)."""

    return {
        "approval_request_id": str(plan.approval_request_id),
        "agent_tool_call_id": str(plan.agent_tool_call_id),
        "decision_status": plan.decision_status.value,
    }


def _parse_interrupt(
    interrupt_record: object,
) -> ApprovalInterruptPayload | IncompatibleGraphState:
    value = getattr(interrupt_record, "value", None)
    if not isinstance(value, Mapping):
        return _incompatible(
            "approval_interrupt_payload_invalid",
            "The approval interrupt payload is not a mapping.",
        )
    try:
        return ApprovalInterruptPayload.model_validate(dict(value))
    except ValueError:
        return _incompatible(
            "approval_interrupt_payload_invalid",
            "The approval interrupt payload is invalid.",
        )


def _validate_checkpoint_payload(
    *,
    state: HumanApprovedSupportGraphStateSnapshot,
    payload: ApprovalInterruptPayload,
) -> IncompatibleGraphState | None:
    if (
        state.approval_request_id != payload.approval_request_id
        or state.agent_tool_call_id != payload.agent_tool_call_id
        or state.agent_run_id != payload.agent_run_id
        or state.ticket_id != payload.ticket_id
        or state.proposed_tool_name != payload.tool_name
        or state.proposed_tool_version != payload.tool_version
        or dict(state.proposed_tool_input or {}) != dict(payload.proposed_input)
        or state.approval_request_reason != payload.request_reason
        or state.approval_expires_at != payload.expires_at
    ):
        return _incompatible(
            "approval_interrupt_state_mismatch",
            "The interrupt payload does not match checkpoint state.",
        )
    if state.approval_status is not HumanApprovalCheckpointStatus.PENDING:
        return _incompatible(
            "approval_checkpoint_status_invalid",
            "Interrupted checkpoint state must remain pending.",
        )
    return None


def _validate_durable_state(
    *,
    context: HumanApprovedResumePlanningContext,
    state: HumanApprovedSupportGraphStateSnapshot,
    payload: ApprovalInterruptPayload,
    approval: ApprovalRequest,
    tool_call: AgentToolCall,
) -> IncompatibleGraphState | None:
    checks = (
        approval.workspace_id == context.workspace_id,
        approval.ticket_id == context.ticket_id,
        approval.agent_run_id == context.agent_run_id,
        approval.agent_tool_call_id == payload.agent_tool_call_id,
        approval.id == payload.approval_request_id,
        approval.tool_name == payload.tool_name,
        approval.tool_version == payload.tool_version,
        dict(approval.proposed_input) == dict(payload.proposed_input),
        approval.request_reason == payload.request_reason,
        approval.expires_at.isoformat() == payload.expires_at,
        tool_call.id == payload.agent_tool_call_id,
        tool_call.workspace_id == context.workspace_id,
        tool_call.ticket_id == context.ticket_id,
        tool_call.agent_run_id == context.agent_run_id,
        tool_call.tool_name == payload.tool_name,
        tool_call.tool_version == payload.tool_version,
        tool_call.input_fingerprint == approval.input_fingerprint,
        dict(tool_call.safe_input) == dict(payload.proposed_input),
        state.proposed_tool_fingerprint == approval.input_fingerprint,
    )
    if not all(checks):
        return _incompatible(
            "approval_durable_state_mismatch",
            "PostgreSQL approval state does not match the interrupted graph proposal.",
        )
    return None


def _validate_tool_call_lifecycle(
    *,
    approval_status: ApprovalRequestStatus,
    tool_call_status: object,
) -> IncompatibleGraphState | None:
    expected_statuses = {
        ApprovalRequestStatus.APPROVED: (AgentToolCallStatus.PENDING_APPROVAL),
        ApprovalRequestStatus.REJECTED: AgentToolCallStatus.REJECTED,
        ApprovalRequestStatus.EXPIRED: AgentToolCallStatus.EXPIRED,
    }
    expected = expected_statuses.get(approval_status)
    if expected is None:
        return None
    if tool_call_status != expected:
        return _incompatible(
            "approval_tool_call_status_mismatch",
            "The AgentToolCall status does not match the ApprovalRequest decision.",
        )
    return None


def _incompatible(
    error_code: str,
    error_summary: str,
) -> IncompatibleGraphState:
    return IncompatibleGraphState(
        error_code=error_code,
        error_summary=error_summary,
    )
