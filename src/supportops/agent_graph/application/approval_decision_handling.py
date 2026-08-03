"""Approval decision projection for resumed human-approved workflows."""

from dataclasses import dataclass
from enum import StrEnum
from uuid import UUID

from pydantic import BaseModel, ConfigDict, JsonValue

from supportops.agent_graph.domain.resume_planning import (
    ApprovalResumeDecisionStatus,
)
from supportops.modules.approvals.domain.models import (
    ApprovalRequest,
    ApprovalRequestStatus,
)


class ApprovalDecisionAction(StrEnum):
    """Graph actions derived from terminal approval state."""

    EXECUTE_SENSITIVE_TOOL = "execute_sensitive_tool"
    CONTINUE_WITHOUT_EXECUTION = "continue_without_execution"


class ApprovalDecisionResumePayload(BaseModel):
    """Strict value accepted from Command(resume=...)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    approval_request_id: UUID
    agent_tool_call_id: UUID
    decision_status: ApprovalResumeDecisionStatus

    def to_json_value(self) -> dict[str, JsonValue]:
        """Return the JSON-compatible resume representation."""

        return self.model_dump(mode="json")


@dataclass(frozen=True, slots=True)
class ApprovalDecisionHandlingResult:
    """Validated action selected from durable approval state."""

    approval_request_id: UUID
    agent_tool_call_id: UUID
    decision_status: ApprovalResumeDecisionStatus
    action: ApprovalDecisionAction
    decision_summary: str

    def __post_init__(self) -> None:
        if not self.decision_summary:
            raise ValueError("decision_summary is required.")
        if self.decision_summary != self.decision_summary.strip():
            raise ValueError(
                "decision_summary must not contain surrounding whitespace.",
            )


class ApprovalDecisionHandlingError(RuntimeError):
    """Raised when resume payload and PostgreSQL disagree."""


def handle_approval_decision(
    *,
    payload: ApprovalDecisionResumePayload,
    approval_request: ApprovalRequest,
) -> ApprovalDecisionHandlingResult:
    """Validate one resume value against durable approval state."""

    if approval_request.id != payload.approval_request_id:
        raise ApprovalDecisionHandlingError(
            "Resume payload references a different approval request.",
        )
    if approval_request.agent_tool_call_id != payload.agent_tool_call_id:
        raise ApprovalDecisionHandlingError(
            "Resume payload references a different AgentToolCall.",
        )

    durable_status = _map_status(approval_request.status)
    if durable_status is None:
        raise ApprovalDecisionHandlingError(
            "Pending approval cannot resume workflow execution.",
        )
    if durable_status is not payload.decision_status:
        raise ApprovalDecisionHandlingError(
            "Resume payload status does not match PostgreSQL.",
        )

    match durable_status:
        case ApprovalResumeDecisionStatus.APPROVED:
            return ApprovalDecisionHandlingResult(
                approval_request_id=approval_request.id,
                agent_tool_call_id=approval_request.agent_tool_call_id,
                decision_status=durable_status,
                action=ApprovalDecisionAction.EXECUTE_SENSITIVE_TOOL,
                decision_summary=("The sensitive action was approved for execution."),
            )
        case ApprovalResumeDecisionStatus.REJECTED:
            return ApprovalDecisionHandlingResult(
                approval_request_id=approval_request.id,
                agent_tool_call_id=approval_request.agent_tool_call_id,
                decision_status=durable_status,
                action=ApprovalDecisionAction.CONTINUE_WITHOUT_EXECUTION,
                decision_summary=("The sensitive action was rejected and was not executed."),
            )
        case ApprovalResumeDecisionStatus.EXPIRED:
            return ApprovalDecisionHandlingResult(
                approval_request_id=approval_request.id,
                agent_tool_call_id=approval_request.agent_tool_call_id,
                decision_status=durable_status,
                action=ApprovalDecisionAction.CONTINUE_WITHOUT_EXECUTION,
                decision_summary=("The sensitive action approval expired and was not executed."),
            )


def _map_status(
    status: ApprovalRequestStatus,
) -> ApprovalResumeDecisionStatus | None:
    match status:
        case ApprovalRequestStatus.APPROVED:
            return ApprovalResumeDecisionStatus.APPROVED
        case ApprovalRequestStatus.REJECTED:
            return ApprovalResumeDecisionStatus.REJECTED
        case ApprovalRequestStatus.EXPIRED:
            return ApprovalResumeDecisionStatus.EXPIRED
        case ApprovalRequestStatus.PENDING:
            return None
