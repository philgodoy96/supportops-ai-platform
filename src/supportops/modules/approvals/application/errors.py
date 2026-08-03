"""Stable application errors for human approval workflows."""

from uuid import UUID

from supportops.modules.approvals.domain.models import ApprovalRequest


class ApprovalApplicationError(RuntimeError):
    """Base error for approval application use cases."""

    code = "approval_application_error"


class ApprovalRequestNotFoundError(ApprovalApplicationError):
    """Raised when a workspace-scoped approval request is absent."""

    code = "approval_request_not_found"

    def __init__(self, approval_request_id: UUID) -> None:
        super().__init__("The approval request was not found.")
        self.approval_request_id = approval_request_id


class ApprovalDecisionConflictError(ApprovalApplicationError):
    """Raised when an immutable terminal decision conflicts."""

    code = "approval_decision_conflict"

    def __init__(self, approval_request_id: UUID) -> None:
        super().__init__(
            "The approval request already has a conflicting decision.",
        )
        self.approval_request_id = approval_request_id


class ApprovalRequestExpiredError(ApprovalApplicationError):
    """Raised after an overdue pending request is durably expired."""

    code = "approval_request_expired"

    def __init__(self, approval_request: ApprovalRequest) -> None:
        super().__init__("The approval request has expired.")
        self.approval_request = approval_request


class ApprovalRunStateConflictError(ApprovalApplicationError):
    """Raised when the related AgentRun cannot be requeued safely."""

    code = "approval_run_state_conflict"

    def __init__(self, agent_run_id: UUID) -> None:
        super().__init__(
            "The AgentRun is not waiting for this approval decision.",
        )
        self.agent_run_id = agent_run_id


class ApprovalToolCallStateConflictError(ApprovalApplicationError):
    """Raised when the proposed sensitive tool call is inconsistent."""

    code = "approval_tool_call_state_conflict"

    def __init__(self, agent_tool_call_id: UUID) -> None:
        super().__init__(
            "The proposed sensitive tool call is inconsistent.",
        )
        self.agent_tool_call_id = agent_tool_call_id
