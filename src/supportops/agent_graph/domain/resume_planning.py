"""Framework-independent resume planning outcomes."""

from dataclasses import dataclass
from enum import StrEnum
from uuid import UUID


class ApprovalResumeDecisionStatus(StrEnum):
    """Terminal approval outcomes accepted for graph resume."""

    APPROVED = "approved"
    REJECTED = "rejected"
    EXPIRED = "expired"


@dataclass(frozen=True, slots=True)
class InitialGraphExecution:
    """Start a workflow with a new initial graph state."""


@dataclass(frozen=True, slots=True)
class ContinueGraphExecution:
    """Continue a checkpointed workflow without resuming an interrupt."""


@dataclass(frozen=True, slots=True)
class ResumeGraphExecution:
    """Resume one interrupted graph from durable approval state."""

    approval_request_id: UUID
    agent_tool_call_id: UUID
    decision_status: ApprovalResumeDecisionStatus

    def __post_init__(self) -> None:
        if not isinstance(self.approval_request_id, UUID):
            raise TypeError("approval_request_id must be a UUID.")
        if not isinstance(self.agent_tool_call_id, UUID):
            raise TypeError("agent_tool_call_id must be a UUID.")
        if not isinstance(
            self.decision_status,
            ApprovalResumeDecisionStatus,
        ):
            raise TypeError(
                "decision_status must be ApprovalResumeDecisionStatus.",
            )


@dataclass(frozen=True, slots=True)
class CompletedGraphExecution:
    """Return the already completed workflow result without replay."""


@dataclass(frozen=True, slots=True)
class IncompatibleGraphState:
    """Fail closed when PostgreSQL and checkpoint state disagree."""

    error_code: str
    error_summary: str

    def __post_init__(self) -> None:
        if not isinstance(self.error_code, str):
            raise TypeError("error_code must be a string.")
        if not self.error_code or self.error_code != self.error_code.strip():
            raise ValueError(
                "error_code is required without surrounding whitespace.",
            )
        if not isinstance(self.error_summary, str):
            raise TypeError("error_summary must be a string.")
        if not self.error_summary or self.error_summary != self.error_summary.strip():
            raise ValueError(
                "error_summary is required without surrounding whitespace.",
            )


type HumanApprovedGraphExecutionPlan = (
    InitialGraphExecution
    | ContinueGraphExecution
    | ResumeGraphExecution
    | CompletedGraphExecution
    | IncompatibleGraphState
)
