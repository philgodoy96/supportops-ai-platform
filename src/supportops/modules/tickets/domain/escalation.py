"""Immutable ticket-escalation domain record."""

from dataclasses import dataclass
from datetime import datetime, timedelta
from uuid import UUID, uuid4

from supportops.agent_tools.domain.grants import (
    SensitiveExecutionGrant,
)
from supportops.agent_tools.tools.escalate_ticket import (
    EscalateTicketInput,
    TicketEscalationTargetQueue,
)


@dataclass(frozen=True, slots=True)
class TicketEscalation:
    """One approved internal ticket-routing mutation."""

    id: UUID
    workspace_id: UUID
    ticket_id: UUID
    agent_run_id: UUID
    executed_by_agent_run_attempt_id: UUID
    approval_request_id: UUID
    agent_tool_call_id: UUID
    target_queue: TicketEscalationTargetQueue
    reason: str
    created_at: datetime

    def __post_init__(self) -> None:
        for field_name in (
            "id",
            "workspace_id",
            "ticket_id",
            "agent_run_id",
            "executed_by_agent_run_attempt_id",
            "approval_request_id",
            "agent_tool_call_id",
        ):
            if not isinstance(getattr(self, field_name), UUID):
                raise TypeError(f"{field_name} must be a UUID.")

        if not isinstance(
            self.target_queue,
            TicketEscalationTargetQueue,
        ):
            raise TypeError(
                "target_queue must be TicketEscalationTargetQueue.",
            )
        if not isinstance(self.reason, str):
            raise TypeError("reason must be a string.")
        if not self.reason:
            raise ValueError("reason is required.")
        if self.reason != self.reason.strip():
            raise ValueError(
                "reason must not contain surrounding whitespace.",
            )
        if len(self.reason) > 1000:
            raise ValueError(
                "reason exceeds the maximum length.",
            )
        _validate_utc(self.created_at, "created_at")

    @classmethod
    def create_from_grant(
        cls,
        *,
        grant: SensitiveExecutionGrant,
        input_data: EscalateTicketInput,
        created_at: datetime,
        escalation_id: UUID | None = None,
    ) -> "TicketEscalation":
        """Create one escalation from an exact execution grant."""

        if grant.tool_name != "escalate_ticket":
            raise ValueError(
                "TicketEscalation requires an escalate_ticket grant.",
            )
        if grant.tool_version != 1:
            raise ValueError(
                "TicketEscalation requires escalate_ticket v1.",
            )

        granted_input = EscalateTicketInput.model_validate(
            dict(grant.granted_input),
        )
        if granted_input != input_data:
            raise ValueError(
                "Escalation input must match the execution grant.",
            )
        _validate_utc(created_at, "created_at")
        if created_at < grant.created_at:
            raise ValueError(
                "Escalation creation cannot precede the grant.",
            )

        return cls(
            id=escalation_id or uuid4(),
            workspace_id=grant.workspace_id,
            ticket_id=grant.ticket_id,
            agent_run_id=grant.agent_run_id,
            executed_by_agent_run_attempt_id=(grant.executed_by_agent_run_attempt_id),
            approval_request_id=grant.approval_request_id,
            agent_tool_call_id=grant.agent_tool_call_id,
            target_queue=input_data.target_queue,
            reason=input_data.reason,
            created_at=created_at,
        )

    def matches_escalation(
        self,
        candidate: "TicketEscalation",
    ) -> bool:
        """Compare immutable escalation identity and content."""

        return (
            self.workspace_id == candidate.workspace_id
            and self.ticket_id == candidate.ticket_id
            and self.agent_run_id == candidate.agent_run_id
            and self.executed_by_agent_run_attempt_id == candidate.executed_by_agent_run_attempt_id
            and self.approval_request_id == candidate.approval_request_id
            and self.agent_tool_call_id == candidate.agent_tool_call_id
            and self.target_queue is candidate.target_queue
            and self.reason == candidate.reason
        )


def _validate_utc(value: datetime, field_name: str) -> None:
    if value.tzinfo is None or value.utcoffset() != timedelta(0):
        raise ValueError(f"{field_name} must be UTC-aware.")
