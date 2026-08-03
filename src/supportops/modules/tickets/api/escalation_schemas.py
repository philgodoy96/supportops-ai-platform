"""HTTP schemas for ticket escalation inspection."""

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict

from supportops.agent_tools.tools.escalate_ticket import (
    TicketEscalationTargetQueue,
)
from supportops.modules.tickets.domain.escalation import (
    TicketEscalation,
)


class TicketEscalationResponse(BaseModel):
    """Workspace-scoped immutable ticket escalation representation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

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

    @classmethod
    def from_domain(
        cls,
        escalation: TicketEscalation,
    ) -> "TicketEscalationResponse":
        """Create a safe API response from the domain entity."""

        return cls(
            id=escalation.id,
            workspace_id=escalation.workspace_id,
            ticket_id=escalation.ticket_id,
            agent_run_id=escalation.agent_run_id,
            executed_by_agent_run_attempt_id=(escalation.executed_by_agent_run_attempt_id),
            approval_request_id=escalation.approval_request_id,
            agent_tool_call_id=escalation.agent_tool_call_id,
            target_queue=escalation.target_queue,
            reason=escalation.reason,
            created_at=escalation.created_at,
        )


class TicketEscalationListResponse(BaseModel):
    """One keyset-paginated ticket escalation page."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    items: tuple[TicketEscalationResponse, ...]
    next_cursor: str | None = None
