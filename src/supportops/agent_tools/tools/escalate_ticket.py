"""Sensitive internal ticket-escalation tool contract."""

from collections.abc import Mapping
from enum import StrEnum
from typing import Annotated, Literal
from uuid import UUID

from pydantic import JsonValue, StringConstraints

from supportops.agent_tools.application.sensitive_bindings import (
    SensitiveToolBinding,
)
from supportops.agent_tools.domain.contracts import (
    StrictToolSchema,
    ToolAuditPolicy,
    ToolDefinition,
    ToolFailurePolicy,
    ToolSafetyLevel,
)

ESCALATE_TICKET_TOOL_NAME = "escalate_ticket"
ESCALATE_TICKET_TOOL_VERSION = 1
ESCALATE_TICKET_DEFAULT_TIMEOUT_SECONDS = 5.0


class TicketEscalationTargetQueue(StrEnum):
    """Bounded internal queues visible to the model."""

    BILLING_OPERATIONS = "billing_operations"
    ENGINEERING_SUPPORT = "engineering_support"
    SECURITY_OPERATIONS = "security_operations"
    SUPPORT_OPERATIONS = "support_operations"


EscalationReason = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        min_length=1,
        max_length=1000,
    ),
]


class EscalateTicketInput(StrictToolSchema):
    """Strict model-visible escalation proposal."""

    target_queue: TicketEscalationTargetQueue
    reason: EscalationReason


class EscalateTicketOutput(StrictToolSchema):
    """Safe output contract for future granted execution."""

    escalation_id: UUID
    ticket_id: UUID
    target_queue: TicketEscalationTargetQueue
    status: Literal["escalated"]


def create_escalate_ticket_definition(
    *,
    timeout_seconds: float = (ESCALATE_TICKET_DEFAULT_TIMEOUT_SECONDS),
) -> ToolDefinition:
    """Create immutable policy metadata for escalation."""

    return ToolDefinition(
        name=ESCALATE_TICKET_TOOL_NAME,
        version=ESCALATE_TICKET_TOOL_VERSION,
        description=(
            "Propose routing this ticket to one bounded internal "
            "support queue. This action requires durable human "
            "approval before execution."
        ),
        input_schema=EscalateTicketInput,
        output_schema=EscalateTicketOutput,
        safety_level=ToolSafetyLevel.SENSITIVE_WRITE,
        timeout_seconds=timeout_seconds,
        failure_policy=ToolFailurePolicy.FAIL_AGENT_RUN,
        audit_policy=ToolAuditPolicy.SAFE_PROJECTION,
    )


def project_escalate_ticket_safe_input(
    arguments: StrictToolSchema,
) -> Mapping[str, JsonValue]:
    """Project only approved escalation input fields."""

    if not isinstance(arguments, EscalateTicketInput):
        raise TypeError(
            "Escalation input projection requires EscalateTicketInput.",
        )

    return {
        "target_queue": arguments.target_queue.value,
        "reason": arguments.reason,
    }


def project_escalate_ticket_approval_reason(
    arguments: StrictToolSchema,
) -> str:
    """Return the bounded human-review reason."""

    if not isinstance(arguments, EscalateTicketInput):
        raise TypeError(
            "Escalation approval reason requires EscalateTicketInput.",
        )

    return arguments.reason


def create_escalate_ticket_binding(
    *,
    timeout_seconds: float = (ESCALATE_TICKET_DEFAULT_TIMEOUT_SECONDS),
) -> SensitiveToolBinding:
    """Create a proposal-only binding without an execution handler."""

    return SensitiveToolBinding(
        definition=create_escalate_ticket_definition(
            timeout_seconds=timeout_seconds,
        ),
        safe_input_projector=(project_escalate_ticket_safe_input),
        approval_reason_projector=(project_escalate_ticket_approval_reason),
    )
