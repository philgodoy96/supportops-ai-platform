"""Unit tests for the sensitive escalation tool contract."""

import pytest

from supportops.agent_tools.domain.contracts import (
    ToolSafetyLevel,
)
from supportops.agent_tools.tools.escalate_ticket import (
    ESCALATE_TICKET_TOOL_NAME,
    ESCALATE_TICKET_TOOL_VERSION,
    EscalateTicketInput,
    TicketEscalationTargetQueue,
    create_escalate_ticket_binding,
    create_escalate_ticket_definition,
    project_escalate_ticket_approval_reason,
    project_escalate_ticket_safe_input,
)


def test_definition_is_sensitive_and_versioned() -> None:
    definition = create_escalate_ticket_definition()

    assert definition.name == ESCALATE_TICKET_TOOL_NAME
    assert definition.version == ESCALATE_TICKET_TOOL_VERSION
    assert definition.safety_level is (ToolSafetyLevel.SENSITIVE_WRITE)


def test_input_accepts_only_bounded_queue_taxonomy() -> None:
    with pytest.raises(ValueError):
        EscalateTicketInput(
            target_queue="arbitrary_queue",
            reason="Route this ticket.",
        )


def test_safe_projection_contains_only_approved_fields() -> None:
    arguments = EscalateTicketInput(
        target_queue=(TicketEscalationTargetQueue.BILLING_OPERATIONS),
        reason="Billing operations must review this case.",
    )

    assert project_escalate_ticket_safe_input(arguments) == {
        "target_queue": "billing_operations",
        "reason": "Billing operations must review this case.",
    }
    assert (
        project_escalate_ticket_approval_reason(arguments)
        == "Billing operations must review this case."
    )


def test_binding_has_no_execution_handler() -> None:
    binding = create_escalate_ticket_binding()

    assert binding.definition.safety_level is (ToolSafetyLevel.SENSITIVE_WRITE)
    assert not hasattr(binding, "handler")
