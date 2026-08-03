"""Unit tests for immutable ticket escalations."""

from dataclasses import replace
from datetime import UTC, datetime, timedelta, timezone
from uuid import uuid4

import pytest

from supportops.agent_tools.domain.audit import AgentToolCall
from supportops.agent_tools.domain.grants import SensitiveExecutionGrant
from supportops.agent_tools.tools.escalate_ticket import (
    EscalateTicketInput,
    TicketEscalationTargetQueue,
)
from supportops.modules.approvals.domain.models import ApprovalRequest
from supportops.modules.tickets.domain.escalation import (
    TicketEscalation,
)
from supportops.modules.tickets.domain.models import (
    Ticket,
    TicketStatus,
)

_NOW = datetime(2026, 8, 3, 18, 30, tzinfo=UTC)


def _grant(
    *,
    tool_name: str = "escalate_ticket",
    tool_version: int = 1,
    safe_input: dict[str, str] | None = None,
) -> SensitiveExecutionGrant:
    if safe_input is None:
        safe_input = {
            "target_queue": "engineering_support",
            "reason": "A product defect requires review.",
        }
    tool_call = AgentToolCall.propose_for_approval(
        workspace_id=uuid4(),
        ticket_id=uuid4(),
        agent_run_id=uuid4(),
        proposed_by_agent_run_attempt_id=uuid4(),
        sequence=1,
        provider_tool_call_id="call-1",
        tool_name=tool_name,
        tool_version=tool_version,
        input_fingerprint="a" * 64,
        safe_input=safe_input,
        proposed_at=_NOW,
    )
    approval = ApprovalRequest.create_pending(
        tool_call=tool_call,
        requested_by_llm_invocation_id=uuid4(),
        request_reason="A product defect requires review.",
        expires_at=_NOW + timedelta(days=1),
        now=_NOW,
    ).approve(
        actor_reference="operator:alice",
        comment=None,
        request_id=uuid4(),
        correlation_id=uuid4(),
        decided_at=_NOW + timedelta(minutes=5),
    )
    return SensitiveExecutionGrant.create(
        approval_request=approval,
        tool_call=tool_call,
        executed_by_agent_run_attempt_id=uuid4(),
        created_at=_NOW + timedelta(minutes=6),
    )


def _input_from_grant(
    grant: SensitiveExecutionGrant,
) -> EscalateTicketInput:
    return EscalateTicketInput.model_validate(
        dict(grant.granted_input),
    )


def test_create_from_grant_copies_ownership() -> None:
    grant = _grant()
    input_data = _input_from_grant(grant)

    escalation = TicketEscalation.create_from_grant(
        grant=grant,
        input_data=input_data,
        created_at=_NOW + timedelta(minutes=7),
    )

    assert escalation.workspace_id == grant.workspace_id
    assert escalation.ticket_id == grant.ticket_id
    assert escalation.agent_run_id == grant.agent_run_id
    assert escalation.approval_request_id == (grant.approval_request_id)
    assert escalation.agent_tool_call_id == (grant.agent_tool_call_id)
    assert escalation.executed_by_agent_run_attempt_id == (grant.executed_by_agent_run_attempt_id)
    assert escalation.target_queue is (TicketEscalationTargetQueue.ENGINEERING_SUPPORT)
    assert escalation.reason == ("A product defect requires review.")


def test_create_requires_escalate_ticket_v1_grant() -> None:
    grant = _grant()
    wrong_tool = replace(grant, tool_name="search_knowledge")
    with pytest.raises(ValueError, match="escalate_ticket"):
        TicketEscalation.create_from_grant(
            grant=wrong_tool,
            input_data=EscalateTicketInput(
                target_queue=(TicketEscalationTargetQueue.ENGINEERING_SUPPORT),
                reason="A product defect requires review.",
            ),
            created_at=_NOW + timedelta(minutes=7),
        )

    wrong_version = replace(grant, tool_version=2)
    with pytest.raises(ValueError, match="v1"):
        TicketEscalation.create_from_grant(
            grant=wrong_version,
            input_data=_input_from_grant(grant),
            created_at=_NOW + timedelta(minutes=7),
        )


def test_create_requires_exact_granted_input() -> None:
    grant = _grant()

    with pytest.raises(ValueError, match="match"):
        TicketEscalation.create_from_grant(
            grant=grant,
            input_data=EscalateTicketInput(
                target_queue=(TicketEscalationTargetQueue.SUPPORT_OPERATIONS),
                reason="Different route.",
            ),
            created_at=_NOW + timedelta(minutes=7),
        )


def test_create_requires_bounded_queue_type() -> None:
    grant = _grant()
    with pytest.raises(TypeError, match="TicketEscalationTargetQueue"):
        TicketEscalation(
            id=uuid4(),
            workspace_id=grant.workspace_id,
            ticket_id=grant.ticket_id,
            agent_run_id=grant.agent_run_id,
            executed_by_agent_run_attempt_id=(grant.executed_by_agent_run_attempt_id),
            approval_request_id=grant.approval_request_id,
            agent_tool_call_id=grant.agent_tool_call_id,
            target_queue="engineering_support",  # type: ignore[arg-type]
            reason="A product defect requires review.",
            created_at=_NOW + timedelta(minutes=7),
        )


def test_create_requires_normalized_reason() -> None:
    grant = _grant(
        safe_input={
            "target_queue": "engineering_support",
            "reason": "Valid reason.",
        },
    )
    with pytest.raises(ValueError, match="surrounding whitespace"):
        TicketEscalation(
            id=uuid4(),
            workspace_id=grant.workspace_id,
            ticket_id=grant.ticket_id,
            agent_run_id=grant.agent_run_id,
            executed_by_agent_run_attempt_id=(grant.executed_by_agent_run_attempt_id),
            approval_request_id=grant.approval_request_id,
            agent_tool_call_id=grant.agent_tool_call_id,
            target_queue=(TicketEscalationTargetQueue.ENGINEERING_SUPPORT),
            reason="  padded  ",
            created_at=_NOW + timedelta(minutes=7),
        )

    with pytest.raises(ValueError, match="required"):
        TicketEscalation(
            id=uuid4(),
            workspace_id=grant.workspace_id,
            ticket_id=grant.ticket_id,
            agent_run_id=grant.agent_run_id,
            executed_by_agent_run_attempt_id=(grant.executed_by_agent_run_attempt_id),
            approval_request_id=grant.approval_request_id,
            agent_tool_call_id=grant.agent_tool_call_id,
            target_queue=(TicketEscalationTargetQueue.ENGINEERING_SUPPORT),
            reason="",
            created_at=_NOW + timedelta(minutes=7),
        )


def test_creation_requires_utc_aware_created_at() -> None:
    grant = _grant()
    naive = datetime(2026, 8, 3, 18, 37)
    with pytest.raises(ValueError, match="UTC-aware"):
        TicketEscalation.create_from_grant(
            grant=grant,
            input_data=_input_from_grant(grant),
            created_at=naive,
        )

    offset = datetime(
        2026,
        8,
        3,
        15,
        37,
        tzinfo=timezone(timedelta(hours=-3)),
    )
    with pytest.raises(ValueError, match="UTC-aware"):
        TicketEscalation.create_from_grant(
            grant=grant,
            input_data=_input_from_grant(grant),
            created_at=offset,
        )


def test_creation_cannot_precede_grant() -> None:
    grant = _grant()

    with pytest.raises(ValueError, match="precede"):
        TicketEscalation.create_from_grant(
            grant=grant,
            input_data=_input_from_grant(grant),
            created_at=grant.created_at - timedelta(seconds=1),
        )


def test_matching_escalation_ignores_generated_id_and_time() -> None:
    grant = _grant()
    first = TicketEscalation.create_from_grant(
        grant=grant,
        input_data=_input_from_grant(grant),
        created_at=_NOW + timedelta(minutes=7),
    )
    second = TicketEscalation(
        id=uuid4(),
        workspace_id=first.workspace_id,
        ticket_id=first.ticket_id,
        agent_run_id=first.agent_run_id,
        executed_by_agent_run_attempt_id=(first.executed_by_agent_run_attempt_id),
        approval_request_id=first.approval_request_id,
        agent_tool_call_id=first.agent_tool_call_id,
        target_queue=first.target_queue,
        reason=first.reason,
        created_at=first.created_at + timedelta(seconds=1),
    )

    assert first.matches_escalation(second)


def test_matching_escalation_conflicts_on_attempt_queue_reason() -> None:
    grant = _grant()
    first = TicketEscalation.create_from_grant(
        grant=grant,
        input_data=_input_from_grant(grant),
        created_at=_NOW + timedelta(minutes=7),
    )

    different_attempt = replace(
        first,
        id=uuid4(),
        executed_by_agent_run_attempt_id=uuid4(),
    )
    assert not first.matches_escalation(different_attempt)

    different_queue = replace(
        first,
        id=uuid4(),
        target_queue=TicketEscalationTargetQueue.SUPPORT_OPERATIONS,
    )
    assert not first.matches_escalation(different_queue)

    different_reason = replace(
        first,
        id=uuid4(),
        reason="A different escalation reason.",
    )
    assert not first.matches_escalation(different_reason)


def test_escalation_does_not_mutate_ticket_status() -> None:
    grant = _grant()
    ticket = Ticket.create(
        ticket_id=grant.ticket_id,
        workspace_id=grant.workspace_id,
        subject="Needs escalation",
        description="Customer reported a product defect.",
        external_reference=None,
        ingestion_request_id=uuid4(),
        correlation_id=uuid4(),
        now=_NOW,
    )
    before_status = ticket.status
    before_updated_at = ticket.updated_at

    TicketEscalation.create_from_grant(
        grant=grant,
        input_data=_input_from_grant(grant),
        created_at=_NOW + timedelta(minutes=7),
    )

    assert ticket.status is TicketStatus.OPEN
    assert ticket.status is before_status
    assert ticket.updated_at == before_updated_at
