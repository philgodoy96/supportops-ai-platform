"""Unit tests for the safe approval interrupt payload."""

from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import uuid4

import pytest

import supportops.agent_graph.application.approval_interrupt as module
from supportops.agent_graph.application.approval_interrupt import (
    ApprovalInterruptPayload,
    interrupt_for_approval,
    parse_approval_interrupt_payload,
)
from supportops.agent_tools.domain.audit import AgentToolCall
from supportops.modules.approvals.domain.models import (
    ApprovalRequest,
)


def _records() -> tuple[AgentToolCall, ApprovalRequest]:
    proposed_at = datetime(2026, 8, 3, 15, 0, tzinfo=UTC)
    tool_call = AgentToolCall.propose_for_approval(
        workspace_id=uuid4(),
        ticket_id=uuid4(),
        agent_run_id=uuid4(),
        proposed_by_agent_run_attempt_id=uuid4(),
        sequence=1,
        provider_tool_call_id="call-1",
        tool_name="escalate_ticket",
        tool_version=1,
        input_fingerprint="a" * 64,
        safe_input={
            "target_queue": "security_operations",
            "reason": "Potential security incident.",
        },
        proposed_at=proposed_at,
    )
    approval = ApprovalRequest.create_pending(
        tool_call=tool_call,
        requested_by_llm_invocation_id=uuid4(),
        request_reason="Potential security incident.",
        expires_at=proposed_at + timedelta(days=1),
        now=proposed_at,
    )
    return tool_call, approval


def test_payload_contains_only_safe_approval_fields() -> None:
    tool_call, approval = _records()

    payload = ApprovalInterruptPayload.from_records(
        tool_call=tool_call,
        approval_request=approval,
    )
    value = payload.to_interrupt_value()

    assert set(value) == {
        "approval_request_id",
        "agent_tool_call_id",
        "agent_run_id",
        "ticket_id",
        "tool_name",
        "tool_version",
        "proposed_input",
        "request_reason",
        "expires_at",
    }
    excluded = {
        "workspace_id",
        "lease_token",
        "lease_owner",
        "checkpoint",
        "provider_tool_call_id",
        "proposed_by_agent_run_attempt_id",
        "executed_by_agent_run_attempt_id",
        "attempt_id",
        "agent_run_attempt_id",
        "prompt",
        "prompt_id",
        "prompt_version",
        "raw_output",
        "safe_output",
        "request_id",
        "correlation_id",
        "ingestion_request_id",
        "actor_reference",
        "decision_actor_reference",
        "input_fingerprint",
        "sequence",
        "proposed_at",
    }
    assert excluded.isdisjoint(value)


def test_payload_round_trips_from_framework_value() -> None:
    tool_call, approval = _records()
    payload = ApprovalInterruptPayload.from_records(
        tool_call=tool_call,
        approval_request=approval,
    )

    parsed = parse_approval_interrupt_payload(
        payload.to_interrupt_value(),
    )

    assert parsed == payload


def test_interrupt_delegates_exact_safe_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tool_call, approval = _records()
    payload = ApprovalInterruptPayload.from_records(
        tool_call=tool_call,
        approval_request=approval,
    )
    captured: dict[str, Any] = {}

    def fake_interrupt(value: object) -> str:
        captured["value"] = value
        return "resume-value"

    monkeypatch.setattr(module, "interrupt", fake_interrupt)

    result = interrupt_for_approval(payload)

    assert result == "resume-value"
    assert captured["value"] == payload.to_interrupt_value()
