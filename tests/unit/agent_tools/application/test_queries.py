"""Unit tests for controlled tool-call query contracts."""

from dataclasses import FrozenInstanceError
from uuid import UUID

import pytest

from supportops.agent_tools.application.queries import (
    AgentToolCallLookup,
    SensitiveAgentToolCallLookup,
)

_WORKSPACE_ID = UUID("11111111-1111-4111-8111-111111111111")
_TICKET_ID = UUID("22222222-2222-4222-8222-222222222222")
_AGENT_RUN_ID = UUID("33333333-3333-4333-8333-333333333333")
_ATTEMPT_ID = UUID("44444444-4444-4444-8444-444444444444")
_FINGERPRINT = "a" * 64


def test_lookup_accepts_proposal_attempt_sequence() -> None:
    query = AgentToolCallLookup(
        workspace_id=_WORKSPACE_ID,
        ticket_id=_TICKET_ID,
        agent_run_id=_AGENT_RUN_ID,
        proposed_by_agent_run_attempt_id=_ATTEMPT_ID,
        sequence=1,
    )

    assert query.proposed_by_agent_run_attempt_id == _ATTEMPT_ID
    assert query.sequence == 1


def test_lookup_rejects_non_positive_sequence() -> None:
    with pytest.raises(ValueError, match="sequence must be positive"):
        AgentToolCallLookup(
            workspace_id=_WORKSPACE_ID,
            ticket_id=_TICKET_ID,
            agent_run_id=_AGENT_RUN_ID,
            proposed_by_agent_run_attempt_id=_ATTEMPT_ID,
            sequence=0,
        )


def test_sensitive_lookup_accepts_identity() -> None:
    query = SensitiveAgentToolCallLookup(
        workspace_id=_WORKSPACE_ID,
        ticket_id=_TICKET_ID,
        agent_run_id=_AGENT_RUN_ID,
        tool_name="escalate_ticket",
        tool_version=1,
        input_fingerprint=_FINGERPRINT,
    )

    assert query.tool_name == "escalate_ticket"
    assert query.input_fingerprint == _FINGERPRINT


def test_sensitive_lookup_rejects_invalid_fingerprint() -> None:
    with pytest.raises(ValueError, match="lowercase SHA-256"):
        SensitiveAgentToolCallLookup(
            workspace_id=_WORKSPACE_ID,
            ticket_id=_TICKET_ID,
            agent_run_id=_AGENT_RUN_ID,
            tool_name="escalate_ticket",
            tool_version=1,
            input_fingerprint="A" * 64,
        )


def test_lookups_are_immutable() -> None:
    query = AgentToolCallLookup(
        workspace_id=_WORKSPACE_ID,
        ticket_id=_TICKET_ID,
        agent_run_id=_AGENT_RUN_ID,
        proposed_by_agent_run_attempt_id=_ATTEMPT_ID,
        sequence=1,
    )

    with pytest.raises(FrozenInstanceError):
        query.sequence = 2  # type: ignore[misc]
