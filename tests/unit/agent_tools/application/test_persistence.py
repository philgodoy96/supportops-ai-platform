"""Unit tests for fenced tool-call persistence commands."""

from dataclasses import FrozenInstanceError, replace
from datetime import UTC, datetime, timedelta
from uuid import UUID

import pytest

from supportops.agent_tools.application.persistence import (
    PersistAgentToolCallCommand,
)
from supportops.agent_tools.domain.audit import (
    AgentToolCall,
    AgentToolCallStatus,
)
from supportops.agent_tools.domain.contracts import (
    ToolSafetyLevel,
)

_WORKSPACE_ID = UUID("11111111-1111-4111-8111-111111111111")
_TICKET_ID = UUID("22222222-2222-4222-8222-222222222222")
_AGENT_RUN_ID = UUID("33333333-3333-4333-8333-333333333333")
_ATTEMPT_ID = UUID("44444444-4444-4444-8444-444444444444")
_LEASE_TOKEN = UUID("55555555-5555-4555-8555-555555555555")
_TOOL_CALL_ID = UUID("66666666-6666-4666-8666-666666666666")

_STARTED_AT = datetime(
    2026,
    8,
    2,
    17,
    30,
    tzinfo=UTC,
)
_FINISHED_AT = _STARTED_AT + timedelta(milliseconds=25)
_PERSISTED_AT = _FINISHED_AT + timedelta(milliseconds=5)


def _tool_call() -> AgentToolCall:
    return AgentToolCall.create(
        tool_call_id=_TOOL_CALL_ID,
        workspace_id=_WORKSPACE_ID,
        ticket_id=_TICKET_ID,
        agent_run_id=_AGENT_RUN_ID,
        agent_run_attempt_id=_ATTEMPT_ID,
        sequence=1,
        provider_tool_call_id="provider-tool-call-1",
        tool_name="search_knowledge",
        tool_version=1,
        safety_level=ToolSafetyLevel.READ_ONLY,
        status=AgentToolCallStatus.SUCCEEDED,
        input_fingerprint="a" * 64,
        safe_input={
            "top_k": 5,
            "document_ids": None,
        },
        safe_output={
            "result_count": 1,
            "chunk_ids": ["77777777-7777-4777-8777-777777777777"],
        },
        latency_ms=25,
        error_code=None,
        started_at=_STARTED_AT,
        finished_at=_FINISHED_AT,
    )


def _command() -> PersistAgentToolCallCommand:
    return PersistAgentToolCallCommand(
        workspace_id=_WORKSPACE_ID,
        ticket_id=_TICKET_ID,
        agent_run_id=_AGENT_RUN_ID,
        agent_run_attempt_id=_ATTEMPT_ID,
        lease_token=_LEASE_TOKEN,
        persisted_at=_PERSISTED_AT,
        tool_call=_tool_call(),
    )


def test_accepts_consistent_terminal_tool_call() -> None:
    command = _command()

    assert command.tool_call.workspace_id == (command.workspace_id)
    assert command.tool_call.ticket_id == command.ticket_id
    assert command.tool_call.agent_run_id == (command.agent_run_id)
    assert command.tool_call.agent_run_attempt_id == (command.agent_run_attempt_id)
    assert command.persisted_at >= command.tool_call.finished_at


@pytest.mark.parametrize(
    ("field_name", "message"),
    [
        (
            "workspace_id",
            "Tool-call workspace ownership",
        ),
        (
            "ticket_id",
            "Tool-call ticket ownership",
        ),
        (
            "agent_run_id",
            "Tool-call AgentRun ownership",
        ),
        (
            "agent_run_attempt_id",
            "Tool-call AgentRunAttempt ownership",
        ),
    ],
)
def test_rejects_tool_call_ownership_mismatch(
    field_name: str,
    message: str,
) -> None:
    mismatched_id = UUID("88888888-8888-4888-8888-888888888888")
    tool_call = _tool_call()

    if field_name == "workspace_id":
        mismatched_call = replace(
            tool_call,
            workspace_id=mismatched_id,
        )
    elif field_name == "ticket_id":
        mismatched_call = replace(
            tool_call,
            ticket_id=mismatched_id,
        )
    elif field_name == "agent_run_id":
        mismatched_call = replace(
            tool_call,
            agent_run_id=mismatched_id,
        )
    else:
        mismatched_call = replace(
            tool_call,
            agent_run_attempt_id=mismatched_id,
        )

    with pytest.raises(
        ValueError,
        match=message,
    ):
        replace(
            _command(),
            tool_call=mismatched_call,
        )


def test_requires_utc_persistence_timestamp() -> None:
    with pytest.raises(
        ValueError,
        match="persisted_at must be a UTC-aware timestamp",
    ):
        replace(
            _command(),
            persisted_at=_PERSISTED_AT.replace(tzinfo=None),
        )


def test_rejects_persistence_before_tool_completion() -> None:
    with pytest.raises(
        ValueError,
        match="must not precede",
    ):
        replace(
            _command(),
            persisted_at=_FINISHED_AT - timedelta(milliseconds=1),
        )


def test_command_is_immutable() -> None:
    command = _command()

    with pytest.raises(FrozenInstanceError):
        command.lease_token = UUID(  # type: ignore[misc]
            "99999999-9999-4999-8999-999999999999",
        )
