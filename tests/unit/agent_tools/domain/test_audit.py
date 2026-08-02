"""Unit tests for controlled tool-call audit records."""

from datetime import UTC, datetime, timedelta
from types import MappingProxyType
from uuid import UUID

import pytest
from pydantic import JsonValue

from supportops.agent_tools.domain.audit import (
    AGENT_TOOL_CALL_SAFE_INPUT_MAX_BYTES,
    AgentToolCall,
    AgentToolCallStatus,
)
from supportops.agent_tools.domain.contracts import (
    ToolSafetyLevel,
)

TOOL_CALL_ID = UUID("11111111-1111-4111-8111-111111111111")
WORKSPACE_ID = UUID("22222222-2222-4222-8222-222222222222")
TICKET_ID = UUID("33333333-3333-4333-8333-333333333333")
AGENT_RUN_ID = UUID("44444444-4444-4444-8444-444444444444")
ATTEMPT_ID = UUID("55555555-5555-4555-8555-555555555555")
STARTED_AT = datetime(
    2026,
    8,
    2,
    15,
    0,
    tzinfo=UTC,
)
FINISHED_AT = STARTED_AT + timedelta(milliseconds=25)
FINGERPRINT = "a" * 64


def create_tool_call(
    *,
    status: AgentToolCallStatus = (AgentToolCallStatus.SUCCEEDED),
    safe_input: dict[str, JsonValue] | None = None,
    safe_output: dict[str, JsonValue] | None = None,
    error_code: str | None = None,
    started_at: datetime = STARTED_AT,
    finished_at: datetime = FINISHED_AT,
) -> AgentToolCall:
    """Create a valid synthetic audit record."""

    if safe_input is None:
        safe_input = {
            "top_k": 5,
            "document_ids": None,
        }

    if safe_output is None and status is AgentToolCallStatus.SUCCEEDED:
        safe_output = {
            "result_count": 2,
            "chunk_ids": [
                "chunk-1",
                "chunk-2",
            ],
        }

    if error_code is None and status is not AgentToolCallStatus.SUCCEEDED:
        error_code = "tool_dependency_unavailable"

    return AgentToolCall.create(
        tool_call_id=TOOL_CALL_ID,
        workspace_id=WORKSPACE_ID,
        ticket_id=TICKET_ID,
        agent_run_id=AGENT_RUN_ID,
        agent_run_attempt_id=ATTEMPT_ID,
        sequence=1,
        provider_tool_call_id="provider-tool-call-1",
        tool_name="search_knowledge",
        tool_version=1,
        safety_level=ToolSafetyLevel.READ_ONLY,
        status=status,
        input_fingerprint=FINGERPRINT,
        safe_input=safe_input,
        safe_output=safe_output,
        latency_ms=25,
        error_code=error_code,
        started_at=started_at,
        finished_at=finished_at,
    )


def test_creates_successful_read_only_tool_audit() -> None:
    tool_call = create_tool_call()

    assert tool_call.id == TOOL_CALL_ID
    assert tool_call.workspace_id == WORKSPACE_ID
    assert tool_call.ticket_id == TICKET_ID
    assert tool_call.agent_run_id == AGENT_RUN_ID
    assert tool_call.agent_run_attempt_id == ATTEMPT_ID
    assert tool_call.sequence == 1
    assert tool_call.tool_name == "search_knowledge"
    assert tool_call.tool_version == 1
    assert tool_call.safety_level is ToolSafetyLevel.READ_ONLY
    assert tool_call.status is AgentToolCallStatus.SUCCEEDED
    assert tool_call.error_code is None
    assert tool_call.safe_output is not None


def test_safe_payloads_are_defensively_copied() -> None:
    safe_input: dict[str, JsonValue] = {
        "top_k": 5,
        "document_ids": [
            "document-1",
        ],
    }
    safe_output: dict[str, JsonValue] = {
        "result_count": 1,
        "chunk_ids": [
            "chunk-1",
        ],
    }

    tool_call = create_tool_call(
        safe_input=safe_input,
        safe_output=safe_output,
    )

    safe_input["top_k"] = 10
    safe_output["result_count"] = 99

    assert tool_call.safe_input["top_k"] == 5
    assert tool_call.safe_output is not None
    assert tool_call.safe_output["result_count"] == 1
    assert isinstance(
        tool_call.safe_input,
        MappingProxyType,
    )

    with pytest.raises(TypeError):
        tool_call.safe_input["top_k"] = 3  # type: ignore[index]


@pytest.mark.parametrize(
    "status",
    [
        AgentToolCallStatus.FAILED,
        AgentToolCallStatus.TIMED_OUT,
        AgentToolCallStatus.REJECTED,
    ],
)
def test_unsuccessful_outcomes_require_error_code(
    status: AgentToolCallStatus,
) -> None:
    tool_call = create_tool_call(
        status=status,
        safe_output=None,
    )

    assert tool_call.status is status
    assert tool_call.error_code == "tool_dependency_unavailable"
    assert tool_call.safe_output is None


def test_success_rejects_error_code() -> None:
    with pytest.raises(
        ValueError,
        match="cannot define an error_code",
    ):
        create_tool_call(
            error_code="tool_unexpected",
        )


def test_success_requires_safe_output() -> None:
    with pytest.raises(
        ValueError,
        match="require safe_output",
    ):
        AgentToolCall.create(
            tool_call_id=TOOL_CALL_ID,
            workspace_id=WORKSPACE_ID,
            ticket_id=TICKET_ID,
            agent_run_id=AGENT_RUN_ID,
            agent_run_attempt_id=ATTEMPT_ID,
            sequence=1,
            provider_tool_call_id=None,
            tool_name="search_knowledge",
            tool_version=1,
            safety_level=ToolSafetyLevel.READ_ONLY,
            status=AgentToolCallStatus.SUCCEEDED,
            input_fingerprint=FINGERPRINT,
            safe_input={},
            safe_output=None,
            latency_ms=0,
            error_code=None,
            started_at=STARTED_AT,
            finished_at=STARTED_AT,
        )


def test_failure_rejects_safe_output() -> None:
    with pytest.raises(
        ValueError,
        match="cannot define safe_output",
    ):
        create_tool_call(
            status=AgentToolCallStatus.FAILED,
            safe_output={
                "unexpected": "partial output",
            },
        )


@pytest.mark.parametrize(
    "fingerprint",
    [
        "",
        "a" * 63,
        "A" * 64,
        "g" * 64,
    ],
)
def test_rejects_invalid_fingerprint(
    fingerprint: str,
) -> None:
    with pytest.raises(
        ValueError,
        match="lowercase SHA-256",
    ):
        AgentToolCall.create(
            tool_call_id=TOOL_CALL_ID,
            workspace_id=WORKSPACE_ID,
            ticket_id=TICKET_ID,
            agent_run_id=AGENT_RUN_ID,
            agent_run_attempt_id=ATTEMPT_ID,
            sequence=1,
            provider_tool_call_id=None,
            tool_name="search_knowledge",
            tool_version=1,
            safety_level=ToolSafetyLevel.READ_ONLY,
            status=AgentToolCallStatus.SUCCEEDED,
            input_fingerprint=fingerprint,
            safe_input={},
            safe_output={},
            latency_ms=0,
            error_code=None,
            started_at=STARTED_AT,
            finished_at=STARTED_AT,
        )


def test_rejects_non_json_safe_payload() -> None:
    with pytest.raises(
        ValueError,
        match="JSON-compatible object",
    ):
        create_tool_call(
            safe_input={
                "workspace_id": WORKSPACE_ID,  # type: ignore[dict-item]
            },
        )


def test_rejects_oversized_safe_payload() -> None:
    with pytest.raises(
        ValueError,
        match="exceeds the supported size",
    ):
        create_tool_call(
            safe_input={
                "value": "x" * AGENT_TOOL_CALL_SAFE_INPUT_MAX_BYTES,
            },
        )


def test_rejects_non_utc_timestamp() -> None:
    with pytest.raises(
        ValueError,
        match="UTC-aware",
    ):
        create_tool_call(
            started_at=STARTED_AT.replace(tzinfo=None),
        )


def test_rejects_finished_time_before_start() -> None:
    with pytest.raises(
        ValueError,
        match="must not precede",
    ):
        create_tool_call(
            finished_at=STARTED_AT - timedelta(milliseconds=1),
        )


def test_rejects_unstable_error_code() -> None:
    with pytest.raises(
        ValueError,
        match="lowercase snake case",
    ):
        create_tool_call(
            status=AgentToolCallStatus.REJECTED,
            safe_output=None,
            error_code="Tool Rejected",
        )


def test_record_is_immutable() -> None:
    tool_call = create_tool_call()

    with pytest.raises(
        AttributeError,
    ):
        tool_call.sequence = 2  # type: ignore[misc]
