"""Unit tests for controlled tool-call lifecycle records."""

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
DECIDED_AT = STARTED_AT + timedelta(minutes=5)
FINGERPRINT = "a" * 64


def create_terminal_tool_call(
    *,
    status: AgentToolCallStatus = (AgentToolCallStatus.SUCCEEDED),
    safe_input: dict[str, JsonValue] | None = None,
    safe_output: dict[str, JsonValue] | None = None,
    error_code: str | None = None,
    started_at: datetime = STARTED_AT,
    finished_at: datetime = FINISHED_AT,
    latency_ms: int = 25,
) -> AgentToolCall:
    """Create a valid synthetic terminal audit record."""

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

    return AgentToolCall.create_terminal(
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
        latency_ms=latency_ms,
        error_code=error_code,
        started_at=started_at,
        finished_at=finished_at,
    )


def create_pending_proposal(
    *,
    proposed_at: datetime = STARTED_AT,
) -> AgentToolCall:
    """Create one valid sensitive proposal."""

    return AgentToolCall.propose_for_approval(
        tool_call_id=TOOL_CALL_ID,
        workspace_id=WORKSPACE_ID,
        ticket_id=TICKET_ID,
        agent_run_id=AGENT_RUN_ID,
        proposed_by_agent_run_attempt_id=ATTEMPT_ID,
        sequence=1,
        provider_tool_call_id="provider-tool-call-1",
        tool_name="escalate_ticket",
        tool_version=1,
        input_fingerprint=FINGERPRINT,
        safe_input={
            "reason_code": "policy_required",
        },
        proposed_at=proposed_at,
    )


def test_creates_successful_read_only_tool_audit() -> None:
    tool_call = create_terminal_tool_call()

    assert tool_call.id == TOOL_CALL_ID
    assert tool_call.workspace_id == WORKSPACE_ID
    assert tool_call.ticket_id == TICKET_ID
    assert tool_call.agent_run_id == AGENT_RUN_ID
    assert tool_call.proposed_by_agent_run_attempt_id == ATTEMPT_ID
    assert tool_call.executed_by_agent_run_attempt_id == ATTEMPT_ID
    assert tool_call.sequence == 1
    assert tool_call.tool_name == "search_knowledge"
    assert tool_call.tool_version == 1
    assert tool_call.safety_level is ToolSafetyLevel.READ_ONLY
    assert tool_call.status is AgentToolCallStatus.SUCCEEDED
    assert tool_call.error_code is None
    assert tool_call.safe_output is not None
    assert tool_call.proposed_at == STARTED_AT
    assert tool_call.execution_started_at == STARTED_AT
    assert tool_call.finished_at == FINISHED_AT
    assert tool_call.latency_ms == 25


@pytest.mark.parametrize(
    "status",
    [
        AgentToolCallStatus.FAILED,
        AgentToolCallStatus.TIMED_OUT,
        AgentToolCallStatus.REJECTED,
    ],
)
def test_terminal_unsuccessful_outcomes_require_error_code(
    status: AgentToolCallStatus,
) -> None:
    tool_call = create_terminal_tool_call(
        status=status,
        safe_output=None,
    )

    assert tool_call.status is status
    assert tool_call.error_code == "tool_dependency_unavailable"
    assert tool_call.safe_output is None
    assert tool_call.executed_by_agent_run_attempt_id == ATTEMPT_ID
    assert tool_call.execution_started_at == STARTED_AT
    assert tool_call.finished_at == FINISHED_AT
    assert tool_call.latency_ms == 25


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

    tool_call = create_terminal_tool_call(
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


def test_success_rejects_error_code() -> None:
    with pytest.raises(
        ValueError,
        match="cannot define an error_code",
    ):
        create_terminal_tool_call(
            error_code="tool_unexpected",
        )


def test_success_requires_safe_output() -> None:
    with pytest.raises(
        ValueError,
        match="require safe_output",
    ):
        AgentToolCall.create_terminal(
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
        create_terminal_tool_call(
            status=AgentToolCallStatus.FAILED,
            safe_output={
                "unexpected": "partial output",
            },
        )


def test_creates_pending_sensitive_proposal() -> None:
    tool_call = create_pending_proposal()

    assert tool_call.status is AgentToolCallStatus.PENDING_APPROVAL
    assert tool_call.safety_level is ToolSafetyLevel.SENSITIVE_WRITE
    assert tool_call.proposed_by_agent_run_attempt_id == ATTEMPT_ID
    assert tool_call.executed_by_agent_run_attempt_id is None
    assert tool_call.safe_output is None
    assert tool_call.latency_ms is None
    assert tool_call.error_code is None
    assert tool_call.execution_started_at is None
    assert tool_call.finished_at is None
    assert tool_call.proposed_at == STARTED_AT
    assert tool_call.safe_input == {
        "reason_code": "policy_required",
    }


def test_pending_rejects_read_only_safety_level() -> None:
    with pytest.raises(
        ValueError,
        match="sensitive_write",
    ):
        AgentToolCall(
            id=TOOL_CALL_ID,
            workspace_id=WORKSPACE_ID,
            ticket_id=TICKET_ID,
            agent_run_id=AGENT_RUN_ID,
            proposed_by_agent_run_attempt_id=ATTEMPT_ID,
            executed_by_agent_run_attempt_id=None,
            sequence=1,
            provider_tool_call_id=None,
            tool_name="search_knowledge",
            tool_version=1,
            safety_level=ToolSafetyLevel.READ_ONLY,
            status=AgentToolCallStatus.PENDING_APPROVAL,
            input_fingerprint=FINGERPRINT,
            safe_input={},
            safe_output=None,
            latency_ms=None,
            error_code=None,
            proposed_at=STARTED_AT,
            execution_started_at=None,
            finished_at=None,
        )


def test_pending_rejects_execution_fields() -> None:
    with pytest.raises(
        ValueError,
        match="Non-executed lifecycle states cannot define",
    ):
        AgentToolCall(
            id=TOOL_CALL_ID,
            workspace_id=WORKSPACE_ID,
            ticket_id=TICKET_ID,
            agent_run_id=AGENT_RUN_ID,
            proposed_by_agent_run_attempt_id=ATTEMPT_ID,
            executed_by_agent_run_attempt_id=ATTEMPT_ID,
            sequence=1,
            provider_tool_call_id=None,
            tool_name="escalate_ticket",
            tool_version=1,
            safety_level=ToolSafetyLevel.SENSITIVE_WRITE,
            status=AgentToolCallStatus.PENDING_APPROVAL,
            input_fingerprint=FINGERPRINT,
            safe_input={},
            safe_output=None,
            latency_ms=None,
            error_code=None,
            proposed_at=STARTED_AT,
            execution_started_at=None,
            finished_at=None,
        )


def test_human_rejection_transition() -> None:
    pending = create_pending_proposal()

    rejected = pending.reject_for_approval(decided_at=DECIDED_AT)

    assert rejected.status is AgentToolCallStatus.REJECTED
    assert rejected.safety_level is ToolSafetyLevel.SENSITIVE_WRITE
    assert rejected.executed_by_agent_run_attempt_id is None
    assert rejected.safe_output is None
    assert rejected.latency_ms is None
    assert rejected.error_code is None
    assert rejected.execution_started_at is None
    assert rejected.finished_at == DECIDED_AT
    assert rejected.id == pending.id
    assert rejected.proposed_by_agent_run_attempt_id == (pending.proposed_by_agent_run_attempt_id)
    assert rejected.safe_input == pending.safe_input
    assert rejected.proposed_at == pending.proposed_at


def test_expiration_transition() -> None:
    pending = create_pending_proposal()

    expired = pending.expire_for_approval(decided_at=DECIDED_AT)

    assert expired.status is AgentToolCallStatus.EXPIRED
    assert expired.executed_by_agent_run_attempt_id is None
    assert expired.error_code is None
    assert expired.latency_ms is None
    assert expired.execution_started_at is None
    assert expired.finished_at == DECIDED_AT


def test_reject_from_non_pending_is_rejected() -> None:
    terminal = create_terminal_tool_call()

    with pytest.raises(
        ValueError,
        match="Only pending approval proposals can be rejected",
    ):
        terminal.reject_for_approval(decided_at=DECIDED_AT)


def test_expire_from_non_pending_is_rejected() -> None:
    terminal = create_terminal_tool_call()

    with pytest.raises(
        ValueError,
        match="Only pending approval proposals can expire",
    ):
        terminal.expire_for_approval(decided_at=DECIDED_AT)


def test_granted_execution_success_transition() -> None:
    pending = create_pending_proposal()
    resume_attempt_id = UUID("aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa")
    started_at = DECIDED_AT
    finished_at = DECIDED_AT + timedelta(milliseconds=40)
    safe_output: dict[str, JsonValue] = {
        "escalation_id": str(UUID("bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb")),
        "ticket_id": str(TICKET_ID),
        "target_queue": "engineering_support",
        "status": "escalated",
    }

    completed = pending.complete_granted_execution_success(
        executed_by_agent_run_attempt_id=resume_attempt_id,
        execution_started_at=started_at,
        finished_at=finished_at,
        safe_output=safe_output,
    )

    assert completed.status is AgentToolCallStatus.SUCCEEDED
    assert completed.executed_by_agent_run_attempt_id == (resume_attempt_id)
    assert completed.proposed_by_agent_run_attempt_id == ATTEMPT_ID
    assert resume_attempt_id != ATTEMPT_ID
    assert completed.execution_started_at == started_at
    assert completed.finished_at == finished_at
    assert completed.latency_ms == 40
    assert completed.error_code is None
    assert dict(completed.safe_output or {}) == safe_output
    assert completed.id == pending.id
    assert completed.workspace_id == pending.workspace_id
    assert completed.ticket_id == pending.ticket_id
    assert completed.agent_run_id == pending.agent_run_id
    assert completed.sequence == pending.sequence
    assert completed.provider_tool_call_id == (pending.provider_tool_call_id)
    assert completed.tool_name == pending.tool_name
    assert completed.tool_version == pending.tool_version
    assert completed.safety_level is pending.safety_level
    assert completed.input_fingerprint == pending.input_fingerprint
    assert completed.safe_input == pending.safe_input
    assert completed.proposed_at == pending.proposed_at
    assert isinstance(completed.safe_output, MappingProxyType)


def test_granted_execution_success_requires_utc_timestamps() -> None:
    pending = create_pending_proposal()

    with pytest.raises(
        ValueError,
        match="execution_started_at must be a UTC-aware timestamp",
    ):
        pending.complete_granted_execution_success(
            executed_by_agent_run_attempt_id=ATTEMPT_ID,
            execution_started_at=DECIDED_AT.replace(tzinfo=None),
            finished_at=DECIDED_AT,
            safe_output={"status": "escalated"},
        )


def test_granted_execution_success_rejects_finished_before_started() -> None:
    pending = create_pending_proposal()

    with pytest.raises(
        ValueError,
        match="finished_at must not precede execution_started_at",
    ):
        pending.complete_granted_execution_success(
            executed_by_agent_run_attempt_id=ATTEMPT_ID,
            execution_started_at=DECIDED_AT,
            finished_at=DECIDED_AT - timedelta(milliseconds=1),
            safe_output={
                "escalation_id": str(TOOL_CALL_ID),
                "ticket_id": str(TICKET_ID),
                "target_queue": "engineering_support",
                "status": "escalated",
            },
        )


@pytest.mark.parametrize(
    "status",
    [
        AgentToolCallStatus.REJECTED,
        AgentToolCallStatus.EXPIRED,
        AgentToolCallStatus.SUCCEEDED,
        AgentToolCallStatus.FAILED,
        AgentToolCallStatus.TIMED_OUT,
    ],
)
def test_granted_execution_success_rejects_non_pending(
    status: AgentToolCallStatus,
) -> None:
    if status in {
        AgentToolCallStatus.REJECTED,
        AgentToolCallStatus.EXPIRED,
    }:
        pending = create_pending_proposal()
        tool_call = (
            pending.reject_for_approval(decided_at=DECIDED_AT)
            if status is AgentToolCallStatus.REJECTED
            else pending.expire_for_approval(decided_at=DECIDED_AT)
        )
    else:
        tool_call = create_terminal_tool_call(status=status)

    with pytest.raises(
        ValueError,
        match="Only pending approval proposals can complete",
    ):
        tool_call.complete_granted_execution_success(
            executed_by_agent_run_attempt_id=ATTEMPT_ID,
            execution_started_at=DECIDED_AT,
            finished_at=DECIDED_AT,
            safe_output={
                "escalation_id": str(TOOL_CALL_ID),
                "ticket_id": str(TICKET_ID),
                "target_queue": "engineering_support",
                "status": "escalated",
            },
        )


def test_rejects_execution_started_before_proposed() -> None:
    with pytest.raises(
        ValueError,
        match="execution_started_at must not precede proposed_at",
    ):
        AgentToolCall(
            id=TOOL_CALL_ID,
            workspace_id=WORKSPACE_ID,
            ticket_id=TICKET_ID,
            agent_run_id=AGENT_RUN_ID,
            proposed_by_agent_run_attempt_id=ATTEMPT_ID,
            executed_by_agent_run_attempt_id=ATTEMPT_ID,
            sequence=1,
            provider_tool_call_id=None,
            tool_name="search_knowledge",
            tool_version=1,
            safety_level=ToolSafetyLevel.READ_ONLY,
            status=AgentToolCallStatus.SUCCEEDED,
            input_fingerprint=FINGERPRINT,
            safe_input={},
            safe_output={},
            latency_ms=0,
            error_code=None,
            proposed_at=STARTED_AT,
            execution_started_at=STARTED_AT - timedelta(milliseconds=1),
            finished_at=FINISHED_AT,
        )


def test_rejects_finished_before_execution_started() -> None:
    with pytest.raises(
        ValueError,
        match="finished_at must not precede",
    ):
        create_terminal_tool_call(
            finished_at=STARTED_AT - timedelta(milliseconds=1),
        )


def test_rejects_negative_latency() -> None:
    with pytest.raises(
        ValueError,
        match="latency_ms must be non-negative",
    ):
        create_terminal_tool_call(latency_ms=-1)


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
        AgentToolCall.create_terminal(
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
        create_terminal_tool_call(
            safe_input={
                "workspace_id": WORKSPACE_ID,  # type: ignore[dict-item]
            },
        )


def test_rejects_oversized_safe_payload() -> None:
    with pytest.raises(
        ValueError,
        match="exceeds the supported size",
    ):
        create_terminal_tool_call(
            safe_input={
                "value": "x" * AGENT_TOOL_CALL_SAFE_INPUT_MAX_BYTES,
            },
        )


def test_rejects_non_utc_timestamp() -> None:
    with pytest.raises(
        ValueError,
        match="UTC-aware",
    ):
        create_terminal_tool_call(
            started_at=STARTED_AT.replace(tzinfo=None),
        )


def test_rejects_unstable_error_code() -> None:
    with pytest.raises(
        ValueError,
        match="lowercase snake case",
    ):
        create_terminal_tool_call(
            status=AgentToolCallStatus.REJECTED,
            safe_output=None,
            error_code="Tool Rejected",
        )


def test_record_is_immutable() -> None:
    tool_call = create_terminal_tool_call()

    with pytest.raises(
        AttributeError,
    ):
        tool_call.sequence = 2  # type: ignore[misc]
