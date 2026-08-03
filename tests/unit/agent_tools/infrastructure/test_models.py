"""Unit tests for controlled tool-call persistence models."""

from datetime import UTC, datetime, timedelta
from typing import cast
from uuid import UUID

from sqlalchemy import Table

from supportops.agent_tools.domain.audit import (
    AgentToolCall,
    AgentToolCallStatus,
)
from supportops.agent_tools.domain.contracts import (
    ToolSafetyLevel,
)
from supportops.agent_tools.infrastructure.models import (
    AgentToolCallRecord,
)
from supportops.infrastructure.postgresql.model_registry import (
    register_persistence_models,
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
    16,
    0,
    tzinfo=UTC,
)
FINISHED_AT = STARTED_AT + timedelta(milliseconds=25)


def create_tool_call() -> AgentToolCall:
    """Create one valid terminal tool-call audit."""

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
        status=AgentToolCallStatus.SUCCEEDED,
        input_fingerprint="a" * 64,
        safe_input={
            "top_k": 5,
            "document_ids": None,
        },
        safe_output={
            "result_count": 1,
            "chunk_ids": ["66666666-6666-4666-8666-666666666666"],
        },
        latency_ms=25,
        error_code=None,
        started_at=STARTED_AT,
        finished_at=FINISHED_AT,
    )


def test_round_trips_tool_call_record() -> None:
    tool_call = create_tool_call()

    record = AgentToolCallRecord.from_domain(tool_call)
    reconstructed = record.to_domain()

    assert reconstructed == tool_call
    assert record.safe_input is not tool_call.safe_input
    assert record.safe_output is not tool_call.safe_output


def test_tool_call_table_has_expected_constraints() -> None:
    register_persistence_models()

    table = cast(Table, AgentToolCallRecord.__table__)
    constraint_names = {constraint.name for constraint in table.constraints}
    index_names = {index.name for index in table.indexes}

    assert {
        "fk_agent_tool_calls_workspace_ticket_agent_run",
        "fk_agent_tool_calls_proposed_by_attempt",
        "fk_agent_tool_calls_executed_by_attempt",
        "uq_agent_tool_calls_proposal_attempt_sequence",
        "uq_agent_tool_calls_proposal_attempt_provider_call",
        "ck_agent_tool_calls_agent_tool_call_sequence_positive",
        ("ck_agent_tool_calls_agent_tool_call_provider_call_id_format"),
        "ck_agent_tool_calls_agent_tool_call_tool_name_format",
        ("ck_agent_tool_calls_agent_tool_call_tool_version_positive"),
        "ck_agent_tool_calls_agent_tool_call_safety_level",
        "ck_agent_tool_calls_agent_tool_call_status",
        ("ck_agent_tool_calls_agent_tool_call_input_fingerprint"),
        "ck_agent_tool_calls_agent_tool_call_safe_input_object",
        "ck_agent_tool_calls_agent_tool_call_safe_input_size",
        ("ck_agent_tool_calls_agent_tool_call_safe_output_object"),
        "ck_agent_tool_calls_agent_tool_call_safe_output_size",
        ("ck_agent_tool_calls_agent_tool_call_latency_non_negative"),
        ("ck_agent_tool_calls_agent_tool_call_error_code_format"),
        "ck_agent_tool_calls_agent_tool_call_timestamp_order",
        ("ck_agent_tool_calls_agent_tool_call_sensitive_pending_state"),
        "ck_agent_tool_calls_agent_tool_call_lifecycle_state",
    }.issubset(constraint_names)

    assert "ix_agent_tool_calls_workspace_run_sequence" in index_names
    assert "uq_agent_tool_calls_sensitive_proposal_identity" in index_names
    assert "ck_agent_tool_calls_agent_tool_call_terminal_outcome" not in (constraint_names)
    assert "proposed_by_agent_run_attempt_id" in {column.name for column in table.c}
    assert "executed_by_agent_run_attempt_id" in {column.name for column in table.c}
    assert "proposed_at" in {column.name for column in table.c}
    assert "execution_started_at" in {column.name for column in table.c}
    assert table.c.latency_ms.nullable is True
    assert table.c.finished_at.nullable is True
    assert table.c.execution_started_at.nullable is True
