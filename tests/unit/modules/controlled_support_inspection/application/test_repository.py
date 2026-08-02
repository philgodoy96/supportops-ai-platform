"""Unit tests for controlled-support repository contracts."""

from datetime import UTC, datetime
from uuid import UUID

import pytest

from supportops.agent_graph.domain.state import (
    CONTROLLED_SUPPORT_WORKFLOW_VERSION,
)
from supportops.agent_tools.domain.audit import (
    AgentToolCall,
    AgentToolCallStatus,
)
from supportops.agent_tools.domain.contracts import (
    ToolSafetyLevel,
)
from supportops.modules.agent_runs.domain.models import (
    AgentRun,
)
from supportops.modules.controlled_support_inspection.application.repository import (
    ControlledSupportInspectionData,
    ControlledSupportInspectionIdentity,
)

_NOW = datetime(
    2026,
    8,
    2,
    18,
    0,
    tzinfo=UTC,
)
_WORKSPACE_ID = UUID("10000000-0000-4000-8000-000000000001")
_TICKET_ID = UUID("20000000-0000-4000-8000-000000000002")
_AGENT_RUN_ID = UUID("30000000-0000-4000-8000-000000000003")
_UNKNOWN_ATTEMPT_ID = UUID("40000000-0000-4000-8000-000000000004")
_TOOL_CALL_ID = UUID("50000000-0000-4000-8000-000000000005")


def _queued_run() -> AgentRun:
    return AgentRun.create_initial(
        agent_run_id=_AGENT_RUN_ID,
        workspace_id=_WORKSPACE_ID,
        ticket_id=_TICKET_ID,
        ingestion_request_id=UUID("60000000-0000-4000-8000-000000000006"),
        correlation_id=UUID("70000000-0000-4000-8000-000000000007"),
        workflow_version=(CONTROLLED_SUPPORT_WORKFLOW_VERSION),
        max_attempts=3,
        now=_NOW,
    )


def _tool_call() -> AgentToolCall:
    return AgentToolCall.create(
        tool_call_id=_TOOL_CALL_ID,
        workspace_id=_WORKSPACE_ID,
        ticket_id=_TICKET_ID,
        agent_run_id=_AGENT_RUN_ID,
        agent_run_attempt_id=_UNKNOWN_ATTEMPT_ID,
        sequence=1,
        provider_tool_call_id="provider-call-1",
        tool_name="lookup_service_status",
        tool_version=1,
        safety_level=ToolSafetyLevel.READ_ONLY,
        status=AgentToolCallStatus.SUCCEEDED,
        input_fingerprint="a" * 64,
        safe_input={
            "service_name": "payments-api",
        },
        safe_output={
            "service_name": "payments-api",
            "status": "operational",
            "incident_reference": None,
            "has_incident": False,
            "source": "deterministic_catalog",
        },
        latency_ms=1,
        error_code=None,
        started_at=_NOW,
        finished_at=_NOW,
    )


def test_identity_accepts_exact_uuid_scope() -> None:
    identity = ControlledSupportInspectionIdentity(
        workspace_id=_WORKSPACE_ID,
        ticket_id=_TICKET_ID,
        agent_run_id=_AGENT_RUN_ID,
    )

    assert identity.workspace_id == _WORKSPACE_ID
    assert identity.ticket_id == _TICKET_ID
    assert identity.agent_run_id == _AGENT_RUN_ID


def test_identity_rejects_non_uuid_value() -> None:
    with pytest.raises(
        TypeError,
        match="must be UUIDs",
    ):
        ControlledSupportInspectionIdentity(
            workspace_id="workspace",  # type: ignore[arg-type]
            ticket_id=_TICKET_ID,
            agent_run_id=_AGENT_RUN_ID,
        )


def test_accepts_empty_queued_snapshot() -> None:
    data = ControlledSupportInspectionData(
        agent_run=_queued_run(),
        attempts=(),
        classification=None,
        tool_calls=(),
        llm_invocations=(),
        recommendation=None,
        citations=(),
    )

    assert data.agent_run.id == _AGENT_RUN_ID
    assert data.attempts == ()
    assert data.tool_calls == ()


def test_rejects_tool_call_for_unknown_attempt() -> None:
    with pytest.raises(
        ValueError,
        match="unknown AgentRun attempt",
    ):
        ControlledSupportInspectionData(
            agent_run=_queued_run(),
            attempts=(),
            classification=None,
            tool_calls=(_tool_call(),),
            llm_invocations=(),
            recommendation=None,
            citations=(),
        )
