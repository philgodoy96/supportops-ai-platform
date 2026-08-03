"""Unit tests for the human-approved sensitive execution node."""

from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from supportops.agent_graph.application.sensitive_tool_execution import (
    SensitiveToolExecutionNode,
)
from supportops.agent_graph.domain.human_approved_state import (
    create_initial_human_approved_support_state,
)
from supportops.agent_tools.application.sensitive_execution import (
    ExecuteApprovedTicketEscalation,
)
from supportops.modules.agent_runs.application.execution import (
    AgentRunExecutionContext,
)


@pytest.mark.asyncio
async def test_node_projects_safe_execution_output() -> None:
    workspace_id = uuid4()
    ticket_id = uuid4()
    agent_run_id = uuid4()
    approval_request_id = uuid4()
    agent_tool_call_id = uuid4()
    state = create_initial_human_approved_support_state(
        workspace_id=workspace_id,
        ticket_id=ticket_id,
        agent_run_id=agent_run_id,
    )
    state.update(
        {
            "run_context_loaded": True,
            "decision_kind": "sensitive_tool",
            "decision_invocation_id": str(uuid4()),
            "decision_summary": "Escalate.",
            "proposed_tool_provider_call_id": "call-1",
            "proposed_tool_name": "escalate_ticket",
            "proposed_tool_version": 1,
            "proposed_tool_input": {
                "target_queue": "engineering_support",
                "reason": "A product defect requires review.",
            },
            "proposed_tool_fingerprint": "a" * 64,
            "approval_request_reason": ("A product defect requires review."),
            "agent_tool_call_id": str(agent_tool_call_id),
            "approval_request_id": str(approval_request_id),
            "approval_status": "pending",
            "approval_expires_at": ("2026-08-04T19:00:00+00:00"),
        },
    )
    output = SimpleNamespace(
        model_dump=lambda mode: {
            "escalation_id": str(uuid4()),
            "ticket_id": str(ticket_id),
            "target_queue": "engineering_support",
            "status": "escalated",
        },
    )
    executor = cast(
        ExecuteApprovedTicketEscalation,
        SimpleNamespace(
            execute=AsyncMock(
                return_value=SimpleNamespace(output=output),
            ),
        ),
    )
    node = SensitiveToolExecutionNode(executor=executor)
    context = cast(AgentRunExecutionContext, SimpleNamespace())

    result = await node.execute(state, context)

    assert result["approval_status"] == "approved"
    sensitive_output = cast(dict[str, Any], result["sensitive_execution_output"])
    assert sensitive_output["status"] == "escalated"
    cast(Any, executor.execute).assert_awaited_once_with(
        context=context,
        approval_request_id=approval_request_id,
        agent_tool_call_id=agent_tool_call_id,
    )
