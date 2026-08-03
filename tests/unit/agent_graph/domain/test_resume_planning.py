"""Unit tests for resume planning outcome contracts."""

from uuid import uuid4

import pytest

from supportops.agent_graph.domain.resume_planning import (
    ApprovalResumeDecisionStatus,
    IncompatibleGraphState,
    ResumeGraphExecution,
)


def test_resume_plan_requires_uuid_identities() -> None:
    with pytest.raises(TypeError, match="approval_request_id"):
        ResumeGraphExecution(
            approval_request_id="invalid",  # type: ignore[arg-type]
            agent_tool_call_id=uuid4(),
            decision_status=(ApprovalResumeDecisionStatus.APPROVED),
        )


def test_resume_plan_accepts_terminal_approval_status() -> None:
    plan = ResumeGraphExecution(
        approval_request_id=uuid4(),
        agent_tool_call_id=uuid4(),
        decision_status=ApprovalResumeDecisionStatus.REJECTED,
    )

    assert plan.decision_status is (ApprovalResumeDecisionStatus.REJECTED)


def test_incompatible_state_requires_stable_message() -> None:
    with pytest.raises(ValueError, match="error_code"):
        IncompatibleGraphState(
            error_code="",
            error_summary="Checkpoint is incompatible.",
        )
