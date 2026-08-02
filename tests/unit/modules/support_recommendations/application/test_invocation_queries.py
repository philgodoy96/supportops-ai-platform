"""Unit tests for attempt-scoped LLM invocation queries."""

from dataclasses import FrozenInstanceError
from uuid import UUID

import pytest

from supportops.modules.support_recommendations.application.invocation_queries import (
    AttemptLLMInvocationQuery,
)

_WORKSPACE_ID = UUID("10000000-0000-4000-8000-000000000001")
_TICKET_ID = UUID("20000000-0000-4000-8000-000000000002")
_AGENT_RUN_ID = UUID("30000000-0000-4000-8000-000000000003")
_ATTEMPT_ID = UUID("40000000-0000-4000-8000-000000000004")


def test_accepts_exact_attempt_ownership() -> None:
    query = AttemptLLMInvocationQuery(
        workspace_id=_WORKSPACE_ID,
        ticket_id=_TICKET_ID,
        agent_run_id=_AGENT_RUN_ID,
        agent_run_attempt_id=_ATTEMPT_ID,
    )

    assert query.workspace_id == _WORKSPACE_ID
    assert query.ticket_id == _TICKET_ID
    assert query.agent_run_id == _AGENT_RUN_ID
    assert query.agent_run_attempt_id == _ATTEMPT_ID


def test_requires_uuid_identifiers() -> None:
    with pytest.raises(
        TypeError,
        match="must be UUID values",
    ):
        AttemptLLMInvocationQuery(
            workspace_id="workspace",  # type: ignore[arg-type]
            ticket_id=_TICKET_ID,
            agent_run_id=_AGENT_RUN_ID,
            agent_run_attempt_id=_ATTEMPT_ID,
        )


def test_query_is_immutable() -> None:
    query = AttemptLLMInvocationQuery(
        workspace_id=_WORKSPACE_ID,
        ticket_id=_TICKET_ID,
        agent_run_id=_AGENT_RUN_ID,
        agent_run_attempt_id=_ATTEMPT_ID,
    )

    with pytest.raises(FrozenInstanceError):
        query.workspace_id = UUID(  # type: ignore[misc]
            "50000000-0000-4000-8000-000000000005",
        )
