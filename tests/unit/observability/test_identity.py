"""Unit tests for deterministic observability identities."""

from uuid import UUID

import pytest

from supportops.observability.identity import (
    agent_run_trace_identity,
    knowledge_index_trace_identity,
    semantic_search_trace_identity,
    ticket_session_id,
)

AGENT_RUN_ID = UUID("11111111-1111-4111-8111-111111111111")
OTHER_AGENT_RUN_ID = UUID("22222222-2222-4222-8222-222222222222")
TICKET_ID = UUID("33333333-3333-4333-8333-333333333333")


def test_agent_run_identity_is_deterministic() -> None:
    first = agent_run_trace_identity(
        agent_run_id=AGENT_RUN_ID,
        ticket_id=TICKET_ID,
    )
    second = agent_run_trace_identity(
        agent_run_id=AGENT_RUN_ID,
        ticket_id=TICKET_ID,
    )

    assert first == second
    assert first.trace_seed == f"agent-run:{AGENT_RUN_ID}"
    assert first.session_id == f"ticket:{TICKET_ID}"


def test_agent_run_identity_is_stable_across_attempts_and_resume() -> None:
    initial_attempt = agent_run_trace_identity(
        agent_run_id=AGENT_RUN_ID,
        ticket_id=TICKET_ID,
    )
    resumed_attempt = agent_run_trace_identity(
        agent_run_id=AGENT_RUN_ID,
        ticket_id=TICKET_ID,
    )

    assert initial_attempt.trace_seed == resumed_attempt.trace_seed
    assert initial_attempt.session_id == resumed_attempt.session_id


def test_different_agent_runs_produce_different_trace_seeds() -> None:
    first = agent_run_trace_identity(
        agent_run_id=AGENT_RUN_ID,
        ticket_id=TICKET_ID,
    )
    second = agent_run_trace_identity(
        agent_run_id=OTHER_AGENT_RUN_ID,
        ticket_id=TICKET_ID,
    )

    assert first.trace_seed != second.trace_seed
    assert first.session_id == second.session_id


def test_ticket_session_identity_is_not_user_identity() -> None:
    identity = agent_run_trace_identity(
        agent_run_id=AGENT_RUN_ID,
        ticket_id=TICKET_ID,
    )

    assert identity.session_id == ticket_session_id(TICKET_ID)
    assert not hasattr(identity, "user_id")


def test_agent_run_tags_do_not_contain_high_cardinality_ids() -> None:
    identity = agent_run_trace_identity(
        agent_run_id=AGENT_RUN_ID,
        ticket_id=TICKET_ID,
    )

    assert identity.tags == ("supportops", "agent-run")
    assert str(AGENT_RUN_ID) not in identity.tags
    assert str(TICKET_ID) not in identity.tags


def test_semantic_search_identity_uses_server_request_id() -> None:
    identity = semantic_search_trace_identity(
        request_id="aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa",
    )

    assert identity.trace_seed == ("semantic-search:aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa")
    assert identity.session_id is None


def test_knowledge_index_identity_accepts_uuid_execution_id() -> None:
    execution_id = UUID("bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb")

    identity = knowledge_index_trace_identity(execution_id=execution_id)

    assert identity.trace_seed == f"knowledge-index:{execution_id}"
    assert identity.session_id is None


def test_knowledge_index_identity_accepts_bounded_safe_string() -> None:
    identity = knowledge_index_trace_identity(
        execution_id="index-run_2026.08.03",
    )

    assert identity.trace_seed == "knowledge-index:index-run_2026.08.03"


@pytest.mark.parametrize(
    "request_id",
    [
        "",
        " ",
        "request with spaces",
        "request/with/slashes",
        "request?with=query",
    ],
)
def test_semantic_search_identity_rejects_unsafe_request_id(
    request_id: str,
) -> None:
    with pytest.raises(ValueError, match="request_id"):
        semantic_search_trace_identity(request_id=request_id)


def test_identity_converts_to_trace_attributes() -> None:
    identity = agent_run_trace_identity(
        agent_run_id=AGENT_RUN_ID,
        ticket_id=TICKET_ID,
    )

    attributes = identity.to_trace_attributes(
        metadata={
            "workflow_name": "ticket-processing",
            "workflow_version": "human-approved-support-v1",
        }
    )

    assert attributes.trace_seed == identity.trace_seed
    assert attributes.name == "agent-run"
    assert attributes.session_id == identity.session_id
    assert attributes.tags == identity.tags
    assert attributes.metadata["workflow_name"] == "ticket-processing"
