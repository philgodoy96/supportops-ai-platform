"""Unit tests for controlled support checkpoint identity."""

from uuid import UUID

from supportops.agent_graph.domain.identity import (
    derive_controlled_support_graph_identity,
)

AGENT_RUN_ID = UUID("11111111-1111-4111-8111-111111111111")
OTHER_AGENT_RUN_ID = UUID("22222222-2222-4222-8222-222222222222")


def test_graph_identity_uses_agent_run_id_as_thread_id() -> None:
    identity = derive_controlled_support_graph_identity(AGENT_RUN_ID)

    assert identity.thread_id == str(AGENT_RUN_ID)
    assert len(identity.thread_id) < 255


def test_graph_identity_uses_versioned_checkpoint_namespace() -> None:
    identity = derive_controlled_support_graph_identity(AGENT_RUN_ID)

    assert identity.checkpoint_namespace == "ticket-processing:controlled-support-v1:graph-v1"


def test_graph_identity_is_deterministic_for_same_agent_run() -> None:
    first_identity = derive_controlled_support_graph_identity(AGENT_RUN_ID)
    second_identity = derive_controlled_support_graph_identity(AGENT_RUN_ID)

    assert first_identity == second_identity


def test_graph_identity_isolated_between_agent_runs() -> None:
    first_identity = derive_controlled_support_graph_identity(AGENT_RUN_ID)
    second_identity = derive_controlled_support_graph_identity(OTHER_AGENT_RUN_ID)

    assert first_identity.thread_id != second_identity.thread_id
    assert first_identity.checkpoint_namespace == second_identity.checkpoint_namespace
