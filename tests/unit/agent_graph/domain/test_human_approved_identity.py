"""Unit tests for human-approved checkpoint identity."""

from typing import Any, cast
from uuid import uuid4

import pytest

from supportops.agent_graph.domain.human_approved_identity import (
    derive_human_approved_support_graph_identity,
)


def test_identity_is_stable_and_versioned() -> None:
    agent_run_id = uuid4()

    first = derive_human_approved_support_graph_identity(
        agent_run_id,
    )
    second = derive_human_approved_support_graph_identity(
        agent_run_id,
    )

    assert first == second
    assert first.thread_id == (
        f"ticket-processing:human-approved-support-v1:graph-v1:{agent_run_id}"
    )
    assert first.checkpoint_namespace == ""


def test_identity_changes_with_agent_run() -> None:
    first = derive_human_approved_support_graph_identity(
        uuid4(),
    )
    second = derive_human_approved_support_graph_identity(
        uuid4(),
    )

    assert first.thread_id != second.thread_id


def test_identity_rejects_non_uuid() -> None:
    with pytest.raises(TypeError, match="UUID"):
        derive_human_approved_support_graph_identity(
            cast(Any, "not-a-uuid"),
        )
