"""Deterministic checkpoint identity for the controlled support graph."""

from dataclasses import dataclass
from uuid import UUID

from supportops.agent_graph.domain.state import (
    CONTROLLED_SUPPORT_GRAPH_VERSION,
    CONTROLLED_SUPPORT_WORKFLOW_NAME,
    CONTROLLED_SUPPORT_WORKFLOW_VERSION,
)

_CHECKPOINT_NAMESPACE_SEPARATOR = ":"


@dataclass(frozen=True, slots=True)
class ControlledSupportGraphIdentity:
    """Framework-independent identity used for graph checkpoint access.

    LangGraph reserves ``checkpoint_ns`` for subgraph routing on the root
    graph, so version isolation lives in ``thread_id`` and the root
    ``checkpoint_namespace`` is always empty.
    """

    thread_id: str
    checkpoint_namespace: str


def derive_controlled_support_graph_identity(
    agent_run_id: UUID,
) -> ControlledSupportGraphIdentity:
    """Derive stable checkpoint identity from an authoritative AgentRun."""

    version_key = _CHECKPOINT_NAMESPACE_SEPARATOR.join(
        (
            CONTROLLED_SUPPORT_WORKFLOW_NAME,
            CONTROLLED_SUPPORT_WORKFLOW_VERSION,
            CONTROLLED_SUPPORT_GRAPH_VERSION,
        )
    )
    thread_id = _CHECKPOINT_NAMESPACE_SEPARATOR.join(
        (
            version_key,
            str(agent_run_id),
        )
    )

    return ControlledSupportGraphIdentity(
        thread_id=thread_id,
        checkpoint_namespace="",
    )
