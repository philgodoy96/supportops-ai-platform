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
    """Framework-independent identity used for graph checkpoint access."""

    thread_id: str
    checkpoint_namespace: str


def derive_controlled_support_graph_identity(
    agent_run_id: UUID,
) -> ControlledSupportGraphIdentity:
    """Derive stable checkpoint identity from an authoritative AgentRun."""

    checkpoint_namespace = _CHECKPOINT_NAMESPACE_SEPARATOR.join(
        (
            CONTROLLED_SUPPORT_WORKFLOW_NAME,
            CONTROLLED_SUPPORT_WORKFLOW_VERSION,
            CONTROLLED_SUPPORT_GRAPH_VERSION,
        )
    )

    return ControlledSupportGraphIdentity(
        thread_id=str(agent_run_id),
        checkpoint_namespace=checkpoint_namespace,
    )
