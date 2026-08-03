"""Deterministic checkpoint identity for the human-approved graph."""

from dataclasses import dataclass
from uuid import UUID

from supportops.agent_graph.domain.human_approved_state import (
    HUMAN_APPROVED_SUPPORT_GRAPH_VERSION,
    HUMAN_APPROVED_SUPPORT_WORKFLOW_NAME,
    HUMAN_APPROVED_SUPPORT_WORKFLOW_VERSION,
)

_SEPARATOR = ":"


@dataclass(frozen=True, slots=True)
class HumanApprovedSupportGraphIdentity:
    """Framework-independent checkpoint identity."""

    thread_id: str
    checkpoint_namespace: str


def derive_human_approved_support_graph_identity(
    agent_run_id: UUID,
) -> HumanApprovedSupportGraphIdentity:
    """Derive one stable versioned thread identity."""

    if not isinstance(agent_run_id, UUID):
        raise TypeError("agent_run_id must be a UUID.")

    thread_id = _SEPARATOR.join(
        (
            HUMAN_APPROVED_SUPPORT_WORKFLOW_NAME,
            HUMAN_APPROVED_SUPPORT_WORKFLOW_VERSION,
            HUMAN_APPROVED_SUPPORT_GRAPH_VERSION,
            str(agent_run_id),
        ),
    )
    return HumanApprovedSupportGraphIdentity(
        thread_id=thread_id,
        checkpoint_namespace="",
    )
