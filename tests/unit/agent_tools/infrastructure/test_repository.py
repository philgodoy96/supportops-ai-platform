"""Unit tests for fenced tool-call repository protocol surface."""

from supportops.agent_tools.infrastructure.repository import (
    SqlAlchemyAgentToolCallExecutionRepository,
)


def test_repository_exposes_fenced_persistence() -> None:
    assert callable(
        getattr(
            SqlAlchemyAgentToolCallExecutionRepository,
            "persist_fenced",
            None,
        )
    )


def test_repository_exposes_approval_outcome_persistence() -> None:
    assert callable(
        getattr(
            SqlAlchemyAgentToolCallExecutionRepository,
            "get_by_id_for_update",
            None,
        )
    )
    assert callable(
        getattr(
            SqlAlchemyAgentToolCallExecutionRepository,
            "save_approval_outcome",
            None,
        )
    )
