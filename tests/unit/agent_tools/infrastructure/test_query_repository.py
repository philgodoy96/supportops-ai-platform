"""Unit tests for tool-call query repository protocol surface."""

from supportops.agent_tools.infrastructure.query_repository import (
    SqlAlchemyAgentToolCallQueryRepository,
)


def test_query_repository_exposes_proposal_and_sensitive_lookups() -> None:
    assert callable(
        getattr(
            SqlAlchemyAgentToolCallQueryRepository,
            "get_by_proposal_attempt_sequence",
            None,
        )
    )
    assert callable(
        getattr(
            SqlAlchemyAgentToolCallQueryRepository,
            "get_sensitive_by_identity",
            None,
        )
    )
