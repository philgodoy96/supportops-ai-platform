"""Unit tests for deterministic SQLAlchemy model registration."""

from supportops.infrastructure.postgresql.base import Base
from supportops.infrastructure.postgresql.model_registry import (
    register_persistence_models,
)


def test_register_persistence_models_populates_shared_metadata() -> None:
    register_persistence_models()

    assert set(Base.metadata.tables) == {
        "workspaces",
        "knowledge_documents",
        "knowledge_document_versions",
        "knowledge_document_chunks",
        "tickets",
        "agent_runs",
        "agent_run_attempts",
        "llm_invocations",
        "ticket_classifications",
        "agent_tool_calls",
        "support_recommendations",
        "support_recommendation_citations",
    }
