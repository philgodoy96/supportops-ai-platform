"""Unit tests for deterministic SQLAlchemy model registration."""

from supportops.infrastructure.postgresql.base import Base
from supportops.infrastructure.postgresql.model_registry import (
    register_persistence_models,
)


def test_register_persistence_models_populates_shared_metadata() -> None:
    register_persistence_models()

    assert set(Base.metadata.tables) == {
        "workspaces",
        "tickets",
    }
