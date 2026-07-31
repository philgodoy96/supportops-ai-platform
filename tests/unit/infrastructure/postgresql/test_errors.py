"""Unit tests for PostgreSQL exception inspection."""

from supportops.infrastructure.postgresql.errors import (
    get_constraint_name,
)


class ConstraintError(Exception):
    """Test exception exposing a database constraint name."""

    def __init__(
        self,
        constraint_name: str,
    ) -> None:
        super().__init__("constraint failure")
        self.constraint_name = constraint_name


class WrappedDatabaseError(Exception):
    """Test exception exposing an original provider error."""

    def __init__(
        self,
        original_error: BaseException,
    ) -> None:
        super().__init__("wrapped database failure")
        self.orig = original_error


def test_get_constraint_name_reads_direct_attribute() -> None:
    error = ConstraintError("uq_workspaces_slug")

    assert get_constraint_name(error) == "uq_workspaces_slug"


def test_get_constraint_name_reads_wrapped_original_error() -> None:
    error = WrappedDatabaseError(
        ConstraintError("uq_workspaces_slug"),
    )

    assert get_constraint_name(error) == "uq_workspaces_slug"


def test_get_constraint_name_reads_exception_cause() -> None:
    provider_error = ConstraintError(
        "uq_workspaces_slug",
    )
    wrapper = RuntimeError("wrapper")
    wrapper.__cause__ = provider_error

    assert get_constraint_name(wrapper) == ("uq_workspaces_slug")


def test_get_constraint_name_returns_none_without_constraint() -> None:
    assert (
        get_constraint_name(
            RuntimeError("database failure"),
        )
        is None
    )
