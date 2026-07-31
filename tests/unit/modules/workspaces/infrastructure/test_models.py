"""Unit tests for the workspace persistence model."""

from datetime import UTC, datetime
from typing import cast
from uuid import UUID

from sqlalchemy import CheckConstraint, Table, UniqueConstraint

from supportops.modules.workspaces.domain.models import Workspace
from supportops.modules.workspaces.infrastructure.models import (
    WorkspaceRecord,
)


def test_workspace_record_round_trip_preserves_domain_values() -> None:
    workspace = Workspace(
        id=UUID("50617047-0dbd-4704-b44f-c22da75b0595"),
        name="Platform Support",
        slug="platform-support",
        created_at=datetime(2026, 7, 31, 12, 0, tzinfo=UTC),
        updated_at=datetime(2026, 7, 31, 12, 0, tzinfo=UTC),
    )

    record = WorkspaceRecord.from_domain(workspace)

    assert record.to_domain() == workspace


def test_workspace_table_declares_expected_constraints() -> None:
    table = cast(Table, WorkspaceRecord.__table__)
    constraint_names = {
        constraint.name
        for constraint in table.constraints
        if isinstance(
            constraint,
            (CheckConstraint, UniqueConstraint),
        )
    }

    assert {
        "ck_workspaces_workspace_name_trimmed",
        "ck_workspaces_workspace_name_length",
        "ck_workspaces_workspace_slug_length",
        "ck_workspaces_workspace_slug_format",
        "ck_workspaces_workspace_timestamp_order",
        "uq_workspaces_slug",
    }.issubset(constraint_names)


def test_workspace_table_uses_nonnullable_business_columns() -> None:
    table = cast(Table, WorkspaceRecord.__table__)

    assert table.c.id.primary_key
    assert not table.c.name.nullable
    assert not table.c.slug.nullable
    assert not table.c.created_at.nullable
    assert not table.c.updated_at.nullable
