"""Unit tests for workspace domain entities."""

from datetime import UTC, datetime, timedelta, timezone
from re import escape
from uuid import UUID

import pytest

from supportops.modules.workspaces.domain.models import Workspace


def test_workspace_create_normalizes_name_and_assigns_identifiers() -> None:
    now = datetime(2026, 7, 31, 12, 0, tzinfo=UTC)

    workspace = Workspace.create(
        name="  Platform Support  ",
        slug="platform-support",
        now=now,
    )

    assert workspace.name == "Platform Support"
    assert workspace.slug == "platform-support"
    assert workspace.id.version == 4
    assert workspace.created_at == now
    assert workspace.updated_at == now


@pytest.mark.parametrize(
    ("name", "expected_message"),
    [
        ("", "Workspace name is required."),
        ("   ", "Workspace name is required."),
        (
            "a" * 121,
            "Workspace name exceeds the maximum length.",
        ),
    ],
)
def test_workspace_create_rejects_invalid_name(
    name: str,
    expected_message: str,
) -> None:
    with pytest.raises(ValueError, match=expected_message):
        Workspace.create(
            name=name,
            slug="platform-support",
        )


@pytest.mark.parametrize(
    "slug",
    [
        "ab",
        "a" * 64,
        "Platform-Support",
        "platform_support",
        "-platform-support",
        "platform-support-",
        "platform--support",
        " platform-support ",
    ],
)
def test_workspace_create_rejects_noncanonical_slug(
    slug: str,
) -> None:
    with pytest.raises(ValueError):
        Workspace.create(
            name="Platform Support",
            slug=slug,
        )


def test_workspace_rejects_non_utc_timestamp() -> None:
    non_utc = datetime(
        2026,
        7,
        31,
        9,
        0,
        tzinfo=timezone(timedelta(hours=-3)),
    )

    with pytest.raises(
        ValueError,
        match=escape("created_at must be a UTC-aware timestamp."),
    ):
        Workspace(
            id=UUID("50617047-0dbd-4704-b44f-c22da75b0595"),
            name="Platform Support",
            slug="platform-support",
            created_at=non_utc,
            updated_at=non_utc,
        )


def test_workspace_rejects_updated_at_before_created_at() -> None:
    created_at = datetime(2026, 7, 31, 12, 0, tzinfo=UTC)
    updated_at = created_at - timedelta(seconds=1)

    with pytest.raises(
        ValueError,
        match=escape("updated_at must not be earlier than created_at."),
    ):
        Workspace(
            id=UUID("50617047-0dbd-4704-b44f-c22da75b0595"),
            name="Platform Support",
            slug="platform-support",
            created_at=created_at,
            updated_at=updated_at,
        )
