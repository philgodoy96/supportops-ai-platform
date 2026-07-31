"""Workspace domain entities and invariants."""

import re
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from uuid import UUID, uuid4

WORKSPACE_NAME_MAX_LENGTH = 120
WORKSPACE_SLUG_MIN_LENGTH = 3
WORKSPACE_SLUG_MAX_LENGTH = 63

_WORKSPACE_SLUG_PATTERN = re.compile(
    r"^[a-z0-9]+(?:-[a-z0-9]+)*$",
)


@dataclass(frozen=True, slots=True)
class Workspace:
    """A top-level data ownership boundary for support operations."""

    id: UUID
    name: str
    slug: str
    created_at: datetime
    updated_at: datetime

    def __post_init__(self) -> None:
        _validate_name(self.name)
        _validate_slug(self.slug)
        _validate_utc_timestamp(
            self.created_at,
            field_name="created_at",
        )
        _validate_utc_timestamp(
            self.updated_at,
            field_name="updated_at",
        )

        if self.updated_at < self.created_at:
            raise ValueError(
                "updated_at must not be earlier than created_at.",
            )

    @classmethod
    def create(
        cls,
        *,
        name: str,
        slug: str,
        workspace_id: UUID | None = None,
        now: datetime | None = None,
    ) -> "Workspace":
        """Create a workspace with normalized human-authored text."""

        normalized_name = name.strip()
        created_at = now or datetime.now(UTC)

        return cls(
            id=workspace_id or uuid4(),
            name=normalized_name,
            slug=slug,
            created_at=created_at,
            updated_at=created_at,
        )


def _validate_name(name: str) -> None:
    if not name:
        raise ValueError("Workspace name is required.")

    if name != name.strip():
        raise ValueError(
            "Workspace name must not contain surrounding whitespace.",
        )

    if len(name) > WORKSPACE_NAME_MAX_LENGTH:
        raise ValueError(
            "Workspace name exceeds the maximum length.",
        )


def _validate_slug(slug: str) -> None:
    if slug != slug.strip():
        raise ValueError(
            "Workspace slug must not contain surrounding whitespace.",
        )

    if not (WORKSPACE_SLUG_MIN_LENGTH <= len(slug) <= WORKSPACE_SLUG_MAX_LENGTH):
        raise ValueError(
            "Workspace slug length is outside the allowed range.",
        )

    if _WORKSPACE_SLUG_PATTERN.fullmatch(slug) is None:
        raise ValueError(
            "Workspace slug must use canonical lowercase hyphenated format.",
        )


def _validate_utc_timestamp(
    value: datetime,
    *,
    field_name: str,
) -> None:
    if value.tzinfo is None or value.utcoffset() != timedelta(0):
        raise ValueError(f"{field_name} must be a UTC-aware timestamp.")
