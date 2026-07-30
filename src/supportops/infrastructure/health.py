"""Shared contracts for infrastructure dependency health checks."""

from dataclasses import dataclass
from enum import StrEnum


class DependencyStatus(StrEnum):
    """Supported dependency health states."""

    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"


@dataclass(frozen=True, slots=True)
class DependencyCheckResult:
    """Sanitized result returned by an infrastructure health check."""

    dependency: str
    status: DependencyStatus
    detail: str | None = None

    @property
    def is_healthy(self) -> bool:
        """Return whether the dependency check succeeded."""

        return self.status is DependencyStatus.HEALTHY
