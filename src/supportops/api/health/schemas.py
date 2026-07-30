"""Response schemas for operational health endpoints."""

from enum import StrEnum

from pydantic import BaseModel

from supportops.infrastructure.health import DependencyStatus


class ApplicationHealthStatus(StrEnum):
    """Supported aggregate application health states."""

    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"


class LivenessResponse(BaseModel):
    """Response returned by the liveness endpoint."""

    status: ApplicationHealthStatus


class DependencyHealthResponse(BaseModel):
    """Sanitized health state for one required dependency."""

    status: DependencyStatus
    detail: str | None = None


class ReadinessResponse(BaseModel):
    """Aggregated application readiness response."""

    status: ApplicationHealthStatus
    dependencies: dict[str, DependencyHealthResponse]
