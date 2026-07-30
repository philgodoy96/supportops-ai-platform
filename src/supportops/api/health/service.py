"""Application readiness aggregation."""

import asyncio
from collections.abc import Awaitable, Callable

from supportops.api.health.schemas import (
    ApplicationHealthStatus,
    DependencyHealthResponse,
    ReadinessResponse,
)
from supportops.infrastructure.health import DependencyCheckResult

DependencyCheck = Callable[[], Awaitable[DependencyCheckResult]]


async def build_readiness_response(
    dependency_checks: tuple[DependencyCheck, ...],
) -> ReadinessResponse:
    """Run dependency checks concurrently and aggregate readiness."""

    results = await asyncio.gather(*(check() for check in dependency_checks))

    dependencies = {
        result.dependency: DependencyHealthResponse(
            status=result.status,
            detail=result.detail,
        )
        for result in results
    }

    application_status = (
        ApplicationHealthStatus.HEALTHY
        if all(result.is_healthy for result in results)
        else ApplicationHealthStatus.UNHEALTHY
    )

    return ReadinessResponse(
        status=application_status,
        dependencies=dependencies,
    )
