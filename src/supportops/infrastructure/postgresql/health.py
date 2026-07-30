"""Bounded PostgreSQL connectivity checks."""

import asyncio

from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncEngine

from supportops.infrastructure.health import (
    DependencyCheckResult,
    DependencyStatus,
)

POSTGRESQL_DEPENDENCY_NAME = "postgresql"


async def check_postgresql_health(
    engine: AsyncEngine,
    timeout_seconds: float,
) -> DependencyCheckResult:
    """Check PostgreSQL connectivity within a bounded timeout."""

    try:
        async with asyncio.timeout(timeout_seconds):
            async with engine.connect() as connection:
                await connection.execute(text("SELECT 1"))
    except TimeoutError:
        return DependencyCheckResult(
            dependency=POSTGRESQL_DEPENDENCY_NAME,
            status=DependencyStatus.UNHEALTHY,
            detail="connectivity check timed out",
        )
    except SQLAlchemyError:
        return DependencyCheckResult(
            dependency=POSTGRESQL_DEPENDENCY_NAME,
            status=DependencyStatus.UNHEALTHY,
            detail="connectivity check failed",
        )

    return DependencyCheckResult(
        dependency=POSTGRESQL_DEPENDENCY_NAME,
        status=DependencyStatus.HEALTHY,
    )
