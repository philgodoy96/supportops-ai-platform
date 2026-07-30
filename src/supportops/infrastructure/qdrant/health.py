"""Bounded Qdrant connectivity checks."""

import asyncio

from qdrant_client import AsyncQdrantClient
from qdrant_client.http.exceptions import (
    ResponseHandlingException,
    UnexpectedResponse,
)

from supportops.infrastructure.health import (
    DependencyCheckResult,
    DependencyStatus,
)

QDRANT_DEPENDENCY_NAME = "qdrant"


async def check_qdrant_health(
    client: AsyncQdrantClient,
    timeout_seconds: float,
) -> DependencyCheckResult:
    """Check Qdrant connectivity within a bounded timeout."""

    try:
        async with asyncio.timeout(timeout_seconds):
            await client.get_collections()
    except TimeoutError:
        return DependencyCheckResult(
            dependency=QDRANT_DEPENDENCY_NAME,
            status=DependencyStatus.UNHEALTHY,
            detail="connectivity check timed out",
        )
    except (UnexpectedResponse, ResponseHandlingException, OSError):
        return DependencyCheckResult(
            dependency=QDRANT_DEPENDENCY_NAME,
            status=DependencyStatus.UNHEALTHY,
            detail="connectivity check failed",
        )

    return DependencyCheckResult(
        dependency=QDRANT_DEPENDENCY_NAME,
        status=DependencyStatus.HEALTHY,
    )
