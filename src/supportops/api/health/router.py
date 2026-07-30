"""Operational health HTTP routes."""

from functools import partial
from typing import Annotated

from fastapi import APIRouter, Depends, Request, Response, status

from supportops.api.health.schemas import (
    ApplicationHealthStatus,
    LivenessResponse,
    ReadinessResponse,
)
from supportops.api.health.service import build_readiness_response
from supportops.api.state import ApplicationState
from supportops.infrastructure.postgresql import check_postgresql_health
from supportops.infrastructure.qdrant import check_qdrant_health

router = APIRouter(prefix="/health", tags=["health"])


def get_application_state(request: Request) -> ApplicationState:
    """Return resources owned by the application lifecycle."""

    state: ApplicationState = request.app.state.supportops
    return state


ApplicationStateDependency = Annotated[
    ApplicationState,
    Depends(get_application_state),
]


@router.get(
    "/live",
    response_model=LivenessResponse,
    status_code=status.HTTP_200_OK,
    summary="Check application liveness",
)
async def get_liveness() -> LivenessResponse:
    """Verify that the application process can respond."""

    return LivenessResponse(status=ApplicationHealthStatus.HEALTHY)


@router.get(
    "/ready",
    response_model=ReadinessResponse,
    responses={
        status.HTTP_503_SERVICE_UNAVAILABLE: {
            "model": ReadinessResponse,
            "description": "One or more required dependencies are unavailable.",
        },
    },
    summary="Check application readiness",
)
async def get_readiness(
    response: Response,
    application_state: ApplicationStateDependency,
) -> ReadinessResponse:
    """Verify that required infrastructure dependencies are available."""

    timeout_seconds = application_state.settings.dependency_health_timeout_seconds

    readiness = await build_readiness_response(
        (
            partial(
                check_postgresql_health,
                application_state.postgresql_engine,
                timeout_seconds,
            ),
            partial(
                check_qdrant_health,
                application_state.qdrant_client,
                timeout_seconds,
            ),
        )
    )

    if readiness.status is ApplicationHealthStatus.UNHEALTHY:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE

    return readiness
