"""Top-level API router composition."""

from fastapi import APIRouter

from supportops.api.health.router import router as health_router
from supportops.modules.agent_runs.api.router import router as agent_runs_router
from supportops.modules.ticket_classifications.api.router import (
    router as ticket_classifications_router,
)
from supportops.modules.tickets.api.router import router as tickets_router
from supportops.modules.workspaces.api.router import router as workspaces_router

api_router = APIRouter()
api_router.include_router(health_router)
api_router.include_router(workspaces_router, prefix="/api/v1")
api_router.include_router(tickets_router, prefix="/api/v1")
api_router.include_router(agent_runs_router, prefix="/api/v1")
api_router.include_router(
    ticket_classifications_router,
    prefix="/api/v1",
)
