"""Top-level API router composition."""

from fastapi import APIRouter

from supportops.api.health.router import router as health_router

api_router = APIRouter()
api_router.include_router(health_router)
