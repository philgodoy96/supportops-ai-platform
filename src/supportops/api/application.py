"""FastAPI application factory."""

from fastapi import FastAPI

from supportops.api.lifespan import application_lifespan
from supportops.api.router import api_router
from supportops.core.settings import Settings, get_settings


def create_application(settings: Settings | None = None) -> FastAPI:
    """Create and configure the FastAPI application."""

    resolved_settings = settings or get_settings()

    app = FastAPI(
        title=resolved_settings.application_name,
        version=resolved_settings.application_version,
        description=(
            "Production-minded support operations platform foundation for controlled AI workflows."
        ),
        lifespan=lambda application: application_lifespan(
            application,
            settings=resolved_settings,
        ),
    )

    app.include_router(api_router)

    return app
