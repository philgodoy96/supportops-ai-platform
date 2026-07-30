"""FastAPI application lifecycle management."""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from supportops.api.state import ApplicationState
from supportops.core.logging import configure_logging
from supportops.core.settings import Settings
from supportops.infrastructure.postgresql import (
    create_postgresql_engine,
    create_postgresql_session_factory,
    dispose_postgresql_engine,
)
from supportops.infrastructure.qdrant import (
    close_qdrant_client,
    create_qdrant_client,
)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def application_lifespan(
    app: FastAPI,
    *,
    settings: Settings,
) -> AsyncIterator[None]:
    """Create and release resources owned by the API process."""

    configure_logging(
        environment=settings.environment,
        log_level=settings.log_level,
    )

    postgresql_engine = create_postgresql_engine(settings)
    postgresql_session_factory = create_postgresql_session_factory(
        postgresql_engine,
    )
    qdrant_client = create_qdrant_client(settings)

    app.state.supportops = ApplicationState(
        settings=settings,
        postgresql_engine=postgresql_engine,
        postgresql_session_factory=postgresql_session_factory,
        qdrant_client=qdrant_client,
    )

    logger.info(
        "application_started",
        extra={
            "application_name": settings.application_name,
            "application_version": settings.application_version,
        },
    )

    try:
        yield
    finally:
        logger.info("application_stopping")

        try:
            await close_qdrant_client(qdrant_client)
        except Exception:
            logger.exception(
                "qdrant_client_close_failed",
                extra={"dependency": "qdrant"},
            )

        try:
            await dispose_postgresql_engine(postgresql_engine)
        except Exception:
            logger.exception(
                "postgresql_engine_disposal_failed",
                extra={"dependency": "postgresql"},
            )

        logger.info("application_stopped")
