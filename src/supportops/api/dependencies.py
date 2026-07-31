"""Shared FastAPI dependency construction."""

from collections.abc import AsyncIterator

from fastapi import Request
from sqlalchemy.ext.asyncio import AsyncSession

from supportops.api.state import ApplicationState


def get_application_state(
    request: Request,
) -> ApplicationState:
    """Return typed process-owned application resources."""

    state: ApplicationState = request.app.state.supportops

    return state


async def get_postgresql_session(
    request: Request,
) -> AsyncIterator[AsyncSession]:
    """Provide one async SQLAlchemy session for an HTTP request."""

    state = get_application_state(request)

    async with state.postgresql_session_factory() as session:
        yield session
