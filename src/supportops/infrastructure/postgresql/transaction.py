"""SQLAlchemy transaction boundary adapter."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from sqlalchemy.ext.asyncio import AsyncSession


class SqlAlchemyTransactionManager:
    """Manage atomic use-case transactions with one async session."""

    def __init__(
        self,
        session: AsyncSession,
    ) -> None:
        self._session = session

    @asynccontextmanager
    async def transaction(
        self,
    ) -> AsyncIterator[None]:
        """Commit on success and roll back on failure."""

        async with self._session.begin():
            yield
