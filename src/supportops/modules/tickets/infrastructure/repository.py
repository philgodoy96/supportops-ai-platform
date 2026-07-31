"""PostgreSQL repository for workspace-scoped support tickets."""

from uuid import UUID

from sqlalchemy import literal, select, tuple_
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from supportops.infrastructure.postgresql.errors import (
    get_constraint_name,
)
from supportops.modules.tickets.domain.models import Ticket
from supportops.modules.tickets.domain.repositories import (
    TicketExternalReferenceConflictError,
    TicketRepository,
)
from supportops.modules.tickets.infrastructure.models import (
    TicketRecord,
)

_TICKET_EXTERNAL_REFERENCE_CONSTRAINT = "uq_tickets_workspace_external_reference"


class SqlAlchemyTicketRepository(TicketRepository):
    """Persist tickets through an explicit workspace boundary."""

    def __init__(
        self,
        session: AsyncSession,
    ) -> None:
        self._session = session

    async def add(
        self,
        ticket: Ticket,
    ) -> None:
        """Add and flush a ticket inside the active transaction."""

        self._session.add(
            TicketRecord.from_domain(ticket),
        )

        try:
            await self._session.flush()
        except IntegrityError as error:
            if get_constraint_name(error) == _TICKET_EXTERNAL_REFERENCE_CONSTRAINT:
                raise TicketExternalReferenceConflictError(
                    "Ticket external reference already exists in the workspace.",
                ) from error

            raise

    async def get(
        self,
        workspace_id: UUID,
        ticket_id: UUID,
    ) -> Ticket | None:
        """Return a ticket only when it belongs to the workspace."""

        statement = select(TicketRecord).where(
            TicketRecord.workspace_id == workspace_id,
            TicketRecord.id == ticket_id,
        )
        result = await self._session.execute(statement)
        record = result.scalar_one_or_none()

        if record is None:
            return None

        return record.to_domain()

    async def list(
        self,
        workspace_id: UUID,
        *,
        limit: int,
        after_created_at: object | None = None,
        after_ticket_id: UUID | None = None,
    ) -> list[Ticket]:
        """List workspace tickets in deterministic descending order."""

        from datetime import datetime

        if limit < 1:
            raise ValueError("Ticket list limit must be positive.")

        if isinstance(after_created_at, datetime) != (after_ticket_id is not None):
            raise ValueError(
                "Ticket pagination position requires both timestamp and ID.",
            )

        statement = select(TicketRecord).where(
            TicketRecord.workspace_id == workspace_id,
        )

        if isinstance(after_created_at, datetime):
            assert after_ticket_id is not None

            statement = statement.where(
                tuple_(
                    TicketRecord.created_at,
                    TicketRecord.id,
                )
                < tuple_(
                    literal(after_created_at),
                    literal(after_ticket_id),
                )
            )

        statement = statement.order_by(
            TicketRecord.created_at.desc(),
            TicketRecord.id.desc(),
        ).limit(limit)

        result = await self._session.execute(statement)

        return [record.to_domain() for record in result.scalars().all()]
