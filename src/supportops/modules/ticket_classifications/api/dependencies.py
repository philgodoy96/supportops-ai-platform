"""FastAPI dependencies for ticket-classification inspection."""

from typing import Annotated

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from supportops.api.dependencies import get_postgresql_session
from supportops.modules.ticket_classifications.application.services import (
    GetTicketClassification,
    ListTicketClassifications,
)
from supportops.modules.ticket_classifications.infrastructure.repository import (
    SqlAlchemyTicketClassificationQueryRepository,
)
from supportops.modules.tickets.infrastructure.repository import (
    SqlAlchemyTicketRepository,
)

PostgresqlSessionDependency = Annotated[
    AsyncSession,
    Depends(get_postgresql_session),
]


def get_get_ticket_classification(
    session: PostgresqlSessionDependency,
) -> GetTicketClassification:
    """Construct the classification-detail inspection use case."""

    return GetTicketClassification(
        repository=(
            SqlAlchemyTicketClassificationQueryRepository(
                session,
            )
        ),
    )


def get_list_ticket_classifications(
    session: PostgresqlSessionDependency,
) -> ListTicketClassifications:
    """Construct the ticket-scoped classification-list use case."""

    return ListTicketClassifications(
        ticket_repository=SqlAlchemyTicketRepository(
            session,
        ),
        classification_repository=(
            SqlAlchemyTicketClassificationQueryRepository(
                session,
            )
        ),
    )


GetTicketClassificationDependency = Annotated[
    GetTicketClassification,
    Depends(get_get_ticket_classification),
]

ListTicketClassificationsDependency = Annotated[
    ListTicketClassifications,
    Depends(get_list_ticket_classifications),
]
