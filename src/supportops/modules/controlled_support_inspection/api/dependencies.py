"""FastAPI dependencies for controlled-support inspection."""

from typing import Annotated

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from supportops.api.dependencies import (
    get_postgresql_session,
)
from supportops.infrastructure.postgresql.transaction import (
    SqlAlchemyTransactionManager,
)
from supportops.modules.controlled_support_inspection.application.services import (
    GetControlledSupportInspection,
)
from supportops.modules.controlled_support_inspection.infrastructure.repository import (
    SqlAlchemyControlledSupportInspectionRepository,
)

PostgresqlSessionDependency = Annotated[
    AsyncSession,
    Depends(get_postgresql_session),
]


def get_controlled_support_inspection(
    session: PostgresqlSessionDependency,
) -> GetControlledSupportInspection:
    """Construct the controlled-support inspection use case."""

    return GetControlledSupportInspection(
        repository=(SqlAlchemyControlledSupportInspectionRepository(session)),
        transaction_manager=SqlAlchemyTransactionManager(session),
    )


GetControlledSupportInspectionDependency = Annotated[
    GetControlledSupportInspection,
    Depends(get_controlled_support_inspection),
]
