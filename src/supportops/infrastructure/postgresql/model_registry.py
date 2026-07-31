"""Deterministic registration of SQLAlchemy persistence models."""


def register_persistence_models() -> None:
    """Import all persistence records into the shared SQLAlchemy metadata."""

    from supportops.modules.tickets.infrastructure.models import (
        TicketRecord,
    )
    from supportops.modules.workspaces.infrastructure.models import (
        WorkspaceRecord,
    )

    _ = (
        WorkspaceRecord,
        TicketRecord,
    )
