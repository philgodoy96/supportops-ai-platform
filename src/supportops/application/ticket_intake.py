"""Transactional ticket intake and durable processing scheduling."""

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from uuid import UUID

from supportops.core.transactions import TransactionManager
from supportops.modules.agent_runs.domain.models import AgentRun
from supportops.modules.agent_runs.domain.repositories import (
    AgentRunRepository,
)
from supportops.modules.tickets.application.errors import (
    TicketExternalReferenceConflictApplicationError,
)
from supportops.modules.tickets.domain.models import Ticket
from supportops.modules.tickets.domain.repositories import (
    TicketExternalReferenceConflictError,
    TicketRepository,
)
from supportops.modules.workspaces.application.errors import (
    WorkspaceNotFoundError,
)
from supportops.modules.workspaces.domain.repositories import (
    WorkspaceRepository,
)

UtcNowProvider = Callable[[], datetime]


@dataclass(frozen=True, slots=True)
class TicketIntakeResult:
    """Ticket intake result with its durable processing reference."""

    ticket: Ticket
    processing_run: AgentRun


class CreateTicketWithInitialRun:
    """Create a ticket and its initial AgentRun atomically."""

    def __init__(
        self,
        *,
        workspace_repository: WorkspaceRepository,
        ticket_repository: TicketRepository,
        agent_run_repository: AgentRunRepository,
        transaction_manager: TransactionManager,
        workflow_version: str,
        max_attempts: int,
        utc_now: UtcNowProvider | None = None,
    ) -> None:
        if max_attempts < 1:
            raise ValueError("max_attempts must be at least one.")

        self._workspace_repository = workspace_repository
        self._ticket_repository = ticket_repository
        self._agent_run_repository = agent_run_repository
        self._transaction_manager = transaction_manager
        self._workflow_version = workflow_version
        self._max_attempts = max_attempts
        self._utc_now = utc_now or _utc_now

    async def execute(
        self,
        *,
        workspace_id: UUID,
        subject: str,
        description: str,
        ingestion_request_id: UUID,
        correlation_id: UUID,
        external_reference: str | None = None,
    ) -> TicketIntakeResult:
        """Persist a ticket and its initial durable run in one transaction."""

        now = self._utc_now()

        ticket = Ticket.create(
            workspace_id=workspace_id,
            subject=subject,
            description=description,
            external_reference=external_reference,
            ingestion_request_id=ingestion_request_id,
            correlation_id=correlation_id,
            now=now,
        )
        processing_run = AgentRun.create_initial(
            workspace_id=workspace_id,
            ticket_id=ticket.id,
            ingestion_request_id=ingestion_request_id,
            correlation_id=correlation_id,
            workflow_version=self._workflow_version,
            max_attempts=self._max_attempts,
            now=now,
        )

        try:
            async with self._transaction_manager.transaction():
                if not await self._workspace_repository.exists(
                    workspace_id,
                ):
                    raise WorkspaceNotFoundError(
                        "Workspace was not found.",
                    )

                await self._ticket_repository.add(ticket)
                await self._agent_run_repository.add(processing_run)
        except TicketExternalReferenceConflictError as error:
            raise TicketExternalReferenceConflictApplicationError(
                "Ticket external reference already exists in the workspace.",
            ) from error

        return TicketIntakeResult(
            ticket=ticket,
            processing_run=processing_run,
        )


def _utc_now() -> datetime:
    return datetime.now(UTC)
