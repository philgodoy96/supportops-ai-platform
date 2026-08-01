"""Workspace-scoped ticket-classification inspection use cases."""

from collections.abc import Sequence
from datetime import datetime
from uuid import UUID

from supportops.modules.agent_runs.application.errors import (
    AgentRunNotFoundError,
)
from supportops.modules.agent_runs.domain.repositories import (
    AgentRunQueryRepository,
)
from supportops.modules.ticket_classifications.application.errors import (
    TicketClassificationNotFoundError,
)
from supportops.modules.ticket_classifications.domain.inspection import (
    AgentRunClassificationReference,
    LLMInvocationInspection,
)
from supportops.modules.ticket_classifications.domain.models import (
    TicketClassification,
)
from supportops.modules.ticket_classifications.domain.repositories import (
    TicketClassificationQueryRepository,
)
from supportops.modules.tickets.application.errors import (
    TicketNotFoundError,
)
from supportops.modules.tickets.domain.repositories import (
    TicketRepository,
)


class GetTicketClassification:
    """Retrieve one accepted classification through its workspace boundary."""

    def __init__(
        self,
        *,
        repository: TicketClassificationQueryRepository,
    ) -> None:
        self._repository = repository

    async def execute(
        self,
        *,
        workspace_id: UUID,
        classification_id: UUID,
    ) -> TicketClassification:
        """Return the scoped classification or raise stable not-found."""

        classification = await self._repository.get(
            workspace_id=workspace_id,
            classification_id=classification_id,
        )
        if classification is None:
            raise TicketClassificationNotFoundError(
                "Ticket classification was not found.",
            )

        return classification


class ListTicketClassifications:
    """List accepted classifications for one workspace-owned ticket."""

    def __init__(
        self,
        *,
        ticket_repository: TicketRepository,
        classification_repository: (TicketClassificationQueryRepository),
    ) -> None:
        self._ticket_repository = ticket_repository
        self._classification_repository = classification_repository

    async def execute(
        self,
        *,
        workspace_id: UUID,
        ticket_id: UUID,
        limit: int,
        after_created_at: datetime | None = None,
        after_classification_id: UUID | None = None,
    ) -> Sequence[TicketClassification]:
        """Return a deterministic page after validating ticket ownership."""

        ticket = await self._ticket_repository.get(
            workspace_id,
            ticket_id,
        )
        if ticket is None:
            raise TicketNotFoundError(
                "Ticket was not found.",
            )

        return await self._classification_repository.list_by_ticket(
            workspace_id=workspace_id,
            ticket_id=ticket.id,
            limit=limit,
            after_created_at=after_created_at,
            after_classification_id=(after_classification_id),
        )


class ListAgentRunLLMInvocations:
    """List safe logical invocation history for one scoped AgentRun."""

    def __init__(
        self,
        *,
        agent_run_repository: AgentRunQueryRepository,
        classification_repository: (TicketClassificationQueryRepository),
    ) -> None:
        self._agent_run_repository = agent_run_repository
        self._classification_repository = classification_repository

    async def execute(
        self,
        *,
        workspace_id: UUID,
        agent_run_id: UUID,
    ) -> Sequence[LLMInvocationInspection]:
        """Return invocation history after validating AgentRun ownership."""

        agent_run = await self._agent_run_repository.get(
            workspace_id=workspace_id,
            agent_run_id=agent_run_id,
        )
        if agent_run is None:
            raise AgentRunNotFoundError(
                "AgentRun was not found.",
            )

        return await self._classification_repository.list_invocations_by_agent_run(
            workspace_id=workspace_id,
            agent_run_id=agent_run.id,
        )


class GetAgentRunClassificationReference:
    """Retrieve an optional classification reference for one AgentRun."""

    def __init__(
        self,
        *,
        agent_run_repository: AgentRunQueryRepository,
        classification_repository: (TicketClassificationQueryRepository),
    ) -> None:
        self._agent_run_repository = agent_run_repository
        self._classification_repository = classification_repository

    async def execute(
        self,
        *,
        workspace_id: UUID,
        agent_run_id: UUID,
    ) -> AgentRunClassificationReference | None:
        """Return the optional reference after validating run ownership."""

        agent_run = await self._agent_run_repository.get(
            workspace_id=workspace_id,
            agent_run_id=agent_run_id,
        )
        if agent_run is None:
            raise AgentRunNotFoundError(
                "AgentRun was not found.",
            )

        return await self._classification_repository.get_reference_by_agent_run_id(
            workspace_id=workspace_id,
            agent_run_id=agent_run.id,
        )
