"""Support ticket HTTP request and response schemas."""

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator

from supportops.modules.agent_runs.domain.models import AgentRun
from supportops.modules.tickets.domain.models import Ticket


class TicketCreateRequest(BaseModel):
    """Payload accepted when creating a support ticket."""

    model_config = ConfigDict(extra="forbid")

    subject: str = Field(
        min_length=1,
        max_length=200,
    )
    description: str = Field(
        min_length=1,
        max_length=20_000,
    )
    external_reference: str | None = Field(
        default=None,
        min_length=1,
        max_length=128,
    )

    @field_validator(
        "subject",
        "description",
        mode="before",
    )
    @classmethod
    def normalize_human_authored_text(
        cls,
        value: object,
    ) -> object:
        """Trim surrounding whitespace from human-authored text."""

        if isinstance(value, str):
            return value.strip()

        return value

    @field_validator("external_reference")
    @classmethod
    def reject_noncanonical_external_reference(
        cls,
        value: str | None,
    ) -> str | None:
        """Preserve upstream references without silent transformation."""

        if value is not None and value != value.strip():
            raise ValueError(
                "External reference must not contain surrounding whitespace.",
            )

        return value


class TicketResponse(BaseModel):
    """Stable support ticket representation."""

    id: UUID
    workspace_id: UUID
    subject: str
    description: str
    status: str
    external_reference: str | None
    ingestion_request_id: UUID
    correlation_id: UUID
    created_at: datetime
    updated_at: datetime

    @classmethod
    def from_domain(
        cls,
        ticket: Ticket,
    ) -> "TicketResponse":
        """Create an API response from a ticket entity."""

        return cls(
            id=ticket.id,
            workspace_id=ticket.workspace_id,
            subject=ticket.subject,
            description=ticket.description,
            status=ticket.status.value,
            external_reference=ticket.external_reference,
            ingestion_request_id=ticket.ingestion_request_id,
            correlation_id=ticket.correlation_id,
            created_at=ticket.created_at,
            updated_at=ticket.updated_at,
        )


class TicketProcessingRunResponse(BaseModel):
    """Initial processing reference returned with a created ticket."""

    id: UUID
    status: str
    workflow_name: str
    workflow_version: str

    @classmethod
    def from_domain(
        cls,
        agent_run: AgentRun,
    ) -> "TicketProcessingRunResponse":
        """Create a processing reference from an AgentRun entity."""

        return cls(
            id=agent_run.id,
            status=agent_run.status.value,
            workflow_name=agent_run.workflow_name,
            workflow_version=agent_run.workflow_version,
        )


class TicketCreateResponse(BaseModel):
    """Created ticket and its durable processing reference."""

    ticket: TicketResponse
    processing_run: TicketProcessingRunResponse

    @classmethod
    def from_domains(
        cls,
        *,
        ticket: Ticket,
        processing_run: AgentRun,
    ) -> "TicketCreateResponse":
        """Create the ticket-intake API response."""

        return cls(
            ticket=TicketResponse.from_domain(ticket),
            processing_run=TicketProcessingRunResponse.from_domain(
                processing_run,
            ),
        )


class TicketListResponse(BaseModel):
    """One bounded page of workspace tickets."""

    items: list[TicketResponse]
    next_cursor: str | None
