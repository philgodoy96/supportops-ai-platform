"""SQLAlchemy persistence model for support tickets."""

from datetime import datetime
from uuid import UUID

from sqlalchemy import (
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
    String,
    UniqueConstraint,
    Uuid,
)
from sqlalchemy.orm import Mapped, mapped_column

from supportops.infrastructure.postgresql.base import Base
from supportops.modules.tickets.domain.models import (
    Ticket,
    TicketStatus,
)


class TicketRecord(Base):
    """Persisted support ticket record."""

    id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True),
        primary_key=True,
    )
    workspace_id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey(
            "workspaces.id",
            ondelete="RESTRICT",
        ),
        nullable=False,
    )
    subject: Mapped[str] = mapped_column(
        String(200),
        nullable=False,
    )
    description: Mapped[str] = mapped_column(
        String(20_000),
        nullable=False,
    )
    status: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
    )
    external_reference: Mapped[str | None] = mapped_column(
        String(128),
        nullable=True,
    )
    ingestion_request_id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True),
        nullable=False,
    )
    correlation_id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True),
        nullable=False,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
    )

    __tablename__ = "tickets"
    __table_args__ = (
        UniqueConstraint(
            "workspace_id",
            "id",
            name="uq_tickets_workspace_id",
        ),
        UniqueConstraint(
            "workspace_id",
            "external_reference",
            name="uq_tickets_workspace_external_reference",
        ),
        CheckConstraint(
            "subject = btrim(subject)",
            name="ticket_subject_trimmed",
        ),
        CheckConstraint(
            "char_length(subject) BETWEEN 1 AND 200",
            name="ticket_subject_length",
        ),
        CheckConstraint(
            "description = btrim(description)",
            name="ticket_description_trimmed",
        ),
        CheckConstraint(
            "char_length(description) BETWEEN 1 AND 20000",
            name="ticket_description_length",
        ),
        CheckConstraint(
            (
                "external_reference IS NULL OR "
                "("
                "external_reference = btrim(external_reference) "
                "AND char_length(external_reference) BETWEEN 1 AND 128"
                ")"
            ),
            name="ticket_external_reference_format",
        ),
        CheckConstraint(
            "status IN ('open')",
            name="ticket_status",
        ),
        CheckConstraint(
            "updated_at >= created_at",
            name="ticket_timestamp_order",
        ),
        Index(
            "ix_tickets_workspace_created_id",
            "workspace_id",
            created_at.desc(),
            id.desc(),
        ),
    )

    @classmethod
    def from_domain(
        cls,
        ticket: Ticket,
    ) -> "TicketRecord":
        """Create a persistence record from a ticket entity."""

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

    def to_domain(self) -> Ticket:
        """Map the persistence record to a ticket entity."""

        return Ticket(
            id=self.id,
            workspace_id=self.workspace_id,
            subject=self.subject,
            description=self.description,
            status=TicketStatus(self.status),
            external_reference=self.external_reference,
            ingestion_request_id=self.ingestion_request_id,
            correlation_id=self.correlation_id,
            created_at=self.created_at,
            updated_at=self.updated_at,
        )
