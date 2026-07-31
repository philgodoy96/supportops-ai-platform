"""Support ticket domain entities and invariants."""

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from uuid import UUID, uuid4

TICKET_SUBJECT_MAX_LENGTH = 200
TICKET_DESCRIPTION_MAX_LENGTH = 20_000
TICKET_EXTERNAL_REFERENCE_MAX_LENGTH = 128


class TicketStatus(StrEnum):
    """Current support ticket lifecycle states."""

    OPEN = "open"


@dataclass(frozen=True, slots=True)
class Ticket:
    """A support request owned by exactly one workspace."""

    id: UUID
    workspace_id: UUID
    subject: str
    description: str
    status: TicketStatus
    external_reference: str | None
    ingestion_request_id: UUID
    correlation_id: UUID
    created_at: datetime
    updated_at: datetime

    def __post_init__(self) -> None:
        _validate_subject(self.subject)
        _validate_description(self.description)
        _validate_external_reference(self.external_reference)
        _validate_status(self.status)
        _validate_utc_timestamp(
            self.created_at,
            field_name="created_at",
        )
        _validate_utc_timestamp(
            self.updated_at,
            field_name="updated_at",
        )

        if self.updated_at < self.created_at:
            raise ValueError(
                "updated_at must not be earlier than created_at.",
            )

    @classmethod
    def create(
        cls,
        *,
        workspace_id: UUID,
        subject: str,
        description: str,
        ingestion_request_id: UUID,
        correlation_id: UUID,
        external_reference: str | None = None,
        ticket_id: UUID | None = None,
        now: datetime | None = None,
    ) -> "Ticket":
        """Create an open ticket with normalized human-authored content."""

        created_at = now or datetime.now(UTC)

        return cls(
            id=ticket_id or uuid4(),
            workspace_id=workspace_id,
            subject=subject.strip(),
            description=description.strip(),
            status=TicketStatus.OPEN,
            external_reference=external_reference,
            ingestion_request_id=ingestion_request_id,
            correlation_id=correlation_id,
            created_at=created_at,
            updated_at=created_at,
        )


def _validate_subject(subject: str) -> None:
    if not subject:
        raise ValueError("Ticket subject is required.")

    if subject != subject.strip():
        raise ValueError(
            "Ticket subject must not contain surrounding whitespace.",
        )

    if len(subject) > TICKET_SUBJECT_MAX_LENGTH:
        raise ValueError(
            "Ticket subject exceeds the maximum length.",
        )


def _validate_description(description: str) -> None:
    if not description:
        raise ValueError("Ticket description is required.")

    if description != description.strip():
        raise ValueError(
            "Ticket description must not contain surrounding whitespace.",
        )

    if len(description) > TICKET_DESCRIPTION_MAX_LENGTH:
        raise ValueError(
            "Ticket description exceeds the maximum length.",
        )


def _validate_external_reference(
    external_reference: str | None,
) -> None:
    if external_reference is None:
        return

    if not external_reference:
        raise ValueError(
            "Ticket external reference must not be empty.",
        )

    if external_reference != external_reference.strip():
        raise ValueError(
            "Ticket external reference must not contain surrounding whitespace.",
        )

    if len(external_reference) > TICKET_EXTERNAL_REFERENCE_MAX_LENGTH:
        raise ValueError(
            "Ticket external reference exceeds the maximum length.",
        )


def _validate_status(status: TicketStatus) -> None:
    if status is not TicketStatus.OPEN:
        raise ValueError(
            "Ticket status is not supported by the current lifecycle.",
        )


def _validate_utc_timestamp(
    value: datetime,
    *,
    field_name: str,
) -> None:
    if value.tzinfo is None or value.utcoffset() != timedelta(0):
        raise ValueError(f"{field_name} must be a UTC-aware timestamp.")
