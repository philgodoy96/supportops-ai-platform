"""Validated models for ticket-classification evaluation cases."""

from typing import Annotated, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StringConstraints,
    model_validator,
)

from supportops.ai.schemas.ticket_classification import (
    TicketCategory,
    TicketClassificationSchemaVersion,
    TicketIntent,
    TicketSentiment,
    TicketUrgency,
)
from supportops.modules.tickets.domain.models import (
    TICKET_DESCRIPTION_MAX_LENGTH,
    TICKET_SUBJECT_MAX_LENGTH,
)

EvaluationCaseId = Annotated[
    str,
    StringConstraints(
        strip_whitespace=False,
        min_length=1,
        max_length=128,
        pattern=r"^[a-z0-9]+(?:-[a-z0-9]+)*$",
    ),
]

EvaluationTag = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        min_length=1,
        max_length=64,
        pattern=r"^[a-z0-9]+(?:-[a-z0-9]+)*$",
    ),
]

EvaluationTicketSubject = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        min_length=1,
        max_length=TICKET_SUBJECT_MAX_LENGTH,
    ),
]

EvaluationTicketDescription = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        min_length=1,
        max_length=TICKET_DESCRIPTION_MAX_LENGTH,
    ),
]

EvaluationTags = Annotated[
    tuple[EvaluationTag, ...],
    Field(
        min_length=1,
        max_length=12,
    ),
]


class TicketClassificationEvaluationTicket(BaseModel):
    """Synthetic ticket input used by one evaluation case."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    subject: EvaluationTicketSubject
    description: EvaluationTicketDescription


class TicketClassificationExpectedLabels(BaseModel):
    """Expected bounded labels for one evaluation case."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    category: TicketCategory
    intent: TicketIntent
    urgency: TicketUrgency
    sentiment: TicketSentiment
    requires_human_review: StrictBool
    schema_version: TicketClassificationSchemaVersion


class TicketClassificationEvaluationCase(BaseModel):
    """One immutable synthetic ticket and its expected labels."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    case_id: EvaluationCaseId
    tags: EvaluationTags
    ticket: TicketClassificationEvaluationTicket
    expected: TicketClassificationExpectedLabels

    @model_validator(mode="after")
    def validate_unique_tags(self) -> Self:
        """Reject duplicate tags that add no dataset information."""

        if len(set(self.tags)) != len(self.tags):
            raise ValueError(
                "Evaluation case tags must be unique.",
            )

        return self
