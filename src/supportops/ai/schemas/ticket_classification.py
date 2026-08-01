"""Application-owned structured output schema for ticket classification."""

from enum import StrEnum
from typing import Annotated, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    StrictBool,
    StringConstraints,
)


class TicketCategory(StrEnum):
    """Bounded support-ticket category taxonomy."""

    ACCOUNT_ACCESS = "account_access"
    SERVICE_INCIDENT = "service_incident"
    BILLING = "billing"
    PRODUCT_BUG = "product_bug"
    HOW_TO = "how_to"
    SECURITY = "security"
    FEATURE_REQUEST = "feature_request"
    OTHER = "other"


class TicketIntent(StrEnum):
    """Bounded support-ticket intent taxonomy."""

    REQUEST_ACCESS = "request_access"
    REPORT_INCIDENT = "report_incident"
    REPORT_PROBLEM = "report_problem"
    ASK_QUESTION = "ask_question"
    REQUEST_CHANGE = "request_change"
    PROVIDE_FEEDBACK = "provide_feedback"
    OTHER = "other"


class TicketUrgency(StrEnum):
    """Bounded operational urgency taxonomy."""

    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


class TicketSentiment(StrEnum):
    """Bounded ticket sentiment taxonomy."""

    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    POSITIVE = "positive"
    MIXED = "mixed"


TicketClassificationSchemaVersion = Literal["ticket-classification-v1"]

TICKET_CLASSIFICATION_SCHEMA_VERSION: TicketClassificationSchemaVersion = "ticket-classification-v1"

TicketSummary = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        min_length=1,
        max_length=500,
    ),
]


class TicketClassificationResult(BaseModel):
    """Validated structured result produced by ticket classification."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    category: TicketCategory
    intent: TicketIntent
    urgency: TicketUrgency
    sentiment: TicketSentiment
    requires_human_review: StrictBool
    summary: TicketSummary
    schema_version: TicketClassificationSchemaVersion
