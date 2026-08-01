"""HTTP response schemas for ticket-classification inspection."""

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel

from supportops.ai.schemas.ticket_classification import (
    TicketCategory,
    TicketClassificationSchemaVersion,
    TicketIntent,
    TicketSentiment,
    TicketUrgency,
)
from supportops.modules.ticket_classifications.domain.models import (
    TicketClassification,
)


class ClassificationPromptResponse(BaseModel):
    """Public prompt provenance for an accepted classification."""

    id: str
    version: int
    content_hash: str


class TicketClassificationResponse(BaseModel):
    """Public representation of one accepted ticket classification."""

    id: UUID
    workspace_id: UUID
    ticket_id: UUID
    agent_run_id: UUID
    accepted_invocation_id: UUID
    category: TicketCategory
    intent: TicketIntent
    urgency: TicketUrgency
    sentiment: TicketSentiment
    requires_human_review: bool
    summary: str
    schema_version: TicketClassificationSchemaVersion
    prompt: ClassificationPromptResponse
    provider: str
    model: str
    created_at: datetime

    @classmethod
    def from_domain(
        cls,
        classification: TicketClassification,
    ) -> "TicketClassificationResponse":
        """Project an immutable accepted classification."""

        return cls(
            id=classification.id,
            workspace_id=classification.workspace_id,
            ticket_id=classification.ticket_id,
            agent_run_id=classification.agent_run_id,
            accepted_invocation_id=(classification.accepted_llm_invocation_id),
            category=classification.category,
            intent=classification.intent,
            urgency=classification.urgency,
            sentiment=classification.sentiment,
            requires_human_review=(classification.requires_human_review),
            summary=classification.summary,
            schema_version=classification.schema_version,
            prompt=ClassificationPromptResponse(
                id=classification.prompt_id,
                version=classification.prompt_version,
                content_hash=(classification.prompt_content_hash),
            ),
            provider=classification.provider,
            model=classification.model,
            created_at=classification.created_at,
        )


class TicketClassificationListResponse(BaseModel):
    """One bounded page of accepted ticket classifications."""

    items: list[TicketClassificationResponse]
    next_cursor: str | None
