"""Read-only projections for ticket-classification inspection."""

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from uuid import UUID

from supportops.ai.gateway.errors import LLMErrorCode
from supportops.ai.gateway.results import LLMInvocationStatus
from supportops.ai.schemas.ticket_classification import (
    TicketClassificationSchemaVersion,
)
from supportops.modules.ticket_classifications.domain.models import (
    LLMInvocation,
    TicketClassification,
)


@dataclass(frozen=True, slots=True)
class AgentRunClassificationReference:
    """Minimal accepted-classification reference for AgentRun inspection."""

    id: UUID
    schema_version: TicketClassificationSchemaVersion
    created_at: datetime

    @classmethod
    def from_domain(
        cls,
        classification: TicketClassification,
    ) -> "AgentRunClassificationReference":
        """Create a reference from an accepted classification."""

        return cls(
            id=classification.id,
            schema_version=classification.schema_version,
            created_at=classification.created_at,
        )


@dataclass(frozen=True, slots=True)
class LLMInvocationInspection:
    """Safe read projection for one durable logical LLM invocation."""

    id: UUID
    workspace_id: UUID
    ticket_id: UUID
    agent_run_id: UUID
    agent_run_attempt_id: UUID
    attempt_number: int
    invocation_sequence: int
    status: LLMInvocationStatus
    provider: str
    model: str
    prompt_id: str
    prompt_version: int
    prompt_content_hash: str
    schema_version: str
    input_tokens: int | None
    cached_input_tokens: int | None
    output_tokens: int | None
    reasoning_tokens: int | None
    total_tokens: int | None
    pricing_catalog_version: str
    pricing_found: bool
    estimated_input_cost_usd: Decimal | None
    estimated_cached_input_cost_usd: Decimal | None
    estimated_output_cost_usd: Decimal | None
    estimated_total_cost_usd: Decimal | None
    latency_ms: int
    error_code: LLMErrorCode | None
    created_at: datetime

    def __post_init__(self) -> None:
        if self.attempt_number <= 0:
            raise ValueError(
                "attempt_number must be positive.",
            )

        if self.invocation_sequence <= 0:
            raise ValueError(
                "invocation_sequence must be positive.",
            )

    @classmethod
    def from_domain(
        cls,
        *,
        invocation: LLMInvocation,
        attempt_number: int,
    ) -> "LLMInvocationInspection":
        """Create a safe projection without provider-internal identifiers."""

        return cls(
            id=invocation.id,
            workspace_id=invocation.workspace_id,
            ticket_id=invocation.ticket_id,
            agent_run_id=invocation.agent_run_id,
            agent_run_attempt_id=(invocation.agent_run_attempt_id),
            attempt_number=attempt_number,
            invocation_sequence=(invocation.invocation_sequence),
            status=invocation.status,
            provider=invocation.provider,
            model=invocation.model,
            prompt_id=invocation.prompt_id,
            prompt_version=invocation.prompt_version,
            prompt_content_hash=(invocation.prompt_content_hash),
            schema_version=invocation.schema_version,
            input_tokens=invocation.input_tokens,
            cached_input_tokens=(invocation.cached_input_tokens),
            output_tokens=invocation.output_tokens,
            reasoning_tokens=invocation.reasoning_tokens,
            total_tokens=invocation.total_tokens,
            pricing_catalog_version=(invocation.pricing_catalog_version),
            pricing_found=invocation.pricing_found,
            estimated_input_cost_usd=(invocation.estimated_input_cost_usd),
            estimated_cached_input_cost_usd=(invocation.estimated_cached_input_cost_usd),
            estimated_output_cost_usd=(invocation.estimated_output_cost_usd),
            estimated_total_cost_usd=(invocation.estimated_total_cost_usd),
            latency_ms=invocation.latency_ms,
            error_code=invocation.error_code,
            created_at=invocation.created_at,
        )
