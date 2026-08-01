"""HTTP response schemas for AgentRun inspection."""

from datetime import datetime
from decimal import Decimal
from uuid import UUID

from pydantic import BaseModel

from supportops.ai.gateway.errors import LLMErrorCode
from supportops.ai.gateway.results import LLMInvocationStatus
from supportops.ai.schemas.ticket_classification import (
    TicketClassificationSchemaVersion,
)
from supportops.modules.agent_runs.domain.models import (
    AgentRun,
    AgentRunAttempt,
    AgentRunAttemptOutcome,
    AgentRunStatus,
)
from supportops.modules.ticket_classifications.domain.inspection import (
    AgentRunClassificationReference,
    LLMInvocationInspection,
)


class AgentRunWorkflowResponse(BaseModel):
    """Public workflow identity for an AgentRun."""

    name: str
    version: str
    trigger_key: str


class AgentRunErrorResponse(BaseModel):
    """Safe persisted processing error metadata."""

    code: str
    summary: str


class AgentRunClassificationReferenceResponse(BaseModel):
    """Minimal accepted-classification state for AgentRun inspection."""

    id: UUID
    schema_version: TicketClassificationSchemaVersion
    created_at: datetime

    @classmethod
    def from_domain(
        cls,
        reference: AgentRunClassificationReference,
    ) -> "AgentRunClassificationReferenceResponse":
        """Project a classification reference."""

        return cls(
            id=reference.id,
            schema_version=reference.schema_version,
            created_at=reference.created_at,
        )


class AgentRunResponse(BaseModel):
    """Public workspace-scoped AgentRun representation."""

    id: UUID
    workspace_id: UUID
    ticket_id: UUID
    status: AgentRunStatus
    workflow: AgentRunWorkflowResponse
    classification: AgentRunClassificationReferenceResponse | None
    attempt_count: int
    max_attempts: int
    available_at: datetime
    first_started_at: datetime | None
    completed_at: datetime | None
    last_error: AgentRunErrorResponse | None
    correlation_id: UUID
    created_at: datetime
    updated_at: datetime

    @classmethod
    def from_domain(
        cls,
        agent_run: AgentRun,
        *,
        classification: AgentRunClassificationReference | None = None,
    ) -> "AgentRunResponse":
        """Project an AgentRun without exposing lease ownership secrets."""

        last_error = None

        if agent_run.last_error_code is not None and agent_run.last_error_summary is not None:
            last_error = AgentRunErrorResponse(
                code=agent_run.last_error_code,
                summary=agent_run.last_error_summary,
            )

        return cls(
            id=agent_run.id,
            workspace_id=agent_run.workspace_id,
            ticket_id=agent_run.ticket_id,
            status=agent_run.status,
            workflow=AgentRunWorkflowResponse(
                name=agent_run.workflow_name,
                version=agent_run.workflow_version,
                trigger_key=agent_run.trigger_key,
            ),
            classification=(
                AgentRunClassificationReferenceResponse.from_domain(
                    classification,
                )
                if classification is not None
                else None
            ),
            attempt_count=agent_run.attempt_count,
            max_attempts=agent_run.max_attempts,
            available_at=agent_run.available_at,
            first_started_at=agent_run.first_started_at,
            completed_at=agent_run.completed_at,
            last_error=last_error,
            correlation_id=agent_run.correlation_id,
            created_at=agent_run.created_at,
            updated_at=agent_run.updated_at,
        )


class AgentRunAttemptResponse(BaseModel):
    """Public representation of one AgentRun execution attempt."""

    id: UUID
    attempt_number: int
    worker_id: str
    started_at: datetime
    finished_at: datetime | None
    outcome: AgentRunAttemptOutcome | None
    error: AgentRunErrorResponse | None

    @classmethod
    def from_domain(
        cls,
        attempt: AgentRunAttempt,
    ) -> "AgentRunAttemptResponse":
        """Project an attempt without exposing fencing identifiers."""

        error = None

        if attempt.error_code is not None and attempt.error_summary is not None:
            error = AgentRunErrorResponse(
                code=attempt.error_code,
                summary=attempt.error_summary,
            )

        return cls(
            id=attempt.id,
            attempt_number=attempt.attempt_number,
            worker_id=attempt.worker_id,
            started_at=attempt.started_at,
            finished_at=attempt.finished_at,
            outcome=attempt.outcome,
            error=error,
        )


class AgentRunAttemptListResponse(BaseModel):
    """Ordered execution-attempt history."""

    items: list[AgentRunAttemptResponse]


class AgentRunLLMInvocationPromptResponse(BaseModel):
    """Prompt provenance for one logical invocation."""

    id: str
    version: int
    content_hash: str


class AgentRunLLMInvocationUsageResponse(BaseModel):
    """Known provider token usage for one logical invocation."""

    input_tokens: int
    cached_input_tokens: int
    output_tokens: int
    reasoning_tokens: int | None
    total_tokens: int


class AgentRunLLMInvocationEstimatedCostResponse(BaseModel):
    """Application-estimated cost provenance for one logical invocation."""

    pricing_catalog_version: str
    pricing_found: bool
    input_cost_usd: Decimal | None
    cached_input_cost_usd: Decimal | None
    output_cost_usd: Decimal | None
    total_cost_usd: Decimal | None


class AgentRunLLMInvocationResponse(BaseModel):
    """Safe public representation of one logical LLM invocation."""

    id: UUID
    agent_run_attempt_id: UUID
    attempt_number: int
    invocation_sequence: int
    status: LLMInvocationStatus
    provider: str
    model: str
    prompt: AgentRunLLMInvocationPromptResponse
    schema_version: str
    usage: AgentRunLLMInvocationUsageResponse | None
    estimated_cost: AgentRunLLMInvocationEstimatedCostResponse
    latency_ms: int
    error_code: LLMErrorCode | None
    created_at: datetime

    @classmethod
    def from_domain(
        cls,
        invocation: LLMInvocationInspection,
    ) -> "AgentRunLLMInvocationResponse":
        """Project safe invocation, usage, cost, and provenance metadata."""

        usage = None
        input_tokens = invocation.input_tokens
        cached_input_tokens = invocation.cached_input_tokens
        output_tokens = invocation.output_tokens
        total_tokens = invocation.total_tokens

        if (
            input_tokens is not None
            and cached_input_tokens is not None
            and output_tokens is not None
            and total_tokens is not None
        ):
            usage = AgentRunLLMInvocationUsageResponse(
                input_tokens=input_tokens,
                cached_input_tokens=cached_input_tokens,
                output_tokens=output_tokens,
                reasoning_tokens=invocation.reasoning_tokens,
                total_tokens=total_tokens,
            )
        elif any(
            value is not None
            for value in (
                input_tokens,
                cached_input_tokens,
                output_tokens,
                total_tokens,
            )
        ):
            raise RuntimeError(
                "LLM invocation inspection contains partial token usage.",
            )

        return cls(
            id=invocation.id,
            agent_run_attempt_id=(
                invocation.agent_run_attempt_id
            ),
            attempt_number=invocation.attempt_number,
            invocation_sequence=(
                invocation.invocation_sequence
            ),
            status=invocation.status,
            provider=invocation.provider,
            model=invocation.model,
            prompt=AgentRunLLMInvocationPromptResponse(
                id=invocation.prompt_id,
                version=invocation.prompt_version,
                content_hash=(
                    invocation.prompt_content_hash
                ),
            ),
            schema_version=invocation.schema_version,
            usage=usage,
            estimated_cost=(
                AgentRunLLMInvocationEstimatedCostResponse(
                    pricing_catalog_version=(
                        invocation.pricing_catalog_version
                    ),
                    pricing_found=invocation.pricing_found,
                    input_cost_usd=(
                        invocation.estimated_input_cost_usd
                    ),
                    cached_input_cost_usd=(
                        invocation
                        .estimated_cached_input_cost_usd
                    ),
                    output_cost_usd=(
                        invocation.estimated_output_cost_usd
                    ),
                    total_cost_usd=(
                        invocation.estimated_total_cost_usd
                    ),
                )
            ),
            latency_ms=invocation.latency_ms,
            error_code=invocation.error_code,
            created_at=invocation.created_at,
        )


class AgentRunLLMInvocationListResponse(BaseModel):
    """Ordered logical LLM invocation history for an AgentRun."""

    items: list[AgentRunLLMInvocationResponse]
