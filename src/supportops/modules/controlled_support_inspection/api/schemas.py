"""HTTP response schemas for controlled-support inspection."""

from datetime import datetime
from decimal import Decimal
from uuid import UUID

from pydantic import BaseModel

from supportops.agent_tools.domain.audit import (
    AgentToolCallStatus,
)
from supportops.agent_tools.domain.contracts import (
    ToolSafetyLevel,
)
from supportops.agent_tools.tools.service_status import (
    ServiceOperationalStatus,
)
from supportops.ai.gateway.errors import LLMErrorCode
from supportops.ai.gateway.results import (
    LLMInvocationStatus,
)
from supportops.ai.schemas.ticket_classification import (
    TicketCategory,
    TicketIntent,
    TicketSentiment,
    TicketUrgency,
)
from supportops.modules.controlled_support_inspection.domain.models import (
    AgentRunInspectionSummary,
    ClassificationInspection,
    ControlledSupportInspection,
    ControlledSupportInspectionStatus,
    KnowledgeSearchEvidenceSummary,
    LLMInvocationInspection,
    LLMUsageSummary,
    RecommendationCitationInspection,
    RecommendationInspection,
    ServiceStatusSummary,
    ToolCallInspection,
)
from supportops.modules.support_recommendations.domain.models import (
    SupportRecommendationAction,
)


class AgentRunInspectionResponse(BaseModel):
    """Safe workflow lifecycle summary."""

    id: UUID
    workspace_id: UUID
    ticket_id: UUID
    workflow_name: str
    workflow_version: str
    status: ControlledSupportInspectionStatus
    attempt_count: int
    max_attempts: int
    created_at: datetime
    first_started_at: datetime | None
    completed_at: datetime | None
    last_error_code: str | None

    @classmethod
    def from_domain(
        cls,
        value: AgentRunInspectionSummary,
    ) -> "AgentRunInspectionResponse":
        """Project one AgentRun inspection summary."""

        return cls(
            id=value.id,
            workspace_id=value.workspace_id,
            ticket_id=value.ticket_id,
            workflow_name=value.workflow_name,
            workflow_version=value.workflow_version,
            status=value.status,
            attempt_count=value.attempt_count,
            max_attempts=value.max_attempts,
            created_at=value.created_at,
            first_started_at=value.first_started_at,
            completed_at=value.completed_at,
            last_error_code=value.last_error_code,
        )


class ClassificationInspectionResponse(BaseModel):
    """Safe accepted classification details."""

    id: UUID
    category: TicketCategory
    intent: TicketIntent
    urgency: TicketUrgency
    sentiment: TicketSentiment
    requires_human_review: bool
    summary: str
    created_at: datetime

    @classmethod
    def from_domain(
        cls,
        value: ClassificationInspection,
    ) -> "ClassificationInspectionResponse":
        """Project one persisted classification."""

        return cls(
            id=value.id,
            category=value.category,
            intent=value.intent,
            urgency=value.urgency,
            sentiment=value.sentiment,
            requires_human_review=(value.requires_human_review),
            summary=value.summary,
            created_at=value.created_at,
        )


class KnowledgeSearchEvidenceSummaryResponse(BaseModel):
    """Bounded knowledge-search provenance."""

    retrieval_query_id: UUID
    result_count: int
    chunk_ids: list[UUID]

    @classmethod
    def from_domain(
        cls,
        value: KnowledgeSearchEvidenceSummary,
    ) -> "KnowledgeSearchEvidenceSummaryResponse":
        """Project safe knowledge-search evidence identity."""

        return cls(
            retrieval_query_id=value.retrieval_query_id,
            result_count=value.result_count,
            chunk_ids=list(value.chunk_ids),
        )


class ServiceStatusSummaryResponse(BaseModel):
    """Bounded deterministic service-status result."""

    service_name: str
    status: ServiceOperationalStatus
    incident_reference: str | None

    @classmethod
    def from_domain(
        cls,
        value: ServiceStatusSummary,
    ) -> "ServiceStatusSummaryResponse":
        """Project one service-status summary."""

        return cls(
            service_name=value.service_name,
            status=value.status,
            incident_reference=value.incident_reference,
        )


class ToolCallInspectionResponse(BaseModel):
    """Safe terminal tool-call history item."""

    id: UUID
    agent_run_attempt_id: UUID
    attempt_number: int
    sequence: int
    tool_name: str
    tool_version: int
    safety_level: ToolSafetyLevel
    status: AgentToolCallStatus
    latency_ms: int
    error_code: str | None
    started_at: datetime
    finished_at: datetime
    result_summary: KnowledgeSearchEvidenceSummaryResponse | ServiceStatusSummaryResponse | None

    @classmethod
    def from_domain(
        cls,
        value: ToolCallInspection,
    ) -> "ToolCallInspectionResponse":
        """Project one tool call without raw audit payloads."""

        result_summary: (
            KnowledgeSearchEvidenceSummaryResponse | ServiceStatusSummaryResponse | None
        ) = None

        if isinstance(
            value.result_summary,
            KnowledgeSearchEvidenceSummary,
        ):
            result_summary = KnowledgeSearchEvidenceSummaryResponse.from_domain(
                value.result_summary
            )
        elif isinstance(
            value.result_summary,
            ServiceStatusSummary,
        ):
            result_summary = ServiceStatusSummaryResponse.from_domain(value.result_summary)

        return cls(
            id=value.id,
            agent_run_attempt_id=(value.agent_run_attempt_id),
            attempt_number=value.attempt_number,
            sequence=value.sequence,
            tool_name=value.tool_name,
            tool_version=value.tool_version,
            safety_level=value.safety_level,
            status=value.status,
            latency_ms=value.latency_ms,
            error_code=value.error_code,
            started_at=value.started_at,
            finished_at=value.finished_at,
            result_summary=result_summary,
        )


class RecommendationPromptResponse(BaseModel):
    """Safe recommendation prompt provenance."""

    id: str
    version: int
    content_hash: str


class RecommendationCitationResponse(BaseModel):
    """Safe ordered recommendation citation."""

    citation_order: int
    retrieval_query_id: UUID
    retrieval_rank: int
    retrieval_score: float
    document_id: UUID
    document_version_id: UUID
    chunk_id: UUID

    @classmethod
    def from_domain(
        cls,
        value: RecommendationCitationInspection,
    ) -> "RecommendationCitationResponse":
        """Project one recommendation citation."""

        return cls(
            citation_order=value.citation_order,
            retrieval_query_id=value.retrieval_query_id,
            retrieval_rank=value.retrieval_rank,
            retrieval_score=value.retrieval_score,
            document_id=value.document_id,
            document_version_id=(value.document_version_id),
            chunk_id=value.chunk_id,
        )


class RecommendationInspectionResponse(BaseModel):
    """Persisted controlled-support recommendation."""

    id: UUID
    recommended_action: SupportRecommendationAction
    response_text: str
    requires_human_review: bool
    decision_summary: str
    prompt: RecommendationPromptResponse
    provider: str
    model: str
    created_at: datetime
    citations: list[RecommendationCitationResponse]

    @classmethod
    def from_domain(
        cls,
        value: RecommendationInspection,
    ) -> "RecommendationInspectionResponse":
        """Project recommendation and citation provenance."""

        return cls(
            id=value.id,
            recommended_action=value.recommended_action,
            response_text=value.response_text,
            requires_human_review=(value.requires_human_review),
            decision_summary=value.decision_summary,
            prompt=RecommendationPromptResponse(
                id=value.prompt.id,
                version=value.prompt.version,
                content_hash=value.prompt.content_hash,
            ),
            provider=value.provider,
            model=value.model,
            created_at=value.created_at,
            citations=[
                RecommendationCitationResponse.from_domain(citation) for citation in value.citations
            ],
        )


class LLMUsageSummaryResponse(BaseModel):
    """Aggregate persisted model usage and estimated cost."""

    invocation_count: int
    successful_invocation_count: int
    failed_invocation_count: int
    input_tokens: int
    cached_input_tokens: int
    output_tokens: int
    reasoning_tokens: int
    total_tokens: int
    estimated_cost_usd: Decimal
    unpriced_invocation_count: int

    @classmethod
    def from_domain(
        cls,
        value: LLMUsageSummary,
    ) -> "LLMUsageSummaryResponse":
        """Project persisted historical usage totals."""

        return cls(
            invocation_count=value.invocation_count,
            successful_invocation_count=(value.successful_invocation_count),
            failed_invocation_count=(value.failed_invocation_count),
            input_tokens=value.input_tokens,
            cached_input_tokens=(value.cached_input_tokens),
            output_tokens=value.output_tokens,
            reasoning_tokens=value.reasoning_tokens,
            total_tokens=value.total_tokens,
            estimated_cost_usd=(value.estimated_cost_usd),
            unpriced_invocation_count=(value.unpriced_invocation_count),
        )


class LLMInvocationInspectionResponse(BaseModel):
    """Safe persisted logical invocation details."""

    id: UUID
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
    estimated_total_cost_usd: Decimal | None
    pricing_found: bool
    latency_ms: int
    error_code: LLMErrorCode | None
    created_at: datetime

    @classmethod
    def from_domain(
        cls,
        value: LLMInvocationInspection,
    ) -> "LLMInvocationInspectionResponse":
        """Project one invocation without provider request data."""

        return cls(
            id=value.id,
            agent_run_attempt_id=(value.agent_run_attempt_id),
            attempt_number=value.attempt_number,
            invocation_sequence=(value.invocation_sequence),
            status=value.status,
            provider=value.provider,
            model=value.model,
            prompt_id=value.prompt_id,
            prompt_version=value.prompt_version,
            prompt_content_hash=(value.prompt_content_hash),
            schema_version=value.schema_version,
            input_tokens=value.input_tokens,
            cached_input_tokens=(value.cached_input_tokens),
            output_tokens=value.output_tokens,
            reasoning_tokens=value.reasoning_tokens,
            total_tokens=value.total_tokens,
            estimated_total_cost_usd=(value.estimated_total_cost_usd),
            pricing_found=value.pricing_found,
            latency_ms=value.latency_ms,
            error_code=value.error_code,
            created_at=value.created_at,
        )


class ControlledSupportInspectionResponse(BaseModel):
    """Complete controlled-support inspection response."""

    agent_run: AgentRunInspectionResponse
    classification: ClassificationInspectionResponse | None
    tool_calls: list[ToolCallInspectionResponse]
    recommendation: RecommendationInspectionResponse | None
    llm_usage: LLMUsageSummaryResponse
    llm_invocations: list[LLMInvocationInspectionResponse]

    @classmethod
    def from_domain(
        cls,
        value: ControlledSupportInspection,
    ) -> "ControlledSupportInspectionResponse":
        """Project the complete immutable inspection view."""

        return cls(
            agent_run=AgentRunInspectionResponse.from_domain(value.agent_run),
            classification=(
                ClassificationInspectionResponse.from_domain(value.classification)
                if value.classification is not None
                else None
            ),
            tool_calls=[
                ToolCallInspectionResponse.from_domain(tool_call) for tool_call in value.tool_calls
            ],
            recommendation=(
                RecommendationInspectionResponse.from_domain(value.recommendation)
                if value.recommendation is not None
                else None
            ),
            llm_usage=LLMUsageSummaryResponse.from_domain(value.llm_usage),
            llm_invocations=[
                LLMInvocationInspectionResponse.from_domain(invocation)
                for invocation in value.llm_invocations
            ],
        )
