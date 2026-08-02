"""Immutable read models for controlled-support inspection."""

from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal
from enum import StrEnum
from math import isfinite
from uuid import UUID

from supportops.agent_graph.domain.state import (
    CONTROLLED_SUPPORT_WORKFLOW_NAME,
    CONTROLLED_SUPPORT_WORKFLOW_VERSION,
)
from supportops.agent_tools.domain.audit import (
    AgentToolCallStatus,
)
from supportops.agent_tools.domain.contracts import (
    ToolSafetyLevel,
)
from supportops.agent_tools.tools.search_knowledge import (
    SEARCH_KNOWLEDGE_TOOL_NAME,
    SEARCH_KNOWLEDGE_TOOL_VERSION,
)
from supportops.agent_tools.tools.service_status import (
    LOOKUP_SERVICE_STATUS_TOOL_NAME,
    LOOKUP_SERVICE_STATUS_TOOL_VERSION,
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
from supportops.modules.agent_runs.domain.models import (
    AgentRun,
    AgentRunStatus,
)
from supportops.modules.support_recommendations.domain.models import (
    SupportRecommendationAction,
)

_ZERO_COST = Decimal("0")


class ControlledSupportInspectionStatus(StrEnum):
    """Stable public lifecycle vocabulary for inspection."""

    QUEUED = "queued"
    RUNNING = "running"
    RETRYING = "retrying"
    FAILED = "failed"
    COMPLETED = "completed"


def map_agent_run_inspection_status(
    status: AgentRunStatus,
) -> ControlledSupportInspectionStatus:
    """Map durable AgentRun status to public inspection status."""

    mapping = {
        AgentRunStatus.QUEUED: (ControlledSupportInspectionStatus.QUEUED),
        AgentRunStatus.RUNNING: (ControlledSupportInspectionStatus.RUNNING),
        AgentRunStatus.RETRY_SCHEDULED: (ControlledSupportInspectionStatus.RETRYING),
        AgentRunStatus.FAILED: (ControlledSupportInspectionStatus.FAILED),
        AgentRunStatus.SUCCEEDED: (ControlledSupportInspectionStatus.COMPLETED),
    }

    try:
        return mapping[status]
    except KeyError as exc:
        raise ValueError("Unsupported AgentRun status for inspection.") from exc


@dataclass(frozen=True, slots=True)
class AgentRunInspectionSummary:
    """Safe workflow and lifecycle summary for one AgentRun."""

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

    def __post_init__(self) -> None:
        _validate_uuid(self.id, field_name="id")
        _validate_uuid(
            self.workspace_id,
            field_name="workspace_id",
        )
        _validate_uuid(
            self.ticket_id,
            field_name="ticket_id",
        )

        if self.workflow_name != CONTROLLED_SUPPORT_WORKFLOW_NAME:
            raise ValueError("workflow_name must identify the controlled support workflow.")

        if self.workflow_version != CONTROLLED_SUPPORT_WORKFLOW_VERSION:
            raise ValueError(
                "workflow_version must identify the controlled support workflow version."
            )

        if not isinstance(
            self.status,
            ControlledSupportInspectionStatus,
        ):
            raise ValueError("status must be a supported inspection status.")

        if self.attempt_count < 0:
            raise ValueError("attempt_count must not be negative.")

        if self.max_attempts < 1:
            raise ValueError("max_attempts must be at least one.")

        if self.attempt_count > self.max_attempts:
            raise ValueError("attempt_count must not exceed max_attempts.")

        _validate_utc_timestamp(
            self.created_at,
            field_name="created_at",
        )
        _validate_optional_utc_timestamp(
            self.first_started_at,
            field_name="first_started_at",
        )
        _validate_optional_utc_timestamp(
            self.completed_at,
            field_name="completed_at",
        )
        _validate_optional_text(
            self.last_error_code,
            field_name="last_error_code",
        )

        if self.attempt_count == 0 and self.first_started_at is not None:
            raise ValueError("first_started_at must be null before the first attempt.")

        if self.attempt_count > 0 and self.first_started_at is None:
            raise ValueError("first_started_at is required after an attempt has started.")

        terminal_statuses = {
            ControlledSupportInspectionStatus.COMPLETED,
            ControlledSupportInspectionStatus.FAILED,
        }

        if self.status in terminal_statuses and self.completed_at is None:
            raise ValueError("Terminal inspection statuses require completed_at.")

        if self.status not in terminal_statuses and self.completed_at is not None:
            raise ValueError("Non-terminal inspection statuses must not define completed_at.")

        if (
            self.status is ControlledSupportInspectionStatus.COMPLETED
            and self.last_error_code is not None
        ):
            raise ValueError("Completed inspection status must not define last_error_code.")

        if (
            self.status
            in {
                ControlledSupportInspectionStatus.RETRYING,
                ControlledSupportInspectionStatus.FAILED,
            }
            and self.last_error_code is None
        ):
            raise ValueError("Retrying and failed inspection statuses require last_error_code.")

        if (
            self.status is ControlledSupportInspectionStatus.QUEUED
            and self.last_error_code is not None
        ):
            raise ValueError("Queued inspection status must not define last_error_code.")

    @classmethod
    def from_agent_run(
        cls,
        agent_run: AgentRun,
    ) -> "AgentRunInspectionSummary":
        """Project one AgentRun without lease or fencing data."""

        return cls(
            id=agent_run.id,
            workspace_id=agent_run.workspace_id,
            ticket_id=agent_run.ticket_id,
            workflow_name=agent_run.workflow_name,
            workflow_version=agent_run.workflow_version,
            status=map_agent_run_inspection_status(agent_run.status),
            attempt_count=agent_run.attempt_count,
            max_attempts=agent_run.max_attempts,
            created_at=agent_run.created_at,
            first_started_at=agent_run.first_started_at,
            completed_at=agent_run.completed_at,
            last_error_code=agent_run.last_error_code,
        )


@dataclass(frozen=True, slots=True)
class ClassificationInspection:
    """Safe accepted ticket-classification projection."""

    id: UUID
    category: TicketCategory
    intent: TicketIntent
    urgency: TicketUrgency
    sentiment: TicketSentiment
    requires_human_review: bool
    summary: str
    created_at: datetime

    def __post_init__(self) -> None:
        _validate_uuid(self.id, field_name="id")

        taxonomy_values = (
            (self.category, TicketCategory, "category"),
            (self.intent, TicketIntent, "intent"),
            (self.urgency, TicketUrgency, "urgency"),
            (
                self.sentiment,
                TicketSentiment,
                "sentiment",
            ),
        )

        for value, expected_type, field_name in taxonomy_values:
            if not isinstance(value, expected_type):
                raise ValueError(f"{field_name} has an unsupported value.")

        if type(self.requires_human_review) is not bool:
            raise ValueError("requires_human_review must be a boolean.")

        _validate_required_text(
            self.summary,
            field_name="summary",
        )
        _validate_utc_timestamp(
            self.created_at,
            field_name="created_at",
        )


@dataclass(frozen=True, slots=True)
class KnowledgeSearchEvidenceSummary:
    """Bounded identity summary for one knowledge search."""

    retrieval_query_id: UUID
    result_count: int
    chunk_ids: tuple[UUID, ...]

    def __post_init__(self) -> None:
        _validate_uuid(
            self.retrieval_query_id,
            field_name="retrieval_query_id",
        )

        if self.result_count < 0:
            raise ValueError("result_count must not be negative.")

        if self.result_count != len(self.chunk_ids):
            raise ValueError("result_count must match chunk_ids.")

        for chunk_id in self.chunk_ids:
            _validate_uuid(
                chunk_id,
                field_name="chunk_id",
            )

        if len(set(self.chunk_ids)) != len(self.chunk_ids):
            raise ValueError("chunk_ids must not contain duplicates.")


@dataclass(frozen=True, slots=True)
class ServiceStatusSummary:
    """Bounded safe result for a service-status lookup."""

    service_name: str
    status: ServiceOperationalStatus
    incident_reference: str | None

    def __post_init__(self) -> None:
        _validate_required_text(
            self.service_name,
            field_name="service_name",
        )

        if not isinstance(
            self.status,
            ServiceOperationalStatus,
        ):
            raise ValueError("status must be a supported operational status.")

        _validate_optional_text(
            self.incident_reference,
            field_name="incident_reference",
        )


type ToolResultSummary = KnowledgeSearchEvidenceSummary | ServiceStatusSummary


@dataclass(frozen=True, slots=True)
class ToolCallInspection:
    """Safe terminal audit projection for one controlled tool."""

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
    result_summary: ToolResultSummary | None

    def __post_init__(self) -> None:
        _validate_uuid(self.id, field_name="id")
        _validate_uuid(
            self.agent_run_attempt_id,
            field_name="agent_run_attempt_id",
        )

        if self.attempt_number < 1:
            raise ValueError("attempt_number must be at least one.")

        if self.sequence < 1:
            raise ValueError("sequence must be at least one.")

        supported_tools = {
            SEARCH_KNOWLEDGE_TOOL_NAME: (SEARCH_KNOWLEDGE_TOOL_VERSION),
            LOOKUP_SERVICE_STATUS_TOOL_NAME: (LOOKUP_SERVICE_STATUS_TOOL_VERSION),
        }

        if self.tool_name not in supported_tools:
            raise ValueError("tool_name must identify a controlled support tool.")

        if self.tool_version != supported_tools[self.tool_name]:
            raise ValueError("tool_version does not match tool_name.")

        if self.safety_level is not ToolSafetyLevel.READ_ONLY:
            raise ValueError("Controlled inspection supports only read-only tool calls.")

        if not isinstance(
            self.status,
            AgentToolCallStatus,
        ):
            raise ValueError("status must be a supported tool-call status.")

        if self.latency_ms < 0:
            raise ValueError("latency_ms must not be negative.")

        _validate_optional_text(
            self.error_code,
            field_name="error_code",
        )
        _validate_utc_timestamp(
            self.started_at,
            field_name="started_at",
        )
        _validate_utc_timestamp(
            self.finished_at,
            field_name="finished_at",
        )

        if self.finished_at < self.started_at:
            raise ValueError("finished_at must not be earlier than started_at.")

        if self.status is AgentToolCallStatus.SUCCEEDED:
            if self.error_code is not None:
                raise ValueError("Successful tool calls must not define error_code.")

            if self.result_summary is None:
                raise ValueError("Successful tool calls require a result summary.")
        else:
            if self.error_code is None:
                raise ValueError("Unsuccessful tool calls require error_code.")

            if self.result_summary is not None:
                raise ValueError("Unsuccessful tool calls must not expose a result summary.")

        if (
            self.tool_name == SEARCH_KNOWLEDGE_TOOL_NAME
            and self.result_summary is not None
            and not isinstance(
                self.result_summary,
                KnowledgeSearchEvidenceSummary,
            )
        ):
            raise ValueError("Knowledge search requires a knowledge evidence summary.")

        if (
            self.tool_name == LOOKUP_SERVICE_STATUS_TOOL_NAME
            and self.result_summary is not None
            and not isinstance(
                self.result_summary,
                ServiceStatusSummary,
            )
        ):
            raise ValueError("Service-status lookup requires a service status summary.")


@dataclass(frozen=True, slots=True)
class TerminalAnalysisInspection:
    """Validated terminal analysis that ended the tool loop."""

    recommended_action: SupportRecommendationAction
    evidence_sufficient: bool
    requires_human_review: bool
    decision_summary: str

    def __post_init__(self) -> None:
        if not isinstance(
            self.recommended_action,
            SupportRecommendationAction,
        ):
            raise ValueError("recommended_action must be supported.")

        if type(self.evidence_sufficient) is not bool:
            raise ValueError("evidence_sufficient must be a boolean.")

        if type(self.requires_human_review) is not bool:
            raise ValueError("requires_human_review must be a boolean.")

        _validate_required_text(
            self.decision_summary,
            field_name="decision_summary",
        )

        if (
            self.recommended_action is SupportRecommendationAction.RESPOND
            and not self.evidence_sufficient
        ):
            raise ValueError("Respond action requires sufficient evidence.")

        if (
            self.recommended_action is SupportRecommendationAction.REQUEST_MORE_INFORMATION
            and self.evidence_sufficient
        ):
            raise ValueError("Request-more-information action requires insufficient evidence.")

        if (
            self.recommended_action is SupportRecommendationAction.RECOMMEND_ESCALATION
            and not self.requires_human_review
        ):
            raise ValueError("Escalation action requires human review.")


@dataclass(frozen=True, slots=True)
class RecommendationPromptInspection:
    """Safe prompt provenance without prompt content."""

    id: str
    version: int
    content_hash: str

    def __post_init__(self) -> None:
        _validate_required_text(
            self.id,
            field_name="id",
        )

        if self.version < 1:
            raise ValueError("version must be positive.")

        _validate_sha256_hash(
            self.content_hash,
            field_name="content_hash",
        )


@dataclass(frozen=True, slots=True)
class RecommendationCitationInspection:
    """Safe citation identity and retrieval provenance."""

    id: UUID
    citation_order: int
    retrieval_query_id: UUID
    retrieval_rank: int
    retrieval_score: float
    document_id: UUID
    document_version_id: UUID
    chunk_id: UUID

    def __post_init__(self) -> None:
        identifiers = {
            "id": self.id,
            "retrieval_query_id": self.retrieval_query_id,
            "document_id": self.document_id,
            "document_version_id": (self.document_version_id),
            "chunk_id": self.chunk_id,
        }

        for field_name, value in identifiers.items():
            _validate_uuid(
                value,
                field_name=field_name,
            )

        if self.citation_order < 0:
            raise ValueError("citation_order must not be negative.")

        if self.retrieval_rank < 0:
            raise ValueError("retrieval_rank must not be negative.")

        if not isfinite(self.retrieval_score):
            raise ValueError("retrieval_score must be finite.")


@dataclass(frozen=True, slots=True)
class RecommendationInspection:
    """Persisted recommendation and ordered citations."""

    id: UUID
    recommended_action: SupportRecommendationAction
    response_text: str
    requires_human_review: bool
    decision_summary: str
    prompt: RecommendationPromptInspection
    provider: str
    model: str
    created_at: datetime
    citations: tuple[
        RecommendationCitationInspection,
        ...,
    ]

    def __post_init__(self) -> None:
        _validate_uuid(self.id, field_name="id")

        if not isinstance(
            self.recommended_action,
            SupportRecommendationAction,
        ):
            raise ValueError("recommended_action must be supported.")

        _validate_required_text(
            self.response_text,
            field_name="response_text",
        )

        if type(self.requires_human_review) is not bool:
            raise ValueError("requires_human_review must be a boolean.")

        _validate_required_text(
            self.decision_summary,
            field_name="decision_summary",
        )
        _validate_required_text(
            self.provider,
            field_name="provider",
        )
        _validate_required_text(
            self.model,
            field_name="model",
        )
        _validate_utc_timestamp(
            self.created_at,
            field_name="created_at",
        )

        citation_orders = tuple(citation.citation_order for citation in self.citations)
        expected_orders = tuple(range(len(self.citations)))

        if citation_orders != expected_orders:
            raise ValueError("Citation order must be contiguous and zero-based.")

        citation_ids = tuple(citation.id for citation in self.citations)
        chunk_ids = tuple(citation.chunk_id for citation in self.citations)

        if len(set(citation_ids)) != len(citation_ids):
            raise ValueError("Citation IDs must be unique.")

        if len(set(chunk_ids)) != len(chunk_ids):
            raise ValueError("Recommendation citations must not repeat chunks.")


@dataclass(frozen=True, slots=True)
class LLMInvocationInspection:
    """Safe logical invocation and historical cost projection."""

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

    def __post_init__(self) -> None:
        _validate_uuid(self.id, field_name="id")
        _validate_uuid(
            self.agent_run_attempt_id,
            field_name="agent_run_attempt_id",
        )

        if self.attempt_number < 1:
            raise ValueError("attempt_number must be at least one.")

        if self.invocation_sequence < 1:
            raise ValueError("invocation_sequence must be at least one.")

        if not isinstance(
            self.status,
            LLMInvocationStatus,
        ):
            raise ValueError("status must be a supported invocation status.")

        for field_name, value in (
            ("provider", self.provider),
            ("model", self.model),
            ("prompt_id", self.prompt_id),
            ("schema_version", self.schema_version),
        ):
            _validate_required_text(
                value,
                field_name=field_name,
            )

        if self.prompt_version < 1:
            raise ValueError("prompt_version must be positive.")

        _validate_sha256_hash(
            self.prompt_content_hash,
            field_name="prompt_content_hash",
        )

        for token_field_name, token_value in (
            ("input_tokens", self.input_tokens),
            (
                "cached_input_tokens",
                self.cached_input_tokens,
            ),
            ("output_tokens", self.output_tokens),
            ("reasoning_tokens", self.reasoning_tokens),
            ("total_tokens", self.total_tokens),
        ):
            _validate_optional_nonnegative_integer(
                token_value,
                field_name=token_field_name,
            )

        if self.estimated_total_cost_usd is not None and self.estimated_total_cost_usd < _ZERO_COST:
            raise ValueError("estimated_total_cost_usd must not be negative.")

        if type(self.pricing_found) is not bool:
            raise ValueError("pricing_found must be a boolean.")

        if self.latency_ms < 0:
            raise ValueError("latency_ms must not be negative.")

        if self.status is LLMInvocationStatus.SUCCEEDED:
            if self.error_code is not None:
                raise ValueError("Successful invocations must not define error_code.")
        elif self.error_code is None:
            raise ValueError("Unsuccessful invocations require error_code.")

        _validate_utc_timestamp(
            self.created_at,
            field_name="created_at",
        )


@dataclass(frozen=True, slots=True)
class LLMUsageSummary:
    """Aggregate persisted invocation usage and estimated cost."""

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

    def __post_init__(self) -> None:
        integer_fields = (
            self.invocation_count,
            self.successful_invocation_count,
            self.failed_invocation_count,
            self.input_tokens,
            self.cached_input_tokens,
            self.output_tokens,
            self.reasoning_tokens,
            self.total_tokens,
            self.unpriced_invocation_count,
        )

        if any(value < 0 for value in integer_fields):
            raise ValueError("Usage summary values must not be negative.")

        if self.successful_invocation_count + self.failed_invocation_count != self.invocation_count:
            raise ValueError("Invocation status counts must match invocation_count.")

        if self.unpriced_invocation_count > self.invocation_count:
            raise ValueError("unpriced_invocation_count must not exceed invocation_count.")

        if self.estimated_cost_usd < _ZERO_COST:
            raise ValueError("estimated_cost_usd must not be negative.")

    @classmethod
    def from_invocations(
        cls,
        invocations: tuple[
            LLMInvocationInspection,
            ...,
        ],
    ) -> "LLMUsageSummary":
        """Aggregate only persisted historical estimates."""

        successful_count = sum(
            invocation.status is LLMInvocationStatus.SUCCEEDED for invocation in invocations
        )

        return cls(
            invocation_count=len(invocations),
            successful_invocation_count=successful_count,
            failed_invocation_count=(len(invocations) - successful_count),
            input_tokens=sum(invocation.input_tokens or 0 for invocation in invocations),
            cached_input_tokens=sum(
                invocation.cached_input_tokens or 0 for invocation in invocations
            ),
            output_tokens=sum(invocation.output_tokens or 0 for invocation in invocations),
            reasoning_tokens=sum(invocation.reasoning_tokens or 0 for invocation in invocations),
            total_tokens=sum(invocation.total_tokens or 0 for invocation in invocations),
            estimated_cost_usd=sum(
                (invocation.estimated_total_cost_usd or _ZERO_COST for invocation in invocations),
                start=_ZERO_COST,
            ),
            unpriced_invocation_count=sum(
                not invocation.pricing_found for invocation in invocations
            ),
        )


@dataclass(frozen=True, slots=True)
class ControlledSupportInspection:
    """Complete bounded inspection view for one AgentRun."""

    agent_run: AgentRunInspectionSummary
    classification: ClassificationInspection | None
    tool_calls: tuple[ToolCallInspection, ...]
    terminal_analysis: TerminalAnalysisInspection | None
    recommendation: RecommendationInspection | None
    llm_usage: LLMUsageSummary
    llm_invocations: tuple[
        LLMInvocationInspection,
        ...,
    ]

    def __post_init__(self) -> None:
        _validate_attempt_scoped_tool_order(self.tool_calls)
        _validate_attempt_scoped_invocation_order(self.llm_invocations)

        tool_call_ids = tuple(tool_call.id for tool_call in self.tool_calls)
        invocation_ids = tuple(invocation.id for invocation in self.llm_invocations)

        if len(set(tool_call_ids)) != len(tool_call_ids):
            raise ValueError("Tool-call IDs must be unique.")

        if len(set(invocation_ids)) != len(invocation_ids):
            raise ValueError("Invocation IDs must be unique.")

        attempt_numbers = tuple(
            [tool_call.attempt_number for tool_call in self.tool_calls]
            + [invocation.attempt_number for invocation in self.llm_invocations]
        )

        if attempt_numbers and max(attempt_numbers) > self.agent_run.attempt_count:
            raise ValueError("Inspection children cannot reference an unstarted attempt.")

        expected_usage = LLMUsageSummary.from_invocations(self.llm_invocations)

        if self.llm_usage != expected_usage:
            raise ValueError("llm_usage must match llm_invocations.")

        if self.tool_calls and self.classification is None:
            raise ValueError("Tool calls require a persisted classification.")

        if self.terminal_analysis is not None and self.classification is None:
            raise ValueError("Terminal analysis requires a persisted classification.")

        if self.recommendation is not None:
            if self.classification is None:
                raise ValueError("Recommendation requires a persisted classification.")

            if self.terminal_analysis is None:
                raise ValueError("Recommendation requires terminal analysis.")

            if self.recommendation.recommended_action != self.terminal_analysis.recommended_action:
                raise ValueError("Recommendation action must match terminal analysis.")

            if (
                self.terminal_analysis.requires_human_review
                and not self.recommendation.requires_human_review
            ):
                raise ValueError(
                    "Recommendation must not weaken the terminal human-review requirement."
                )

        if (
            self.agent_run.status is ControlledSupportInspectionStatus.COMPLETED
            and self.recommendation is None
        ):
            raise ValueError("Completed controlled workflows require a persisted recommendation.")

        if self.agent_run.status is ControlledSupportInspectionStatus.QUEUED and (
            self.classification is not None
            or self.tool_calls
            or self.terminal_analysis is not None
            or self.recommendation is not None
            or self.llm_invocations
        ):
            raise ValueError("Queued controlled workflows must not contain execution progress.")


def _validate_attempt_scoped_tool_order(
    tool_calls: tuple[ToolCallInspection, ...],
) -> None:
    keys = tuple(
        (
            tool_call.attempt_number,
            tool_call.sequence,
        )
        for tool_call in tool_calls
    )

    if keys != tuple(sorted(keys)):
        raise ValueError("Tool calls must be ordered by attempt and sequence.")

    sequences_by_attempt: dict[int, list[int]] = {}

    for tool_call in tool_calls:
        sequences_by_attempt.setdefault(
            tool_call.attempt_number,
            [],
        ).append(tool_call.sequence)

    for sequences in sequences_by_attempt.values():
        if tuple(sequences) != tuple(range(1, len(sequences) + 1)):
            raise ValueError(
                "Tool-call sequences must be contiguous and one-based within each attempt."
            )


def _validate_attempt_scoped_invocation_order(
    invocations: tuple[
        LLMInvocationInspection,
        ...,
    ],
) -> None:
    keys = tuple(
        (
            invocation.attempt_number,
            invocation.invocation_sequence,
        )
        for invocation in invocations
    )

    if keys != tuple(sorted(keys)):
        raise ValueError("LLM invocations must be ordered by attempt and sequence.")

    sequences_by_attempt: dict[int, list[int]] = {}

    for invocation in invocations:
        sequences_by_attempt.setdefault(
            invocation.attempt_number,
            [],
        ).append(invocation.invocation_sequence)

    for sequences in sequences_by_attempt.values():
        if tuple(sequences) != tuple(range(1, len(sequences) + 1)):
            raise ValueError(
                "Invocation sequences must be contiguous and one-based within each attempt."
            )


def _validate_uuid(
    value: UUID,
    *,
    field_name: str,
) -> None:
    if not isinstance(value, UUID):
        raise TypeError(f"{field_name} must be a UUID.")


def _validate_required_text(
    value: str,
    *,
    field_name: str,
) -> None:
    if not value:
        raise ValueError(f"{field_name} is required.")

    if value != value.strip():
        raise ValueError(f"{field_name} must not contain surrounding whitespace.")


def _validate_optional_text(
    value: str | None,
    *,
    field_name: str,
) -> None:
    if value is not None:
        _validate_required_text(
            value,
            field_name=field_name,
        )


def _validate_sha256_hash(
    value: str,
    *,
    field_name: str,
) -> None:
    if len(value) != 64:
        raise ValueError(f"{field_name} must contain 64 hexadecimal characters.")

    if any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{field_name} must be lowercase hexadecimal.")


def _validate_optional_nonnegative_integer(
    value: int | None,
    *,
    field_name: str,
) -> None:
    if value is None:
        return

    if type(value) is not int:
        raise TypeError(f"{field_name} must be an integer.")

    if value < 0:
        raise ValueError(f"{field_name} must not be negative.")


def _validate_optional_utc_timestamp(
    value: datetime | None,
    *,
    field_name: str,
) -> None:
    if value is not None:
        _validate_utc_timestamp(
            value,
            field_name=field_name,
        )


def _validate_utc_timestamp(
    value: datetime,
    *,
    field_name: str,
) -> None:
    if value.tzinfo is None or value.utcoffset() != timedelta(0):
        raise ValueError(f"{field_name} must be a UTC-aware timestamp.")
