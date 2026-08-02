"""Controlled-support inspection application use cases."""

from collections.abc import Mapping, Sequence
from uuid import UUID

from pydantic import JsonValue

from supportops.agent_graph.domain.state import (
    CONTROLLED_SUPPORT_WORKFLOW_NAME,
    CONTROLLED_SUPPORT_WORKFLOW_VERSION,
)
from supportops.agent_tools.domain.audit import (
    AgentToolCall,
    AgentToolCallStatus,
)
from supportops.agent_tools.tools.search_knowledge import (
    SEARCH_KNOWLEDGE_TOOL_NAME,
)
from supportops.agent_tools.tools.service_status import (
    LOOKUP_SERVICE_STATUS_TOOL_NAME,
    ServiceOperationalStatus,
)
from supportops.core.transactions import TransactionManager
from supportops.modules.controlled_support_inspection.application.errors import (
    ControlledSupportInspectionError,
    ControlledSupportInspectionInconsistentError,
    ControlledSupportInspectionNotFoundError,
    UnsupportedAgentRunInspectionError,
)
from supportops.modules.controlled_support_inspection.application.repository import (
    ControlledSupportInspectionData,
    ControlledSupportInspectionIdentity,
    ControlledSupportInspectionRepository,
)
from supportops.modules.controlled_support_inspection.domain.models import (
    AgentRunInspectionSummary,
    ClassificationInspection,
    ControlledSupportInspection,
    KnowledgeSearchEvidenceSummary,
    LLMInvocationInspection,
    LLMUsageSummary,
    RecommendationCitationInspection,
    RecommendationInspection,
    RecommendationPromptInspection,
    ServiceStatusSummary,
    TerminalAnalysisInspection,
    ToolCallInspection,
    ToolResultSummary,
)
from supportops.modules.support_recommendations.domain.models import (
    SupportRecommendation,
    SupportRecommendationAction,
    SupportRecommendationCitation,
)
from supportops.modules.ticket_classifications.domain.models import (
    LLMInvocation,
    TicketClassification,
)


class GetControlledSupportInspection:
    """Retrieve one immutable workspace-scoped workflow inspection."""

    def __init__(
        self,
        *,
        repository: ControlledSupportInspectionRepository,
        transaction_manager: TransactionManager,
    ) -> None:
        self._repository = repository
        self._transaction_manager = transaction_manager

    async def execute(
        self,
        *,
        workspace_id: UUID,
        ticket_id: UUID,
        agent_run_id: UUID,
    ) -> ControlledSupportInspection:
        """Return the scoped inspection or raise a stable error."""

        identity = ControlledSupportInspectionIdentity(
            workspace_id=workspace_id,
            ticket_id=ticket_id,
            agent_run_id=agent_run_id,
        )

        try:
            async with self._transaction_manager.transaction():
                data = await self._repository.get_inspection_data(identity)

                if data is None:
                    raise (ControlledSupportInspectionNotFoundError())

                _validate_supported_workflow(data)

                return _project_inspection(data)
        except ControlledSupportInspectionError:
            raise
        except (KeyError, TypeError, ValueError) as error:
            raise (ControlledSupportInspectionInconsistentError()) from error


def _validate_supported_workflow(
    data: ControlledSupportInspectionData,
) -> None:
    agent_run = data.agent_run

    if (
        agent_run.workflow_name != CONTROLLED_SUPPORT_WORKFLOW_NAME
        or agent_run.workflow_version != CONTROLLED_SUPPORT_WORKFLOW_VERSION
    ):
        raise UnsupportedAgentRunInspectionError()


def _project_inspection(
    data: ControlledSupportInspectionData,
) -> ControlledSupportInspection:
    attempt_numbers = {attempt.id: attempt.attempt_number for attempt in data.attempts}
    classification = (
        _project_classification(data.classification) if data.classification is not None else None
    )
    tool_calls = tuple(
        _project_tool_call(
            tool_call,
            attempt_numbers=attempt_numbers,
        )
        for tool_call in data.tool_calls
    )
    invocations = tuple(
        _project_invocation(
            invocation,
            attempt_numbers=attempt_numbers,
        )
        for invocation in data.llm_invocations
    )
    recommendation = (
        _project_recommendation(
            data.recommendation,
            citations=data.citations,
        )
        if data.recommendation is not None
        else None
    )
    terminal_analysis = (
        _project_terminal_analysis(data.recommendation) if data.recommendation is not None else None
    )

    return ControlledSupportInspection(
        agent_run=AgentRunInspectionSummary.from_agent_run(data.agent_run),
        classification=classification,
        tool_calls=tool_calls,
        terminal_analysis=terminal_analysis,
        recommendation=recommendation,
        llm_usage=LLMUsageSummary.from_invocations(invocations),
        llm_invocations=invocations,
    )


def _project_classification(
    classification: TicketClassification,
) -> ClassificationInspection:
    return ClassificationInspection(
        id=classification.id,
        category=classification.category,
        intent=classification.intent,
        urgency=classification.urgency,
        sentiment=classification.sentiment,
        requires_human_review=(classification.requires_human_review),
        summary=classification.summary,
        created_at=classification.created_at,
    )


def _project_tool_call(
    tool_call: AgentToolCall,
    *,
    attempt_numbers: Mapping[UUID, int],
) -> ToolCallInspection:
    attempt_number = attempt_numbers.get(tool_call.agent_run_attempt_id)

    if attempt_number is None:
        raise ValueError("Tool call references an unknown AgentRun attempt.")

    return ToolCallInspection(
        id=tool_call.id,
        agent_run_attempt_id=(tool_call.agent_run_attempt_id),
        attempt_number=attempt_number,
        sequence=tool_call.sequence,
        tool_name=tool_call.tool_name,
        tool_version=tool_call.tool_version,
        safety_level=tool_call.safety_level,
        status=tool_call.status,
        latency_ms=tool_call.latency_ms,
        error_code=tool_call.error_code,
        started_at=tool_call.started_at,
        finished_at=tool_call.finished_at,
        result_summary=_project_tool_result(tool_call),
    )


def _project_tool_result(
    tool_call: AgentToolCall,
) -> ToolResultSummary | None:
    if tool_call.status is not AgentToolCallStatus.SUCCEEDED:
        return None

    if tool_call.safe_output is None:
        raise ValueError("Successful tool call has no safe output.")

    if tool_call.tool_name == SEARCH_KNOWLEDGE_TOOL_NAME:
        return _project_knowledge_search_result(tool_call.safe_output)

    if tool_call.tool_name == LOOKUP_SERVICE_STATUS_TOOL_NAME:
        return _project_service_status_result(tool_call.safe_output)

    raise ValueError("Tool call references an unsupported controlled tool.")


def _project_knowledge_search_result(
    output: Mapping[str, JsonValue],
) -> KnowledgeSearchEvidenceSummary:
    retrieval_query_id = _require_uuid(
        output,
        key="retrieval_query_id",
    )
    result_count = _require_integer(
        output,
        key="result_count",
    )
    evidence = _require_sequence(
        output,
        key="evidence",
    )
    chunk_ids = tuple(
        _require_uuid_from_value(
            _require_mapping(item)["chunk_id"],
            field_name="evidence.chunk_id",
        )
        for item in evidence
    )

    if result_count != len(evidence):
        raise ValueError("Knowledge result_count does not match evidence.")

    return KnowledgeSearchEvidenceSummary(
        retrieval_query_id=retrieval_query_id,
        result_count=result_count,
        chunk_ids=chunk_ids,
    )


def _project_service_status_result(
    output: Mapping[str, JsonValue],
) -> ServiceStatusSummary:
    service_name = _require_string(
        output,
        key="service_name",
    )
    status = ServiceOperationalStatus(
        _require_string(
            output,
            key="status",
        )
    )
    incident_reference_value = output.get("incident_reference")

    if incident_reference_value is not None and not isinstance(
        incident_reference_value,
        str,
    ):
        raise TypeError("incident_reference must be a string or null.")

    return ServiceStatusSummary(
        service_name=service_name,
        status=status,
        incident_reference=incident_reference_value,
    )


def _project_invocation(
    invocation: LLMInvocation,
    *,
    attempt_numbers: Mapping[UUID, int],
) -> LLMInvocationInspection:
    attempt_number = attempt_numbers.get(invocation.agent_run_attempt_id)

    if attempt_number is None:
        raise ValueError("LLM invocation references an unknown AgentRun attempt.")

    return LLMInvocationInspection(
        id=invocation.id,
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
        estimated_total_cost_usd=(invocation.estimated_total_cost_usd),
        pricing_found=invocation.pricing_found,
        latency_ms=invocation.latency_ms,
        error_code=invocation.error_code,
        created_at=invocation.created_at,
    )


def _project_terminal_analysis(
    recommendation: SupportRecommendation,
) -> TerminalAnalysisInspection:
    return TerminalAnalysisInspection(
        recommended_action=(recommendation.recommended_action),
        evidence_sufficient=(
            recommendation.recommended_action
            is not SupportRecommendationAction.REQUEST_MORE_INFORMATION
        ),
        requires_human_review=(recommendation.requires_human_review),
        decision_summary=(recommendation.decision_summary),
    )


def _project_recommendation(
    recommendation: SupportRecommendation,
    *,
    citations: tuple[
        SupportRecommendationCitation,
        ...,
    ],
) -> RecommendationInspection:
    return RecommendationInspection(
        id=recommendation.id,
        recommended_action=(recommendation.recommended_action),
        response_text=recommendation.response_text,
        requires_human_review=(recommendation.requires_human_review),
        decision_summary=(recommendation.decision_summary),
        prompt=RecommendationPromptInspection(
            id=recommendation.prompt_id,
            version=recommendation.prompt_version,
            content_hash=(recommendation.prompt_content_hash),
        ),
        provider=recommendation.provider,
        model=recommendation.model,
        created_at=recommendation.created_at,
        citations=tuple(_project_citation(citation) for citation in citations),
    )


def _project_citation(
    citation: SupportRecommendationCitation,
) -> RecommendationCitationInspection:
    return RecommendationCitationInspection(
        id=citation.id,
        citation_order=citation.ordinal - 1,
        retrieval_query_id=citation.retrieval_query_id,
        retrieval_rank=citation.retrieval_rank,
        retrieval_score=citation.retrieval_score,
        document_id=citation.document_id,
        document_version_id=citation.document_version_id,
        chunk_id=citation.chunk_id,
    )


def _require_mapping(
    value: JsonValue,
) -> Mapping[str, JsonValue]:
    if not isinstance(value, Mapping):
        raise TypeError("Expected a JSON object.")

    return value


def _require_sequence(
    mapping: Mapping[str, JsonValue],
    *,
    key: str,
) -> Sequence[JsonValue]:
    value = mapping[key]

    if not isinstance(value, Sequence) or isinstance(value, str):
        raise TypeError(f"{key} must be a JSON array.")

    return value


def _require_string(
    mapping: Mapping[str, JsonValue],
    *,
    key: str,
) -> str:
    value = mapping[key]

    if not isinstance(value, str):
        raise TypeError(f"{key} must be a string.")

    return value


def _require_integer(
    mapping: Mapping[str, JsonValue],
    *,
    key: str,
) -> int:
    value = mapping[key]

    if type(value) is not int:
        raise TypeError(f"{key} must be an integer.")

    return value


def _require_uuid(
    mapping: Mapping[str, JsonValue],
    *,
    key: str,
) -> UUID:
    return _require_uuid_from_value(
        mapping[key],
        field_name=key,
    )


def _require_uuid_from_value(
    value: JsonValue,
    *,
    field_name: str,
) -> UUID:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a UUID string.")

    return UUID(value)
