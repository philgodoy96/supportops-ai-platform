"""Unit tests for controlled-support inspection HTTP schemas."""

from datetime import UTC, datetime
from decimal import Decimal
from uuid import UUID

from supportops.ai.schemas.ticket_classification import (
    TicketCategory,
    TicketIntent,
    TicketSentiment,
    TicketUrgency,
)
from supportops.modules.controlled_support_inspection.api.schemas import (
    ControlledSupportInspectionResponse,
)
from supportops.modules.controlled_support_inspection.domain.models import (
    AgentRunInspectionSummary,
    ClassificationInspection,
    ControlledSupportInspection,
    ControlledSupportInspectionStatus,
    LLMUsageSummary,
    RecommendationCitationInspection,
    RecommendationInspection,
    RecommendationPromptInspection,
    TerminalAnalysisInspection,
)
from supportops.modules.support_recommendations.domain.models import (
    SupportRecommendationAction,
)

_NOW = datetime(
    2026,
    8,
    2,
    18,
    30,
    tzinfo=UTC,
)

_WORKSPACE_ID = UUID("10000000-0000-4000-8000-000000000001")
_TICKET_ID = UUID("20000000-0000-4000-8000-000000000002")
_AGENT_RUN_ID = UUID("30000000-0000-4000-8000-000000000003")
_CLASSIFICATION_ID = UUID("40000000-0000-4000-8000-000000000004")
_RECOMMENDATION_ID = UUID("50000000-0000-4000-8000-000000000005")


def _citation(
    *,
    citation_id: UUID,
    order: int,
    chunk_id: UUID,
) -> RecommendationCitationInspection:
    return RecommendationCitationInspection(
        id=citation_id,
        citation_order=order,
        retrieval_query_id=UUID(f"60000000-0000-4000-8000-{order + 1:012d}"),
        retrieval_rank=order,
        retrieval_score=0.9 - (order * 0.1),
        document_id=UUID(f"70000000-0000-4000-8000-{order + 1:012d}"),
        document_version_id=UUID(f"80000000-0000-4000-8000-{order + 1:012d}"),
        chunk_id=chunk_id,
    )


def test_serializes_ordered_citations_without_internal_ids() -> None:
    citations = (
        _citation(
            citation_id=UUID("90000000-0000-4000-8000-000000000001"),
            order=0,
            chunk_id=UUID("a0000000-0000-4000-8000-000000000001"),
        ),
        _citation(
            citation_id=UUID("90000000-0000-4000-8000-000000000002"),
            order=1,
            chunk_id=UUID("a0000000-0000-4000-8000-000000000002"),
        ),
    )
    inspection = ControlledSupportInspection(
        agent_run=AgentRunInspectionSummary(
            id=_AGENT_RUN_ID,
            workspace_id=_WORKSPACE_ID,
            ticket_id=_TICKET_ID,
            workflow_name="ticket-processing",
            workflow_version="controlled-support-v1",
            status=(ControlledSupportInspectionStatus.COMPLETED),
            attempt_count=1,
            retryable_failure_count=0,
            max_retryable_failures=3,
            created_at=_NOW,
            first_started_at=_NOW,
            completed_at=_NOW,
            last_error_code=None,
        ),
        classification=ClassificationInspection(
            id=_CLASSIFICATION_ID,
            category=TicketCategory.ACCOUNT_ACCESS,
            intent=TicketIntent.REQUEST_ACCESS,
            urgency=TicketUrgency.NORMAL,
            sentiment=TicketSentiment.NEUTRAL,
            requires_human_review=False,
            summary=("The customer needs account recovery guidance."),
            created_at=_NOW,
        ),
        tool_calls=(),
        terminal_analysis=TerminalAnalysisInspection(
            recommended_action=(SupportRecommendationAction.RESPOND),
            evidence_sufficient=True,
            requires_human_review=False,
            decision_summary=("The persisted evidence supports a direct response."),
        ),
        recommendation=RecommendationInspection(
            id=_RECOMMENDATION_ID,
            recommended_action=(SupportRecommendationAction.RESPOND),
            response_text=("Follow the documented account recovery steps."),
            requires_human_review=False,
            decision_summary=("The persisted evidence supports a direct response."),
            prompt=RecommendationPromptInspection(
                id="support-recommendation-draft",
                version=1,
                content_hash="b" * 64,
            ),
            provider="mock",
            model="mock-support-model-v1",
            created_at=_NOW,
            citations=citations,
        ),
        llm_usage=LLMUsageSummary.from_invocations(()),
        llm_invocations=(),
    )

    payload = ControlledSupportInspectionResponse.from_domain(inspection).model_dump(mode="json")
    serialized_citations = payload["recommendation"]["citations"]

    assert [citation["citation_order"] for citation in serialized_citations] == [
        0,
        1,
    ]
    assert [citation["chunk_id"] for citation in serialized_citations] == [
        "a0000000-0000-4000-8000-000000000001",
        "a0000000-0000-4000-8000-000000000002",
    ]

    assert all("id" not in citation for citation in serialized_citations)
    assert payload["llm_usage"]["estimated_cost_usd"] == (str(Decimal("0")))
