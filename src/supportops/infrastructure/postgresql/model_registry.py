"""Deterministic registration of SQLAlchemy persistence models."""


def register_persistence_models() -> None:
    """Import all persistence records into the shared SQLAlchemy metadata."""

    from supportops.agent_tools.infrastructure.models import AgentToolCallRecord
    from supportops.modules.agent_runs.infrastructure.models import (
        AgentRunAttemptRecord,
        AgentRunRecord,
    )
    from supportops.modules.knowledge_documents.infrastructure.models import (
        DocumentChunkRecord,
        DocumentRecord,
        DocumentVersionRecord,
    )
    from supportops.modules.support_recommendations.infrastructure.models import (
        SupportRecommendationCitationRecord,
        SupportRecommendationRecord,
    )
    from supportops.modules.ticket_classifications.infrastructure.models import (
        LLMInvocationRecord,
        TicketClassificationRecord,
    )
    from supportops.modules.tickets.infrastructure.models import (
        TicketRecord,
    )
    from supportops.modules.workspaces.infrastructure.models import (
        WorkspaceRecord,
    )

    _ = (
        WorkspaceRecord,
        DocumentRecord,
        DocumentVersionRecord,
        DocumentChunkRecord,
        TicketRecord,
        AgentRunRecord,
        AgentRunAttemptRecord,
        LLMInvocationRecord,
        TicketClassificationRecord,
        AgentToolCallRecord,
        SupportRecommendationRecord,
        SupportRecommendationCitationRecord,
    )
