"""Workspace-scoped semantic knowledge search routes."""

from uuid import UUID

from fastapi import APIRouter

from supportops.knowledge_retrieval.api.dependencies import (
    SearchKnowledgeDependency,
)
from supportops.knowledge_retrieval.api.schemas import (
    KnowledgeSearchRequestBody,
    KnowledgeSearchResponse,
)

router = APIRouter(
    prefix="/workspaces/{workspace_id}/knowledge",
    tags=["knowledge retrieval"],
)


@router.post(
    "/search",
    response_model=KnowledgeSearchResponse,
)
async def search_knowledge(
    workspace_id: UUID,
    request: KnowledgeSearchRequestBody,
    service: SearchKnowledgeDependency,
) -> KnowledgeSearchResponse:
    """Retrieve authoritative evidence from active knowledge versions."""

    result = await service.execute(request.to_domain(workspace_id=workspace_id))

    return KnowledgeSearchResponse.from_domain(result)
