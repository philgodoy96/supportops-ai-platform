"""FastAPI dependency construction for semantic knowledge retrieval."""

from typing import Annotated

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from supportops.api.dependencies import (
    get_application_state,
    get_postgresql_session,
)
from supportops.api.state import ApplicationState
from supportops.knowledge_index.vector_store.qdrant import (
    QdrantKnowledgeVectorStore,
)
from supportops.knowledge_retrieval.postgresql import (
    SqlAlchemyActiveKnowledgeVersionResolver,
    SqlAlchemyKnowledgeChunkHydrator,
)
from supportops.knowledge_retrieval.qdrant import (
    QdrantKnowledgeVectorSearcher,
)
from supportops.knowledge_retrieval.service import (
    SearchKnowledge,
)

PostgresqlSessionDependency = Annotated[
    AsyncSession,
    Depends(get_postgresql_session),
]
ApplicationStateDependency = Annotated[
    ApplicationState,
    Depends(get_application_state),
]


def get_search_knowledge(
    session: PostgresqlSessionDependency,
    state: ApplicationStateDependency,
) -> SearchKnowledge:
    """Construct one request-scoped retrieval service."""

    vector_store = QdrantKnowledgeVectorStore(client=state.qdrant_client)
    vector_searcher = QdrantKnowledgeVectorSearcher(
        client=state.qdrant_client,
        collection_guard=vector_store,
    )

    return SearchKnowledge(
        active_version_resolver=(SqlAlchemyActiveKnowledgeVersionResolver(session)),
        chunk_hydrator=(SqlAlchemyKnowledgeChunkHydrator(session)),
        embedding_provider=(state.embedding_provider),
        vector_searcher=vector_searcher,
        index_profile=(state.knowledge_index_profile),
        embedding_timeout_seconds=(state.settings.embedding_request_timeout_seconds),
        observability_client=state.observability_client,
    )


SearchKnowledgeDependency = Annotated[
    SearchKnowledge,
    Depends(get_search_knowledge),
]
