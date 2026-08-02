"""Process-scoped LLM and session-scoped executor composition."""

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import cast

from pydantic import SecretStr
from qdrant_client import AsyncQdrantClient
from sqlalchemy.ext.asyncio import AsyncSession

from supportops.agent_graph.application.decision_execution import (
    ControlledSupportDecisionExecutor,
)
from supportops.agent_graph.application.recommendation_execution import (
    ControlledSupportRecommendationExecutor,
)
from supportops.agent_graph.application.tool_execution import (
    ControlledToolDecisionExecutor,
)
from supportops.agent_graph.application.tool_observations import (
    ControlledToolObservationAssembler,
)
from supportops.agent_graph.application.workflow import (
    ControlledSupportWorkflowExecutor,
    ControlledSupportWorkflowNodes,
    compile_controlled_support_graph,
)
from supportops.agent_graph.domain.state import (
    CONTROLLED_SUPPORT_WORKFLOW_VERSION,
)
from supportops.agent_graph.infrastructure.checkpoints import (
    PostgresCheckpointRuntime,
    create_postgres_checkpoint_runtime,
)
from supportops.agent_tools.application.execution import (
    BoundedReadOnlyToolExecutor,
)
from supportops.agent_tools.infrastructure.query_repository import (
    SqlAlchemyAgentToolCallQueryRepository,
)
from supportops.agent_tools.infrastructure.repository import (
    SqlAlchemyAgentToolCallExecutionRepository,
)
from supportops.agent_tools.tools.registry import (
    create_controlled_support_tool_registry,
)
from supportops.agent_tools.tools.service_status import (
    DeterministicServiceStatusCatalog,
)
from supportops.ai.embeddings.contracts import EmbeddingProvider
from supportops.ai.gateway.contracts import LLMProvider
from supportops.ai.gateway.service import LLMGateway
from supportops.ai.gateway.tool_decisions import (
    LLMToolDecisionGateway,
    LLMToolDecisionProvider,
)
from supportops.ai.providers.mock import (
    MOCK_TICKET_CLASSIFIER_MODEL,
    MockLLMProvider,
)
from supportops.ai.providers.openai import OpenAILLMProvider
from supportops.core.settings import Settings
from supportops.core.transactions import TransactionManager
from supportops.infrastructure.qdrant import (
    close_qdrant_client,
    create_qdrant_client,
)
from supportops.knowledge_index.composition import (
    build_knowledge_index_profile,
    create_embedding_provider,
)
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
from supportops.knowledge_retrieval.service import SearchKnowledge
from supportops.modules.agent_runs.application.deterministic_executor import (
    DeterministicTicketProcessingExecutor,
)
from supportops.modules.agent_runs.application.executor_registry import (
    AgentRunExecutorRegistration,
    AgentRunExecutorRegistry,
)
from supportops.modules.agent_runs.domain.models import (
    DETERMINISTIC_BASELINE_WORKFLOW_VERSION,
    INITIAL_TICKET_PROCESSING_WORKFLOW_NAME,
    TICKET_CLASSIFICATION_WORKFLOW_VERSION,
)
from supportops.modules.knowledge_documents.domain.models import (
    KnowledgeIndexProfile,
)
from supportops.modules.support_recommendations.infrastructure.invocation_query_repository import (
    SqlAlchemyAttemptLLMInvocationQueryRepository,
)
from supportops.modules.support_recommendations.infrastructure.query_repository import (
    SqlAlchemySupportRecommendationQueryRepository,
)
from supportops.modules.support_recommendations.infrastructure.repository import (
    SqlAlchemySupportRecommendationExecutionRepository,
)
from supportops.modules.ticket_classifications.application.executor import (
    TicketClassificationExecutor,
)
from supportops.modules.ticket_classifications.domain.repositories import (
    TicketClassificationRepository,
)
from supportops.modules.ticket_classifications.infrastructure.repository import (
    SqlAlchemyClassificationPersistenceRepository,
    SqlAlchemyTicketClassificationQueryRepository,
)

MOCK_LLM_PROVIDER_NAME = "mock"
OPENAI_LLM_PROVIDER_NAME = "openai"

type CheckpointRuntimeFactory = Callable[
    ...,
    Awaitable[PostgresCheckpointRuntime],
]
type EmbeddingProviderFactory = Callable[[Settings], EmbeddingProvider]
type QdrantClientFactory = Callable[[Settings], AsyncQdrantClient]
type KnowledgeIndexProfileFactory = Callable[
    [Settings],
    KnowledgeIndexProfile,
]


@dataclass(frozen=True, slots=True)
class WorkerLLMRuntime:
    """Own one process-scoped provider and application LLM Gateway."""

    provider: LLMProvider
    gateway: LLMGateway
    model: str

    async def close(self) -> None:
        """Release provider-owned process resources."""

        await self.provider.close()


@dataclass(slots=True)
class WorkerControlledSupportRuntime:
    """Own process-scoped controlled-support infrastructure resources."""

    checkpoint_runtime: PostgresCheckpointRuntime
    embedding_provider: EmbeddingProvider
    qdrant_client: AsyncQdrantClient
    index_profile: KnowledgeIndexProfile
    vector_store: QdrantKnowledgeVectorStore
    vector_searcher: QdrantKnowledgeVectorSearcher
    _closed: bool = field(
        default=False,
        init=False,
        repr=False,
    )

    async def close(self) -> None:
        """Release process-owned resources idempotently."""

        if self._closed:
            return

        self._closed = True
        failures: list[Exception] = []

        try:
            await self.checkpoint_runtime.close()
        except Exception as error:
            failures.append(error)

        try:
            await self.embedding_provider.close()
        except Exception as error:
            failures.append(error)

        try:
            await close_qdrant_client(self.qdrant_client)
        except Exception as error:
            failures.append(error)

        if failures:
            primary_failure = failures[0]
            for secondary_failure in failures[1:]:
                primary_failure.add_note(
                    "An additional controlled-support runtime "
                    "resource failed to close: "
                    f"{type(secondary_failure).__name__}."
                )
            raise primary_failure


def create_worker_llm_runtime(
    *,
    provider_name: str,
    openai_api_key: str | None,
    openai_model: str,
    openai_base_url: str | None,
    request_timeout_seconds: float,
    transport_max_retries: int,
    max_repair_attempts: int,
) -> WorkerLLMRuntime:
    """Create one explicitly configured process-scoped LLM runtime."""

    _validate_required_text(
        provider_name,
        field_name="provider_name",
    )

    provider: LLMProvider
    model: str

    if provider_name == MOCK_LLM_PROVIDER_NAME:
        provider = MockLLMProvider(
            model=MOCK_TICKET_CLASSIFIER_MODEL,
        )
        model = MOCK_TICKET_CLASSIFIER_MODEL
    elif provider_name == OPENAI_LLM_PROVIDER_NAME:
        if openai_api_key is None:
            raise ValueError(
                "openai_api_key is required when the OpenAI provider is selected.",
            )

        provider = OpenAILLMProvider.create(
            api_key=openai_api_key,
            model=openai_model,
            timeout_seconds=request_timeout_seconds,
            transport_max_retries=transport_max_retries,
            base_url=openai_base_url,
        )
        model = openai_model
    else:
        raise ValueError(
            f"Unsupported LLM provider: {provider_name}.",
        )

    gateway = LLMGateway(
        provider=provider,
        max_repair_attempts=max_repair_attempts,
    )

    return WorkerLLMRuntime(
        provider=provider,
        gateway=gateway,
        model=model,
    )


async def create_worker_controlled_support_runtime(
    *,
    settings: Settings,
    checkpoint_runtime_factory: CheckpointRuntimeFactory = (create_postgres_checkpoint_runtime),
    embedding_provider_factory: EmbeddingProviderFactory = (create_embedding_provider),
    qdrant_client_factory: QdrantClientFactory = (create_qdrant_client),
    index_profile_factory: KnowledgeIndexProfileFactory = (build_knowledge_index_profile),
) -> WorkerControlledSupportRuntime:
    """Create one process-scoped controlled-support runtime."""

    checkpoint_database_url = SecretStr(
        _to_psycopg_connection_url(
            str(settings.postgresql_url),
        ),
    )
    checkpoint_runtime: PostgresCheckpointRuntime | None = None
    embedding_provider: EmbeddingProvider | None = None
    qdrant_client: AsyncQdrantClient | None = None

    try:
        checkpoint_runtime = await checkpoint_runtime_factory(
            database_url=checkpoint_database_url,
        )
        await checkpoint_runtime.setup()

        index_profile = index_profile_factory(settings)
        embedding_provider = embedding_provider_factory(settings)
        qdrant_client = qdrant_client_factory(settings)
        vector_store = QdrantKnowledgeVectorStore(
            client=qdrant_client,
        )
        vector_searcher = QdrantKnowledgeVectorSearcher(
            client=qdrant_client,
            collection_guard=vector_store,
        )
    except Exception:
        await _close_partial_controlled_support_resources(
            checkpoint_runtime=checkpoint_runtime,
            embedding_provider=embedding_provider,
            qdrant_client=qdrant_client,
        )
        raise

    return WorkerControlledSupportRuntime(
        checkpoint_runtime=checkpoint_runtime,
        embedding_provider=embedding_provider,
        qdrant_client=qdrant_client,
        index_profile=index_profile,
        vector_store=vector_store,
        vector_searcher=vector_searcher,
    )


def create_session_scoped_executor_registry(
    *,
    session: AsyncSession,
    transaction_manager: TransactionManager,
    gateway: LLMGateway,
    provider: LLMProvider,
    model: str,
    request_timeout_seconds: float,
    controlled_runtime: WorkerControlledSupportRuntime,
    embedding_timeout_seconds: float,
) -> AgentRunExecutorRegistry:
    """Create all workflow executors owned by one database session."""

    classification_repository = SqlAlchemyClassificationPersistenceRepository(
        session,
    )
    classification_executor = TicketClassificationExecutor(
        gateway=gateway,
        model=model,
        request_timeout_seconds=request_timeout_seconds,
        transaction_manager=transaction_manager,
        classification_repository=(classification_repository),
        execution_repository=classification_repository,
    )
    classification_query_repository = SqlAlchemyTicketClassificationQueryRepository(
        session,
    )

    active_version_resolver = SqlAlchemyActiveKnowledgeVersionResolver(
        session,
    )
    chunk_hydrator = SqlAlchemyKnowledgeChunkHydrator(session)
    knowledge_search = SearchKnowledge(
        active_version_resolver=active_version_resolver,
        chunk_hydrator=chunk_hydrator,
        embedding_provider=(controlled_runtime.embedding_provider),
        vector_searcher=controlled_runtime.vector_searcher,
        index_profile=controlled_runtime.index_profile,
        embedding_timeout_seconds=embedding_timeout_seconds,
    )
    tool_registry = create_controlled_support_tool_registry(
        knowledge_search=knowledge_search,
        service_status_catalog=(DeterministicServiceStatusCatalog(())),
    )
    bounded_tool_executor = BoundedReadOnlyToolExecutor(
        registry=tool_registry,
    )
    tool_call_execution_repository = SqlAlchemyAgentToolCallExecutionRepository(
        session,
    )
    tool_call_query_repository = SqlAlchemyAgentToolCallQueryRepository(
        session,
    )
    invocation_query_repository = SqlAlchemyAttemptLLMInvocationQueryRepository(
        session,
    )
    recommendation_execution_repository = SqlAlchemySupportRecommendationExecutionRepository(
        session,
    )
    recommendation_query_repository = SqlAlchemySupportRecommendationQueryRepository(
        session,
    )
    observation_assembler = ControlledToolObservationAssembler(
        transaction_manager=transaction_manager,
        tool_call_repository=tool_call_query_repository,
        chunk_hydrator=chunk_hydrator,
    )
    decision_executor = ControlledSupportDecisionExecutor(
        gateway=LLMToolDecisionGateway(
            provider=cast(LLMToolDecisionProvider, provider),
        ),
        tool_registry=tool_registry,
        model=model,
        request_timeout_seconds=request_timeout_seconds,
        transaction_manager=transaction_manager,
        invocation_query_repository=(invocation_query_repository),
        execution_repository=(recommendation_execution_repository),
    )
    tool_executor = ControlledToolDecisionExecutor(
        executor=bounded_tool_executor,
        transaction_manager=transaction_manager,
        execution_repository=(tool_call_execution_repository),
        query_repository=tool_call_query_repository,
    )
    recommendation_executor = ControlledSupportRecommendationExecutor(
        gateway=gateway,
        model=model,
        request_timeout_seconds=request_timeout_seconds,
        transaction_manager=transaction_manager,
        observation_assembler=observation_assembler,
        invocation_query_repository=(invocation_query_repository),
        recommendation_query_repository=(recommendation_query_repository),
        execution_repository=(recommendation_execution_repository),
    )
    nodes = ControlledSupportWorkflowNodes(
        transaction_manager=transaction_manager,
        classification_repository=cast(
            TicketClassificationRepository,
            classification_query_repository,
        ),
        classification_executor=classification_executor,
        observation_assembler=observation_assembler,
        decision_executor=decision_executor,
        tool_executor=tool_executor,
        recommendation_executor=recommendation_executor,
    )
    graph = compile_controlled_support_graph(
        nodes=nodes,
        checkpointer=(controlled_runtime.checkpoint_runtime.checkpointer),
    )
    controlled_executor = ControlledSupportWorkflowExecutor(
        graph=graph,
    )

    return AgentRunExecutorRegistry(
        (
            AgentRunExecutorRegistration(
                workflow_name=(INITIAL_TICKET_PROCESSING_WORKFLOW_NAME),
                workflow_version=(DETERMINISTIC_BASELINE_WORKFLOW_VERSION),
                executor=(DeterministicTicketProcessingExecutor()),
            ),
            AgentRunExecutorRegistration(
                workflow_name=(INITIAL_TICKET_PROCESSING_WORKFLOW_NAME),
                workflow_version=(TICKET_CLASSIFICATION_WORKFLOW_VERSION),
                executor=classification_executor,
            ),
            AgentRunExecutorRegistration(
                workflow_name=(INITIAL_TICKET_PROCESSING_WORKFLOW_NAME),
                workflow_version=(CONTROLLED_SUPPORT_WORKFLOW_VERSION),
                executor=controlled_executor,
            ),
        ),
    )


def _to_psycopg_connection_url(database_url: str) -> str:
    for sqlalchemy_prefix in (
        "postgresql+asyncpg://",
        "postgresql+psycopg://",
    ):
        if database_url.startswith(sqlalchemy_prefix):
            return "postgresql://" + database_url[len(sqlalchemy_prefix) :]

    if database_url.startswith("postgresql://"):
        return database_url

    raise ValueError(
        "postgresql_url must use a PostgreSQL connection URL scheme.",
    )


async def _close_partial_controlled_support_resources(
    *,
    checkpoint_runtime: PostgresCheckpointRuntime | None,
    embedding_provider: EmbeddingProvider | None,
    qdrant_client: AsyncQdrantClient | None,
) -> None:
    failures: list[Exception] = []

    if checkpoint_runtime is not None:
        try:
            await checkpoint_runtime.close()
        except Exception as error:
            failures.append(error)

    if embedding_provider is not None:
        try:
            await embedding_provider.close()
        except Exception as error:
            failures.append(error)

    if qdrant_client is not None:
        try:
            await close_qdrant_client(qdrant_client)
        except Exception as error:
            failures.append(error)

    if failures:
        primary_failure = failures[0]
        for secondary_failure in failures[1:]:
            primary_failure.add_note(
                "An additional partially created resource "
                "failed to close: "
                f"{type(secondary_failure).__name__}."
            )
        raise primary_failure


def _validate_required_text(
    value: str,
    *,
    field_name: str,
) -> None:
    if not value:
        raise ValueError(f"{field_name} is required.")

    if value != value.strip():
        raise ValueError(
            f"{field_name} must not contain surrounding whitespace.",
        )
