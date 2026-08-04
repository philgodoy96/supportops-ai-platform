"""Process composition for explicit knowledge indexing commands."""

from contextlib import suppress
from dataclasses import dataclass, field
from typing import cast
from uuid import UUID

from qdrant_client import AsyncQdrantClient
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
)

from supportops.ai.embeddings.contracts import EmbeddingProvider
from supportops.ai.embeddings.mock import (
    MOCK_HASHING_EMBEDDING_MODEL,
    MockEmbeddingProvider,
)
from supportops.ai.embeddings.observability import (
    ObservingEmbeddingProvider,
)
from supportops.ai.embeddings.openai import (
    OpenAIEmbeddingProvider,
)
from supportops.ai.embeddings.pricing import (
    DEFAULT_EMBEDDING_PRICING_CATALOG,
)
from supportops.core.settings import (
    EmbeddingProviderName,
    Settings,
)
from supportops.infrastructure.postgresql import (
    create_postgresql_engine,
    create_postgresql_session_factory,
    dispose_postgresql_engine,
)
from supportops.infrastructure.postgresql.transaction import (
    SqlAlchemyTransactionManager,
)
from supportops.infrastructure.qdrant import (
    close_qdrant_client,
    create_qdrant_client,
)
from supportops.knowledge_index.chunking.contracts import (
    ChunkingPolicy,
)
from supportops.knowledge_index.chunking.markdown import (
    MarkdownTokenChunker,
)
from supportops.knowledge_index.chunking.tokenizer import (
    TiktokenTokenizer,
)
from supportops.knowledge_index.indexing.results import (
    IndexDocumentVersionResult,
)
from supportops.knowledge_index.indexing.service import (
    IndexDocumentVersion,
)
from supportops.knowledge_index.vector_store.contracts import (
    KnowledgeCollectionProfile,
)
from supportops.knowledge_index.vector_store.qdrant import (
    QdrantKnowledgeVectorStore,
)
from supportops.modules.knowledge_documents.domain.models import (
    KnowledgeIndexProfile,
)
from supportops.modules.knowledge_documents.infrastructure.repository import (
    SqlAlchemyDocumentChunkRepository,
    SqlAlchemyDocumentVersionRepository,
)
from supportops.observability.composition import create_observability_client
from supportops.observability.contracts import ObservabilityClient

MOCK_KNOWLEDGE_COLLECTION = "supportops-knowledge-mock-v1"
OPENAI_KNOWLEDGE_COLLECTION = "supportops-knowledge-openai-v1"
KNOWLEDGE_VECTOR_NAME = "dense"

OPENAI_KNOWLEDGE_EMBEDDING_MODEL = "text-embedding-3-small"
OPENAI_KNOWLEDGE_EMBEDDING_DIMENSIONS = 1536


class KnowledgeIndexCompositionError(ValueError):
    """Raised when runtime configuration cannot form a stable profile."""


@dataclass(slots=True)
class KnowledgeIndexRuntime:
    """Own process-scoped indexing infrastructure and provider resources."""

    settings: Settings
    engine: AsyncEngine
    session_factory: async_sessionmaker[AsyncSession]
    qdrant_client: AsyncQdrantClient
    embedding_provider: EmbeddingProvider
    index_profile: KnowledgeIndexProfile
    chunker: MarkdownTokenChunker
    vector_store: QdrantKnowledgeVectorStore
    observability_client: ObservabilityClient
    _closed: bool = field(
        default=False,
        init=False,
        repr=False,
    )

    async def ensure_collection(self) -> None:
        """Create or validate the configured knowledge collection."""

        self._require_open()

        await self.vector_store.ensure_collection(self.collection_profile)

    async def index_version(
        self,
        *,
        workspace_id: UUID,
        document_id: UUID,
        document_version_id: UUID,
    ) -> IndexDocumentVersionResult:
        """Index one workspace-owned document version."""

        self._require_open()

        async with self.session_factory() as session:
            service = IndexDocumentVersion(
                version_repository=(SqlAlchemyDocumentVersionRepository(session)),
                chunk_repository=(SqlAlchemyDocumentChunkRepository(session)),
                transaction_manager=(SqlAlchemyTransactionManager(session)),
                chunker=self.chunker,
                embedding_provider=self.embedding_provider,
                vector_store=self.vector_store,
                pricing_catalog=(DEFAULT_EMBEDDING_PRICING_CATALOG),
                index_profile=self.index_profile,
                embedding_timeout_seconds=(self.settings.embedding_request_timeout_seconds),
                observability_client=self.observability_client,
            )

            return await service.execute(
                workspace_id=workspace_id,
                document_id=document_id,
                document_version_id=document_version_id,
            )

    @property
    def collection_profile(
        self,
    ) -> KnowledgeCollectionProfile:
        """Return the Qdrant profile derived from persisted identity."""

        return KnowledgeCollectionProfile(
            collection_name=(self.index_profile.knowledge_collection),
            vector_name=(self.index_profile.knowledge_vector_name),
            dimensions=(self.index_profile.embedding_dimensions),
        )

    async def close(self) -> None:
        """Release all process-scoped resources idempotently."""

        if self._closed:
            return

        self._closed = True
        failures: list[Exception] = []

        try:
            await self.embedding_provider.close()
        except Exception as error:
            failures.append(error)

        try:
            await close_qdrant_client(self.qdrant_client)
        except Exception as error:
            failures.append(error)

        try:
            await dispose_postgresql_engine(self.engine)
        except Exception as error:
            failures.append(error)

        with suppress(Exception):
            self.observability_client.shutdown()

        if failures:
            primary_failure = failures[0]
            for secondary_failure in failures[1:]:
                primary_failure.add_note(
                    "An additional indexing runtime resource "
                    "failed to close: "
                    f"{type(secondary_failure).__name__}."
                )
            raise primary_failure

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("Knowledge indexing runtime is closed.")


async def create_knowledge_index_runtime(
    *,
    settings: Settings,
) -> KnowledgeIndexRuntime:
    """Create one process-scoped knowledge indexing runtime."""

    index_profile = build_knowledge_index_profile(settings)
    observability_client = create_observability_client(settings)

    embedding_provider: EmbeddingProvider | None = None
    engine: AsyncEngine | None = None
    qdrant_client: AsyncQdrantClient | None = None

    try:
        embedding_provider = create_embedding_provider(
            settings,
            observability_client=observability_client,
        )
        engine = create_postgresql_engine(settings)
        session_factory = create_postgresql_session_factory(engine)
        qdrant_client = create_qdrant_client(settings)

        policy = ChunkingPolicy()
        chunker = MarkdownTokenChunker(
            policy=policy,
            tokenizer=TiktokenTokenizer(encoding_name=(policy.tokenizer_encoding)),
        )
        vector_store = QdrantKnowledgeVectorStore(client=qdrant_client)

        return KnowledgeIndexRuntime(
            settings=settings,
            engine=engine,
            session_factory=session_factory,
            qdrant_client=qdrant_client,
            embedding_provider=embedding_provider,
            index_profile=index_profile,
            chunker=chunker,
            vector_store=vector_store,
            observability_client=observability_client,
        )
    except Exception:
        await _close_partial_runtime(
            embedding_provider=embedding_provider,
            qdrant_client=qdrant_client,
            engine=engine,
            observability_client=observability_client,
        )
        raise


def build_knowledge_index_profile(
    settings: Settings,
) -> KnowledgeIndexProfile:
    """Build the immutable profile selected by runtime configuration."""

    provider = settings.embedding_provider

    if provider is EmbeddingProviderName.MOCK:
        if settings.embedding_model != MOCK_HASHING_EMBEDDING_MODEL:
            raise KnowledgeIndexCompositionError(
                "Mock knowledge indexing requires mock-hashing-embedding-v1."
            )

        collection_name = MOCK_KNOWLEDGE_COLLECTION
    elif provider is EmbeddingProviderName.OPENAI:
        if settings.embedding_model != OPENAI_KNOWLEDGE_EMBEDDING_MODEL:
            raise KnowledgeIndexCompositionError(
                "OpenAI knowledge indexing requires text-embedding-3-small."
            )
        if settings.embedding_dimensions != OPENAI_KNOWLEDGE_EMBEDDING_DIMENSIONS:
            raise KnowledgeIndexCompositionError(
                "OpenAI knowledge indexing requires 1536 embedding dimensions."
            )

        collection_name = OPENAI_KNOWLEDGE_COLLECTION
    else:
        raise KnowledgeIndexCompositionError("Unsupported embedding provider.")

    policy = ChunkingPolicy()

    return KnowledgeIndexProfile(
        chunking_strategy=policy.strategy,
        chunking_version=policy.version,
        tokenizer_encoding=(policy.tokenizer_encoding),
        embedding_provider=provider.value,
        embedding_model=settings.embedding_model,
        embedding_dimensions=(settings.embedding_dimensions),
        knowledge_collection=collection_name,
        knowledge_vector_name=(KNOWLEDGE_VECTOR_NAME),
    )


def create_embedding_provider(
    settings: Settings,
    *,
    observability_client: ObservabilityClient | None = None,
) -> EmbeddingProvider:
    """Create the configured process-scoped embedding adapter."""

    provider: EmbeddingProvider

    if settings.embedding_provider is EmbeddingProviderName.MOCK:
        provider = MockEmbeddingProvider(
            model=settings.embedding_model,
            dimensions=settings.embedding_dimensions,
        )
    elif settings.embedding_provider is EmbeddingProviderName.OPENAI:
        if settings.openai_api_key is None:
            raise KnowledgeIndexCompositionError(
                "OpenAI API key is required for OpenAI knowledge indexing."
            )

        provider = OpenAIEmbeddingProvider.create(
            api_key=(settings.openai_api_key.get_secret_value()),
            model=settings.embedding_model,
            dimensions=settings.embedding_dimensions,
            timeout_seconds=(settings.embedding_request_timeout_seconds),
            transport_max_retries=(settings.embedding_transport_max_retries),
            base_url=settings.openai_base_url,
        )
    else:
        raise KnowledgeIndexCompositionError("Unsupported embedding provider.")

    return ObservingEmbeddingProvider(
        provider=provider,
        observability_client=observability_client,
        pricing_catalog=DEFAULT_EMBEDDING_PRICING_CATALOG,
    )


async def _close_partial_runtime(
    *,
    embedding_provider: EmbeddingProvider | None,
    qdrant_client: AsyncQdrantClient | None,
    engine: AsyncEngine | None,
    observability_client: ObservabilityClient,
) -> None:
    failures: list[Exception] = []

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

    if engine is not None:
        try:
            await dispose_postgresql_engine(engine)
        except Exception as error:
            failures.append(error)

    with suppress(Exception):
        observability_client.shutdown()

    if failures:
        primary_failure = failures[0]
        for secondary_failure in failures[1:]:
            primary_failure.add_note(
                "An additional partially created resource "
                "failed to close: "
                f"{type(secondary_failure).__name__}."
            )
        raise primary_failure


def require_runtime_type(
    runtime: object,
) -> KnowledgeIndexRuntime:
    """Narrow a composed runtime for typed integration boundaries."""

    return cast(KnowledgeIndexRuntime, runtime)
