"""Explicit, retry-safe document-version indexing orchestration."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import AbstractContextManager
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from time import perf_counter
from typing import Final
from uuid import UUID

from supportops.ai.embeddings.contracts import (
    EmbeddingOperation,
    EmbeddingProvider,
    EmbeddingProviderResponse,
    EmbeddingRequest,
    EmbeddingVector,
)
from supportops.ai.embeddings.errors import (
    EmbeddingError,
    EmbeddingInvalidResponseError,
)
from supportops.ai.embeddings.pricing import (
    EmbeddingPricingCatalog,
    estimate_embedding_cost,
)
from supportops.core.transactions import TransactionManager
from supportops.knowledge_index.chunking.contracts import (
    KnowledgeDocumentChunker,
)
from supportops.knowledge_index.indexing.errors import (
    KnowledgeChunkingError,
    KnowledgeChunkPersistenceError,
    KnowledgeDocumentVersionNotFoundError,
    KnowledgeIndexingError,
    KnowledgeIndexProfileMismatchError,
    KnowledgeProjectionCountMismatchError,
)
from supportops.knowledge_index.indexing.results import (
    IndexDocumentVersionResult,
)
from supportops.knowledge_index.vector_store.contracts import (
    KnowledgeCollectionCompatibilityError,
    KnowledgeCollectionProfile,
    KnowledgeVectorPoint,
    KnowledgeVectorStore,
    KnowledgeVectorStoreError,
    KnowledgeVectorStoreOperationError,
    KnowledgeVectorStoreUnavailableError,
    KnowledgeVersionProjection,
)
from supportops.modules.knowledge_documents.domain.models import (
    DocumentChunk,
    DocumentVersion,
    DocumentVersionStatus,
    KnowledgeIndexProfile,
)
from supportops.modules.knowledge_documents.domain.repositories import (
    DocumentChunkConflictError,
    DocumentChunkRepository,
    DocumentVersionRepository,
)
from supportops.observability.contracts import (
    ObservabilityClient,
    ObservationScope,
)
from supportops.observability.models import (
    JsonValue,
    ObservationAttributes,
    ObservationStatus,
    ObservationType,
    ObservationUpdate,
)
from supportops.observability.noop import NoOpObservabilityClient

DEFAULT_EMBEDDING_BATCH_SIZE = 64

_UNEXPECTED_FAILURE_CODE: Final = "knowledge_index_unexpected_failure"

_STAGE_LOAD: Final = "knowledge-index.load-document-version"
_STAGE_CHUNK: Final = "knowledge-index.chunk-document"
_STAGE_UPSERT: Final = "knowledge-index.upsert-vectors"
_STAGE_VERIFY: Final = "knowledge-index.verify-index"
_STAGE_PERSIST: Final = "knowledge-index.persist-outcome"

_STAGE_METADATA_KEYS: Final = frozenset(
    {
        "chunk_count",
        "batch_count",
        "vector_count",
        "expected_vector_count",
        "verified_vector_count",
        "persisted_status",
        "latency_ms",
        "error_code",
    }
)
_STAGE_METADATA_PATHS: Final = frozenset((key,) for key in _STAGE_METADATA_KEYS)


@dataclass(frozen=True, slots=True)
class _EmbeddingOutcome:
    vectors: tuple[EmbeddingVector, ...]
    input_tokens: int


class IndexDocumentVersion:
    """Build and verify one document-version vector projection."""

    def __init__(
        self,
        *,
        version_repository: DocumentVersionRepository,
        chunk_repository: DocumentChunkRepository,
        transaction_manager: TransactionManager,
        chunker: KnowledgeDocumentChunker,
        embedding_provider: EmbeddingProvider,
        vector_store: KnowledgeVectorStore,
        pricing_catalog: EmbeddingPricingCatalog,
        index_profile: KnowledgeIndexProfile,
        embedding_timeout_seconds: float,
        embedding_batch_size: int = (DEFAULT_EMBEDDING_BATCH_SIZE),
        clock: Callable[[], datetime] | None = None,
        observability_client: ObservabilityClient | None = None,
    ) -> None:
        if embedding_timeout_seconds <= 0:
            raise ValueError("embedding_timeout_seconds must be positive.")
        if embedding_batch_size <= 0:
            raise ValueError("embedding_batch_size must be positive.")
        if (
            chunker.policy.strategy != index_profile.chunking_strategy
            or chunker.policy.version != index_profile.chunking_version
            or chunker.policy.tokenizer_encoding != index_profile.tokenizer_encoding
        ):
            raise ValueError("Chunker policy must match the index profile.")
        if embedding_provider.provider_name != index_profile.embedding_provider:
            raise ValueError("Embedding provider must match the index profile.")

        self._version_repository = version_repository
        self._chunk_repository = chunk_repository
        self._transaction_manager = transaction_manager
        self._chunker = chunker
        self._embedding_provider = embedding_provider
        self._vector_store = vector_store
        self._pricing_catalog = pricing_catalog
        self._index_profile = index_profile
        self._embedding_timeout_seconds = embedding_timeout_seconds
        self._embedding_batch_size = embedding_batch_size
        self._clock = clock or _utc_now
        self._observability_client = (
            observability_client if observability_client is not None else NoOpObservabilityClient()
        )

    async def execute(
        self,
        *,
        workspace_id: UUID,
        document_id: UUID,
        document_version_id: UUID,
    ) -> IndexDocumentVersionResult:
        """Index one version without implicitly activating it."""

        load_stage = _SafeIndexingStage(
            client=self._observability_client,
            name=_STAGE_LOAD,
        )
        load_stage.start()
        try:
            version = await self._prepare_version(
                workspace_id=workspace_id,
                document_id=document_id,
                document_version_id=document_version_id,
            )
            load_stage.complete(_ok_stage_update(load_stage))
        except Exception as error:
            load_stage.complete(_error_stage_update(error))
            raise
        finally:
            load_stage.close()

        if version.status is DocumentVersionStatus.READY:
            return IndexDocumentVersionResult(
                version=version,
                already_ready=True,
            )

        try:
            chunk_stage = _SafeIndexingStage(
                client=self._observability_client,
                name=_STAGE_CHUNK,
            )
            chunk_stage.start()
            try:
                chunks = self._generate_chunks(version)
                chunk_stage.complete(
                    _ok_stage_update(
                        chunk_stage,
                        metadata={
                            "chunk_count": len(chunks),
                        },
                    )
                )
            except Exception as error:
                chunk_stage.complete(_error_stage_update(error))
                raise
            finally:
                chunk_stage.close()

            await self._persist_chunks(
                version=version,
                chunks=chunks,
            )

            embedding_outcome = await self._embed_chunks(
                version=version,
                chunks=chunks,
            )
            cost_estimate = estimate_embedding_cost(
                provider=self._index_profile.embedding_provider,
                model=self._index_profile.embedding_model,
                input_tokens=embedding_outcome.input_tokens,
                catalog=self._pricing_catalog,
            )

            collection_profile = self._collection_profile()
            projection = _version_projection(version)
            points = tuple(
                KnowledgeVectorPoint.from_chunk(
                    chunk=chunk,
                    media_type=version.media_type,
                    vector=vector,
                )
                for chunk, vector in zip(
                    chunks,
                    embedding_outcome.vectors,
                    strict=True,
                )
            )

            upsert_stage = _SafeIndexingStage(
                client=self._observability_client,
                name=_STAGE_UPSERT,
            )
            upsert_stage.start()
            try:
                await self._vector_store.upsert_version_points(
                    profile=collection_profile,
                    projection=projection,
                    points=points,
                )
                upsert_stage.complete(
                    _ok_stage_update(
                        upsert_stage,
                        metadata={
                            "vector_count": len(points),
                            "chunk_count": len(chunks),
                        },
                    )
                )
            except Exception as error:
                upsert_stage.complete(_error_stage_update(error))
                raise
            finally:
                upsert_stage.close()

            verify_stage = _SafeIndexingStage(
                client=self._observability_client,
                name=_STAGE_VERIFY,
            )
            verify_stage.start()
            try:
                projected_count = await self._vector_store.count_version_points(
                    profile=collection_profile,
                    projection=projection,
                )
                if projected_count != len(chunks):
                    raise (KnowledgeProjectionCountMismatchError())

                verify_stage.complete(
                    _ok_stage_update(
                        verify_stage,
                        metadata={
                            "expected_vector_count": len(chunks),
                            "verified_vector_count": projected_count,
                        },
                    )
                )
            except Exception as error:
                verify_stage.complete(_error_stage_update(error))
                raise
            finally:
                verify_stage.close()

            persist_stage = _SafeIndexingStage(
                client=self._observability_client,
                name=_STAGE_PERSIST,
            )
            persist_stage.start()
            try:
                ready_version = await self._mark_ready(
                    workspace_id=workspace_id,
                    document_id=document_id,
                    document_version_id=document_version_id,
                    chunk_count=len(chunks),
                    embedding_input_tokens=(embedding_outcome.input_tokens),
                    embedding_estimated_cost_usd=(cost_estimate.estimated_cost_usd),
                    pricing_catalog_version=(cost_estimate.pricing_catalog_version),
                )
                persist_stage.complete(
                    _ok_stage_update(
                        persist_stage,
                        metadata={
                            "persisted_status": (DocumentVersionStatus.READY.value),
                            "chunk_count": len(chunks),
                        },
                    )
                )
            except Exception as error:
                persist_stage.complete(_error_stage_update(error))
                raise
            finally:
                persist_stage.close()
        except (
            EmbeddingError,
            KnowledgeIndexingError,
            KnowledgeVectorStoreError,
        ) as error:
            await self._observe_failure_persistence(
                workspace_id=workspace_id,
                document_id=document_id,
                document_version_id=document_version_id,
                error=error,
            )
            raise

        return IndexDocumentVersionResult(
            version=ready_version,
            already_ready=False,
        )

    async def _observe_failure_persistence(
        self,
        *,
        workspace_id: UUID,
        document_id: UUID,
        document_version_id: UUID,
        error: Exception,
    ) -> None:
        persist_stage = _SafeIndexingStage(
            client=self._observability_client,
            name=_STAGE_PERSIST,
        )
        persist_stage.start()
        try:
            await self._record_failure_without_masking(
                workspace_id=workspace_id,
                document_id=document_id,
                document_version_id=document_version_id,
                error=error,
            )
            persist_stage.complete(
                _ok_stage_update(
                    persist_stage,
                    metadata={
                        "persisted_status": (DocumentVersionStatus.FAILED.value),
                        "error_code": _failure_error_code(error),
                    },
                )
            )
        finally:
            persist_stage.close()

    async def _prepare_version(
        self,
        *,
        workspace_id: UUID,
        document_id: UUID,
        document_version_id: UUID,
    ) -> DocumentVersion:
        async with self._transaction_manager.transaction():
            version = await self._version_repository.get_for_update(
                workspace_id=workspace_id,
                document_id=document_id,
                document_version_id=document_version_id,
            )
            if version is None:
                raise (KnowledgeDocumentVersionNotFoundError())

            if version.status is DocumentVersionStatus.READY:
                self._require_matching_profile(version)
                return version

            candidate = version
            if version.status is DocumentVersionStatus.FAILED:
                candidate = version.prepare_retry(now=self._clock())

            try:
                prepared = candidate.bind_index_profile(
                    self._index_profile,
                    now=self._clock(),
                )
            except ValueError as error:
                raise (KnowledgeIndexProfileMismatchError()) from error

            if prepared != version:
                await self._version_repository.update(prepared)

            return prepared

    def _generate_chunks(
        self,
        version: DocumentVersion,
    ) -> tuple[DocumentChunk, ...]:
        try:
            chunks = tuple(self._chunker.chunk(version))
        except (
            RuntimeError,
            ValueError,
        ) as error:
            raise KnowledgeChunkingError() from error

        if not chunks:
            raise KnowledgeChunkingError()

        return chunks

    async def _persist_chunks(
        self,
        *,
        version: DocumentVersion,
        chunks: tuple[DocumentChunk, ...],
    ) -> None:
        try:
            async with self._transaction_manager.transaction():
                current = await self._version_repository.get_for_update(
                    workspace_id=version.workspace_id,
                    document_id=version.document_id,
                    document_version_id=version.id,
                )
                if current is None:
                    raise (KnowledgeDocumentVersionNotFoundError())

                self._require_matching_profile(current)

                await self._chunk_repository.add_many(chunks)
                persisted_count = await self._chunk_repository.count_by_version(
                    workspace_id=version.workspace_id,
                    document_id=version.document_id,
                    document_version_id=version.id,
                )
                if persisted_count != len(chunks):
                    raise (KnowledgeProjectionCountMismatchError())
        except DocumentChunkConflictError as error:
            raise KnowledgeChunkPersistenceError() from error

    async def _embed_chunks(
        self,
        *,
        version: DocumentVersion,
        chunks: tuple[DocumentChunk, ...],
    ) -> _EmbeddingOutcome:
        vectors: list[EmbeddingVector] = []
        total_input_tokens = 0

        for start in range(
            0,
            len(chunks),
            self._embedding_batch_size,
        ):
            batch = chunks[start : start + self._embedding_batch_size]
            response = await self._embedding_provider.embed(
                EmbeddingRequest(
                    operation=(EmbeddingOperation.KNOWLEDGE_INDEXING),
                    model=self._index_profile.embedding_model,
                    inputs=tuple(chunk.content for chunk in batch),
                    dimensions=(self._index_profile.embedding_dimensions),
                    timeout_seconds=(self._embedding_timeout_seconds),
                    metadata={
                        "workspace_id": str(version.workspace_id),
                        "document_id": str(version.document_id),
                        "document_version_id": str(version.id),
                    },
                )
            )

            batch_vectors, batch_input_tokens = self._validate_embedding_response(
                response=response,
                expected_count=len(batch),
            )
            vectors.extend(batch_vectors)
            total_input_tokens += batch_input_tokens

        if len(vectors) != len(chunks):
            raise EmbeddingInvalidResponseError()

        return _EmbeddingOutcome(
            vectors=tuple(vectors),
            input_tokens=total_input_tokens,
        )

    def _validate_embedding_response(
        self,
        *,
        response: EmbeddingProviderResponse,
        expected_count: int,
    ) -> tuple[tuple[EmbeddingVector, ...], int]:
        if (
            response.provider != self._index_profile.embedding_provider
            or response.model != self._index_profile.embedding_model
            or response.dimensions != self._index_profile.embedding_dimensions
            or len(response.embeddings) != expected_count
            or response.usage is None
            or response.usage.input_tokens is None
        ):
            raise EmbeddingInvalidResponseError(provider_request_id=(response.provider_request_id))

        return (
            response.embeddings,
            response.usage.input_tokens,
        )

    async def _mark_ready(
        self,
        *,
        workspace_id: UUID,
        document_id: UUID,
        document_version_id: UUID,
        chunk_count: int,
        embedding_input_tokens: int,
        embedding_estimated_cost_usd: Decimal | None,
        pricing_catalog_version: str,
    ) -> DocumentVersion:

        async with self._transaction_manager.transaction():
            version = await self._version_repository.get_for_update(
                workspace_id=workspace_id,
                document_id=document_id,
                document_version_id=document_version_id,
            )
            if version is None:
                raise (KnowledgeDocumentVersionNotFoundError())

            self._require_matching_profile(version)

            if version.status is DocumentVersionStatus.READY:
                return version

            ready = version.mark_ready(
                chunk_count=chunk_count,
                embedding_input_tokens=(embedding_input_tokens),
                embedding_estimated_cost_usd=(embedding_estimated_cost_usd),
                embedding_pricing_catalog_version=(pricing_catalog_version),
                indexed_at=self._clock(),
            )
            await self._version_repository.update(ready)
            return ready

    async def _record_failure_without_masking(
        self,
        *,
        workspace_id: UUID,
        document_id: UUID,
        document_version_id: UUID,
        error: Exception,
    ) -> None:
        try:
            await self._record_failure(
                workspace_id=workspace_id,
                document_id=document_id,
                document_version_id=document_version_id,
                error_code=_failure_error_code(error),
            )
        except Exception:
            error.add_note(
                "The indexing failure could not be persisted without masking the original error."
            )

    async def _record_failure(
        self,
        *,
        workspace_id: UUID,
        document_id: UUID,
        document_version_id: UUID,
        error_code: str,
    ) -> None:
        async with self._transaction_manager.transaction():
            version = await self._version_repository.get_for_update(
                workspace_id=workspace_id,
                document_id=document_id,
                document_version_id=document_version_id,
            )
            if version is None or version.status is DocumentVersionStatus.READY:
                return

            chunk_count = await self._chunk_repository.count_by_version(
                workspace_id=workspace_id,
                document_id=document_id,
                document_version_id=document_version_id,
            )
            failed = version.mark_failed(
                error_code=error_code,
                chunk_count=chunk_count,
                now=self._clock(),
            )
            await self._version_repository.update(failed)

    def _require_matching_profile(
        self,
        version: DocumentVersion,
    ) -> None:
        if version.index_profile != self._index_profile:
            raise KnowledgeIndexProfileMismatchError()

    def _collection_profile(
        self,
    ) -> KnowledgeCollectionProfile:
        return KnowledgeCollectionProfile(
            collection_name=(self._index_profile.knowledge_collection),
            vector_name=(self._index_profile.knowledge_vector_name),
            dimensions=(self._index_profile.embedding_dimensions),
        )


class _SafeIndexingStage:
    """Isolate stage-observation failures from indexing behavior."""

    def __init__(
        self,
        *,
        client: ObservabilityClient,
        name: str,
    ) -> None:
        self._client = client
        self._name = name
        self._started_at = perf_counter()
        self._manager: AbstractContextManager[ObservationScope] | None = None
        self._scope: ObservationScope | None = None
        self._completed = False

    def start(self) -> None:
        self._started_at = perf_counter()
        try:
            self._manager = self._client.start_observation(
                ObservationAttributes(
                    name=self._name,
                    observation_type=ObservationType.SPAN,
                    metadata={},
                    metadata_paths=_STAGE_METADATA_PATHS,
                    input_data=None,
                    input_paths=frozenset(),
                    output_paths=frozenset(),
                )
            )
            self._scope = self._manager.__enter__()
        except Exception:
            self._manager = None
            self._scope = None

    def elapsed_milliseconds(self) -> int:
        return max(
            0,
            round((perf_counter() - self._started_at) * 1000),
        )

    def complete(self, update: ObservationUpdate | None) -> None:
        if self._scope is None or update is None or self._completed:
            return

        try:
            self._scope.update(update)
            self._completed = True
        except Exception:
            return

    def close(self) -> None:
        if self._manager is None:
            return

        try:
            self._manager.__exit__(None, None, None)
        except Exception:
            return
        finally:
            self._manager = None
            self._scope = None


def _ok_stage_update(
    stage: _SafeIndexingStage,
    *,
    metadata: dict[str, JsonValue] | None = None,
) -> ObservationUpdate | None:
    try:
        payload: dict[str, JsonValue] = {
            "latency_ms": stage.elapsed_milliseconds(),
        }
        if metadata is not None:
            payload.update(metadata)

        return ObservationUpdate(
            status=ObservationStatus.OK,
            metadata=payload,
        )
    except Exception:
        return None


def _error_stage_update(error: BaseException) -> ObservationUpdate | None:
    try:
        error_code = _observation_error_code(error)
        return ObservationUpdate(
            status=ObservationStatus.ERROR,
            metadata={
                "error_code": error_code,
            },
            error_code=error_code,
        )
    except Exception:
        return None


def _observation_error_code(error: BaseException) -> str:
    if isinstance(
        error,
        (
            EmbeddingError,
            KnowledgeIndexingError,
            KnowledgeVectorStoreError,
        ),
    ):
        return _failure_error_code(error)

    return _UNEXPECTED_FAILURE_CODE


def _version_projection(
    version: DocumentVersion,
) -> KnowledgeVersionProjection:
    return KnowledgeVersionProjection(
        workspace_id=version.workspace_id,
        document_id=version.document_id,
        document_version_id=version.id,
    )


def _failure_error_code(
    error: Exception,
) -> str:
    if isinstance(error, EmbeddingError):
        return error.error_code.value

    if isinstance(
        error,
        KnowledgeCollectionCompatibilityError,
    ):
        return "knowledge_collection_incompatible"

    if isinstance(
        error,
        KnowledgeVectorStoreUnavailableError,
    ):
        return "knowledge_vector_store_unavailable"

    if isinstance(
        error,
        KnowledgeVectorStoreOperationError,
    ):
        return "knowledge_vector_store_operation_failed"

    error_code = getattr(
        error,
        "error_code",
        None,
    )
    if error_code is not None:
        return str(error_code)

    return "knowledge_indexing_failed"


def _utc_now() -> datetime:
    return datetime.now(UTC)
