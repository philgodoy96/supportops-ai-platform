"""Application orchestration for authoritative semantic retrieval."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass
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
from supportops.knowledge_index.vector_store.contracts import (
    KnowledgeVectorStoreError,
)
from supportops.knowledge_retrieval.contracts import (
    MAX_KNOWLEDGE_VECTOR_CANDIDATES,
    ActiveKnowledgeVersion,
    ActiveKnowledgeVersionResolver,
    KnowledgeChunkHydrator,
    KnowledgeEvidence,
    KnowledgeSearchRequest,
    KnowledgeSearchResult,
    KnowledgeSearchTarget,
    KnowledgeVectorCandidate,
    KnowledgeVectorSearcher,
    KnowledgeVectorSearchRequest,
)
from supportops.modules.knowledge_documents.domain.models import (
    DocumentChunk,
    KnowledgeIndexProfile,
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

DEFAULT_RETRIEVAL_CANDIDATE_MULTIPLIER = 4

_OBSERVATION_NAME: Final = "knowledge.search"
_UNEXPECTED_FAILURE_CODE: Final = "knowledge_retrieval_unexpected_failure"
_VECTOR_STORE_FAILURE_CODE: Final = "knowledge_retrieval_unavailable"

_OBSERVATION_METADATA_KEYS: Final = frozenset(
    {
        "workspace_id",
        "top_k",
        "requested_document_count",
        "requested_document_version_count",
        "embedding_provider",
        "embedding_model",
        "embedding_dimensions",
        "correlation_id",
        "agent_run_id",
        "agent_run_attempt_id",
        "searched_version_count",
        "candidate_count",
        "hydrated_candidate_count",
        "evidence_count",
        "filtered_candidate_count",
        "latency_ms",
        "status",
        "error_code",
    }
)
_OBSERVATION_METADATA_PATHS: Final = frozenset((key,) for key in _OBSERVATION_METADATA_KEYS)

_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class _EligibleScope:
    versions: tuple[ActiveKnowledgeVersion, ...]

    @property
    def targets(
        self,
    ) -> tuple[KnowledgeSearchTarget, ...]:
        return tuple(version.target for version in self.versions)


@dataclass(frozen=True, slots=True)
class _HydrationOutcome:
    evidence: tuple[KnowledgeEvidence, ...]
    hydrated_candidate_count: int
    filtered_candidate_count: int


class SearchKnowledge:
    """Retrieve authoritative evidence from active ready versions."""

    def __init__(
        self,
        *,
        active_version_resolver: (ActiveKnowledgeVersionResolver),
        chunk_hydrator: KnowledgeChunkHydrator,
        embedding_provider: EmbeddingProvider,
        vector_searcher: KnowledgeVectorSearcher,
        index_profile: KnowledgeIndexProfile,
        embedding_timeout_seconds: float,
        candidate_multiplier: int = (DEFAULT_RETRIEVAL_CANDIDATE_MULTIPLIER),
        observability_client: ObservabilityClient | None = None,
    ) -> None:
        if embedding_timeout_seconds <= 0:
            raise ValueError("embedding_timeout_seconds must be positive.")
        if candidate_multiplier <= 0:
            raise ValueError("candidate_multiplier must be positive.")
        if embedding_provider.provider_name != index_profile.embedding_provider:
            raise ValueError("Embedding provider must match the retrieval index profile.")

        self._active_version_resolver = active_version_resolver
        self._chunk_hydrator = chunk_hydrator
        self._embedding_provider = embedding_provider
        self._vector_searcher = vector_searcher
        self._index_profile = index_profile
        self._embedding_timeout_seconds = embedding_timeout_seconds
        self._candidate_multiplier = candidate_multiplier
        self._observability_client = (
            observability_client if observability_client is not None else NoOpObservabilityClient()
        )

    async def execute(
        self,
        request: KnowledgeSearchRequest,
    ) -> KnowledgeSearchResult:
        """Return ranked PostgreSQL-backed evidence for one query."""

        observation = _SafeRetrievalObservation(
            client=self._observability_client,
            attributes=_start_attributes(
                request=request,
                index_profile=self._index_profile,
            ),
        )
        observation.start()
        started_at = perf_counter()

        try:
            resolved_versions = tuple(
                await self._active_version_resolver.resolve(
                    workspace_id=request.workspace_id,
                    document_ids=request.document_ids,
                )
            )
            eligible_scope = self._eligible_scope(resolved_versions)

            if not eligible_scope.versions:
                result = KnowledgeSearchResult(
                    request=request,
                    searched_version_count=0,
                    evidence=(),
                )
                observation.update(
                    _success_update(
                        result=result,
                        candidate_count=0,
                        hydrated_candidate_count=0,
                        filtered_candidate_count=0,
                        latency_ms=_elapsed_milliseconds(started_at),
                    )
                )
                return result

            query_vector = await self._embed_query(request)
            candidates = tuple(
                await self._vector_searcher.search(
                    KnowledgeVectorSearchRequest(
                        workspace_id=(request.workspace_id),
                        profile=self._index_profile,
                        targets=eligible_scope.targets,
                        query_vector=query_vector,
                        limit=self._candidate_limit(request.top_k),
                    )
                )
            )

            hydration = await self._hydrate_evidence(
                request=request,
                scope=eligible_scope,
                candidates=candidates,
            )

            result = KnowledgeSearchResult(
                request=request,
                searched_version_count=len(eligible_scope.versions),
                evidence=hydration.evidence,
            )
            observation.update(
                _success_update(
                    result=result,
                    candidate_count=len(candidates),
                    hydrated_candidate_count=(hydration.hydrated_candidate_count),
                    filtered_candidate_count=(hydration.filtered_candidate_count),
                    latency_ms=_elapsed_milliseconds(started_at),
                )
            )
            return result
        except EmbeddingError as error:
            observation.update(
                _failure_update(
                    latency_ms=_elapsed_milliseconds(started_at),
                    error_code=error.error_code.value,
                )
            )
            raise
        except KnowledgeVectorStoreError:
            observation.update(
                _failure_update(
                    latency_ms=_elapsed_milliseconds(started_at),
                    error_code=_VECTOR_STORE_FAILURE_CODE,
                )
            )
            raise
        except Exception:
            observation.update(
                _failure_update(
                    latency_ms=_elapsed_milliseconds(started_at),
                    error_code=_UNEXPECTED_FAILURE_CODE,
                )
            )
            raise
        finally:
            observation.close()

    def _eligible_scope(
        self,
        versions: Sequence[ActiveKnowledgeVersion],
    ) -> _EligibleScope:
        eligible_by_target: dict[
            KnowledgeSearchTarget,
            ActiveKnowledgeVersion,
        ] = {}

        for version in versions:
            if version.index_profile != self._index_profile:
                _log_discarded_active_version(
                    version=version,
                    reason_type=("IndexProfileMismatch"),
                )
                continue

            if version.target in eligible_by_target:
                _log_discarded_active_version(
                    version=version,
                    reason_type=("DuplicateActiveTarget"),
                )
                continue

            eligible_by_target[version.target] = version

        return _EligibleScope(versions=tuple(eligible_by_target.values()))

    async def _embed_query(
        self,
        request: KnowledgeSearchRequest,
    ) -> EmbeddingVector:
        response = await self._embedding_provider.embed(
            EmbeddingRequest(
                operation=(EmbeddingOperation.KNOWLEDGE_QUERY),
                model=(self._index_profile.embedding_model),
                inputs=(request.query,),
                dimensions=(self._index_profile.embedding_dimensions),
                timeout_seconds=(self._embedding_timeout_seconds),
                metadata={
                    "workspace_id": str(request.workspace_id),
                },
            )
        )

        return self._validate_query_embedding(response)

    def _validate_query_embedding(
        self,
        response: EmbeddingProviderResponse,
    ) -> EmbeddingVector:
        if (
            response.provider != self._index_profile.embedding_provider
            or response.model != self._index_profile.embedding_model
            or response.dimensions != self._index_profile.embedding_dimensions
            or len(response.embeddings) != 1
        ):
            raise EmbeddingInvalidResponseError(provider_request_id=(response.provider_request_id))

        return response.embeddings[0]

    async def _hydrate_evidence(
        self,
        *,
        request: KnowledgeSearchRequest,
        scope: _EligibleScope,
        candidates: tuple[KnowledgeVectorCandidate, ...],
    ) -> _HydrationOutcome:
        ordered_candidates = tuple(
            sorted(
                candidates,
                key=lambda candidate: (
                    -candidate.score,
                    str(candidate.chunk_id),
                ),
            )
        )

        unique_candidates: list[KnowledgeVectorCandidate] = []
        seen_chunk_ids: set[UUID] = set()
        filtered_candidate_count = 0

        for candidate in ordered_candidates:
            if candidate.chunk_id in seen_chunk_ids:
                filtered_candidate_count += 1
                _log_discarded_candidate(
                    candidate=candidate,
                    reason_type="DuplicateChunk",
                )
                continue

            seen_chunk_ids.add(candidate.chunk_id)
            unique_candidates.append(candidate)

        if not unique_candidates:
            return _HydrationOutcome(
                evidence=(),
                hydrated_candidate_count=0,
                filtered_candidate_count=filtered_candidate_count,
            )

        chunks = tuple(
            await self._chunk_hydrator.hydrate(
                workspace_id=request.workspace_id,
                chunk_ids=tuple(candidate.chunk_id for candidate in unique_candidates),
            )
        )
        chunks_by_id = _unique_chunks_by_id(chunks)
        versions_by_target = {version.target: version for version in scope.versions}

        evidence: list[KnowledgeEvidence] = []
        hydrated_candidate_count = 0

        for candidate in unique_candidates:
            active_version = versions_by_target.get(
                KnowledgeSearchTarget(
                    document_id=(candidate.document_id),
                    document_version_id=(candidate.document_version_id),
                )
            )
            chunk = chunks_by_id.get(candidate.chunk_id)

            if active_version is None:
                filtered_candidate_count += 1
                _log_discarded_candidate(
                    candidate=candidate,
                    reason_type=("InactiveVersionCandidate"),
                )
                continue

            if chunk is None:
                filtered_candidate_count += 1
                _log_discarded_candidate(
                    candidate=candidate,
                    reason_type=("AuthoritativeChunkMissing"),
                )
                continue

            hydrated_candidate_count += 1

            try:
                hydrated = KnowledgeEvidence.from_sources(
                    rank=len(evidence) + 1,
                    candidate=candidate,
                    active_version=active_version,
                    chunk=chunk,
                )
            except ValueError as error:
                filtered_candidate_count += 1
                _log_discarded_candidate(
                    candidate=candidate,
                    reason_type=type(error).__name__,
                )
                continue

            evidence.append(hydrated)

            if len(evidence) == request.top_k:
                break

        return _HydrationOutcome(
            evidence=tuple(evidence),
            hydrated_candidate_count=hydrated_candidate_count,
            filtered_candidate_count=filtered_candidate_count,
        )

    def _candidate_limit(
        self,
        top_k: int,
    ) -> int:
        return min(
            MAX_KNOWLEDGE_VECTOR_CANDIDATES,
            max(
                top_k,
                top_k * self._candidate_multiplier,
            ),
        )


class _SafeRetrievalObservation:
    """Isolate observability failures from retrieval behavior."""

    def __init__(
        self,
        *,
        client: ObservabilityClient,
        attributes: ObservationAttributes,
    ) -> None:
        self._client = client
        self._attributes = attributes
        self._manager: AbstractContextManager[ObservationScope] | None = None
        self._scope: ObservationScope | None = None

    def start(self) -> None:
        try:
            self._manager = self._client.start_observation(self._attributes)
            self._scope = self._manager.__enter__()
        except Exception:
            self._manager = None
            self._scope = None

    def update(self, update: ObservationUpdate | None) -> None:
        if self._scope is None or update is None:
            return

        try:
            self._scope.update(update)
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


def _start_attributes(
    *,
    request: KnowledgeSearchRequest,
    index_profile: KnowledgeIndexProfile,
) -> ObservationAttributes:
    try:
        return ObservationAttributes(
            name=_OBSERVATION_NAME,
            observation_type=ObservationType.RETRIEVER,
            provider=index_profile.embedding_provider,
            model=index_profile.embedding_model,
            metadata=_start_metadata(
                request=request,
                index_profile=index_profile,
            ),
            metadata_paths=_OBSERVATION_METADATA_PATHS,
            input_data=None,
            input_paths=frozenset(),
            output_paths=frozenset(),
        )
    except Exception:
        return ObservationAttributes(
            name=_OBSERVATION_NAME,
            observation_type=ObservationType.RETRIEVER,
            input_data=None,
            input_paths=frozenset(),
            output_paths=frozenset(),
        )


def _start_metadata(
    *,
    request: KnowledgeSearchRequest,
    index_profile: KnowledgeIndexProfile,
) -> dict[str, JsonValue]:
    return {
        "workspace_id": str(request.workspace_id),
        "top_k": request.top_k,
        "requested_document_count": len(request.document_ids),
        "embedding_provider": index_profile.embedding_provider,
        "embedding_model": index_profile.embedding_model,
        "embedding_dimensions": index_profile.embedding_dimensions,
    }


def _success_update(
    *,
    result: KnowledgeSearchResult,
    candidate_count: int | None,
    hydrated_candidate_count: int | None,
    filtered_candidate_count: int | None,
    latency_ms: int,
) -> ObservationUpdate | None:
    try:
        metadata: dict[str, JsonValue] = {
            "searched_version_count": result.searched_version_count,
            "evidence_count": len(result.evidence),
            "latency_ms": latency_ms,
            "status": ObservationStatus.OK.value,
        }

        if candidate_count is not None:
            metadata["candidate_count"] = candidate_count
        if hydrated_candidate_count is not None:
            metadata["hydrated_candidate_count"] = hydrated_candidate_count
        if filtered_candidate_count is not None:
            metadata["filtered_candidate_count"] = filtered_candidate_count

        return ObservationUpdate(
            status=ObservationStatus.OK,
            metadata=metadata,
        )
    except Exception:
        return None


def _failure_update(
    *,
    latency_ms: int,
    error_code: str,
) -> ObservationUpdate | None:
    try:
        return ObservationUpdate(
            status=ObservationStatus.ERROR,
            metadata={
                "latency_ms": latency_ms,
                "status": ObservationStatus.ERROR.value,
                "error_code": error_code,
            },
            error_code=error_code,
        )
    except Exception:
        return None


def _elapsed_milliseconds(started_at: float) -> int:
    return max(
        0,
        round((perf_counter() - started_at) * 1000),
    )


def _unique_chunks_by_id(
    chunks: Sequence[DocumentChunk],
) -> dict[UUID, DocumentChunk]:
    chunks_by_id: dict[
        UUID,
        DocumentChunk,
    ] = {}

    for chunk in chunks:
        if chunk.id in chunks_by_id:
            _LOGGER.warning(
                "Discarded duplicate authoritative knowledge chunk.",
                extra={
                    "chunk_id": str(chunk.id),
                    "reason_type": ("DuplicateAuthoritativeChunk"),
                },
            )
            continue

        chunks_by_id[chunk.id] = chunk

    return chunks_by_id


def _log_discarded_active_version(
    *,
    version: ActiveKnowledgeVersion,
    reason_type: str,
) -> None:
    _LOGGER.warning(
        "Discarded ineligible active knowledge version.",
        extra={
            "workspace_id": str(version.workspace_id),
            "document_id": str(version.document_id),
            "document_version_id": str(version.document_version_id),
            "reason_type": reason_type,
        },
    )


def _log_discarded_candidate(
    *,
    candidate: KnowledgeVectorCandidate,
    reason_type: str,
) -> None:
    _LOGGER.warning(
        "Discarded inconsistent semantic knowledge candidate.",
        extra={
            "workspace_id": str(candidate.workspace_id),
            "document_id": str(candidate.document_id),
            "document_version_id": str(candidate.document_version_id),
            "chunk_id": str(candidate.chunk_id),
            "reason_type": reason_type,
        },
    )
