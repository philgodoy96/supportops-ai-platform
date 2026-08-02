"""Application orchestration for authoritative semantic retrieval."""

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from uuid import UUID

from supportops.ai.embeddings.contracts import (
    EmbeddingOperation,
    EmbeddingProvider,
    EmbeddingProviderResponse,
    EmbeddingRequest,
    EmbeddingVector,
)
from supportops.ai.embeddings.errors import (
    EmbeddingInvalidResponseError,
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

DEFAULT_RETRIEVAL_CANDIDATE_MULTIPLIER = 4

_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class _EligibleScope:
    versions: tuple[ActiveKnowledgeVersion, ...]

    @property
    def targets(
        self,
    ) -> tuple[KnowledgeSearchTarget, ...]:
        return tuple(version.target for version in self.versions)


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

    async def execute(
        self,
        request: KnowledgeSearchRequest,
    ) -> KnowledgeSearchResult:
        """Return ranked PostgreSQL-backed evidence for one query."""

        resolved_versions = tuple(
            await self._active_version_resolver.resolve(
                workspace_id=request.workspace_id,
                document_ids=request.document_ids,
            )
        )
        eligible_scope = self._eligible_scope(resolved_versions)

        if not eligible_scope.versions:
            return KnowledgeSearchResult(
                request=request,
                searched_version_count=0,
                evidence=(),
            )

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

        evidence = await self._hydrate_evidence(
            request=request,
            scope=eligible_scope,
            candidates=candidates,
        )

        return KnowledgeSearchResult(
            request=request,
            searched_version_count=len(eligible_scope.versions),
            evidence=evidence,
        )

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
    ) -> tuple[KnowledgeEvidence, ...]:
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

        for candidate in ordered_candidates:
            if candidate.chunk_id in seen_chunk_ids:
                _log_discarded_candidate(
                    candidate=candidate,
                    reason_type="DuplicateChunk",
                )
                continue

            seen_chunk_ids.add(candidate.chunk_id)
            unique_candidates.append(candidate)

        if not unique_candidates:
            return ()

        chunks = tuple(
            await self._chunk_hydrator.hydrate(
                workspace_id=request.workspace_id,
                chunk_ids=tuple(candidate.chunk_id for candidate in unique_candidates),
            )
        )
        chunks_by_id = _unique_chunks_by_id(chunks)
        versions_by_target = {version.target: version for version in scope.versions}

        evidence: list[KnowledgeEvidence] = []

        for candidate in unique_candidates:
            active_version = versions_by_target.get(
                KnowledgeSearchTarget(
                    document_id=(candidate.document_id),
                    document_version_id=(candidate.document_version_id),
                )
            )
            chunk = chunks_by_id.get(candidate.chunk_id)

            if active_version is None:
                _log_discarded_candidate(
                    candidate=candidate,
                    reason_type=("InactiveVersionCandidate"),
                )
                continue

            if chunk is None:
                _log_discarded_candidate(
                    candidate=candidate,
                    reason_type=("AuthoritativeChunkMissing"),
                )
                continue

            try:
                hydrated = KnowledgeEvidence.from_sources(
                    rank=len(evidence) + 1,
                    candidate=candidate,
                    active_version=active_version,
                    chunk=chunk,
                )
            except ValueError as error:
                _log_discarded_candidate(
                    candidate=candidate,
                    reason_type=type(error).__name__,
                )
                continue

            evidence.append(hydrated)

            if len(evidence) == request.top_k:
                break

        return tuple(evidence)

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
