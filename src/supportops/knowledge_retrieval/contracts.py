"""Provider-independent contracts for semantic knowledge retrieval."""

import re
from collections.abc import Sequence
from dataclasses import dataclass
from math import isfinite
from typing import Protocol
from uuid import UUID

from supportops.modules.knowledge_documents.domain.content import (
    compute_content_sha256,
)
from supportops.modules.knowledge_documents.domain.models import (
    DocumentChunk,
    DocumentMediaType,
    KnowledgeIndexProfile,
)

DEFAULT_KNOWLEDGE_SEARCH_TOP_K = 5
MAX_KNOWLEDGE_SEARCH_TOP_K = 20
MAX_KNOWLEDGE_SEARCH_QUERY_LENGTH = 2000
MAX_KNOWLEDGE_SEARCH_DOCUMENT_FILTERS = 20
MAX_KNOWLEDGE_VECTOR_CANDIDATES = 100

_CONTENT_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")

type KnowledgeQueryVector = tuple[float, ...]


@dataclass(frozen=True, slots=True)
class KnowledgeSearchRequest:
    """Workspace-scoped semantic knowledge query."""

    workspace_id: UUID
    query: str
    top_k: int = DEFAULT_KNOWLEDGE_SEARCH_TOP_K
    document_ids: tuple[UUID, ...] = ()

    def __post_init__(self) -> None:
        normalized_query = self.query.strip()
        if not normalized_query:
            raise ValueError("query must contain meaningful text.")
        if len(normalized_query) > MAX_KNOWLEDGE_SEARCH_QUERY_LENGTH:
            raise ValueError("query exceeds the maximum length.")
        if not 1 <= self.top_k <= MAX_KNOWLEDGE_SEARCH_TOP_K:
            raise ValueError(f"top_k must be between 1 and {MAX_KNOWLEDGE_SEARCH_TOP_K}.")

        normalized_document_ids = tuple(self.document_ids)
        if len(normalized_document_ids) > MAX_KNOWLEDGE_SEARCH_DOCUMENT_FILTERS:
            raise ValueError("document_ids exceeds the maximum number of filters.")

        _validate_uuid_sequence(
            normalized_document_ids,
            field_name="document_ids",
        )
        if len(set(normalized_document_ids)) != len(normalized_document_ids):
            raise ValueError("document_ids must not contain duplicates.")

        object.__setattr__(
            self,
            "query",
            normalized_query,
        )
        object.__setattr__(
            self,
            "document_ids",
            normalized_document_ids,
        )


@dataclass(frozen=True, slots=True)
class KnowledgeSearchTarget:
    """One active document-version pair eligible for vector search."""

    document_id: UUID
    document_version_id: UUID


@dataclass(frozen=True, slots=True)
class ActiveKnowledgeVersion:
    """Authoritative active ready version resolved from PostgreSQL."""

    workspace_id: UUID
    document_id: UUID
    document_title: str
    document_external_reference: str | None
    document_version_id: UUID
    version_number: int
    media_type: DocumentMediaType
    index_profile: KnowledgeIndexProfile

    def __post_init__(self) -> None:
        _validate_required_text(
            self.document_title,
            field_name="document_title",
        )
        _validate_optional_text(
            self.document_external_reference,
            field_name="document_external_reference",
        )

        if self.version_number <= 0:
            raise ValueError("version_number must be positive.")
        if not isinstance(
            self.media_type,
            DocumentMediaType,
        ):
            raise TypeError("media_type must be a DocumentMediaType.")
        if not isinstance(
            self.index_profile,
            KnowledgeIndexProfile,
        ):
            raise TypeError("index_profile must be a KnowledgeIndexProfile.")

    @property
    def target(self) -> KnowledgeSearchTarget:
        """Return the vector-search identity for this active version."""

        return KnowledgeSearchTarget(
            document_id=self.document_id,
            document_version_id=(self.document_version_id),
        )


@dataclass(frozen=True, slots=True)
class KnowledgeVectorSearchRequest:
    """Validated vector-store candidate-search request."""

    workspace_id: UUID
    profile: KnowledgeIndexProfile
    targets: tuple[KnowledgeSearchTarget, ...]
    query_vector: KnowledgeQueryVector
    limit: int

    def __post_init__(self) -> None:
        if not isinstance(
            self.profile,
            KnowledgeIndexProfile,
        ):
            raise TypeError("profile must be a KnowledgeIndexProfile.")

        normalized_targets = tuple(self.targets)
        if not normalized_targets:
            raise ValueError("Vector search requires at least one active target.")

        for index, target in enumerate(normalized_targets):
            if not isinstance(
                target,
                KnowledgeSearchTarget,
            ):
                raise TypeError(f"targets[{index}] must be a KnowledgeSearchTarget.")

        if len(set(normalized_targets)) != len(normalized_targets):
            raise ValueError("Vector search targets must not contain duplicates.")

        normalized_vector = _normalize_vector(
            self.query_vector,
            dimensions=(self.profile.embedding_dimensions),
        )

        if not 1 <= self.limit <= (MAX_KNOWLEDGE_VECTOR_CANDIDATES):
            raise ValueError(f"limit must be between 1 and {MAX_KNOWLEDGE_VECTOR_CANDIDATES}.")

        object.__setattr__(
            self,
            "targets",
            normalized_targets,
        )
        object.__setattr__(
            self,
            "query_vector",
            normalized_vector,
        )


@dataclass(frozen=True, slots=True)
class KnowledgeVectorCandidate:
    """One non-authoritative vector-search candidate."""

    chunk_id: UUID
    workspace_id: UUID
    document_id: UUID
    document_version_id: UUID
    ordinal: int
    content_sha256: str
    media_type: DocumentMediaType
    chunking_strategy: str
    chunking_version: str
    score: float

    def __post_init__(self) -> None:
        if self.ordinal < 0:
            raise ValueError("ordinal must be non-negative.")

        _validate_content_sha256(self.content_sha256)

        if not isinstance(
            self.media_type,
            DocumentMediaType,
        ):
            raise TypeError("media_type must be a DocumentMediaType.")

        _validate_required_text(
            self.chunking_strategy,
            field_name="chunking_strategy",
        )
        _validate_required_text(
            self.chunking_version,
            field_name="chunking_version",
        )

        normalized_score = _normalize_score(self.score)
        object.__setattr__(
            self,
            "score",
            normalized_score,
        )


@dataclass(frozen=True, slots=True)
class KnowledgeCitation:
    """Stable citation metadata for one authoritative chunk."""

    workspace_id: UUID
    document_id: UUID
    document_title: str
    document_external_reference: str | None
    document_version_id: UUID
    version_number: int
    chunk_id: UUID
    ordinal: int
    section_path: tuple[str, ...]
    media_type: DocumentMediaType

    def __post_init__(self) -> None:
        _validate_required_text(
            self.document_title,
            field_name="document_title",
        )
        _validate_optional_text(
            self.document_external_reference,
            field_name="document_external_reference",
        )

        if self.version_number <= 0:
            raise ValueError("version_number must be positive.")
        if self.ordinal < 0:
            raise ValueError("ordinal must be non-negative.")
        if not isinstance(
            self.section_path,
            tuple,
        ):
            raise TypeError("section_path must be a tuple.")

        for index, segment in enumerate(self.section_path):
            _validate_required_text(
                segment,
                field_name=f"section_path[{index}]",
            )

        if not isinstance(
            self.media_type,
            DocumentMediaType,
        ):
            raise TypeError("media_type must be a DocumentMediaType.")


@dataclass(frozen=True, slots=True)
class KnowledgeEvidence:
    """Ranked authoritative evidence hydrated from PostgreSQL."""

    rank: int
    score: float
    content: str
    content_sha256: str
    token_count: int
    citation: KnowledgeCitation

    def __post_init__(self) -> None:
        if self.rank <= 0:
            raise ValueError("rank must be positive.")

        normalized_score = _normalize_score(self.score)

        if not self.content.strip():
            raise ValueError("Evidence content must contain meaningful text.")

        _validate_content_sha256(self.content_sha256)
        if compute_content_sha256(self.content) != self.content_sha256:
            raise ValueError("content_sha256 must match evidence content.")

        if self.token_count <= 0:
            raise ValueError("token_count must be positive.")

        object.__setattr__(
            self,
            "score",
            normalized_score,
        )

    @classmethod
    def from_sources(
        cls,
        *,
        rank: int,
        candidate: KnowledgeVectorCandidate,
        active_version: ActiveKnowledgeVersion,
        chunk: DocumentChunk,
    ) -> "KnowledgeEvidence":
        """Hydrate and validate one candidate against authoritative state."""

        if (
            candidate.workspace_id != active_version.workspace_id
            or candidate.document_id != active_version.document_id
            or candidate.document_version_id != active_version.document_version_id
        ):
            raise ValueError("Vector candidate does not belong to the active knowledge version.")

        if (
            chunk.workspace_id != active_version.workspace_id
            or chunk.document_id != active_version.document_id
            or chunk.document_version_id != active_version.document_version_id
            or chunk.id != candidate.chunk_id
        ):
            raise ValueError(
                "Authoritative chunk does not belong to the vector candidate and active version."
            )

        profile = active_version.index_profile
        if (
            chunk.ordinal != candidate.ordinal
            or chunk.content_sha256 != candidate.content_sha256
            or chunk.chunking_strategy != candidate.chunking_strategy
            or chunk.chunking_version != candidate.chunking_version
            or chunk.chunking_strategy != profile.chunking_strategy
            or chunk.chunking_version != profile.chunking_version
            or candidate.media_type is not active_version.media_type
        ):
            raise ValueError(
                "Vector candidate provenance does not match authoritative PostgreSQL state."
            )

        return cls(
            rank=rank,
            score=candidate.score,
            content=chunk.content,
            content_sha256=(chunk.content_sha256),
            token_count=chunk.token_count,
            citation=KnowledgeCitation(
                workspace_id=(active_version.workspace_id),
                document_id=(active_version.document_id),
                document_title=(active_version.document_title),
                document_external_reference=(active_version.document_external_reference),
                document_version_id=(active_version.document_version_id),
                version_number=(active_version.version_number),
                chunk_id=chunk.id,
                ordinal=chunk.ordinal,
                section_path=chunk.section_path,
                media_type=(active_version.media_type),
            ),
        )


@dataclass(frozen=True, slots=True)
class KnowledgeSearchResult:
    """Semantic search result containing authoritative evidence."""

    request: KnowledgeSearchRequest
    searched_version_count: int
    evidence: tuple[KnowledgeEvidence, ...]

    def __post_init__(self) -> None:
        if self.searched_version_count < 0:
            raise ValueError("searched_version_count must be non-negative.")

        normalized_evidence = tuple(self.evidence)
        if len(normalized_evidence) > self.request.top_k:
            raise ValueError("Evidence count must not exceed requested top_k.")

        expected_ranks = tuple(
            range(
                1,
                len(normalized_evidence) + 1,
            )
        )
        actual_ranks = tuple(item.rank for item in normalized_evidence)
        if actual_ranks != expected_ranks:
            raise ValueError("Evidence ranks must be contiguous and one-based.")

        scores = tuple(item.score for item in normalized_evidence)
        if scores != tuple(
            sorted(
                scores,
                reverse=True,
            )
        ):
            raise ValueError("Evidence must be ordered by descending score.")

        chunk_ids: set[UUID] = set()

        for item in normalized_evidence:
            if item.citation.workspace_id != self.request.workspace_id:
                raise ValueError("Evidence must belong to the requested workspace.")

            if item.citation.chunk_id in chunk_ids:
                raise ValueError("Evidence must not contain duplicate chunks.")

            chunk_ids.add(item.citation.chunk_id)

        object.__setattr__(
            self,
            "evidence",
            normalized_evidence,
        )


class ActiveKnowledgeVersionResolver(Protocol):
    """Resolve active ready versions through PostgreSQL ownership."""

    async def resolve(
        self,
        *,
        workspace_id: UUID,
        document_ids: tuple[UUID, ...],
    ) -> Sequence[ActiveKnowledgeVersion]:
        """Return eligible active versions for one workspace."""
        ...


class KnowledgeChunkHydrator(Protocol):
    """Load authoritative PostgreSQL chunks for vector candidates."""

    async def hydrate(
        self,
        *,
        workspace_id: UUID,
        chunk_ids: tuple[UUID, ...],
    ) -> Sequence[DocumentChunk]:
        """Return workspace-owned chunks without Qdrant text."""
        ...


class KnowledgeVectorSearcher(Protocol):
    """Return candidates from the rebuildable vector projection."""

    async def search(
        self,
        request: KnowledgeVectorSearchRequest,
    ) -> Sequence[KnowledgeVectorCandidate]:
        """Search one compatible active-version scope."""
        ...


def _normalize_vector(
    vector: Sequence[float],
    *,
    dimensions: int,
) -> KnowledgeQueryVector:
    if len(vector) != dimensions:
        raise ValueError("query_vector must match profile embedding dimensions.")

    normalized: list[float] = []

    for index, coordinate in enumerate(vector):
        if isinstance(
            coordinate,
            bool,
        ) or not isinstance(
            coordinate,
            (int, float),
        ):
            raise TypeError(f"query_vector[{index}] must be numeric.")

        value = float(coordinate)
        if not isfinite(value):
            raise ValueError(f"query_vector[{index}] must be finite.")

        normalized.append(value)

    return tuple(normalized)


def _normalize_score(
    score: float,
) -> float:
    if isinstance(score, bool) or not isinstance(
        score,
        (int, float),
    ):
        raise TypeError("score must be numeric.")

    normalized_score = float(score)
    if not isfinite(normalized_score):
        raise ValueError("score must be finite.")

    return normalized_score


def _validate_uuid_sequence(
    values: tuple[UUID, ...],
    *,
    field_name: str,
) -> None:
    for index, value in enumerate(values):
        if not isinstance(value, UUID):
            raise TypeError(f"{field_name}[{index}] must be a UUID.")


def _validate_content_sha256(
    value: str,
) -> None:
    if _CONTENT_SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError("content_sha256 must be a lowercase SHA-256 digest.")


def _validate_required_text(
    value: str,
    *,
    field_name: str,
) -> None:
    if not value:
        raise ValueError(f"{field_name} is required.")
    if value != value.strip():
        raise ValueError(f"{field_name} must not contain surrounding whitespace.")


def _validate_optional_text(
    value: str | None,
    *,
    field_name: str,
) -> None:
    if value is not None:
        _validate_required_text(
            value,
            field_name=field_name,
        )
