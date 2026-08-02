"""Provider-independent contracts for the knowledge vector projection."""

import re
from collections.abc import Sequence
from dataclasses import dataclass
from math import isfinite
from typing import Protocol
from uuid import UUID

from supportops.modules.knowledge_documents.domain.models import (
    DocumentChunk,
    DocumentMediaType,
)

type KnowledgeVector = tuple[float, ...]

_CONTENT_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class KnowledgeVectorStoreError(Exception):
    """Base class for safe knowledge-vector-store failures."""


class KnowledgeCollectionCompatibilityError(KnowledgeVectorStoreError):
    """Raised when an existing collection violates the expected profile."""


class KnowledgeVectorStoreUnavailableError(KnowledgeVectorStoreError):
    """Raised when the vector store cannot complete an infrastructure call."""


class KnowledgeVectorStoreOperationError(KnowledgeVectorStoreError):
    """Raised when the vector store rejects an indexing operation."""


@dataclass(frozen=True, slots=True)
class KnowledgeCollectionProfile:
    """Collection identity and dense-vector configuration."""

    collection_name: str
    vector_name: str
    dimensions: int

    def __post_init__(self) -> None:
        _validate_identifier(
            self.collection_name,
            field_name="collection_name",
        )
        _validate_identifier(
            self.vector_name,
            field_name="vector_name",
        )
        if self.dimensions <= 0:
            raise ValueError("dimensions must be positive.")


@dataclass(frozen=True, slots=True)
class KnowledgeVersionProjection:
    """Workspace-owned document version projected into a vector store."""

    workspace_id: UUID
    document_id: UUID
    document_version_id: UUID


@dataclass(frozen=True, slots=True)
class KnowledgeVectorPoint:
    """One deterministic chunk vector and its non-authoritative payload."""

    chunk_id: UUID
    workspace_id: UUID
    document_id: UUID
    document_version_id: UUID
    ordinal: int
    content_sha256: str
    media_type: DocumentMediaType
    chunking_strategy: str
    chunking_version: str
    vector: KnowledgeVector

    def __post_init__(self) -> None:
        if self.ordinal < 0:
            raise ValueError("ordinal must be non-negative.")
        if _CONTENT_SHA256_PATTERN.fullmatch(self.content_sha256) is None:
            raise ValueError("content_sha256 must be a lowercase SHA-256 digest.")

        _validate_identifier(
            self.chunking_strategy,
            field_name="chunking_strategy",
        )
        _validate_identifier(
            self.chunking_version,
            field_name="chunking_version",
        )

        object.__setattr__(
            self,
            "vector",
            _normalize_vector(self.vector),
        )

    @classmethod
    def from_chunk(
        cls,
        *,
        chunk: DocumentChunk,
        media_type: DocumentMediaType,
        vector: Sequence[float],
    ) -> "KnowledgeVectorPoint":
        """Create a projection point from authoritative PostgreSQL data."""

        return cls(
            chunk_id=chunk.id,
            workspace_id=chunk.workspace_id,
            document_id=chunk.document_id,
            document_version_id=chunk.document_version_id,
            ordinal=chunk.ordinal,
            content_sha256=chunk.content_sha256,
            media_type=media_type,
            chunking_strategy=chunk.chunking_strategy,
            chunking_version=chunk.chunking_version,
            vector=tuple(vector),
        )


class KnowledgeVectorStore(Protocol):
    """Maintain the rebuildable vector projection for knowledge chunks."""

    async def ensure_collection(
        self,
        profile: KnowledgeCollectionProfile,
    ) -> None:
        """Create or validate a compatible collection and payload indexes."""
        ...

    async def upsert_version_points(
        self,
        *,
        profile: KnowledgeCollectionProfile,
        projection: KnowledgeVersionProjection,
        points: Sequence[KnowledgeVectorPoint],
    ) -> None:
        """Idempotently upsert one complete document-version projection."""
        ...

    async def count_version_points(
        self,
        *,
        profile: KnowledgeCollectionProfile,
        projection: KnowledgeVersionProjection,
    ) -> int:
        """Count projected points for one owned document version."""
        ...


def _normalize_vector(
    vector: Sequence[float],
) -> KnowledgeVector:
    if not vector:
        raise ValueError("vector must contain at least one coordinate.")

    normalized: list[float] = []

    for index, coordinate in enumerate(vector):
        if isinstance(coordinate, bool) or not isinstance(
            coordinate,
            (int, float),
        ):
            raise TypeError(f"vector[{index}] must be numeric.")

        value = float(coordinate)
        if not isfinite(value):
            raise ValueError(f"vector[{index}] must be finite.")

        normalized.append(value)

    return tuple(normalized)


def _validate_identifier(
    value: str,
    *,
    field_name: str,
) -> None:
    if not value:
        raise ValueError(f"{field_name} is required.")
    if value != value.strip():
        raise ValueError(f"{field_name} must not contain surrounding whitespace.")
