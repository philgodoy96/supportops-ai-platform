"""Stable application-owned errors for knowledge indexing."""

from enum import StrEnum
from typing import ClassVar


class KnowledgeIndexingErrorCode(StrEnum):
    """Durable operational codes for indexing failures."""

    CHUNKING_FAILED = "knowledge_chunking_failed"
    CHUNK_PERSISTENCE_CONFLICT = "knowledge_chunk_persistence_conflict"
    INDEX_PROFILE_MISMATCH = "knowledge_index_profile_mismatch"
    PROJECTION_COUNT_MISMATCH = "knowledge_projection_count_mismatch"
    COLLECTION_INCOMPATIBLE = "knowledge_collection_incompatible"
    VECTOR_STORE_UNAVAILABLE = "knowledge_vector_store_unavailable"
    VECTOR_STORE_OPERATION_FAILED = "knowledge_vector_store_operation_failed"


class KnowledgeIndexingError(Exception):
    """Base class for safe indexing orchestration errors."""

    safe_summary: ClassVar[str]
    retryable: ClassVar[bool]
    terminal: ClassVar[bool]

    def __init__(self) -> None:
        super().__init__(self.safe_summary)


class KnowledgeDocumentVersionNotFoundError(KnowledgeIndexingError):
    """Raised when the scoped version does not exist."""

    safe_summary = "Knowledge document version was not found."
    retryable = False
    terminal = True


class KnowledgeIndexProfileMismatchError(KnowledgeIndexingError):
    """Raised when runtime configuration conflicts with persisted identity."""

    error_code = KnowledgeIndexingErrorCode.INDEX_PROFILE_MISMATCH
    safe_summary = (
        "The configured knowledge index profile does not match "
        "the persisted document version profile."
    )
    retryable = False
    terminal = True


class KnowledgeChunkingError(KnowledgeIndexingError):
    """Raised when deterministic chunk generation fails."""

    error_code = KnowledgeIndexingErrorCode.CHUNKING_FAILED
    safe_summary = "The knowledge document could not be chunked deterministically."
    retryable = False
    terminal = True


class KnowledgeChunkPersistenceError(KnowledgeIndexingError):
    """Raised when authoritative chunk state conflicts with a rerun."""

    error_code = KnowledgeIndexingErrorCode.CHUNK_PERSISTENCE_CONFLICT
    safe_summary = "Persisted knowledge chunks conflict with the deterministic indexing result."
    retryable = False
    terminal = True


class KnowledgeProjectionCountMismatchError(KnowledgeIndexingError):
    """Raised when Qdrant does not contain the complete projection."""

    error_code = KnowledgeIndexingErrorCode.PROJECTION_COUNT_MISMATCH
    safe_summary = (
        "The knowledge vector projection count does not match the authoritative chunk count."
    )
    retryable = True
    terminal = False
