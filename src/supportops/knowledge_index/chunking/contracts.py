"""Contracts and policy for deterministic knowledge-document chunking."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from supportops.modules.knowledge_documents.domain.models import (
    DocumentChunk,
    DocumentVersion,
)

DEFAULT_CHUNKING_STRATEGY = "markdown-token"
DEFAULT_CHUNKING_VERSION = "v1"
DEFAULT_TOKENIZER_ENCODING = "cl100k_base"
DEFAULT_MAX_CHUNK_TOKENS = 500
DEFAULT_CHUNK_OVERLAP_TOKENS = 75


@dataclass(frozen=True, slots=True)
class ChunkingPolicy:
    """Immutable identity and limits for one chunking algorithm."""

    strategy: str = DEFAULT_CHUNKING_STRATEGY
    version: str = DEFAULT_CHUNKING_VERSION
    tokenizer_encoding: str = DEFAULT_TOKENIZER_ENCODING
    max_tokens: int = DEFAULT_MAX_CHUNK_TOKENS
    overlap_tokens: int = DEFAULT_CHUNK_OVERLAP_TOKENS

    def __post_init__(self) -> None:
        _validate_identifier(
            self.strategy,
            field_name="strategy",
        )
        _validate_identifier(
            self.version,
            field_name="version",
        )
        _validate_identifier(
            self.tokenizer_encoding,
            field_name="tokenizer_encoding",
        )

        if self.max_tokens <= 0:
            raise ValueError("max_tokens must be positive.")
        if self.overlap_tokens < 0:
            raise ValueError("overlap_tokens must be non-negative.")
        if self.overlap_tokens >= self.max_tokens:
            raise ValueError("overlap_tokens must be smaller than max_tokens.")


class TextTokenizer(Protocol):
    """Minimal tokenizer boundary owned by the indexing package."""

    @property
    def encoding_name(self) -> str:
        """Return the stable tokenizer encoding identity."""
        ...

    def encode(self, text: str) -> tuple[int, ...]:
        """Encode text into deterministic integer tokens."""
        ...

    def decode(self, tokens: Sequence[int]) -> str:
        """Decode integer tokens into text."""
        ...


class KnowledgeDocumentChunker(Protocol):
    """Convert one profiled pending version into authoritative chunks."""

    @property
    def policy(self) -> ChunkingPolicy:
        """Return the immutable policy used by the chunker."""
        ...

    def chunk(
        self,
        document_version: DocumentVersion,
    ) -> tuple[DocumentChunk, ...]:
        """Create deterministic chunks without performing persistence."""
        ...


def _validate_identifier(
    value: str,
    *,
    field_name: str,
) -> None:
    if not value:
        raise ValueError(f"{field_name} is required.")
    if value != value.strip():
        raise ValueError(f"{field_name} must not contain surrounding whitespace.")
