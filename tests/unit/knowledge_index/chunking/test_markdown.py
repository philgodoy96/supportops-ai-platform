"""Tests for deterministic Markdown-aware token chunking."""

from collections.abc import Sequence
from datetime import UTC, datetime
from decimal import Decimal
from itertools import pairwise
from uuid import UUID

import pytest

from supportops.knowledge_index.chunking.contracts import (
    ChunkingPolicy,
)
from supportops.knowledge_index.chunking.markdown import (
    MarkdownTokenChunker,
    chunk_ids,
)
from supportops.modules.knowledge_documents.domain.models import (
    DocumentChunk,
    DocumentMediaType,
    DocumentVersion,
    KnowledgeIndexProfile,
)

_WORKSPACE_ID = UUID("032c8c87-57cc-4d14-bfbd-04968b4e8cd4")
_DOCUMENT_ID = UUID("276046a2-28ec-4cb1-8bb6-a2ff70f9064b")
_VERSION_ID = UUID("09036916-84cf-4a58-bdf4-09bc52716ec5")
_CREATED_AT = datetime(
    2026,
    8,
    1,
    23,
    45,
    tzinfo=UTC,
)


class CharacterTokenizer:
    """Offline deterministic tokenizer with one token per character."""

    def __init__(
        self,
        *,
        encoding_name: str = "cl100k_base",
    ) -> None:
        self._encoding_name = encoding_name

    @property
    def encoding_name(self) -> str:
        """Return the configured encoding identity."""

        return self._encoding_name

    def encode(self, text: str) -> tuple[int, ...]:
        """Encode each Unicode code point."""

        return tuple(ord(character) for character in text)

    def decode(self, tokens: Sequence[int]) -> str:
        """Decode Unicode code points."""

        return "".join(chr(token) for token in tokens)


def create_profiled_version(
    *,
    content: str,
    media_type: DocumentMediaType = (DocumentMediaType.TEXT_MARKDOWN),
    policy: ChunkingPolicy | None = None,
) -> DocumentVersion:
    """Create one pending version with a matching index profile."""

    resolved_policy = policy or ChunkingPolicy()
    pending = DocumentVersion.create_pending(
        document_version_id=_VERSION_ID,
        workspace_id=_WORKSPACE_ID,
        document_id=_DOCUMENT_ID,
        version_number=1,
        media_type=media_type,
        content=content,
        now=_CREATED_AT,
    )
    return pending.bind_index_profile(
        KnowledgeIndexProfile(
            chunking_strategy=resolved_policy.strategy,
            chunking_version=resolved_policy.version,
            tokenizer_encoding=(resolved_policy.tokenizer_encoding),
            embedding_provider="mock",
            embedding_model="mock-hashing-embedding-v1",
            embedding_dimensions=64,
            knowledge_collection=("supportops-knowledge-mock-v1"),
            knowledge_vector_name="dense",
        ),
        now=_CREATED_AT,
    )


def create_chunker(
    *,
    policy: ChunkingPolicy | None = None,
) -> MarkdownTokenChunker:
    """Create an offline chunker for deterministic tests."""

    resolved_policy = policy or ChunkingPolicy()
    return MarkdownTokenChunker(
        policy=resolved_policy,
        tokenizer=CharacterTokenizer(encoding_name=resolved_policy.tokenizer_encoding),
    )


def assert_chunk_limits(
    chunks: tuple[DocumentChunk, ...],
    *,
    policy: ChunkingPolicy,
) -> None:
    """Assert stored counts match tokenized content and policy limits."""

    tokenizer = CharacterTokenizer(encoding_name=policy.tokenizer_encoding)

    for chunk in chunks:
        assert chunk.token_count == len(tokenizer.encode(chunk.content))
        assert chunk.token_count <= policy.max_tokens
        assert chunk.content.strip()


def test_chunker_tracks_markdown_heading_hierarchy_and_code_blocks() -> None:
    content = (
        "# Overview\n\n"
        "This runbook covers database recovery.\n\n"
        "## Recovery\n\n"
        "```sql\n"
        "SELECT pg_is_in_recovery();\n"
        "```\n\n"
        "Restart the connection pool after promotion.\n"
    )
    version = create_profiled_version(content=content)
    chunks = create_chunker().chunk(version)

    assert len(chunks) == 2
    assert chunks[0].section_path == ("Overview",)
    assert chunks[1].section_path == (
        "Overview",
        "Recovery",
    )
    assert "This runbook covers database recovery." in (chunks[0].content)
    assert ("```sql\nSELECT pg_is_in_recovery();\n```") in chunks[1].content
    assert "".join(chunk.content for chunk in chunks) == content


def test_plain_text_does_not_interpret_markdown_headings() -> None:
    content = "# This is plain text\n\nIt must not create a section path.\n"
    version = create_profiled_version(
        content=content,
        media_type=DocumentMediaType.TEXT_PLAIN,
    )

    chunks = create_chunker().chunk(version)

    assert len(chunks) == 1
    assert chunks[0].section_path == ()
    assert chunks[0].content == content


def test_chunker_enforces_maximum_size_and_overlap() -> None:
    policy = ChunkingPolicy(
        max_tokens=50,
        overlap_tokens=10,
    )
    content = "# Long procedure\n\n" + ("a" * 160) + "\n"
    version = create_profiled_version(
        content=content,
        policy=policy,
    )

    chunks = create_chunker(policy=policy).chunk(version)

    assert len(chunks) > 1
    assert_chunk_limits(chunks, policy=policy)

    for previous, current in pairwise(chunks):
        assert (
            previous.content[-policy.overlap_tokens :] == (current.content[: policy.overlap_tokens])
        )


def test_chunker_prefers_paragraph_boundary_before_hard_limit() -> None:
    policy = ChunkingPolicy(
        max_tokens=50,
        overlap_tokens=5,
    )
    content = "# Recovery\n\nFirst paragraph.\n\nSecond paragraph is deliberately longer.\n"
    version = create_profiled_version(
        content=content,
        policy=policy,
    )

    chunks = create_chunker(policy=policy).chunk(version)

    assert len(chunks) == 2
    assert chunks[0].content == ("# Recovery\n\nFirst paragraph.\n\n")
    assert chunks[0].token_count < policy.max_tokens
    assert chunks[0].section_path == ("Recovery",)
    assert chunks[1].section_path == ("Recovery",)
    assert_chunk_limits(chunks, policy=policy)


def test_oversized_fenced_code_block_is_split_without_exceeding_limit() -> None:
    policy = ChunkingPolicy(
        max_tokens=40,
        overlap_tokens=8,
    )
    content = "# Diagnostics\n\n```text\n" + ("diagnostic-output-" * 8) + "\n```\n"
    version = create_profiled_version(
        content=content,
        policy=policy,
    )

    chunks = create_chunker(policy=policy).chunk(version)

    assert len(chunks) > 1
    assert all(chunk.section_path == ("Diagnostics",) for chunk in chunks)
    assert chunks[0].content.startswith("# Diagnostics")
    assert chunks[-1].content.endswith("```\n")
    assert_chunk_limits(chunks, policy=policy)


def test_chunking_is_deterministic_across_safe_reruns() -> None:
    content = "# Recovery\n\nRestart the database connection pool.\n"
    version = create_profiled_version(content=content)
    chunker = create_chunker()

    first = chunker.chunk(version)
    second = chunker.chunk(version)

    assert first == second
    assert chunk_ids(first) == chunk_ids(second)
    assert all(chunk.created_at == version.created_at for chunk in first)


def test_chunker_rejects_index_profile_mismatch() -> None:
    version_policy = ChunkingPolicy()
    version = create_profiled_version(
        content="# Recovery\n",
        policy=version_policy,
    )
    incompatible_policy = ChunkingPolicy(
        version="v2",
    )
    chunker = create_chunker(policy=incompatible_policy)

    with pytest.raises(
        ValueError,
        match=(
            r"Document version index profile does not match "
            r"the configured chunking policy\."
        ),
    ):
        chunker.chunk(version)


def test_chunker_rejects_ready_version_rewrite() -> None:
    version = create_profiled_version(content="# Recovery\n")
    ready = version.mark_ready(
        chunk_count=1,
        embedding_input_tokens=3,
        embedding_estimated_cost_usd=Decimal("0"),
        embedding_pricing_catalog_version=("supportops-embedding-pricing-2026-08-01"),
        indexed_at=_CREATED_AT,
    )

    with pytest.raises(
        ValueError,
        match=(r"Only pending document versions may be chunked\."),
    ):
        create_chunker().chunk(ready)


@pytest.mark.parametrize(
    ("max_tokens", "overlap_tokens"),
    [
        (0, 0),
        (10, -1),
        (10, 10),
        (10, 11),
    ],
)
def test_chunking_policy_rejects_invalid_limits(
    max_tokens: int,
    overlap_tokens: int,
) -> None:
    with pytest.raises(ValueError):
        ChunkingPolicy(
            max_tokens=max_tokens,
            overlap_tokens=overlap_tokens,
        )
