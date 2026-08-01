"""Tests for deterministic authoritative document chunks."""

from dataclasses import FrozenInstanceError, replace
from datetime import UTC, datetime
from uuid import UUID, uuid4

import pytest

from supportops.modules.knowledge_documents.domain.models import (
    DocumentChunk,
    DocumentMediaType,
    DocumentVersion,
    KnowledgeIndexProfile,
)

WORKSPACE_ID = UUID("032c8c87-57cc-4d14-bfbd-04968b4e8cd4")
DOCUMENT_ID = UUID("276046a2-28ec-4cb1-8bb6-a2ff70f9064b")
VERSION_ID = UUID("09036916-84cf-4a58-bdf4-09bc52716ec5")
NOW = datetime(2026, 8, 1, 18, 0, tzinfo=UTC)


def create_profiled_version(
    *,
    tokenizer_encoding: str = "cl100k_base",
) -> DocumentVersion:
    pending = DocumentVersion.create_pending(
        workspace_id=WORKSPACE_ID,
        document_id=DOCUMENT_ID,
        version_number=1,
        media_type=DocumentMediaType.TEXT_MARKDOWN,
        content=("# Connection exhaustion\nRestart the pool.\n"),
        document_version_id=VERSION_ID,
        now=NOW,
    )

    return pending.bind_index_profile(
        KnowledgeIndexProfile(
            chunking_strategy="markdown-token",
            chunking_version="v1",
            tokenizer_encoding=tokenizer_encoding,
            embedding_provider="mock",
            embedding_model="mock-hashing-embedding-v1",
            embedding_dimensions=64,
            knowledge_collection=("supportops-knowledge-mock-v1"),
            knowledge_vector_name="dense",
        ),
        now=NOW,
    )


def test_chunk_copies_ownership_and_builds_stable_identity() -> None:
    version = create_profiled_version()

    first = DocumentChunk.create(
        document_version=version,
        ordinal=0,
        section_path=(" Connection exhaustion ",),
        content=("Restart the pool before increasing connection limits."),
        token_count=9,
        now=NOW,
    )
    second = DocumentChunk.create(
        document_version=version,
        ordinal=0,
        section_path=("Connection exhaustion",),
        content=("Restart the pool before increasing connection limits."),
        token_count=9,
        now=NOW,
    )

    assert first.id == second.id
    assert first.id.version == 5
    assert first.workspace_id == WORKSPACE_ID
    assert first.document_id == DOCUMENT_ID
    assert first.document_version_id == VERSION_ID
    assert first.section_path == ("Connection exhaustion",)
    assert first.chunking_strategy == "markdown-token"
    assert first.chunking_version == "v1"


def test_chunk_identity_changes_with_ordinal_or_content() -> None:
    version = create_profiled_version()

    first = DocumentChunk.create(
        document_version=version,
        ordinal=0,
        section_path=(),
        content="Restart the pool.",
        token_count=4,
        now=NOW,
    )
    different_ordinal = DocumentChunk.create(
        document_version=version,
        ordinal=1,
        section_path=(),
        content="Restart the pool.",
        token_count=4,
        now=NOW,
    )
    different_content = DocumentChunk.create(
        document_version=version,
        ordinal=0,
        section_path=(),
        content="Escalate before restarting the pool.",
        token_count=6,
        now=NOW,
    )

    assert (
        len(
            {
                first.id,
                different_ordinal.id,
                different_content.id,
            }
        )
        == 3
    )


def test_chunk_identity_includes_tokenizer_encoding() -> None:
    content = "Restart the pool."

    default_encoding_chunk = DocumentChunk.create(
        document_version=create_profiled_version(tokenizer_encoding="cl100k_base"),
        ordinal=0,
        section_path=(),
        content=content,
        token_count=4,
        now=NOW,
    )
    alternate_encoding_chunk = DocumentChunk.create(
        document_version=create_profiled_version(tokenizer_encoding="o200k_base"),
        ordinal=0,
        section_path=(),
        content=content,
        token_count=4,
        now=NOW,
    )

    assert default_encoding_chunk.id != alternate_encoding_chunk.id


def test_chunk_requires_bound_index_profile() -> None:
    version = DocumentVersion.create_pending(
        workspace_id=WORKSPACE_ID,
        document_id=DOCUMENT_ID,
        version_number=1,
        media_type=DocumentMediaType.TEXT_PLAIN,
        content="Restart the pool.",
        now=NOW,
    )

    with pytest.raises(
        ValueError,
        match="index profile is required",
    ):
        DocumentChunk.create(
            document_version=version,
            ordinal=0,
            section_path=(),
            content="Restart the pool.",
            token_count=4,
            now=NOW,
        )


@pytest.mark.parametrize(
    ("ordinal", "content", "token_count"),
    [
        (-1, "Restart the pool.", 4),
        (0, "   ", 4),
        (0, "Restart the pool.", 0),
    ],
)
def test_chunk_rejects_invalid_core_fields(
    ordinal: int,
    content: str,
    token_count: int,
) -> None:
    with pytest.raises(ValueError):
        DocumentChunk.create(
            document_version=create_profiled_version(),
            ordinal=ordinal,
            section_path=(),
            content=content,
            token_count=token_count,
            now=NOW,
        )


def test_chunk_rejects_non_deterministic_explicit_id() -> None:
    with pytest.raises(
        ValueError,
        match="deterministic chunk identity",
    ):
        DocumentChunk.create(
            document_version=create_profiled_version(),
            ordinal=0,
            section_path=(),
            content="Restart the pool.",
            token_count=4,
            chunk_id=uuid4(),
            now=NOW,
        )


def test_chunk_is_immutable() -> None:
    chunk = DocumentChunk.create(
        document_version=create_profiled_version(),
        ordinal=0,
        section_path=(),
        content="Restart the pool.",
        token_count=4,
        now=NOW,
    )

    with pytest.raises(FrozenInstanceError):
        chunk.content = "Changed"  # type: ignore[misc]


def test_chunk_hash_cannot_be_replaced_independently() -> None:
    chunk = DocumentChunk.create(
        document_version=create_profiled_version(),
        ordinal=0,
        section_path=(),
        content="Restart the pool.",
        token_count=4,
        now=NOW,
    )

    with pytest.raises(
        ValueError,
        match="content_sha256 must match",
    ):
        replace(
            chunk,
            content_sha256="0" * 64,
        )
