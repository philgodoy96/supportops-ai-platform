"""Unit tests for knowledge vector-store contracts."""

from datetime import UTC, datetime
from hashlib import sha256
from uuid import UUID

import pytest

from supportops.knowledge_index.vector_store.contracts import (
    KnowledgeCollectionProfile,
    KnowledgeVectorPoint,
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
    2,
    0,
    30,
    tzinfo=UTC,
)


def create_chunk() -> DocumentChunk:
    """Create one deterministic authoritative chunk."""

    version = DocumentVersion.create_pending(
        document_version_id=_VERSION_ID,
        workspace_id=_WORKSPACE_ID,
        document_id=_DOCUMENT_ID,
        version_number=1,
        media_type=DocumentMediaType.TEXT_MARKDOWN,
        content="# Recovery\nRestart the connection pool.\n",
        now=_CREATED_AT,
    ).bind_index_profile(
        KnowledgeIndexProfile(
            chunking_strategy="markdown-token",
            chunking_version="v1",
            tokenizer_encoding="cl100k_base",
            embedding_provider="mock",
            embedding_model="mock-hashing-embedding-v1",
            embedding_dimensions=3,
            knowledge_collection=("supportops-knowledge-mock-v1"),
            knowledge_vector_name="dense",
        ),
        now=_CREATED_AT,
    )

    return DocumentChunk.create(
        document_version=version,
        ordinal=0,
        section_path=("Recovery",),
        content="Restart the connection pool.",
        token_count=6,
        now=_CREATED_AT,
    )


def test_collection_profile_requires_explicit_identity_and_dimensions() -> None:
    profile = KnowledgeCollectionProfile(
        collection_name="supportops-knowledge-mock-v1",
        vector_name="dense",
        dimensions=64,
    )

    assert profile.collection_name == ("supportops-knowledge-mock-v1")
    assert profile.vector_name == "dense"
    assert profile.dimensions == 64


@pytest.mark.parametrize(
    ("collection_name", "vector_name", "dimensions"),
    [
        ("", "dense", 64),
        (" collection", "dense", 64),
        ("collection", "", 64),
        ("collection", "dense ", 64),
        ("collection", "dense", 0),
    ],
)
def test_collection_profile_rejects_invalid_values(
    collection_name: str,
    vector_name: str,
    dimensions: int,
) -> None:
    with pytest.raises(ValueError):
        KnowledgeCollectionProfile(
            collection_name=collection_name,
            vector_name=vector_name,
            dimensions=dimensions,
        )


def test_vector_point_maps_authoritative_chunk_ownership() -> None:
    chunk = create_chunk()

    point = KnowledgeVectorPoint.from_chunk(
        chunk=chunk,
        media_type=DocumentMediaType.TEXT_MARKDOWN,
        vector=(1, 0, -1),
    )

    assert point.chunk_id == chunk.id
    assert point.workspace_id == chunk.workspace_id
    assert point.document_id == chunk.document_id
    assert point.document_version_id == chunk.document_version_id
    assert point.ordinal == chunk.ordinal
    assert point.content_sha256 == chunk.content_sha256
    assert point.media_type is (DocumentMediaType.TEXT_MARKDOWN)
    assert point.chunking_strategy == "markdown-token"
    assert point.chunking_version == "v1"
    assert point.vector == (1.0, 0.0, -1.0)


@pytest.mark.parametrize(
    ("ordinal", "content_sha256", "vector"),
    [
        (-1, sha256(b"chunk").hexdigest(), (0.1,)),
        (0, "not-a-sha256", (0.1,)),
        (0, sha256(b"chunk").hexdigest(), ()),
        (
            0,
            sha256(b"chunk").hexdigest(),
            (float("nan"),),
        ),
        (
            0,
            sha256(b"chunk").hexdigest(),
            (float("inf"),),
        ),
    ],
)
def test_vector_point_rejects_invalid_projection_values(
    ordinal: int,
    content_sha256: str,
    vector: tuple[float, ...],
) -> None:
    with pytest.raises(ValueError):
        KnowledgeVectorPoint(
            chunk_id=UUID(int=1),
            workspace_id=_WORKSPACE_ID,
            document_id=_DOCUMENT_ID,
            document_version_id=_VERSION_ID,
            ordinal=ordinal,
            content_sha256=content_sha256,
            media_type=DocumentMediaType.TEXT_MARKDOWN,
            chunking_strategy="markdown-token",
            chunking_version="v1",
            vector=vector,
        )


def test_vector_point_rejects_nonnumeric_coordinate() -> None:
    with pytest.raises(
        TypeError,
        match="must be numeric",
    ):
        KnowledgeVectorPoint(
            chunk_id=UUID(int=1),
            workspace_id=_WORKSPACE_ID,
            document_id=_DOCUMENT_ID,
            document_version_id=_VERSION_ID,
            ordinal=0,
            content_sha256=sha256(b"chunk").hexdigest(),
            media_type=DocumentMediaType.TEXT_MARKDOWN,
            chunking_strategy="markdown-token",
            chunking_version="v1",
            vector=("invalid",),  # type: ignore[arg-type]
        )
