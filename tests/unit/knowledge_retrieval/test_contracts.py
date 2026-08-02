"""Unit tests for semantic knowledge retrieval contracts."""

from dataclasses import replace
from datetime import UTC, datetime
from hashlib import sha256
from typing import Any
from uuid import UUID

import pytest

from supportops.knowledge_retrieval.contracts import (
    MAX_KNOWLEDGE_SEARCH_DOCUMENT_FILTERS,
    ActiveKnowledgeVersion,
    KnowledgeEvidence,
    KnowledgeSearchRequest,
    KnowledgeSearchResult,
    KnowledgeSearchTarget,
    KnowledgeVectorCandidate,
    KnowledgeVectorSearchRequest,
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
    1,
    30,
    tzinfo=UTC,
)


def create_profile(
    *,
    dimensions: int = 3,
) -> KnowledgeIndexProfile:
    """Create one compatible retrieval profile."""

    return KnowledgeIndexProfile(
        chunking_strategy="markdown-token",
        chunking_version="v1",
        tokenizer_encoding="cl100k_base",
        embedding_provider="mock",
        embedding_model="mock-hashing-embedding-v1",
        embedding_dimensions=dimensions,
        knowledge_collection=("supportops-knowledge-mock-v1"),
        knowledge_vector_name="dense",
    )


def create_active_version() -> ActiveKnowledgeVersion:
    """Create one active ready-version descriptor."""

    return ActiveKnowledgeVersion(
        workspace_id=_WORKSPACE_ID,
        document_id=_DOCUMENT_ID,
        document_title=("Database Recovery Runbook"),
        document_external_reference=("database-recovery"),
        document_version_id=_VERSION_ID,
        version_number=2,
        media_type=(DocumentMediaType.TEXT_MARKDOWN),
        index_profile=create_profile(),
    )


def create_chunk() -> DocumentChunk:
    """Create one authoritative deterministic chunk."""

    version = DocumentVersion.create_pending(
        document_version_id=_VERSION_ID,
        workspace_id=_WORKSPACE_ID,
        document_id=_DOCUMENT_ID,
        version_number=2,
        media_type=(DocumentMediaType.TEXT_MARKDOWN),
        content=("# Recovery\n\nRestart the connection pool.\n"),
        now=_CREATED_AT,
    ).bind_index_profile(
        create_profile(),
        now=_CREATED_AT,
    )

    return DocumentChunk.create(
        document_version=version,
        ordinal=0,
        section_path=("Recovery",),
        content=("Restart the connection pool."),
        token_count=6,
        now=_CREATED_AT,
    )


def create_candidate(
    chunk: DocumentChunk,
    *,
    score: float = 0.91,
) -> KnowledgeVectorCandidate:
    """Create one candidate matching authoritative state."""

    return KnowledgeVectorCandidate(
        chunk_id=chunk.id,
        workspace_id=chunk.workspace_id,
        document_id=chunk.document_id,
        document_version_id=(chunk.document_version_id),
        ordinal=chunk.ordinal,
        content_sha256=(chunk.content_sha256),
        media_type=(DocumentMediaType.TEXT_MARKDOWN),
        chunking_strategy=(chunk.chunking_strategy),
        chunking_version=(chunk.chunking_version),
        score=score,
    )


def create_evidence(
    *,
    rank: int = 1,
    score: float = 0.91,
) -> KnowledgeEvidence:
    """Create one hydrated evidence item."""

    active_version = create_active_version()
    chunk = create_chunk()
    candidate = create_candidate(
        chunk,
        score=score,
    )

    return KnowledgeEvidence.from_sources(
        rank=rank,
        candidate=candidate,
        active_version=active_version,
        chunk=chunk,
    )


def test_search_request_normalizes_query_and_preserves_filters() -> None:
    document_ids = (
        _DOCUMENT_ID,
        UUID(int=2),
    )

    request = KnowledgeSearchRequest(
        workspace_id=_WORKSPACE_ID,
        query=("  How do I recover the database?  "),
        top_k=7,
        document_ids=document_ids,
    )

    assert request.query == ("How do I recover the database?")
    assert request.top_k == 7
    assert request.document_ids == document_ids


@pytest.mark.parametrize(
    ("query", "top_k"),
    [
        ("", 5),
        (" \n\t", 5),
        ("query", 0),
        ("query", 21),
        ("x" * 2001, 5),
    ],
)
def test_search_request_rejects_invalid_query_or_top_k(
    query: str,
    top_k: int,
) -> None:
    with pytest.raises(ValueError):
        KnowledgeSearchRequest(
            workspace_id=_WORKSPACE_ID,
            query=query,
            top_k=top_k,
        )


def test_search_request_rejects_duplicate_document_filters() -> None:
    with pytest.raises(
        ValueError,
        match="must not contain duplicates",
    ):
        KnowledgeSearchRequest(
            workspace_id=_WORKSPACE_ID,
            query="database recovery",
            document_ids=(
                _DOCUMENT_ID,
                _DOCUMENT_ID,
            ),
        )


def test_search_request_rejects_too_many_document_filters() -> None:
    with pytest.raises(
        ValueError,
        match="maximum number of filters",
    ):
        KnowledgeSearchRequest(
            workspace_id=_WORKSPACE_ID,
            query="database recovery",
            document_ids=tuple(
                UUID(int=index + 1) for index in range(MAX_KNOWLEDGE_SEARCH_DOCUMENT_FILTERS + 1)
            ),
        )


def test_active_version_exposes_search_target() -> None:
    version = create_active_version()

    assert version.target == KnowledgeSearchTarget(
        document_id=_DOCUMENT_ID,
        document_version_id=_VERSION_ID,
    )


@pytest.mark.parametrize(
    ("title", "version_number"),
    [
        ("", 2),
        (" Database Recovery", 2),
        ("Database Recovery", 0),
    ],
)
def test_active_version_rejects_invalid_metadata(
    title: str,
    version_number: int,
) -> None:
    with pytest.raises(ValueError):
        ActiveKnowledgeVersion(
            workspace_id=_WORKSPACE_ID,
            document_id=_DOCUMENT_ID,
            document_title=title,
            document_external_reference=None,
            document_version_id=_VERSION_ID,
            version_number=version_number,
            media_type=(DocumentMediaType.TEXT_MARKDOWN),
            index_profile=create_profile(),
        )


def test_vector_search_request_normalizes_vector_and_targets() -> None:
    target = create_active_version().target

    request = KnowledgeVectorSearchRequest(
        workspace_id=_WORKSPACE_ID,
        profile=create_profile(),
        targets=(target,),
        query_vector=(1, 0, -1),
        limit=12,
    )

    assert request.targets == (target,)
    assert request.query_vector == (
        1.0,
        0.0,
        -1.0,
    )
    assert request.limit == 12


@pytest.mark.parametrize(
    ("query_vector", "limit"),
    [
        ((0.1, 0.2), 5),
        ((0.1, 0.2, float("nan")), 5),
        ((0.1, 0.2, float("inf")), 5),
        ((0.1, 0.2, 0.3), 0),
        ((0.1, 0.2, 0.3), 101),
    ],
)
def test_vector_search_request_rejects_invalid_vector_or_limit(
    query_vector: tuple[float, ...],
    limit: int,
) -> None:
    with pytest.raises(ValueError):
        KnowledgeVectorSearchRequest(
            workspace_id=_WORKSPACE_ID,
            profile=create_profile(),
            targets=(create_active_version().target,),
            query_vector=query_vector,
            limit=limit,
        )


def test_vector_search_request_requires_unique_active_targets() -> None:
    target = create_active_version().target

    with pytest.raises(
        ValueError,
        match="must not contain duplicates",
    ):
        KnowledgeVectorSearchRequest(
            workspace_id=_WORKSPACE_ID,
            profile=create_profile(),
            targets=(target, target),
            query_vector=(0.1, 0.2, 0.3),
            limit=5,
        )


def test_candidate_preserves_projection_provenance() -> None:
    chunk = create_chunk()

    candidate = create_candidate(chunk)

    assert candidate.chunk_id == chunk.id
    assert candidate.content_sha256 == chunk.content_sha256
    assert candidate.ordinal == 0
    assert candidate.score == 0.91


@pytest.mark.parametrize(
    ("ordinal", "content_sha256", "score"),
    [
        (
            -1,
            sha256(b"chunk").hexdigest(),
            0.5,
        ),
        (
            0,
            "not-a-digest",
            0.5,
        ),
        (
            0,
            sha256(b"chunk").hexdigest(),
            float("nan"),
        ),
        (
            0,
            sha256(b"chunk").hexdigest(),
            float("inf"),
        ),
    ],
)
def test_candidate_rejects_invalid_projection_metadata(
    ordinal: int,
    content_sha256: str,
    score: float,
) -> None:
    with pytest.raises(ValueError):
        KnowledgeVectorCandidate(
            chunk_id=UUID(int=1),
            workspace_id=_WORKSPACE_ID,
            document_id=_DOCUMENT_ID,
            document_version_id=_VERSION_ID,
            ordinal=ordinal,
            content_sha256=content_sha256,
            media_type=(DocumentMediaType.TEXT_MARKDOWN),
            chunking_strategy="markdown-token",
            chunking_version="v1",
            score=score,
        )


def test_evidence_hydrates_authoritative_content_and_citation() -> None:
    active_version = create_active_version()
    chunk = create_chunk()
    candidate = create_candidate(chunk)

    evidence = KnowledgeEvidence.from_sources(
        rank=1,
        candidate=candidate,
        active_version=active_version,
        chunk=chunk,
    )

    assert evidence.content == chunk.content
    assert evidence.content_sha256 == chunk.content_sha256
    assert evidence.token_count == (chunk.token_count)
    assert evidence.score == candidate.score
    assert evidence.citation.document_title == ("Database Recovery Runbook")
    assert evidence.citation.document_external_reference == "database-recovery"
    assert evidence.citation.document_version_id == _VERSION_ID
    assert evidence.citation.chunk_id == chunk.id
    assert evidence.citation.section_path == ("Recovery",)


@pytest.mark.parametrize(
    "candidate_change",
    [
        {
            "workspace_id": UUID(int=999),
        },
        {
            "document_id": UUID(int=999),
        },
        {
            "document_version_id": UUID(int=999),
        },
        {
            "ordinal": 1,
        },
        {
            "content_sha256": sha256(b"different").hexdigest(),
        },
        {
            "chunking_version": "v2",
        },
        {
            "media_type": (DocumentMediaType.TEXT_PLAIN),
        },
    ],
)
def test_evidence_rejects_inconsistent_vector_candidate(
    candidate_change: dict[str, Any],
) -> None:
    active_version = create_active_version()
    chunk = create_chunk()
    candidate = replace(
        create_candidate(chunk),
        **candidate_change,
    )

    with pytest.raises(ValueError):
        KnowledgeEvidence.from_sources(
            rank=1,
            candidate=candidate,
            active_version=active_version,
            chunk=chunk,
        )


def test_search_result_accepts_ranked_unique_evidence() -> None:
    request = KnowledgeSearchRequest(
        workspace_id=_WORKSPACE_ID,
        query="database recovery",
        top_k=2,
    )
    first = create_evidence(
        rank=1,
        score=0.91,
    )
    second_source = create_evidence(
        rank=2,
        score=0.80,
    )
    second = replace(
        second_source,
        citation=replace(
            second_source.citation,
            chunk_id=UUID(int=998),
            ordinal=1,
        ),
    )

    result = KnowledgeSearchResult(
        request=request,
        searched_version_count=1,
        evidence=(first, second),
    )

    assert result.evidence == (
        first,
        second,
    )
    assert result.searched_version_count == 1


def test_search_result_rejects_non_contiguous_ranks() -> None:
    request = KnowledgeSearchRequest(
        workspace_id=_WORKSPACE_ID,
        query="database recovery",
        top_k=2,
    )

    with pytest.raises(
        ValueError,
        match="contiguous",
    ):
        KnowledgeSearchResult(
            request=request,
            searched_version_count=1,
            evidence=(create_evidence(rank=2),),
        )


def test_search_result_rejects_ascending_scores() -> None:
    request = KnowledgeSearchRequest(
        workspace_id=_WORKSPACE_ID,
        query="database recovery",
        top_k=2,
    )
    first = create_evidence(
        rank=1,
        score=0.50,
    )
    second_source = create_evidence(
        rank=2,
        score=0.90,
    )
    second = replace(
        second_source,
        citation=replace(
            second_source.citation,
            chunk_id=UUID(int=997),
        ),
    )

    with pytest.raises(
        ValueError,
        match="descending score",
    ):
        KnowledgeSearchResult(
            request=request,
            searched_version_count=1,
            evidence=(first, second),
        )


def test_search_result_rejects_duplicate_chunks() -> None:
    request = KnowledgeSearchRequest(
        workspace_id=_WORKSPACE_ID,
        query="database recovery",
        top_k=2,
    )

    with pytest.raises(
        ValueError,
        match="duplicate chunks",
    ):
        KnowledgeSearchResult(
            request=request,
            searched_version_count=1,
            evidence=(
                create_evidence(
                    rank=1,
                    score=0.91,
                ),
                create_evidence(
                    rank=2,
                    score=0.80,
                ),
            ),
        )
