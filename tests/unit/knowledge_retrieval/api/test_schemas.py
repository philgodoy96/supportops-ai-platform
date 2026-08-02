"""Unit tests for semantic knowledge retrieval HTTP schemas."""

from hashlib import sha256
from uuid import UUID

import pytest
from pydantic import ValidationError

from supportops.knowledge_retrieval.api.schemas import (
    KnowledgeSearchRequestBody,
    KnowledgeSearchResponse,
)
from supportops.knowledge_retrieval.contracts import (
    KnowledgeCitation,
    KnowledgeEvidence,
    KnowledgeSearchRequest,
    KnowledgeSearchResult,
)
from supportops.modules.knowledge_documents.domain.models import (
    DocumentMediaType,
)

_WORKSPACE_ID = UUID("032c8c87-57cc-4d14-bfbd-04968b4e8cd4")
_DOCUMENT_ID = UUID("276046a2-28ec-4cb1-8bb6-a2ff70f9064b")
_VERSION_ID = UUID("09036916-84cf-4a58-bdf4-09bc52716ec5")
_CHUNK_ID = UUID("13ee455e-0e53-46a1-a316-010b5e14f8cc")


def create_evidence() -> KnowledgeEvidence:
    """Create one authoritative evidence item."""

    content = "Restart the database connection pool."

    return KnowledgeEvidence(
        rank=1,
        score=0.91,
        content=content,
        content_sha256=sha256(content.encode("utf-8")).hexdigest(),
        token_count=6,
        citation=KnowledgeCitation(
            workspace_id=_WORKSPACE_ID,
            document_id=_DOCUMENT_ID,
            document_title=("Database Recovery Runbook"),
            document_external_reference=("database-recovery"),
            document_version_id=_VERSION_ID,
            version_number=2,
            chunk_id=_CHUNK_ID,
            ordinal=0,
            section_path=("Recovery",),
            media_type=(DocumentMediaType.TEXT_MARKDOWN),
        ),
    )


def test_request_schema_normalizes_query_and_creates_domain_request() -> None:
    body = KnowledgeSearchRequestBody(
        query="  How do I recover the database?  ",
        top_k=7,
        document_ids=[_DOCUMENT_ID],
    )

    request = body.to_domain(workspace_id=_WORKSPACE_ID)

    assert body.query == ("How do I recover the database?")
    assert request == KnowledgeSearchRequest(
        workspace_id=_WORKSPACE_ID,
        query="How do I recover the database?",
        top_k=7,
        document_ids=(_DOCUMENT_ID,),
    )


@pytest.mark.parametrize(
    "payload",
    [
        {
            "query": "   ",
        },
        {
            "query": "database recovery",
            "top_k": 0,
        },
        {
            "query": "database recovery",
            "top_k": 21,
        },
        {
            "query": "database recovery",
            "document_ids": [
                str(_DOCUMENT_ID),
                str(_DOCUMENT_ID),
            ],
        },
        {
            "query": "database recovery",
            "unknown_field": True,
        },
    ],
)
def test_request_schema_rejects_invalid_payload(
    payload: dict[str, object],
) -> None:
    with pytest.raises(ValidationError):
        KnowledgeSearchRequestBody.model_validate(payload)


def test_response_schema_maps_ranked_evidence_and_citation() -> None:
    request = KnowledgeSearchRequest(
        workspace_id=_WORKSPACE_ID,
        query="database recovery",
        top_k=5,
        document_ids=(_DOCUMENT_ID,),
    )
    result = KnowledgeSearchResult(
        request=request,
        searched_version_count=1,
        evidence=(create_evidence(),),
    )

    response = KnowledgeSearchResponse.from_domain(result)

    assert response.workspace_id == (_WORKSPACE_ID)
    assert response.query == "database recovery"
    assert response.top_k == 5
    assert response.document_ids == [_DOCUMENT_ID]
    assert response.searched_version_count == 1
    assert len(response.evidence) == 1

    evidence = response.evidence[0]
    assert evidence.rank == 1
    assert evidence.score == 0.91
    assert evidence.content == ("Restart the database connection pool.")
    assert evidence.token_count == 6
    assert evidence.citation.document_id == (_DOCUMENT_ID)
    assert evidence.citation.document_version_id == _VERSION_ID
    assert evidence.citation.chunk_id == (_CHUNK_ID)
    assert evidence.citation.section_path == ["Recovery"]
    assert evidence.citation.media_type == ("text/markdown")


def test_empty_result_maps_without_placeholder_evidence() -> None:
    request = KnowledgeSearchRequest(
        workspace_id=_WORKSPACE_ID,
        query="database recovery",
    )

    response = KnowledgeSearchResponse.from_domain(
        KnowledgeSearchResult(
            request=request,
            searched_version_count=0,
            evidence=(),
        )
    )

    assert response.searched_version_count == 0
    assert response.evidence == []
