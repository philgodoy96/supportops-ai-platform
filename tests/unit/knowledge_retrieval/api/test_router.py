"""Unit tests for the semantic knowledge retrieval HTTP route."""

from collections.abc import Sequence
from hashlib import sha256
from typing import cast
from uuid import UUID

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from supportops.ai.embeddings.errors import (
    EmbeddingTimeoutError,
)
from supportops.api.errors import (
    register_error_handlers,
)
from supportops.api.middleware.request_context import (
    RequestContextMiddleware,
)
from supportops.knowledge_index.vector_store.contracts import (
    KnowledgeVectorStoreUnavailableError,
)
from supportops.knowledge_retrieval.api.dependencies import (
    get_search_knowledge,
)
from supportops.knowledge_retrieval.api.router import (
    router,
)
from supportops.knowledge_retrieval.contracts import (
    KnowledgeCitation,
    KnowledgeEvidence,
    KnowledgeSearchRequest,
    KnowledgeSearchResult,
)
from supportops.knowledge_retrieval.service import (
    SearchKnowledge,
)
from supportops.modules.knowledge_documents.domain.models import (
    DocumentMediaType,
)

_WORKSPACE_ID = UUID("032c8c87-57cc-4d14-bfbd-04968b4e8cd4")
_DOCUMENT_ID = UUID("276046a2-28ec-4cb1-8bb6-a2ff70f9064b")
_VERSION_ID = UUID("09036916-84cf-4a58-bdf4-09bc52716ec5")
_CHUNK_ID = UUID("13ee455e-0e53-46a1-a316-010b5e14f8cc")


def create_result(
    request: KnowledgeSearchRequest,
) -> KnowledgeSearchResult:
    """Create one successful search result."""

    content = "Restart the database connection pool."

    return KnowledgeSearchResult(
        request=request,
        searched_version_count=1,
        evidence=(
            KnowledgeEvidence(
                rank=1,
                score=0.93,
                content=content,
                content_sha256=sha256(content.encode("utf-8")).hexdigest(),
                token_count=6,
                citation=KnowledgeCitation(
                    workspace_id=_WORKSPACE_ID,
                    document_id=_DOCUMENT_ID,
                    document_title=("Database Recovery Runbook"),
                    document_external_reference=("database-recovery"),
                    document_version_id=(_VERSION_ID),
                    version_number=2,
                    chunk_id=_CHUNK_ID,
                    ordinal=0,
                    section_path=("Recovery",),
                    media_type=(DocumentMediaType.TEXT_MARKDOWN),
                ),
            ),
        ),
    )


class FakeSearchKnowledge:
    """Record search requests and return one configured result."""

    def __init__(self) -> None:
        self.requests: list[KnowledgeSearchRequest] = []
        self.error: Exception | None = None

    async def execute(
        self,
        request: KnowledgeSearchRequest,
    ) -> KnowledgeSearchResult:
        """Return or raise the configured outcome."""

        self.requests.append(request)

        if self.error is not None:
            raise self.error

        return create_result(request)


def create_app(
    service: FakeSearchKnowledge,
) -> FastAPI:
    """Create an isolated FastAPI app for route tests."""

    app = FastAPI()
    register_error_handlers(app)
    app.add_middleware(RequestContextMiddleware)
    app.include_router(
        router,
        prefix="/api/v1",
    )
    app.dependency_overrides[get_search_knowledge] = lambda: cast(
        SearchKnowledge,
        service,
    )

    return app


async def post_search(
    *,
    app: FastAPI,
    payload: dict[str, object],
) -> tuple[int, dict[str, object], dict[str, str]]:
    """Execute one isolated semantic-search request."""

    transport = ASGITransport(
        app=app,
        raise_app_exceptions=False,
    )

    async with AsyncClient(
        transport=transport,
        base_url="http://test",
    ) as client:
        response = await client.post(
            (f"/api/v1/workspaces/{_WORKSPACE_ID}/knowledge/search"),
            json=payload,
        )

    return (
        response.status_code,
        response.json(),
        dict(response.headers),
    )


async def test_route_executes_workspace_scoped_search() -> None:
    service = FakeSearchKnowledge()
    app = create_app(service)

    status_code, payload, headers = await post_search(
        app=app,
        payload={
            "query": ("  recover the database  "),
            "top_k": 3,
            "document_ids": [str(_DOCUMENT_ID)],
        },
    )

    assert status_code == 200
    assert len(service.requests) == 1
    request = service.requests[0]

    assert request.workspace_id == (_WORKSPACE_ID)
    assert request.query == ("recover the database")
    assert request.top_k == 3
    assert request.document_ids == (_DOCUMENT_ID,)

    assert payload["workspace_id"] == str(_WORKSPACE_ID)
    assert payload["query"] == ("recover the database")
    assert payload["searched_version_count"] == 1

    evidence = cast(
        Sequence[dict[str, object]],
        payload["evidence"],
    )
    assert len(evidence) == 1
    assert evidence[0]["content"] == ("Restart the database connection pool.")

    citation = cast(
        dict[str, object],
        evidence[0]["citation"],
    )
    assert citation["document_id"] == str(_DOCUMENT_ID)
    assert citation["chunk_id"] == str(_CHUNK_ID)

    request_id = UUID(headers["x-request-id"])
    correlation_id = UUID(headers["x-correlation-id"])
    assert request_id.version == 4
    assert correlation_id == request_id


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
            "document_ids": [
                str(_DOCUMENT_ID),
                str(_DOCUMENT_ID),
            ],
        },
        {
            "query": "database recovery",
            "unexpected": "value",
        },
    ],
)
async def test_route_rejects_invalid_request_before_service(
    payload: dict[str, object],
) -> None:
    service = FakeSearchKnowledge()
    app = create_app(service)

    status_code, _, _ = await post_search(
        app=app,
        payload=payload,
    )

    assert status_code == 422
    assert service.requests == []


@pytest.mark.parametrize(
    "error",
    [
        EmbeddingTimeoutError(),
        KnowledgeVectorStoreUnavailableError("The knowledge vector store is unavailable."),
    ],
)
async def test_route_maps_expected_dependency_failures_to_503(
    error: Exception,
) -> None:
    service = FakeSearchKnowledge()
    service.error = error
    app = create_app(service)

    status_code, payload, headers = await post_search(
        app=app,
        payload={
            "query": "database recovery",
        },
    )

    assert status_code == 503

    error_payload = cast(
        dict[str, object],
        payload["error"],
    )
    assert error_payload["code"] == ("knowledge_retrieval_unavailable")
    assert error_payload["message"] == ("Knowledge retrieval is temporarily unavailable.")
    assert error_payload["request_id"] == (headers["x-request-id"])

    assert "embedding" not in str(payload).casefold()
    assert "vector store" not in str(payload).casefold()
