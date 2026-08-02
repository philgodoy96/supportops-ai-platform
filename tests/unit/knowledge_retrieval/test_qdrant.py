"""Unit tests for the Qdrant semantic knowledge search adapter."""

from collections.abc import Sequence
from hashlib import sha256
from typing import cast
from uuid import UUID

import pytest
from qdrant_client import AsyncQdrantClient, models
from qdrant_client.http.models import QueryResponse

from supportops.knowledge_index.vector_store.contracts import (
    KnowledgeCollectionCompatibilityError,
    KnowledgeCollectionProfile,
    KnowledgeVectorStoreUnavailableError,
)
from supportops.knowledge_index.vector_store.qdrant import (
    CHUNK_ID_PAYLOAD,
    CHUNK_ORDINAL_PAYLOAD,
    CHUNKING_STRATEGY_PAYLOAD,
    CHUNKING_VERSION_PAYLOAD,
    CONTENT_SHA256_PAYLOAD,
    DOCUMENT_ID_PAYLOAD,
    DOCUMENT_VERSION_ID_PAYLOAD,
    MEDIA_TYPE_PAYLOAD,
    WORKSPACE_ID_PAYLOAD,
)
from supportops.knowledge_retrieval.contracts import (
    KnowledgeSearchTarget,
    KnowledgeVectorSearchRequest,
)
from supportops.knowledge_retrieval.qdrant import (
    QdrantKnowledgeVectorSearcher,
)
from supportops.modules.knowledge_documents.domain.models import (
    KnowledgeIndexProfile,
)

_WORKSPACE_ID = UUID("032c8c87-57cc-4d14-bfbd-04968b4e8cd4")
_DOCUMENT_A_ID = UUID("276046a2-28ec-4cb1-8bb6-a2ff70f9064b")
_DOCUMENT_B_ID = UUID("10bea98d-dfb1-4c33-884f-a4f36789a2ab")
_VERSION_A_ID = UUID("09036916-84cf-4a58-bdf4-09bc52716ec5")
_VERSION_B_ID = UUID("8ac21e0c-f869-4d84-a9e1-37d8074c9e54")


def create_profile() -> KnowledgeIndexProfile:
    """Create one deterministic search profile."""

    return KnowledgeIndexProfile(
        chunking_strategy="markdown-token",
        chunking_version="v1",
        tokenizer_encoding="cl100k_base",
        embedding_provider="mock",
        embedding_model="mock-hashing-embedding-v1",
        embedding_dimensions=3,
        knowledge_collection=("supportops-knowledge-mock-v1"),
        knowledge_vector_name="dense",
    )


def create_request() -> KnowledgeVectorSearchRequest:
    """Create one multi-document active search scope."""

    return KnowledgeVectorSearchRequest(
        workspace_id=_WORKSPACE_ID,
        profile=create_profile(),
        targets=(
            KnowledgeSearchTarget(
                document_id=_DOCUMENT_A_ID,
                document_version_id=_VERSION_A_ID,
            ),
            KnowledgeSearchTarget(
                document_id=_DOCUMENT_B_ID,
                document_version_id=_VERSION_B_ID,
            ),
        ),
        query_vector=(1, 0, -1),
        limit=12,
    )


def create_payload(
    *,
    chunk_id: UUID,
    workspace_id: UUID = _WORKSPACE_ID,
    document_id: UUID = _DOCUMENT_A_ID,
    document_version_id: UUID = _VERSION_A_ID,
    ordinal: int = 0,
    chunking_version: str = "v1",
) -> dict[str, object]:
    """Create one complete Qdrant knowledge payload."""

    return {
        WORKSPACE_ID_PAYLOAD: str(workspace_id),
        DOCUMENT_ID_PAYLOAD: str(document_id),
        DOCUMENT_VERSION_ID_PAYLOAD: str(document_version_id),
        CHUNK_ID_PAYLOAD: str(chunk_id),
        CHUNK_ORDINAL_PAYLOAD: ordinal,
        CONTENT_SHA256_PAYLOAD: sha256(f"chunk-{chunk_id}".encode()).hexdigest(),
        MEDIA_TYPE_PAYLOAD: "text/markdown",
        CHUNKING_STRATEGY_PAYLOAD: ("markdown-token"),
        CHUNKING_VERSION_PAYLOAD: (chunking_version),
    }


def create_scored_point(
    *,
    chunk_id: UUID,
    score: float = 0.91,
    payload: dict[str, object] | None = None,
) -> models.ScoredPoint:
    """Create one SDK-compatible scored point."""

    return models.ScoredPoint(
        id=str(chunk_id),
        version=1,
        score=score,
        payload=(payload if payload is not None else create_payload(chunk_id=chunk_id)),
        vector=None,
    )


class FakeCollectionGuard:
    """Record collection compatibility checks."""

    def __init__(self) -> None:
        self.profiles: list[KnowledgeCollectionProfile] = []
        self.error: Exception | None = None

    async def ensure_collection(
        self,
        profile: KnowledgeCollectionProfile,
    ) -> None:
        """Record or fail the compatibility check."""

        self.profiles.append(profile)

        if self.error is not None:
            raise self.error


class FakeQdrantClient:
    """Record semantic search calls without network access."""

    def __init__(
        self,
        *,
        points: Sequence[models.ScoredPoint] = (),
    ) -> None:
        self.points = tuple(points)
        self.calls: list[dict[str, object]] = []
        self.error: Exception | None = None

    async def query_points(
        self,
        **kwargs: object,
    ) -> QueryResponse:
        """Return or raise the configured search result."""

        self.calls.append(dict(kwargs))

        if self.error is not None:
            raise self.error

        return QueryResponse(points=list(self.points))


def create_searcher(
    *,
    client: FakeQdrantClient,
    guard: FakeCollectionGuard,
    timeout_seconds: int = 10,
) -> QdrantKnowledgeVectorSearcher:
    """Create the adapter around test doubles."""

    return QdrantKnowledgeVectorSearcher(
        client=cast(
            AsyncQdrantClient,
            client,
        ),
        collection_guard=guard,
        search_timeout_seconds=timeout_seconds,
    )


async def test_search_uses_named_vector_scope_and_payload_selection() -> None:
    first_chunk_id = UUID(int=101)
    second_chunk_id = UUID(int=102)
    client = FakeQdrantClient(
        points=(
            create_scored_point(
                chunk_id=first_chunk_id,
                score=0.95,
            ),
            create_scored_point(
                chunk_id=second_chunk_id,
                score=0.80,
                payload=create_payload(
                    chunk_id=second_chunk_id,
                    document_id=_DOCUMENT_B_ID,
                    document_version_id=_VERSION_B_ID,
                    ordinal=1,
                ),
            ),
        )
    )
    guard = FakeCollectionGuard()
    request = create_request()

    candidates = await create_searcher(
        client=client,
        guard=guard,
    ).search(request)

    assert [candidate.chunk_id for candidate in candidates] == [
        first_chunk_id,
        second_chunk_id,
    ]
    assert [candidate.score for candidate in candidates] == [
        0.95,
        0.80,
    ]

    assert guard.profiles == [
        KnowledgeCollectionProfile(
            collection_name=("supportops-knowledge-mock-v1"),
            vector_name="dense",
            dimensions=3,
        )
    ]

    assert len(client.calls) == 1
    call = client.calls[0]

    assert call["collection_name"] == ("supportops-knowledge-mock-v1")
    assert call["query"] == [
        1.0,
        0.0,
        -1.0,
    ]
    assert call["using"] == "dense"
    assert call["limit"] == 12
    assert call["with_vectors"] is False
    assert call["timeout"] == 10

    requested_payload = cast(
        list[str],
        call["with_payload"],
    )
    assert set(requested_payload) == {
        WORKSPACE_ID_PAYLOAD,
        DOCUMENT_ID_PAYLOAD,
        DOCUMENT_VERSION_ID_PAYLOAD,
        CHUNK_ID_PAYLOAD,
        CHUNK_ORDINAL_PAYLOAD,
        CONTENT_SHA256_PAYLOAD,
        MEDIA_TYPE_PAYLOAD,
        CHUNKING_STRATEGY_PAYLOAD,
        CHUNKING_VERSION_PAYLOAD,
    }
    assert "content" not in requested_payload


async def test_search_filter_uses_workspace_and_exact_target_pairs() -> None:
    client = FakeQdrantClient()
    guard = FakeCollectionGuard()

    await create_searcher(
        client=client,
        guard=guard,
    ).search(create_request())

    query_filter = cast(
        models.Filter,
        client.calls[0]["query_filter"],
    )
    must_conditions = cast(
        list[models.Condition],
        query_filter.must,
    )

    assert len(must_conditions) == 2

    workspace_condition = cast(
        models.FieldCondition,
        must_conditions[0],
    )
    assert workspace_condition.key == (WORKSPACE_ID_PAYLOAD)
    assert cast(
        models.MatchValue,
        workspace_condition.match,
    ).value == str(_WORKSPACE_ID)

    target_scope = cast(
        models.Filter,
        must_conditions[1],
    )
    target_filters = cast(
        list[models.Condition],
        target_scope.should,
    )

    assert len(target_filters) == 2

    resolved_pairs: set[tuple[str, str]] = set()

    for target_filter in target_filters:
        pair_filter = cast(
            models.Filter,
            target_filter,
        )
        pair_conditions = cast(
            list[models.Condition],
            pair_filter.must,
        )
        values = {
            cast(
                models.FieldCondition,
                condition,
            ).key: cast(
                models.MatchValue,
                cast(
                    models.FieldCondition,
                    condition,
                ).match,
            ).value
            for condition in pair_conditions
        }
        resolved_pairs.add(
            (
                cast(
                    str,
                    values[DOCUMENT_ID_PAYLOAD],
                ),
                cast(
                    str,
                    values[DOCUMENT_VERSION_ID_PAYLOAD],
                ),
            )
        )

    assert resolved_pairs == {
        (
            str(_DOCUMENT_A_ID),
            str(_VERSION_A_ID),
        ),
        (
            str(_DOCUMENT_B_ID),
            str(_VERSION_B_ID),
        ),
    }


async def test_search_discards_inconsistent_candidates(
    caplog: pytest.LogCaptureFixture,
) -> None:
    valid_chunk_id = UUID(int=201)
    mismatched_point_id = UUID(int=202)
    missing_payload_id = UUID(int=203)
    other_workspace_id = UUID(int=204)
    inactive_pair_id = UUID(int=205)
    incompatible_profile_id = UUID(int=206)

    client = FakeQdrantClient(
        points=(
            create_scored_point(
                chunk_id=valid_chunk_id,
            ),
            create_scored_point(
                chunk_id=mismatched_point_id,
                payload=create_payload(
                    chunk_id=UUID(int=999),
                ),
            ),
            models.ScoredPoint(
                id=str(missing_payload_id),
                version=1,
                score=0.88,
                payload=None,
                vector=None,
            ),
            create_scored_point(
                chunk_id=other_workspace_id,
                payload=create_payload(
                    chunk_id=other_workspace_id,
                    workspace_id=UUID(int=999),
                ),
            ),
            create_scored_point(
                chunk_id=inactive_pair_id,
                payload=create_payload(
                    chunk_id=inactive_pair_id,
                    document_id=_DOCUMENT_A_ID,
                    document_version_id=(_VERSION_B_ID),
                ),
            ),
            create_scored_point(
                chunk_id=incompatible_profile_id,
                payload=create_payload(
                    chunk_id=(incompatible_profile_id),
                    chunking_version="v2",
                ),
            ),
            create_scored_point(
                chunk_id=valid_chunk_id,
                score=0.50,
            ),
        )
    )

    candidates = await create_searcher(
        client=client,
        guard=FakeCollectionGuard(),
    ).search(create_request())

    assert tuple(candidate.chunk_id for candidate in candidates) == (valid_chunk_id,)
    assert "Discarded inconsistent Qdrant" in (caplog.text)


async def test_collection_compatibility_error_stops_search() -> None:
    client = FakeQdrantClient()
    guard = FakeCollectionGuard()
    guard.error = KnowledgeCollectionCompatibilityError("incompatible collection")

    with pytest.raises(KnowledgeCollectionCompatibilityError):
        await create_searcher(
            client=client,
            guard=guard,
        ).search(create_request())

    assert client.calls == []


async def test_connection_failure_becomes_owned_unavailable_error() -> None:
    client = FakeQdrantClient()
    client.error = OSError("connection refused")

    with pytest.raises(
        KnowledgeVectorStoreUnavailableError,
        match="vector store is unavailable",
    ):
        await create_searcher(
            client=client,
            guard=FakeCollectionGuard(),
        ).search(create_request())


@pytest.mark.parametrize(
    "timeout_seconds",
    [
        0,
        -1,
    ],
)
def test_searcher_rejects_invalid_timeout(
    timeout_seconds: int,
) -> None:
    with pytest.raises(
        ValueError,
        match="search_timeout_seconds must be positive",
    ):
        create_searcher(
            client=FakeQdrantClient(),
            guard=FakeCollectionGuard(),
            timeout_seconds=timeout_seconds,
        )
