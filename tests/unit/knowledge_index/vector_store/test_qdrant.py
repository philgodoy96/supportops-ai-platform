"""Unit tests for the Qdrant knowledge vector-store adapter."""

from hashlib import sha256
from types import SimpleNamespace
from typing import cast
from uuid import UUID

import pytest
from qdrant_client import AsyncQdrantClient, models

from supportops.knowledge_index.vector_store.contracts import (
    KnowledgeCollectionCompatibilityError,
    KnowledgeCollectionProfile,
    KnowledgeVectorPoint,
    KnowledgeVectorStoreUnavailableError,
    KnowledgeVersionProjection,
)
from supportops.knowledge_index.vector_store.qdrant import (
    CHUNK_ID_PAYLOAD,
    CONTENT_SHA256_PAYLOAD,
    DOCUMENT_ID_PAYLOAD,
    DOCUMENT_VERSION_ID_PAYLOAD,
    KNOWLEDGE_PAYLOAD_INDEXES,
    WORKSPACE_ID_PAYLOAD,
    QdrantKnowledgeVectorStore,
)
from supportops.modules.knowledge_documents.domain.models import (
    DocumentMediaType,
)

_WORKSPACE_ID = UUID("032c8c87-57cc-4d14-bfbd-04968b4e8cd4")
_DOCUMENT_ID = UUID("276046a2-28ec-4cb1-8bb6-a2ff70f9064b")
_VERSION_ID = UUID("09036916-84cf-4a58-bdf4-09bc52716ec5")


def create_profile(
    *,
    dimensions: int = 3,
) -> KnowledgeCollectionProfile:
    """Create one deterministic collection profile."""

    return KnowledgeCollectionProfile(
        collection_name="supportops-knowledge-mock-v1",
        vector_name="dense",
        dimensions=dimensions,
    )


def create_projection() -> KnowledgeVersionProjection:
    """Create one deterministic version projection identity."""

    return KnowledgeVersionProjection(
        workspace_id=_WORKSPACE_ID,
        document_id=_DOCUMENT_ID,
        document_version_id=_VERSION_ID,
    )


def create_point(
    index: int,
    *,
    projection: KnowledgeVersionProjection | None = None,
    dimensions: int = 3,
) -> KnowledgeVectorPoint:
    """Create one deterministic vector point."""

    resolved_projection = projection or create_projection()

    return KnowledgeVectorPoint(
        chunk_id=UUID(int=index + 1),
        workspace_id=resolved_projection.workspace_id,
        document_id=resolved_projection.document_id,
        document_version_id=(resolved_projection.document_version_id),
        ordinal=index,
        content_sha256=sha256(f"chunk-{index}".encode()).hexdigest(),
        media_type=DocumentMediaType.TEXT_MARKDOWN,
        chunking_strategy="markdown-token",
        chunking_version="v1",
        vector=tuple(float(index + coordinate + 1) for coordinate in range(dimensions)),
    )


def completed_result() -> models.UpdateResult:
    """Return one SDK-like completed update result."""

    return cast(
        models.UpdateResult,
        SimpleNamespace(
            status=models.UpdateStatus.COMPLETED,
        ),
    )


def compatible_collection_info(
    profile: KnowledgeCollectionProfile,
    *,
    payload_indexes: bool,
) -> models.CollectionInfo:
    """Return SDK-like compatible collection metadata."""

    payload_schema = (
        {
            field_name: SimpleNamespace(
                data_type=field_schema,
            )
            for field_name, field_schema in (KNOWLEDGE_PAYLOAD_INDEXES.items())
        }
        if payload_indexes
        else {}
    )

    return cast(
        models.CollectionInfo,
        SimpleNamespace(
            config=SimpleNamespace(
                params=SimpleNamespace(
                    vectors={
                        profile.vector_name: models.VectorParams(
                            size=profile.dimensions,
                            distance=models.Distance.COSINE,
                        )
                    }
                )
            ),
            payload_schema=payload_schema,
        ),
    )


class FakeQdrantClient:
    """Record Qdrant calls without network access."""

    def __init__(
        self,
        *,
        profile: KnowledgeCollectionProfile,
        collection_exists: bool = True,
        payload_indexes: bool = True,
    ) -> None:
        self.collection_exists_result = collection_exists
        self.collection_info = compatible_collection_info(
            profile,
            payload_indexes=payload_indexes,
        )
        self.collection_exists_calls: list[str] = []
        self.create_collection_calls: list[dict[str, object]] = []
        self.get_collection_calls: list[str] = []
        self.payload_index_calls: list[dict[str, object]] = []
        self.upsert_calls: list[dict[str, object]] = []
        self.count_calls: list[dict[str, object]] = []
        self.count_result = 0
        self.collection_exists_error: Exception | None = None

    async def collection_exists(
        self,
        *,
        collection_name: str,
        **kwargs: object,
    ) -> bool:
        """Return or raise the configured existence result."""

        del kwargs
        self.collection_exists_calls.append(collection_name)
        if self.collection_exists_error is not None:
            raise self.collection_exists_error
        return self.collection_exists_result

    async def create_collection(
        self,
        **kwargs: object,
    ) -> bool:
        """Record collection creation."""

        self.create_collection_calls.append(dict(kwargs))
        self.collection_exists_result = True
        return True

    async def get_collection(
        self,
        *,
        collection_name: str,
        **kwargs: object,
    ) -> models.CollectionInfo:
        """Return configured collection metadata."""

        del kwargs
        self.get_collection_calls.append(collection_name)
        return self.collection_info

    async def create_payload_index(
        self,
        **kwargs: object,
    ) -> models.UpdateResult:
        """Record and materialize one payload index."""

        call = dict(kwargs)
        self.payload_index_calls.append(call)
        field_name = cast(str, call["field_name"])
        field_schema = cast(
            models.PayloadSchemaType,
            call["field_schema"],
        )
        self.collection_info.payload_schema[field_name] = cast(
            models.PayloadIndexInfo,
            SimpleNamespace(data_type=field_schema),
        )
        return completed_result()

    async def upsert(
        self,
        **kwargs: object,
    ) -> models.UpdateResult:
        """Record one vector batch."""

        self.upsert_calls.append(dict(kwargs))
        return completed_result()

    async def count(
        self,
        **kwargs: object,
    ) -> models.CountResult:
        """Return the configured exact count."""

        self.count_calls.append(dict(kwargs))
        return cast(
            models.CountResult,
            SimpleNamespace(count=self.count_result),
        )


def create_store(
    client: FakeQdrantClient,
    *,
    batch_size: int = 64,
) -> QdrantKnowledgeVectorStore:
    """Create the adapter around one fake SDK client."""

    return QdrantKnowledgeVectorStore(
        client=cast(AsyncQdrantClient, client),
        batch_size=batch_size,
    )


async def test_ensure_collection_creates_named_vector_and_indexes() -> None:
    profile = create_profile()
    client = FakeQdrantClient(
        profile=profile,
        collection_exists=False,
        payload_indexes=False,
    )
    store = create_store(client)

    await store.ensure_collection(profile)

    assert len(client.create_collection_calls) == 1
    create_call = client.create_collection_calls[0]
    assert create_call["collection_name"] == (profile.collection_name)
    vectors = cast(
        dict[str, models.VectorParams],
        create_call["vectors_config"],
    )
    assert set(vectors) == {"dense"}
    assert vectors["dense"].size == 3
    assert vectors["dense"].distance == models.Distance.COSINE

    assert {cast(str, call["field_name"]) for call in client.payload_index_calls} == set(
        KNOWLEDGE_PAYLOAD_INDEXES
    )
    assert all(call["wait"] is True for call in client.payload_index_calls)


async def test_ensure_collection_is_idempotent_when_compatible() -> None:
    profile = create_profile()
    client = FakeQdrantClient(
        profile=profile,
        payload_indexes=True,
    )
    store = create_store(client)

    await store.ensure_collection(profile)
    await store.ensure_collection(profile)

    assert client.create_collection_calls == []
    assert client.payload_index_calls == []
    assert client.get_collection_calls == [
        profile.collection_name,
        profile.collection_name,
    ]


@pytest.mark.parametrize(
    ("vector_name", "dimensions", "distance"),
    [
        ("other", 3, models.Distance.COSINE),
        ("dense", 4, models.Distance.COSINE),
        ("dense", 3, models.Distance.DOT),
    ],
)
async def test_ensure_collection_rejects_incompatible_vector_profile(
    vector_name: str,
    dimensions: int,
    distance: models.Distance,
) -> None:
    profile = create_profile()
    client = FakeQdrantClient(
        profile=profile,
        payload_indexes=True,
    )
    client.collection_info.config.params.vectors = {
        vector_name: models.VectorParams(
            size=dimensions,
            distance=distance,
        )
    }
    store = create_store(client)

    with pytest.raises(KnowledgeCollectionCompatibilityError):
        await store.ensure_collection(profile)


async def test_ensure_collection_rejects_incompatible_payload_index() -> None:
    profile = create_profile()
    client = FakeQdrantClient(
        profile=profile,
        payload_indexes=True,
    )
    client.collection_info.payload_schema[WORKSPACE_ID_PAYLOAD] = cast(
        models.PayloadIndexInfo,
        SimpleNamespace(
            data_type=models.PayloadSchemaType.KEYWORD,
        ),
    )
    store = create_store(client)

    with pytest.raises(
        KnowledgeCollectionCompatibilityError,
        match="incompatible type",
    ):
        await store.ensure_collection(profile)


async def test_upsert_validates_and_batches_complete_projection() -> None:
    profile = create_profile()
    projection = create_projection()
    client = FakeQdrantClient(
        profile=profile,
        payload_indexes=True,
    )
    store = create_store(
        client,
        batch_size=64,
    )
    points = tuple(create_point(index) for index in range(130))

    await store.upsert_version_points(
        profile=profile,
        projection=projection,
        points=points,
    )

    assert [
        len(cast(list[models.PointStruct], call["points"])) for call in client.upsert_calls
    ] == [64, 64, 2]
    assert all(call["collection_name"] == profile.collection_name for call in client.upsert_calls)
    assert all(call["wait"] is True for call in client.upsert_calls)

    first_point = cast(
        list[models.PointStruct],
        client.upsert_calls[0]["points"],
    )[0]
    assert first_point.id == str(points[0].chunk_id)
    assert first_point.vector == {"dense": list(points[0].vector)}
    assert first_point.payload is not None
    assert first_point.payload[WORKSPACE_ID_PAYLOAD] == str(_WORKSPACE_ID)
    assert first_point.payload[DOCUMENT_ID_PAYLOAD] == str(_DOCUMENT_ID)
    assert first_point.payload[DOCUMENT_VERSION_ID_PAYLOAD] == str(_VERSION_ID)
    assert first_point.payload[CHUNK_ID_PAYLOAD] == str(points[0].chunk_id)
    assert first_point.payload[CONTENT_SHA256_PAYLOAD] == (points[0].content_sha256)
    assert "content" not in first_point.payload


async def test_upsert_rejects_cross_version_point_before_qdrant_call() -> None:
    profile = create_profile()
    projection = create_projection()
    client = FakeQdrantClient(
        profile=profile,
        payload_indexes=True,
    )
    store = create_store(client)
    other_projection = KnowledgeVersionProjection(
        workspace_id=_WORKSPACE_ID,
        document_id=_DOCUMENT_ID,
        document_version_id=UUID(int=999),
    )

    with pytest.raises(
        ValueError,
        match="must belong to the projected version",
    ):
        await store.upsert_version_points(
            profile=profile,
            projection=projection,
            points=(
                create_point(
                    0,
                    projection=other_projection,
                ),
            ),
        )

    assert client.collection_exists_calls == []
    assert client.upsert_calls == []


async def test_upsert_rejects_duplicate_chunk_identity_and_ordinal() -> None:
    profile = create_profile()
    projection = create_projection()
    client = FakeQdrantClient(
        profile=profile,
        payload_indexes=True,
    )
    store = create_store(client)
    first = create_point(0)
    duplicate = KnowledgeVectorPoint(
        chunk_id=first.chunk_id,
        workspace_id=first.workspace_id,
        document_id=first.document_id,
        document_version_id=first.document_version_id,
        ordinal=first.ordinal,
        content_sha256=sha256(b"other").hexdigest(),
        media_type=first.media_type,
        chunking_strategy=first.chunking_strategy,
        chunking_version=first.chunking_version,
        vector=first.vector,
    )

    with pytest.raises(ValueError):
        await store.upsert_version_points(
            profile=profile,
            projection=projection,
            points=(first, duplicate),
        )

    assert client.upsert_calls == []


async def test_count_uses_exact_workspace_owned_version_filter() -> None:
    profile = create_profile()
    projection = create_projection()
    client = FakeQdrantClient(
        profile=profile,
        payload_indexes=True,
    )
    client.count_result = 7
    store = create_store(client)

    result = await store.count_version_points(
        profile=profile,
        projection=projection,
    )

    assert result == 7
    assert len(client.count_calls) == 1
    call = client.count_calls[0]
    assert call["collection_name"] == (profile.collection_name)
    assert call["exact"] is True

    count_filter = cast(
        models.Filter,
        call["count_filter"],
    )
    conditions = cast(
        list[models.FieldCondition],
        count_filter.must,
    )
    values_by_key = {
        condition.key: cast(
            models.MatchValue,
            condition.match,
        ).value
        for condition in conditions
    }
    assert values_by_key == {
        WORKSPACE_ID_PAYLOAD: str(_WORKSPACE_ID),
        DOCUMENT_ID_PAYLOAD: str(_DOCUMENT_ID),
        DOCUMENT_VERSION_ID_PAYLOAD: str(_VERSION_ID),
    }


async def test_connection_failure_becomes_owned_unavailable_error() -> None:
    profile = create_profile()
    client = FakeQdrantClient(
        profile=profile,
        payload_indexes=True,
    )
    client.collection_exists_error = OSError("connection refused")
    store = create_store(client)

    with pytest.raises(
        KnowledgeVectorStoreUnavailableError,
        match="vector store is unavailable",
    ):
        await store.ensure_collection(profile)


@pytest.mark.parametrize("batch_size", [0, -1])
def test_store_rejects_invalid_batch_size(
    batch_size: int,
) -> None:
    profile = create_profile()
    client = FakeQdrantClient(
        profile=profile,
        payload_indexes=True,
    )

    with pytest.raises(
        ValueError,
        match="batch_size must be positive",
    ):
        create_store(
            client,
            batch_size=batch_size,
        )
