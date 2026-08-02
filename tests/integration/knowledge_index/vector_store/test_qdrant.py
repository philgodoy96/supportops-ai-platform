"""Integration tests for the Qdrant knowledge vector projection."""

from hashlib import sha256
from uuid import UUID, uuid4

import pytest
from qdrant_client import models

from supportops.core.settings import Settings
from supportops.infrastructure.qdrant import (
    close_qdrant_client,
    create_qdrant_client,
)
from supportops.knowledge_index.vector_store.contracts import (
    KnowledgeCollectionProfile,
    KnowledgeVectorPoint,
    KnowledgeVersionProjection,
)
from supportops.knowledge_index.vector_store.qdrant import (
    DOCUMENT_VERSION_ID_PAYLOAD,
    KNOWLEDGE_PAYLOAD_INDEXES,
    QdrantKnowledgeVectorStore,
)
from supportops.modules.knowledge_documents.domain.models import (
    DocumentMediaType,
)

pytestmark = pytest.mark.integration

_WORKSPACE_ID = UUID("032c8c87-57cc-4d14-bfbd-04968b4e8cd4")
_DOCUMENT_ID = UUID("276046a2-28ec-4cb1-8bb6-a2ff70f9064b")
_VERSION_A_ID = UUID("09036916-84cf-4a58-bdf4-09bc52716ec5")
_VERSION_B_ID = UUID("8ac21e0c-f869-4d84-a9e1-37d8074c9e54")


def create_projection(
    document_version_id: UUID,
) -> KnowledgeVersionProjection:
    """Create one workspace-owned projection identity."""

    return KnowledgeVersionProjection(
        workspace_id=_WORKSPACE_ID,
        document_id=_DOCUMENT_ID,
        document_version_id=document_version_id,
    )


def create_points(
    *,
    projection: KnowledgeVersionProjection,
    count: int,
    ordinal_offset: int = 0,
) -> tuple[KnowledgeVectorPoint, ...]:
    """Create deterministic points for one projection."""

    return tuple(
        KnowledgeVectorPoint(
            chunk_id=uuid4(),
            workspace_id=projection.workspace_id,
            document_id=projection.document_id,
            document_version_id=(projection.document_version_id),
            ordinal=ordinal_offset + index,
            content_sha256=sha256(
                (f"{projection.document_version_id}:{ordinal_offset + index}").encode()
            ).hexdigest(),
            media_type=DocumentMediaType.TEXT_MARKDOWN,
            chunking_strategy="markdown-token",
            chunking_version="v1",
            vector=(
                float(index + 1),
                float(index + 2),
                float(index + 3),
                float(index + 4),
            ),
        )
        for index in range(count)
    )


async def test_qdrant_adapter_creates_indexes_and_upserts_idempotently(
    integration_settings: Settings,
) -> None:
    collection_name = f"supportops-knowledge-integration-{uuid4().hex}"
    profile = KnowledgeCollectionProfile(
        collection_name=collection_name,
        vector_name="dense",
        dimensions=4,
    )
    projection_a = create_projection(_VERSION_A_ID)
    projection_b = create_projection(_VERSION_B_ID)
    points_a = create_points(
        projection=projection_a,
        count=3,
    )
    points_b = create_points(
        projection=projection_b,
        count=2,
    )

    client = create_qdrant_client(integration_settings)
    store = QdrantKnowledgeVectorStore(
        client=client,
        batch_size=2,
    )

    try:
        await store.ensure_collection(profile)
        await store.ensure_collection(profile)

        collection_info = await client.get_collection(
            collection_name=collection_name,
        )
        vectors = collection_info.config.params.vectors
        assert isinstance(vectors, dict)
        assert set(vectors) == {"dense"}
        assert vectors["dense"].size == 4
        assert vectors["dense"].distance == models.Distance.COSINE

        for field_name, field_schema in KNOWLEDGE_PAYLOAD_INDEXES.items():
            payload_index = (collection_info.payload_schema or {}).get(field_name)
            assert payload_index is not None
            assert payload_index.data_type == field_schema

        await store.upsert_version_points(
            profile=profile,
            projection=projection_a,
            points=points_a,
        )
        await store.upsert_version_points(
            profile=profile,
            projection=projection_b,
            points=points_b,
        )
        await store.upsert_version_points(
            profile=profile,
            projection=projection_a,
            points=points_a,
        )

        assert (
            await store.count_version_points(
                profile=profile,
                projection=projection_a,
            )
            == 3
        )
        assert (
            await store.count_version_points(
                profile=profile,
                projection=projection_b,
            )
            == 2
        )

        records = await client.retrieve(
            collection_name=collection_name,
            ids=[str(point.chunk_id) for point in points_a],
            with_payload=True,
            with_vectors=True,
        )

        assert len(records) == 3
        for record in records:
            assert record.payload is not None
            assert "content" not in record.payload
            assert record.payload[DOCUMENT_VERSION_ID_PAYLOAD] == str(_VERSION_A_ID)
            assert isinstance(record.vector, dict)
            assert set(record.vector) == {"dense"}

        total = await client.count(
            collection_name=collection_name,
            exact=True,
        )
        assert total.count == 5
    finally:
        if await client.collection_exists(
            collection_name=collection_name,
        ):
            await client.delete_collection(
                collection_name=collection_name,
            )
        await close_qdrant_client(client)
