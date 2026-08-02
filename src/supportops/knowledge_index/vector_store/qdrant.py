"""Qdrant adapter for the rebuildable knowledge vector projection."""

from collections.abc import Mapping, Sequence
from types import MappingProxyType
from uuid import UUID

from qdrant_client import AsyncQdrantClient, models
from qdrant_client.http.exceptions import (
    ResponseHandlingException,
    UnexpectedResponse,
)

from supportops.knowledge_index.vector_store.contracts import (
    KnowledgeCollectionCompatibilityError,
    KnowledgeCollectionProfile,
    KnowledgeVectorPoint,
    KnowledgeVectorStoreOperationError,
    KnowledgeVectorStoreUnavailableError,
    KnowledgeVersionProjection,
)

DEFAULT_KNOWLEDGE_VECTOR_BATCH_SIZE = 64

WORKSPACE_ID_PAYLOAD = "workspace_id"
DOCUMENT_ID_PAYLOAD = "document_id"
DOCUMENT_VERSION_ID_PAYLOAD = "document_version_id"
CHUNK_ID_PAYLOAD = "chunk_id"
CHUNK_ORDINAL_PAYLOAD = "ordinal"
CONTENT_SHA256_PAYLOAD = "content_sha256"
MEDIA_TYPE_PAYLOAD = "media_type"
CHUNKING_STRATEGY_PAYLOAD = "chunking_strategy"
CHUNKING_VERSION_PAYLOAD = "chunking_version"

KNOWLEDGE_PAYLOAD_INDEXES: Mapping[
    str,
    models.PayloadSchemaType,
] = MappingProxyType(
    {
        WORKSPACE_ID_PAYLOAD: models.PayloadSchemaType.UUID,
        DOCUMENT_ID_PAYLOAD: models.PayloadSchemaType.UUID,
        DOCUMENT_VERSION_ID_PAYLOAD: models.PayloadSchemaType.UUID,
    }
)


class QdrantKnowledgeVectorStore:
    """Maintain version-scoped Qdrant points without source content."""

    def __init__(
        self,
        *,
        client: AsyncQdrantClient,
        batch_size: int = DEFAULT_KNOWLEDGE_VECTOR_BATCH_SIZE,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")

        self._client = client
        self._batch_size = batch_size

    async def ensure_collection(
        self,
        profile: KnowledgeCollectionProfile,
    ) -> None:
        """Create or validate one named-vector collection."""

        try:
            exists = await self._client.collection_exists(
                collection_name=profile.collection_name,
            )
            if not exists:
                await self._create_collection_safely(profile)

            collection_info = await self._client.get_collection(
                collection_name=profile.collection_name,
            )
            _validate_collection_profile(
                collection_info,
                profile=profile,
            )
            await self._ensure_payload_indexes(
                collection_info=collection_info,
                profile=profile,
            )
        except (
            KnowledgeCollectionCompatibilityError,
            KnowledgeVectorStoreOperationError,
            KnowledgeVectorStoreUnavailableError,
        ):
            raise
        except (
            ResponseHandlingException,
            UnexpectedResponse,
            OSError,
            TimeoutError,
        ) as error:
            raise _normalize_qdrant_error(error) from error

    async def upsert_version_points(
        self,
        *,
        profile: KnowledgeCollectionProfile,
        projection: KnowledgeVersionProjection,
        points: Sequence[KnowledgeVectorPoint],
    ) -> None:
        """Idempotently upsert one complete deterministic projection."""

        normalized_points = tuple(points)
        if not normalized_points:
            raise ValueError("A document version projection requires at least one point.")

        _validate_projection_points(
            profile=profile,
            projection=projection,
            points=normalized_points,
        )
        await self.ensure_collection(profile)

        try:
            for start in range(
                0,
                len(normalized_points),
                self._batch_size,
            ):
                batch = normalized_points[start : start + self._batch_size]
                result = await self._client.upsert(
                    collection_name=profile.collection_name,
                    points=[
                        _to_qdrant_point(
                            point,
                            vector_name=profile.vector_name,
                        )
                        for point in batch
                    ],
                    wait=True,
                )
                _require_completed(
                    result,
                    operation="upsert knowledge vector points",
                )
        except (
            KnowledgeVectorStoreOperationError,
            KnowledgeVectorStoreUnavailableError,
        ):
            raise
        except (
            ResponseHandlingException,
            UnexpectedResponse,
            OSError,
            TimeoutError,
        ) as error:
            raise _normalize_qdrant_error(error) from error

    async def count_version_points(
        self,
        *,
        profile: KnowledgeCollectionProfile,
        projection: KnowledgeVersionProjection,
    ) -> int:
        """Return the exact number of points projected for one version."""

        await self.ensure_collection(profile)

        try:
            result = await self._client.count(
                collection_name=profile.collection_name,
                count_filter=_version_filter(projection),
                exact=True,
            )
        except (
            ResponseHandlingException,
            UnexpectedResponse,
            OSError,
            TimeoutError,
        ) as error:
            raise _normalize_qdrant_error(error) from error

        return result.count

    async def _create_collection_safely(
        self,
        profile: KnowledgeCollectionProfile,
    ) -> None:
        try:
            await self._client.create_collection(
                collection_name=profile.collection_name,
                vectors_config={
                    profile.vector_name: models.VectorParams(
                        size=profile.dimensions,
                        distance=models.Distance.COSINE,
                    )
                },
            )
        except UnexpectedResponse as error:
            if error.status_code not in {400, 409}:
                raise

            if not await self._client.collection_exists(
                collection_name=profile.collection_name,
            ):
                raise

    async def _ensure_payload_indexes(
        self,
        *,
        collection_info: models.CollectionInfo,
        profile: KnowledgeCollectionProfile,
    ) -> None:
        payload_schema = collection_info.payload_schema or {}

        for field_name, field_schema in KNOWLEDGE_PAYLOAD_INDEXES.items():
            existing_index = payload_schema.get(field_name)
            if existing_index is not None:
                if existing_index.data_type != field_schema:
                    raise KnowledgeCollectionCompatibilityError(
                        "Knowledge collection payload index "
                        f"{field_name!r} has an incompatible type."
                    )
                continue

            try:
                result = await self._client.create_payload_index(
                    collection_name=profile.collection_name,
                    field_name=field_name,
                    field_schema=field_schema,
                    wait=True,
                )
                _require_completed(
                    result,
                    operation=(f"create payload index {field_name!r}"),
                )
            except UnexpectedResponse as error:
                if error.status_code not in {400, 409}:
                    raise

                refreshed = await self._client.get_collection(
                    collection_name=profile.collection_name,
                )
                concurrent_index = (refreshed.payload_schema or {}).get(field_name)
                if concurrent_index is None or concurrent_index.data_type != field_schema:
                    raise


def _validate_collection_profile(
    collection_info: models.CollectionInfo,
    *,
    profile: KnowledgeCollectionProfile,
) -> None:
    vectors = collection_info.config.params.vectors
    if not isinstance(vectors, dict):
        raise KnowledgeCollectionCompatibilityError(
            "Knowledge collection must use one named dense vector."
        )
    if set(vectors) != {profile.vector_name}:
        raise KnowledgeCollectionCompatibilityError(
            "Knowledge collection vector names are incompatible."
        )

    vector_params = vectors[profile.vector_name]
    if vector_params.size != profile.dimensions:
        raise KnowledgeCollectionCompatibilityError(
            "Knowledge collection vector dimensions are incompatible."
        )
    if vector_params.distance != models.Distance.COSINE:
        raise KnowledgeCollectionCompatibilityError(
            "Knowledge collection distance metric is incompatible."
        )


def _validate_projection_points(
    *,
    profile: KnowledgeCollectionProfile,
    projection: KnowledgeVersionProjection,
    points: tuple[KnowledgeVectorPoint, ...],
) -> None:
    chunk_ids: set[UUID] = set()
    ordinals: set[int] = set()

    for point in points:
        if (
            point.workspace_id != projection.workspace_id
            or point.document_id != projection.document_id
            or point.document_version_id != projection.document_version_id
        ):
            raise ValueError("Every vector point must belong to the projected version.")
        if len(point.vector) != profile.dimensions:
            raise ValueError("Every vector point must match the collection dimensions.")
        if point.chunk_id in chunk_ids:
            raise ValueError("Vector point batch contains a duplicate chunk identifier.")
        if point.ordinal in ordinals:
            raise ValueError("Vector point batch contains a duplicate chunk ordinal.")

        chunk_ids.add(point.chunk_id)
        ordinals.add(point.ordinal)


def _to_qdrant_point(
    point: KnowledgeVectorPoint,
    *,
    vector_name: str,
) -> models.PointStruct:
    return models.PointStruct(
        id=str(point.chunk_id),
        vector={
            vector_name: list(point.vector),
        },
        payload={
            WORKSPACE_ID_PAYLOAD: str(point.workspace_id),
            DOCUMENT_ID_PAYLOAD: str(point.document_id),
            DOCUMENT_VERSION_ID_PAYLOAD: str(point.document_version_id),
            CHUNK_ID_PAYLOAD: str(point.chunk_id),
            CHUNK_ORDINAL_PAYLOAD: point.ordinal,
            CONTENT_SHA256_PAYLOAD: point.content_sha256,
            MEDIA_TYPE_PAYLOAD: point.media_type.value,
            CHUNKING_STRATEGY_PAYLOAD: point.chunking_strategy,
            CHUNKING_VERSION_PAYLOAD: point.chunking_version,
        },
    )


def _version_filter(
    projection: KnowledgeVersionProjection,
) -> models.Filter:
    return models.Filter(
        must=[
            models.FieldCondition(
                key=WORKSPACE_ID_PAYLOAD,
                match=models.MatchValue(value=str(projection.workspace_id)),
            ),
            models.FieldCondition(
                key=DOCUMENT_ID_PAYLOAD,
                match=models.MatchValue(value=str(projection.document_id)),
            ),
            models.FieldCondition(
                key=DOCUMENT_VERSION_ID_PAYLOAD,
                match=models.MatchValue(value=str(projection.document_version_id)),
            ),
        ]
    )


def _require_completed(
    result: models.UpdateResult,
    *,
    operation: str,
) -> None:
    if result.status != models.UpdateStatus.COMPLETED:
        raise KnowledgeVectorStoreOperationError(f"Qdrant did not complete operation: {operation}.")


def _normalize_qdrant_error(
    error: (ResponseHandlingException | UnexpectedResponse | OSError | TimeoutError),
) -> KnowledgeVectorStoreUnavailableError | KnowledgeVectorStoreOperationError:
    if isinstance(
        error,
        (
            ResponseHandlingException,
            OSError,
            TimeoutError,
        ),
    ):
        return KnowledgeVectorStoreUnavailableError("The knowledge vector store is unavailable.")

    if error.status_code is None or error.status_code in {408, 429} or error.status_code >= 500:
        return KnowledgeVectorStoreUnavailableError("The knowledge vector store is unavailable.")

    return KnowledgeVectorStoreOperationError("The knowledge vector store rejected the operation.")
