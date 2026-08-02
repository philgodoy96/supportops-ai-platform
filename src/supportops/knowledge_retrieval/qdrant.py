"""Qdrant adapter for active-version semantic knowledge search."""

import logging
from collections.abc import Mapping, Sequence
from typing import Protocol
from uuid import UUID

from qdrant_client import AsyncQdrantClient, models
from qdrant_client.http.exceptions import (
    ResponseHandlingException,
    UnexpectedResponse,
)

from supportops.knowledge_index.vector_store.contracts import (
    KnowledgeCollectionProfile,
    KnowledgeVectorStoreOperationError,
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
    KnowledgeVectorCandidate,
    KnowledgeVectorSearchRequest,
)
from supportops.modules.knowledge_documents.domain.models import (
    DocumentMediaType,
)

DEFAULT_KNOWLEDGE_SEARCH_TIMEOUT_SECONDS = 10

_KNOWLEDGE_SEARCH_PAYLOAD_FIELDS = (
    WORKSPACE_ID_PAYLOAD,
    DOCUMENT_ID_PAYLOAD,
    DOCUMENT_VERSION_ID_PAYLOAD,
    CHUNK_ID_PAYLOAD,
    CHUNK_ORDINAL_PAYLOAD,
    CONTENT_SHA256_PAYLOAD,
    MEDIA_TYPE_PAYLOAD,
    CHUNKING_STRATEGY_PAYLOAD,
    CHUNKING_VERSION_PAYLOAD,
)

_LOGGER = logging.getLogger(__name__)


class KnowledgeCollectionGuard(Protocol):
    """Validate one vector collection before semantic search."""

    async def ensure_collection(
        self,
        profile: KnowledgeCollectionProfile,
    ) -> None:
        """Create or validate a compatible collection."""
        ...


class QdrantKnowledgeVectorSearcher:
    """Search active knowledge projections without trusting Qdrant content."""

    def __init__(
        self,
        *,
        client: AsyncQdrantClient,
        collection_guard: KnowledgeCollectionGuard,
        search_timeout_seconds: int = (DEFAULT_KNOWLEDGE_SEARCH_TIMEOUT_SECONDS),
    ) -> None:
        if search_timeout_seconds <= 0:
            raise ValueError("search_timeout_seconds must be positive.")

        self._client = client
        self._collection_guard = collection_guard
        self._search_timeout_seconds = search_timeout_seconds

    async def search(
        self,
        request: KnowledgeVectorSearchRequest,
    ) -> Sequence[KnowledgeVectorCandidate]:
        """Return validated candidates from one compatible collection."""

        collection_profile = KnowledgeCollectionProfile(
            collection_name=(request.profile.knowledge_collection),
            vector_name=(request.profile.knowledge_vector_name),
            dimensions=(request.profile.embedding_dimensions),
        )

        await self._collection_guard.ensure_collection(collection_profile)

        try:
            response = await self._client.query_points(
                collection_name=(collection_profile.collection_name),
                query=list(request.query_vector),
                using=collection_profile.vector_name,
                query_filter=_build_search_filter(
                    workspace_id=request.workspace_id,
                    targets=request.targets,
                ),
                limit=request.limit,
                with_payload=list(_KNOWLEDGE_SEARCH_PAYLOAD_FIELDS),
                with_vectors=False,
                timeout=self._search_timeout_seconds,
            )
        except (
            ResponseHandlingException,
            UnexpectedResponse,
            OSError,
            TimeoutError,
        ) as error:
            raise _normalize_qdrant_error(error) from error

        return _validated_candidates(
            points=response.points,
            request=request,
        )


def _validated_candidates(
    *,
    points: Sequence[models.ScoredPoint],
    request: KnowledgeVectorSearchRequest,
) -> tuple[KnowledgeVectorCandidate, ...]:
    candidates: list[KnowledgeVectorCandidate] = []
    target_set = set(request.targets)
    seen_chunk_ids: set[UUID] = set()

    for point in points:
        try:
            candidate = _candidate_from_point(point)
            _validate_candidate_scope(
                candidate=candidate,
                request=request,
                target_set=target_set,
            )

            if candidate.chunk_id in seen_chunk_ids:
                raise ValueError("Qdrant returned a duplicate chunk candidate.")
        except (
            KeyError,
            TypeError,
            ValueError,
        ) as error:
            _log_discarded_candidate(
                point=point,
                reason=error,
                collection_name=(request.profile.knowledge_collection),
            )
            continue

        seen_chunk_ids.add(candidate.chunk_id)
        candidates.append(candidate)

    return tuple(candidates)


def _candidate_from_point(
    point: models.ScoredPoint,
) -> KnowledgeVectorCandidate:
    payload = point.payload
    if payload is None:
        raise ValueError("Qdrant candidate payload is missing.")

    point_id = _parse_uuid(
        point.id,
        field_name="point.id",
    )
    chunk_id = _parse_uuid(
        _required_payload_value(
            payload,
            CHUNK_ID_PAYLOAD,
        ),
        field_name=CHUNK_ID_PAYLOAD,
    )

    if point_id != chunk_id:
        raise ValueError("Qdrant point ID does not match chunk_id.")

    return KnowledgeVectorCandidate(
        chunk_id=chunk_id,
        workspace_id=_parse_uuid(
            _required_payload_value(
                payload,
                WORKSPACE_ID_PAYLOAD,
            ),
            field_name=WORKSPACE_ID_PAYLOAD,
        ),
        document_id=_parse_uuid(
            _required_payload_value(
                payload,
                DOCUMENT_ID_PAYLOAD,
            ),
            field_name=DOCUMENT_ID_PAYLOAD,
        ),
        document_version_id=_parse_uuid(
            _required_payload_value(
                payload,
                DOCUMENT_VERSION_ID_PAYLOAD,
            ),
            field_name=DOCUMENT_VERSION_ID_PAYLOAD,
        ),
        ordinal=_parse_integer(
            _required_payload_value(
                payload,
                CHUNK_ORDINAL_PAYLOAD,
            ),
            field_name=CHUNK_ORDINAL_PAYLOAD,
        ),
        content_sha256=_parse_text(
            _required_payload_value(
                payload,
                CONTENT_SHA256_PAYLOAD,
            ),
            field_name=CONTENT_SHA256_PAYLOAD,
        ),
        media_type=DocumentMediaType(
            _parse_text(
                _required_payload_value(
                    payload,
                    MEDIA_TYPE_PAYLOAD,
                ),
                field_name=MEDIA_TYPE_PAYLOAD,
            )
        ),
        chunking_strategy=_parse_text(
            _required_payload_value(
                payload,
                CHUNKING_STRATEGY_PAYLOAD,
            ),
            field_name=CHUNKING_STRATEGY_PAYLOAD,
        ),
        chunking_version=_parse_text(
            _required_payload_value(
                payload,
                CHUNKING_VERSION_PAYLOAD,
            ),
            field_name=CHUNKING_VERSION_PAYLOAD,
        ),
        score=point.score,
    )


def _validate_candidate_scope(
    *,
    candidate: KnowledgeVectorCandidate,
    request: KnowledgeVectorSearchRequest,
    target_set: set[KnowledgeSearchTarget],
) -> None:
    if candidate.workspace_id != request.workspace_id:
        raise ValueError("Qdrant candidate belongs to another workspace.")

    candidate_target = KnowledgeSearchTarget(
        document_id=candidate.document_id,
        document_version_id=(candidate.document_version_id),
    )
    if candidate_target not in target_set:
        raise ValueError("Qdrant candidate does not belong to an active search target.")

    if (
        candidate.chunking_strategy != request.profile.chunking_strategy
        or candidate.chunking_version != request.profile.chunking_version
    ):
        raise ValueError("Qdrant candidate chunking profile does not match the search profile.")


def _build_search_filter(
    *,
    workspace_id: UUID,
    targets: tuple[KnowledgeSearchTarget, ...],
) -> models.Filter:
    target_filter = models.Filter(
        should=[
            models.Filter(
                must=[
                    _uuid_condition(
                        key=DOCUMENT_ID_PAYLOAD,
                        value=target.document_id,
                    ),
                    _uuid_condition(
                        key=(DOCUMENT_VERSION_ID_PAYLOAD),
                        value=(target.document_version_id),
                    ),
                ]
            )
            for target in targets
        ]
    )

    return models.Filter(
        must=[
            _uuid_condition(
                key=WORKSPACE_ID_PAYLOAD,
                value=workspace_id,
            ),
            target_filter,
        ]
    )


def _uuid_condition(
    *,
    key: str,
    value: UUID,
) -> models.FieldCondition:
    return models.FieldCondition(
        key=key,
        match=models.MatchValue(value=str(value)),
    )


def _required_payload_value(
    payload: Mapping[str, object],
    key: str,
) -> object:
    if key not in payload:
        raise KeyError(f"Qdrant payload field {key!r} is missing.")

    return payload[key]


def _parse_uuid(
    value: object,
    *,
    field_name: str,
) -> UUID:
    if isinstance(value, UUID):
        return value

    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a UUID string.")

    try:
        return UUID(value)
    except ValueError as error:
        raise ValueError(f"{field_name} must be a valid UUID.") from error


def _parse_integer(
    value: object,
    *,
    field_name: str,
) -> int:
    if isinstance(value, bool) or not isinstance(
        value,
        int,
    ):
        raise TypeError(f"{field_name} must be an integer.")

    return value


def _parse_text(
    value: object,
    *,
    field_name: str,
) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string.")
    if not value:
        raise ValueError(f"{field_name} must not be empty.")
    if value != value.strip():
        raise ValueError(f"{field_name} must not contain surrounding whitespace.")

    return value


def _log_discarded_candidate(
    *,
    point: models.ScoredPoint,
    reason: Exception,
    collection_name: str,
) -> None:
    _LOGGER.warning(
        "Discarded inconsistent Qdrant knowledge candidate.",
        extra={
            "collection_name": collection_name,
            "point_id": str(point.id),
            "reason_type": type(reason).__name__,
        },
    )


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

    return KnowledgeVectorStoreOperationError(
        "The knowledge vector store rejected the search operation."
    )
