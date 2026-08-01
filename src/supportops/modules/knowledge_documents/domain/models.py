"""Versioned knowledge-document domain entities and invariants."""

from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from enum import StrEnum
from uuid import UUID, uuid4, uuid5

from supportops.modules.knowledge_documents.domain.content import (
    compute_content_sha256,
    normalize_document_content,
    validate_content_sha256,
)

DOCUMENT_TITLE_MAX_LENGTH = 200
DOCUMENT_EXTERNAL_REFERENCE_MAX_LENGTH = 128
DOCUMENT_INDEX_IDENTIFIER_MAX_LENGTH = 128
DOCUMENT_INDEX_ERROR_CODE_MAX_LENGTH = 64
DOCUMENT_PRICING_CATALOG_VERSION_MAX_LENGTH = 128
DOCUMENT_CHUNK_SECTION_DEPTH_MAX = 6
DOCUMENT_CHUNK_SECTION_SEGMENT_MAX_LENGTH = 200
DOCUMENT_EMBEDDING_COST_SCALE = 12
DOCUMENT_EMBEDDING_MAX_COST_USD = Decimal("99999999.999999999999")

KNOWLEDGE_CHUNK_ID_NAMESPACE = UUID("8a9fa064-2bea-4c7a-bf7f-c8f9e4abf620")


class DocumentMediaType(StrEnum):
    """Text media types accepted by the knowledge ingestion boundary."""

    TEXT_PLAIN = "text/plain"
    TEXT_MARKDOWN = "text/markdown"


class DocumentVersionStatus(StrEnum):
    """Durable indexing eligibility states for one immutable version."""

    PENDING = "pending"
    READY = "ready"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class KnowledgeIndexProfile:
    """Immutable chunking, embedding, and vector-index identity."""

    chunking_strategy: str
    chunking_version: str
    tokenizer_encoding: str
    embedding_provider: str
    embedding_model: str
    embedding_dimensions: int
    knowledge_collection: str
    knowledge_vector_name: str

    def __post_init__(self) -> None:
        _validate_bounded_identifier(
            self.chunking_strategy,
            field_name="chunking_strategy",
            maximum_length=DOCUMENT_INDEX_IDENTIFIER_MAX_LENGTH,
        )
        _validate_bounded_identifier(
            self.chunking_version,
            field_name="chunking_version",
            maximum_length=DOCUMENT_INDEX_IDENTIFIER_MAX_LENGTH,
        )
        _validate_bounded_identifier(
            self.tokenizer_encoding,
            field_name="tokenizer_encoding",
            maximum_length=DOCUMENT_INDEX_IDENTIFIER_MAX_LENGTH,
        )
        _validate_bounded_identifier(
            self.embedding_provider,
            field_name="embedding_provider",
            maximum_length=DOCUMENT_INDEX_IDENTIFIER_MAX_LENGTH,
        )
        _validate_bounded_identifier(
            self.embedding_model,
            field_name="embedding_model",
            maximum_length=DOCUMENT_INDEX_IDENTIFIER_MAX_LENGTH,
        )
        if self.embedding_dimensions <= 0:
            raise ValueError("embedding_dimensions must be positive.")
        _validate_bounded_identifier(
            self.knowledge_collection,
            field_name="knowledge_collection",
            maximum_length=DOCUMENT_INDEX_IDENTIFIER_MAX_LENGTH,
        )
        _validate_bounded_identifier(
            self.knowledge_vector_name,
            field_name="knowledge_vector_name",
            maximum_length=DOCUMENT_INDEX_IDENTIFIER_MAX_LENGTH,
        )


@dataclass(frozen=True, slots=True)
class Document:
    """Workspace-owned knowledge-document identity and active-version pointer."""

    id: UUID
    workspace_id: UUID
    title: str
    external_reference: str | None
    active_version_id: UUID | None
    created_at: datetime
    updated_at: datetime

    def __post_init__(self) -> None:
        _validate_bounded_text(
            self.title,
            field_name="title",
            maximum_length=DOCUMENT_TITLE_MAX_LENGTH,
        )
        _validate_optional_bounded_identifier(
            self.external_reference,
            field_name="external_reference",
            maximum_length=DOCUMENT_EXTERNAL_REFERENCE_MAX_LENGTH,
        )
        _validate_utc_timestamp(self.created_at, field_name="created_at")
        _validate_utc_timestamp(self.updated_at, field_name="updated_at")
        if self.updated_at < self.created_at:
            raise ValueError("updated_at must not be earlier than created_at.")

    @classmethod
    def create(
        cls,
        *,
        workspace_id: UUID,
        title: str,
        external_reference: str | None = None,
        document_id: UUID | None = None,
        now: datetime | None = None,
    ) -> "Document":
        """Create a workspace-owned document without an active version."""

        created_at = now or datetime.now(UTC)
        normalized_external_reference = (
            external_reference.strip() if external_reference is not None else None
        )
        return cls(
            id=document_id or uuid4(),
            workspace_id=workspace_id,
            title=title.strip(),
            external_reference=normalized_external_reference,
            active_version_id=None,
            created_at=created_at,
            updated_at=created_at,
        )

    def activate_version(
        self,
        version: "DocumentVersion",
        *,
        now: datetime | None = None,
    ) -> "Document":
        """Return a document pointing to one ready owned version."""

        if version.workspace_id != self.workspace_id or version.document_id != self.id:
            raise ValueError("The active version must belong to the same document and workspace.")
        if version.status is not DocumentVersionStatus.READY:
            raise ValueError("Only ready document versions may become active.")
        if self.active_version_id == version.id:
            return self

        updated_at = now or datetime.now(UTC)
        if updated_at < self.updated_at:
            raise ValueError("updated_at must not move backwards.")

        return replace(
            self,
            active_version_id=version.id,
            updated_at=updated_at,
        )


@dataclass(frozen=True, slots=True)
class DocumentVersion:
    """Immutable source content plus indexing state and provenance."""

    id: UUID
    workspace_id: UUID
    document_id: UUID
    version_number: int
    media_type: DocumentMediaType
    content: str
    content_sha256: str
    status: DocumentVersionStatus
    chunking_strategy: str | None
    chunking_version: str | None
    tokenizer_encoding: str | None
    embedding_provider: str | None
    embedding_model: str | None
    embedding_dimensions: int | None
    knowledge_collection: str | None
    knowledge_vector_name: str | None
    embedding_input_tokens: int | None
    embedding_estimated_cost_usd: Decimal | None
    embedding_pricing_catalog_version: str | None
    chunk_count: int | None
    indexed_at: datetime | None
    last_error_code: str | None
    created_at: datetime
    updated_at: datetime

    def __post_init__(self) -> None:
        if self.version_number <= 0:
            raise ValueError("version_number must be positive.")
        if not isinstance(self.media_type, DocumentMediaType):
            raise ValueError("media_type must be a supported DocumentMediaType.")
        if not isinstance(self.status, DocumentVersionStatus):
            raise ValueError("status must be a supported DocumentVersionStatus.")

        normalized_content = normalize_document_content(self.content)
        if normalized_content != self.content:
            raise ValueError("content must use the canonical normalization format.")

        validate_content_sha256(
            self.content_sha256,
            field_name="content_sha256",
        )
        if compute_content_sha256(self.content) != self.content_sha256:
            raise ValueError("content_sha256 must match the stored content.")

        _validate_utc_timestamp(self.created_at, field_name="created_at")
        _validate_utc_timestamp(self.updated_at, field_name="updated_at")
        _validate_optional_utc_timestamp(
            self.indexed_at,
            field_name="indexed_at",
        )

        if self.updated_at < self.created_at:
            raise ValueError("updated_at must not be earlier than created_at.")
        if self.indexed_at is not None:
            if self.indexed_at < self.created_at:
                raise ValueError("indexed_at must not be earlier than created_at.")
            if self.indexed_at > self.updated_at:
                raise ValueError("indexed_at must not be later than updated_at.")

        _validate_optional_non_negative_integer(
            self.embedding_input_tokens,
            field_name="embedding_input_tokens",
        )
        _validate_optional_cost(
            self.embedding_estimated_cost_usd,
            field_name="embedding_estimated_cost_usd",
        )
        _validate_optional_bounded_identifier(
            self.embedding_pricing_catalog_version,
            field_name="embedding_pricing_catalog_version",
            maximum_length=(DOCUMENT_PRICING_CATALOG_VERSION_MAX_LENGTH),
        )
        _validate_optional_non_negative_integer(
            self.chunk_count,
            field_name="chunk_count",
        )
        _validate_optional_bounded_identifier(
            self.last_error_code,
            field_name="last_error_code",
            maximum_length=DOCUMENT_INDEX_ERROR_CODE_MAX_LENGTH,
        )

        _validate_index_profile_state(self)
        _validate_version_status_state(self)

    @classmethod
    def create_pending(
        cls,
        *,
        workspace_id: UUID,
        document_id: UUID,
        version_number: int,
        media_type: DocumentMediaType,
        content: str,
        document_version_id: UUID | None = None,
        now: datetime | None = None,
    ) -> "DocumentVersion":
        """Create a pending version from canonical accepted content."""

        created_at = now or datetime.now(UTC)
        normalized_content = normalize_document_content(content)

        return cls(
            id=document_version_id or uuid4(),
            workspace_id=workspace_id,
            document_id=document_id,
            version_number=version_number,
            media_type=media_type,
            content=normalized_content,
            content_sha256=compute_content_sha256(normalized_content),
            status=DocumentVersionStatus.PENDING,
            chunking_strategy=None,
            chunking_version=None,
            tokenizer_encoding=None,
            embedding_provider=None,
            embedding_model=None,
            embedding_dimensions=None,
            knowledge_collection=None,
            knowledge_vector_name=None,
            embedding_input_tokens=None,
            embedding_estimated_cost_usd=None,
            embedding_pricing_catalog_version=None,
            chunk_count=None,
            indexed_at=None,
            last_error_code=None,
            created_at=created_at,
            updated_at=created_at,
        )

    @property
    def index_profile(self) -> KnowledgeIndexProfile | None:
        """Return the bound immutable index profile when one exists."""

        if self.chunking_strategy is None:
            return None

        assert self.chunking_version is not None
        assert self.tokenizer_encoding is not None
        assert self.embedding_provider is not None
        assert self.embedding_model is not None
        assert self.embedding_dimensions is not None
        assert self.knowledge_collection is not None
        assert self.knowledge_vector_name is not None

        return KnowledgeIndexProfile(
            chunking_strategy=self.chunking_strategy,
            chunking_version=self.chunking_version,
            tokenizer_encoding=self.tokenizer_encoding,
            embedding_provider=self.embedding_provider,
            embedding_model=self.embedding_model,
            embedding_dimensions=self.embedding_dimensions,
            knowledge_collection=self.knowledge_collection,
            knowledge_vector_name=self.knowledge_vector_name,
        )

    def bind_index_profile(
        self,
        profile: KnowledgeIndexProfile,
        *,
        now: datetime | None = None,
    ) -> "DocumentVersion":
        """Bind the first profile or verify an identical retry profile."""

        existing_profile = self.index_profile
        if existing_profile is not None:
            if existing_profile != profile:
                raise ValueError(
                    "Document version index profile does not match the persisted profile."
                )
            return self

        updated_at = _resolve_monotonic_timestamp(self.updated_at, now)
        return replace(
            self,
            chunking_strategy=profile.chunking_strategy,
            chunking_version=profile.chunking_version,
            tokenizer_encoding=profile.tokenizer_encoding,
            embedding_provider=profile.embedding_provider,
            embedding_model=profile.embedding_model,
            embedding_dimensions=profile.embedding_dimensions,
            knowledge_collection=profile.knowledge_collection,
            knowledge_vector_name=profile.knowledge_vector_name,
            updated_at=updated_at,
        )

    def prepare_retry(
        self,
        *,
        now: datetime | None = None,
    ) -> "DocumentVersion":
        """Return a failed compatible version to pending retry state."""

        if self.status is not DocumentVersionStatus.FAILED:
            raise ValueError("Only failed document versions may be prepared for retry.")
        if self.index_profile is None:
            raise ValueError("Failed document versions require a persisted index profile.")

        updated_at = _resolve_monotonic_timestamp(self.updated_at, now)
        return replace(
            self,
            status=DocumentVersionStatus.PENDING,
            embedding_input_tokens=None,
            embedding_estimated_cost_usd=None,
            embedding_pricing_catalog_version=None,
            indexed_at=None,
            last_error_code=None,
            updated_at=updated_at,
        )

    def mark_failed(
        self,
        *,
        error_code: str,
        chunk_count: int,
        now: datetime | None = None,
    ) -> "DocumentVersion":
        """Persist a normalized failed indexing outcome."""

        if self.status is DocumentVersionStatus.READY:
            raise ValueError("Ready document versions cannot transition to failed.")
        if self.index_profile is None:
            raise ValueError("An index profile must be bound before recording failure.")

        _validate_bounded_identifier(
            error_code,
            field_name="error_code",
            maximum_length=DOCUMENT_INDEX_ERROR_CODE_MAX_LENGTH,
        )
        if chunk_count < 0:
            raise ValueError("chunk_count must be non-negative.")

        updated_at = _resolve_monotonic_timestamp(self.updated_at, now)
        return replace(
            self,
            status=DocumentVersionStatus.FAILED,
            embedding_input_tokens=None,
            embedding_estimated_cost_usd=None,
            embedding_pricing_catalog_version=None,
            chunk_count=chunk_count,
            indexed_at=None,
            last_error_code=error_code,
            updated_at=updated_at,
        )

    def mark_ready(
        self,
        *,
        chunk_count: int,
        embedding_input_tokens: int,
        embedding_estimated_cost_usd: Decimal | None,
        embedding_pricing_catalog_version: str,
        indexed_at: datetime | None = None,
    ) -> "DocumentVersion":
        """Persist successful acknowledged indexing provenance."""

        if self.status is DocumentVersionStatus.READY:
            raise ValueError("Ready document versions cannot be rewritten.")
        if self.index_profile is None:
            raise ValueError("An index profile must be bound before marking a version ready.")
        if chunk_count <= 0:
            raise ValueError("Ready document versions require at least one chunk.")
        if embedding_input_tokens < 0:
            raise ValueError("embedding_input_tokens must be non-negative.")

        _validate_optional_cost(
            embedding_estimated_cost_usd,
            field_name="embedding_estimated_cost_usd",
        )
        _validate_bounded_identifier(
            embedding_pricing_catalog_version,
            field_name="embedding_pricing_catalog_version",
            maximum_length=(DOCUMENT_PRICING_CATALOG_VERSION_MAX_LENGTH),
        )

        ready_at = indexed_at or datetime.now(UTC)
        if ready_at < self.updated_at:
            raise ValueError("indexed_at must not move document version time backwards.")

        return replace(
            self,
            status=DocumentVersionStatus.READY,
            embedding_input_tokens=embedding_input_tokens,
            embedding_estimated_cost_usd=embedding_estimated_cost_usd,
            embedding_pricing_catalog_version=(embedding_pricing_catalog_version),
            chunk_count=chunk_count,
            indexed_at=ready_at,
            last_error_code=None,
            updated_at=ready_at,
        )


@dataclass(frozen=True, slots=True)
class DocumentChunk:
    """Immutable authoritative chunk content for one document version."""

    id: UUID
    workspace_id: UUID
    document_id: UUID
    document_version_id: UUID
    ordinal: int
    section_path: tuple[str, ...]
    content: str
    content_sha256: str
    token_count: int
    chunking_strategy: str
    chunking_version: str
    created_at: datetime

    def __post_init__(self) -> None:
        if self.ordinal < 0:
            raise ValueError("ordinal must be non-negative.")

        _validate_section_path(self.section_path)

        if not self.content.strip():
            raise ValueError("Chunk content must contain non-whitespace text.")

        validate_content_sha256(
            self.content_sha256,
            field_name="content_sha256",
        )
        if compute_content_sha256(self.content) != self.content_sha256:
            raise ValueError("content_sha256 must match the stored chunk content.")
        if self.token_count <= 0:
            raise ValueError("token_count must be positive.")

        _validate_bounded_identifier(
            self.chunking_strategy,
            field_name="chunking_strategy",
            maximum_length=DOCUMENT_INDEX_IDENTIFIER_MAX_LENGTH,
        )
        _validate_bounded_identifier(
            self.chunking_version,
            field_name="chunking_version",
            maximum_length=DOCUMENT_INDEX_IDENTIFIER_MAX_LENGTH,
        )
        _validate_utc_timestamp(self.created_at, field_name="created_at")

    @classmethod
    def create(
        cls,
        *,
        document_version: DocumentVersion,
        ordinal: int,
        section_path: tuple[str, ...],
        content: str,
        token_count: int,
        chunk_id: UUID | None = None,
        now: datetime | None = None,
    ) -> "DocumentChunk":
        """Create one deterministic chunk under the bound profile."""

        profile = document_version.index_profile
        if profile is None:
            raise ValueError("A document version index profile is required before chunk creation.")

        normalized_section_path = tuple(segment.strip() for segment in section_path)
        content_sha256 = compute_content_sha256(content)
        deterministic_id = build_document_chunk_id(
            document_version_id=document_version.id,
            chunking_strategy=profile.chunking_strategy,
            chunking_version=profile.chunking_version,
            tokenizer_encoding=profile.tokenizer_encoding,
            ordinal=ordinal,
            content_sha256=content_sha256,
        )

        if chunk_id is not None and chunk_id != deterministic_id:
            raise ValueError("chunk_id must match the deterministic chunk identity.")

        return cls(
            id=deterministic_id,
            workspace_id=document_version.workspace_id,
            document_id=document_version.document_id,
            document_version_id=document_version.id,
            ordinal=ordinal,
            section_path=normalized_section_path,
            content=content,
            content_sha256=content_sha256,
            token_count=token_count,
            chunking_strategy=profile.chunking_strategy,
            chunking_version=profile.chunking_version,
            created_at=now or datetime.now(UTC),
        )


def build_document_chunk_id(
    *,
    document_version_id: UUID,
    chunking_strategy: str,
    chunking_version: str,
    tokenizer_encoding: str,
    ordinal: int,
    content_sha256: str,
) -> UUID:
    """Build the stable UUIDv5 used by PostgreSQL and Qdrant."""

    if ordinal < 0:
        raise ValueError("ordinal must be non-negative.")

    _validate_bounded_identifier(
        chunking_strategy,
        field_name="chunking_strategy",
        maximum_length=DOCUMENT_INDEX_IDENTIFIER_MAX_LENGTH,
    )
    _validate_bounded_identifier(
        chunking_version,
        field_name="chunking_version",
        maximum_length=DOCUMENT_INDEX_IDENTIFIER_MAX_LENGTH,
    )
    _validate_bounded_identifier(
        tokenizer_encoding,
        field_name="tokenizer_encoding",
        maximum_length=DOCUMENT_INDEX_IDENTIFIER_MAX_LENGTH,
    )
    validate_content_sha256(
        content_sha256,
        field_name="content_sha256",
    )

    canonical_identity = ":".join(
        (
            str(document_version_id),
            chunking_strategy,
            chunking_version,
            tokenizer_encoding,
            str(ordinal),
            content_sha256,
        )
    )
    return uuid5(
        KNOWLEDGE_CHUNK_ID_NAMESPACE,
        canonical_identity,
    )


def _validate_index_profile_state(
    version: DocumentVersion,
) -> None:
    profile_fields = (
        version.chunking_strategy,
        version.chunking_version,
        version.tokenizer_encoding,
        version.embedding_provider,
        version.embedding_model,
        version.embedding_dimensions,
        version.knowledge_collection,
        version.knowledge_vector_name,
    )
    populated_count = sum(value is not None for value in profile_fields)

    if populated_count not in {0, len(profile_fields)}:
        raise ValueError("Index profile fields must be populated or cleared together.")

    if populated_count == len(profile_fields):
        assert version.chunking_strategy is not None
        assert version.chunking_version is not None
        assert version.tokenizer_encoding is not None
        assert version.embedding_provider is not None
        assert version.embedding_model is not None
        assert version.embedding_dimensions is not None
        assert version.knowledge_collection is not None
        assert version.knowledge_vector_name is not None

        KnowledgeIndexProfile(
            chunking_strategy=version.chunking_strategy,
            chunking_version=version.chunking_version,
            tokenizer_encoding=version.tokenizer_encoding,
            embedding_provider=version.embedding_provider,
            embedding_model=version.embedding_model,
            embedding_dimensions=version.embedding_dimensions,
            knowledge_collection=version.knowledge_collection,
            knowledge_vector_name=version.knowledge_vector_name,
        )


def _validate_version_status_state(
    version: DocumentVersion,
) -> None:
    profile = version.index_profile

    if profile is None:
        if version.status is not DocumentVersionStatus.PENDING:
            raise ValueError("Only pending versions may exist without an index profile.")

        _require_empty_index_outcome(version)

        if version.chunk_count is not None:
            raise ValueError("Unprofiled pending versions must not define chunk_count.")
        return

    if version.status is DocumentVersionStatus.PENDING:
        if version.last_error_code is not None:
            raise ValueError("Pending versions must not define last_error_code.")
        if version.indexed_at is not None:
            raise ValueError("Pending versions must not define indexed_at.")

        _require_empty_embedding_outcome(version)
        return

    if version.status is DocumentVersionStatus.FAILED:
        if version.last_error_code is None:
            raise ValueError("Failed versions require last_error_code.")
        if version.chunk_count is None:
            raise ValueError("Failed versions require chunk_count.")
        if version.indexed_at is not None:
            raise ValueError("Failed versions must not define indexed_at.")

        _require_empty_embedding_outcome(version)
        return

    if version.last_error_code is not None:
        raise ValueError("Ready versions must not define last_error_code.")
    if version.chunk_count is None or version.chunk_count <= 0:
        raise ValueError("Ready versions require a positive chunk_count.")
    if version.embedding_input_tokens is None:
        raise ValueError("Ready versions require embedding_input_tokens.")
    if version.embedding_pricing_catalog_version is None:
        raise ValueError("Ready versions require embedding pricing provenance.")
    if version.indexed_at is None:
        raise ValueError("Ready versions require indexed_at.")


def _require_empty_index_outcome(
    version: DocumentVersion,
) -> None:
    if version.last_error_code is not None:
        raise ValueError("Unprofiled pending versions must not define last_error_code.")
    if version.indexed_at is not None:
        raise ValueError("Unprofiled pending versions must not define indexed_at.")

    _require_empty_embedding_outcome(version)


def _require_empty_embedding_outcome(
    version: DocumentVersion,
) -> None:
    if version.embedding_input_tokens is not None:
        raise ValueError("Non-ready versions must not define embedding_input_tokens.")
    if version.embedding_estimated_cost_usd is not None:
        raise ValueError("Non-ready versions must not define embedding estimated cost.")
    if version.embedding_pricing_catalog_version is not None:
        raise ValueError("Non-ready versions must not define embedding pricing provenance.")


def _validate_section_path(
    section_path: tuple[str, ...],
) -> None:
    if not isinstance(section_path, tuple):
        raise TypeError("section_path must be an immutable tuple.")
    if len(section_path) > DOCUMENT_CHUNK_SECTION_DEPTH_MAX:
        raise ValueError("section_path exceeds the supported heading depth.")

    for segment in section_path:
        _validate_bounded_text(
            segment,
            field_name="section_path segment",
            maximum_length=DOCUMENT_CHUNK_SECTION_SEGMENT_MAX_LENGTH,
        )


def _validate_bounded_text(
    value: str,
    *,
    field_name: str,
    maximum_length: int,
) -> None:
    _validate_bounded_identifier(
        value,
        field_name=field_name,
        maximum_length=maximum_length,
    )


def _validate_bounded_identifier(
    value: str,
    *,
    field_name: str,
    maximum_length: int,
) -> None:
    if not value:
        raise ValueError(f"{field_name} is required.")
    if value != value.strip():
        raise ValueError(f"{field_name} must not contain surrounding whitespace.")
    if len(value) > maximum_length:
        raise ValueError(f"{field_name} exceeds the maximum length.")


def _validate_optional_bounded_identifier(
    value: str | None,
    *,
    field_name: str,
    maximum_length: int,
) -> None:
    if value is None:
        return

    _validate_bounded_identifier(
        value,
        field_name=field_name,
        maximum_length=maximum_length,
    )


def _validate_optional_non_negative_integer(
    value: int | None,
    *,
    field_name: str,
) -> None:
    if value is not None and value < 0:
        raise ValueError(f"{field_name} must be non-negative.")


def _validate_optional_cost(
    value: Decimal | None,
    *,
    field_name: str,
) -> None:
    if value is None:
        return
    if not isinstance(value, Decimal):
        raise TypeError(f"{field_name} must be a Decimal.")
    if not value.is_finite():
        raise ValueError(f"{field_name} must be finite.")
    if value < 0:
        raise ValueError(f"{field_name} must be non-negative.")
    if value > DOCUMENT_EMBEDDING_MAX_COST_USD:
        raise ValueError(f"{field_name} exceeds the supported maximum.")

    exponent = value.as_tuple().exponent
    if isinstance(exponent, int) and exponent < -DOCUMENT_EMBEDDING_COST_SCALE:
        raise ValueError(f"{field_name} exceeds the supported decimal scale.")


def _resolve_monotonic_timestamp(
    current: datetime,
    proposed: datetime | None,
) -> datetime:
    resolved = proposed or datetime.now(UTC)
    _validate_utc_timestamp(
        resolved,
        field_name="updated_at",
    )

    if resolved < current:
        raise ValueError("updated_at must not move backwards.")

    return resolved


def _validate_optional_utc_timestamp(
    value: datetime | None,
    *,
    field_name: str,
) -> None:
    if value is None:
        return

    _validate_utc_timestamp(
        value,
        field_name=field_name,
    )


def _validate_utc_timestamp(
    value: datetime,
    *,
    field_name: str,
) -> None:
    if value.tzinfo is None or value.utcoffset() != timedelta(0):
        raise ValueError(f"{field_name} must be a UTC-aware timestamp.")
