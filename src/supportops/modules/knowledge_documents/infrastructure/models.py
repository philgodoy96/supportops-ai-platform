"""SQLAlchemy persistence models for versioned knowledge documents."""

from datetime import datetime
from decimal import Decimal
from uuid import UUID

from sqlalchemy import (
    BigInteger,
    CheckConstraint,
    DateTime,
    ForeignKey,
    ForeignKeyConstraint,
    Index,
    Integer,
    Numeric,
    String,
    Text,
    UniqueConstraint,
    Uuid,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from supportops.infrastructure.postgresql.base import Base
from supportops.modules.knowledge_documents.domain.models import (
    DOCUMENT_CHUNK_SECTION_DEPTH_MAX,
    DOCUMENT_EMBEDDING_COST_SCALE,
    DOCUMENT_EXTERNAL_REFERENCE_MAX_LENGTH,
    DOCUMENT_INDEX_ERROR_CODE_MAX_LENGTH,
    DOCUMENT_INDEX_IDENTIFIER_MAX_LENGTH,
    DOCUMENT_PRICING_CATALOG_VERSION_MAX_LENGTH,
    DOCUMENT_TITLE_MAX_LENGTH,
    Document,
    DocumentChunk,
    DocumentMediaType,
    DocumentVersion,
    DocumentVersionStatus,
)

_DOCUMENT_CONTENT_HASH_LENGTH = 64
_DOCUMENT_VERSION_STATUS_MAX_LENGTH = 32
_DOCUMENT_MEDIA_TYPE_MAX_LENGTH = 32
_DOCUMENT_EMBEDDING_COST_PRECISION = 20

_DOCUMENT_MEDIA_TYPE_SQL_VALUES = ", ".join(f"'{member.value}'" for member in DocumentMediaType)
_DOCUMENT_VERSION_STATUS_SQL_VALUES = ", ".join(
    f"'{member.value}'" for member in DocumentVersionStatus
)


def _optional_identifier_sql(
    column_name: str,
    *,
    maximum_length: int,
) -> str:
    return (
        f"{column_name} IS NULL OR ("
        f"{column_name} = btrim({column_name}) "
        f"AND char_length({column_name}) BETWEEN 1 AND {maximum_length}"
        ")"
    )


def _required_identifier_sql(
    column_name: str,
    *,
    maximum_length: int,
) -> str:
    return (
        f"{column_name} = btrim({column_name}) "
        f"AND char_length({column_name}) BETWEEN 1 AND {maximum_length}"
    )


class DocumentRecord(Base):
    """Persisted workspace-owned knowledge-document identity."""

    __tablename__ = "knowledge_documents"

    id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True),
        primary_key=True,
    )
    workspace_id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey(
            "workspaces.id",
            ondelete="RESTRICT",
        ),
        nullable=False,
    )
    title: Mapped[str] = mapped_column(
        String(DOCUMENT_TITLE_MAX_LENGTH),
        nullable=False,
    )
    external_reference: Mapped[str | None] = mapped_column(
        String(DOCUMENT_EXTERNAL_REFERENCE_MAX_LENGTH),
        nullable=True,
    )
    active_version_id: Mapped[UUID | None] = mapped_column(
        Uuid(as_uuid=True),
        nullable=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
    )

    __table_args__ = (
        UniqueConstraint(
            "workspace_id",
            "id",
            name="uq_knowledge_documents_workspace_id",
        ),
        UniqueConstraint(
            "workspace_id",
            "external_reference",
            name="uq_knowledge_documents_workspace_external_reference",
        ),
        ForeignKeyConstraint(
            [
                "workspace_id",
                "id",
                "active_version_id",
            ],
            [
                "knowledge_document_versions.workspace_id",
                "knowledge_document_versions.document_id",
                "knowledge_document_versions.id",
            ],
            name="fk_knowledge_documents_active_version",
            ondelete="RESTRICT",
            use_alter=True,
        ),
        CheckConstraint(
            "title = btrim(title)",
            name="document_title_trimmed",
        ),
        CheckConstraint(
            (f"char_length(title) BETWEEN 1 AND {DOCUMENT_TITLE_MAX_LENGTH}"),
            name="document_title_length",
        ),
        CheckConstraint(
            _optional_identifier_sql(
                "external_reference",
                maximum_length=(DOCUMENT_EXTERNAL_REFERENCE_MAX_LENGTH),
            ),
            name="document_external_reference_format",
        ),
        CheckConstraint(
            "updated_at >= created_at",
            name="document_timestamp_order",
        ),
        Index(
            "ix_knowledge_documents_workspace_created_id",
            "workspace_id",
            created_at.desc(),
            id.desc(),
        ),
        Index(
            "ix_knowledge_documents_workspace_active_version",
            "workspace_id",
            "active_version_id",
            postgresql_where=active_version_id.is_not(None),
        ),
    )

    @classmethod
    def from_domain(
        cls,
        document: Document,
    ) -> "DocumentRecord":
        """Create a persistence record from a document entity."""

        return cls(
            id=document.id,
            workspace_id=document.workspace_id,
            title=document.title,
            external_reference=document.external_reference,
            active_version_id=document.active_version_id,
            created_at=document.created_at,
            updated_at=document.updated_at,
        )

    def to_domain(self) -> Document:
        """Map the persistence record to a document entity."""

        return Document(
            id=self.id,
            workspace_id=self.workspace_id,
            title=self.title,
            external_reference=self.external_reference,
            active_version_id=self.active_version_id,
            created_at=self.created_at,
            updated_at=self.updated_at,
        )


class DocumentVersionRecord(Base):
    """Persisted immutable source version and indexing provenance."""

    __tablename__ = "knowledge_document_versions"

    id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True),
        primary_key=True,
    )
    workspace_id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True),
        nullable=False,
    )
    document_id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True),
        nullable=False,
    )
    version_number: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )
    media_type: Mapped[str] = mapped_column(
        String(_DOCUMENT_MEDIA_TYPE_MAX_LENGTH),
        nullable=False,
    )
    content: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )
    content_sha256: Mapped[str] = mapped_column(
        String(_DOCUMENT_CONTENT_HASH_LENGTH),
        nullable=False,
    )
    status: Mapped[str] = mapped_column(
        String(_DOCUMENT_VERSION_STATUS_MAX_LENGTH),
        nullable=False,
    )
    chunking_strategy: Mapped[str | None] = mapped_column(
        String(DOCUMENT_INDEX_IDENTIFIER_MAX_LENGTH),
        nullable=True,
    )
    chunking_version: Mapped[str | None] = mapped_column(
        String(DOCUMENT_INDEX_IDENTIFIER_MAX_LENGTH),
        nullable=True,
    )
    tokenizer_encoding: Mapped[str | None] = mapped_column(
        String(DOCUMENT_INDEX_IDENTIFIER_MAX_LENGTH),
        nullable=True,
    )
    embedding_provider: Mapped[str | None] = mapped_column(
        String(DOCUMENT_INDEX_IDENTIFIER_MAX_LENGTH),
        nullable=True,
    )
    embedding_model: Mapped[str | None] = mapped_column(
        String(DOCUMENT_INDEX_IDENTIFIER_MAX_LENGTH),
        nullable=True,
    )
    embedding_dimensions: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
    )
    knowledge_collection: Mapped[str | None] = mapped_column(
        String(DOCUMENT_INDEX_IDENTIFIER_MAX_LENGTH),
        nullable=True,
    )
    knowledge_vector_name: Mapped[str | None] = mapped_column(
        String(DOCUMENT_INDEX_IDENTIFIER_MAX_LENGTH),
        nullable=True,
    )
    embedding_input_tokens: Mapped[int | None] = mapped_column(
        BigInteger,
        nullable=True,
    )
    embedding_estimated_cost_usd: Mapped[Decimal | None] = mapped_column(
        Numeric(
            _DOCUMENT_EMBEDDING_COST_PRECISION,
            DOCUMENT_EMBEDDING_COST_SCALE,
            asdecimal=True,
        ),
        nullable=True,
    )
    embedding_pricing_catalog_version: Mapped[str | None] = mapped_column(
        String(DOCUMENT_PRICING_CATALOG_VERSION_MAX_LENGTH),
        nullable=True,
    )
    chunk_count: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
    )
    indexed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    last_error_code: Mapped[str | None] = mapped_column(
        String(DOCUMENT_INDEX_ERROR_CODE_MAX_LENGTH),
        nullable=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
    )

    __table_args__ = (
        ForeignKeyConstraint(
            [
                "workspace_id",
                "document_id",
            ],
            [
                "knowledge_documents.workspace_id",
                "knowledge_documents.id",
            ],
            name="fk_knowledge_document_versions_document",
            ondelete="RESTRICT",
        ),
        UniqueConstraint(
            "workspace_id",
            "document_id",
            "id",
            name=("uq_knowledge_document_versions_workspace_document_id"),
        ),
        UniqueConstraint(
            "document_id",
            "version_number",
            name=("uq_knowledge_document_versions_document_version_number"),
        ),
        UniqueConstraint(
            "document_id",
            "content_sha256",
            name=("uq_knowledge_document_versions_document_content_sha256"),
        ),
        CheckConstraint(
            "version_number >= 1",
            name="document_version_number_positive",
        ),
        CheckConstraint(
            (f"media_type IN ({_DOCUMENT_MEDIA_TYPE_SQL_VALUES})"),
            name="document_version_media_type",
        ),
        CheckConstraint(
            "content ~ '[^[:space:]]'",
            name="document_version_content_non_whitespace",
        ),
        CheckConstraint(
            "content_sha256 ~ '^[0-9a-f]{64}$'",
            name="document_version_content_sha256",
        ),
        CheckConstraint(
            (f"status IN ({_DOCUMENT_VERSION_STATUS_SQL_VALUES})"),
            name="document_version_status",
        ),
        CheckConstraint(
            (
                "("
                "chunking_strategy IS NULL "
                "AND chunking_version IS NULL "
                "AND tokenizer_encoding IS NULL "
                "AND embedding_provider IS NULL "
                "AND embedding_model IS NULL "
                "AND embedding_dimensions IS NULL "
                "AND knowledge_collection IS NULL "
                "AND knowledge_vector_name IS NULL"
                ") OR ("
                "chunking_strategy IS NOT NULL "
                "AND chunking_version IS NOT NULL "
                "AND tokenizer_encoding IS NOT NULL "
                "AND embedding_provider IS NOT NULL "
                "AND embedding_model IS NOT NULL "
                "AND embedding_dimensions IS NOT NULL "
                "AND knowledge_collection IS NOT NULL "
                "AND knowledge_vector_name IS NOT NULL"
                ")"
            ),
            name="document_version_index_profile_completeness",
        ),
        CheckConstraint(
            _optional_identifier_sql(
                "chunking_strategy",
                maximum_length=DOCUMENT_INDEX_IDENTIFIER_MAX_LENGTH,
            ),
            name="document_version_chunking_strategy_format",
        ),
        CheckConstraint(
            _optional_identifier_sql(
                "chunking_version",
                maximum_length=DOCUMENT_INDEX_IDENTIFIER_MAX_LENGTH,
            ),
            name="document_version_chunking_version_format",
        ),
        CheckConstraint(
            _optional_identifier_sql(
                "tokenizer_encoding",
                maximum_length=DOCUMENT_INDEX_IDENTIFIER_MAX_LENGTH,
            ),
            name="document_version_tokenizer_encoding_format",
        ),
        CheckConstraint(
            _optional_identifier_sql(
                "embedding_provider",
                maximum_length=DOCUMENT_INDEX_IDENTIFIER_MAX_LENGTH,
            ),
            name="document_version_embedding_provider_format",
        ),
        CheckConstraint(
            _optional_identifier_sql(
                "embedding_model",
                maximum_length=DOCUMENT_INDEX_IDENTIFIER_MAX_LENGTH,
            ),
            name="document_version_embedding_model_format",
        ),
        CheckConstraint(
            _optional_identifier_sql(
                "knowledge_collection",
                maximum_length=DOCUMENT_INDEX_IDENTIFIER_MAX_LENGTH,
            ),
            name="document_version_knowledge_collection_format",
        ),
        CheckConstraint(
            _optional_identifier_sql(
                "knowledge_vector_name",
                maximum_length=DOCUMENT_INDEX_IDENTIFIER_MAX_LENGTH,
            ),
            name="document_version_knowledge_vector_name_format",
        ),
        CheckConstraint(
            ("embedding_dimensions IS NULL OR embedding_dimensions > 0"),
            name="document_version_embedding_dimensions_positive",
        ),
        CheckConstraint(
            ("embedding_input_tokens IS NULL OR embedding_input_tokens >= 0"),
            name="document_version_embedding_tokens_non_negative",
        ),
        CheckConstraint(
            ("embedding_estimated_cost_usd IS NULL OR embedding_estimated_cost_usd >= 0"),
            name="document_version_embedding_cost_non_negative",
        ),
        CheckConstraint(
            _optional_identifier_sql(
                "embedding_pricing_catalog_version",
                maximum_length=(DOCUMENT_PRICING_CATALOG_VERSION_MAX_LENGTH),
            ),
            name="document_version_pricing_catalog_format",
        ),
        CheckConstraint(
            "chunk_count IS NULL OR chunk_count >= 0",
            name="document_version_chunk_count_non_negative",
        ),
        CheckConstraint(
            _optional_identifier_sql(
                "last_error_code",
                maximum_length=DOCUMENT_INDEX_ERROR_CODE_MAX_LENGTH,
            ),
            name="document_version_error_code_format",
        ),
        CheckConstraint(
            (
                "("
                "status = 'pending' "
                "AND indexed_at IS NULL "
                "AND last_error_code IS NULL "
                "AND embedding_input_tokens IS NULL "
                "AND embedding_estimated_cost_usd IS NULL "
                "AND embedding_pricing_catalog_version IS NULL"
                ") OR ("
                "status = 'failed' "
                "AND chunking_strategy IS NOT NULL "
                "AND chunk_count IS NOT NULL "
                "AND indexed_at IS NULL "
                "AND last_error_code IS NOT NULL "
                "AND embedding_input_tokens IS NULL "
                "AND embedding_estimated_cost_usd IS NULL "
                "AND embedding_pricing_catalog_version IS NULL"
                ") OR ("
                "status = 'ready' "
                "AND chunking_strategy IS NOT NULL "
                "AND chunk_count > 0 "
                "AND embedding_input_tokens IS NOT NULL "
                "AND embedding_pricing_catalog_version IS NOT NULL "
                "AND indexed_at IS NOT NULL "
                "AND last_error_code IS NULL"
                ")"
            ),
            name="document_version_status_state",
        ),
        CheckConstraint(
            "updated_at >= created_at",
            name="document_version_timestamp_order",
        ),
        CheckConstraint(
            ("indexed_at IS NULL OR (indexed_at >= created_at AND indexed_at <= updated_at)"),
            name="document_version_indexed_timestamp_order",
        ),
        Index(
            "ix_knowledge_document_versions_workspace_document_number",
            "workspace_id",
            "document_id",
            version_number.desc(),
            id.desc(),
        ),
    )

    @classmethod
    def from_domain(
        cls,
        version: DocumentVersion,
    ) -> "DocumentVersionRecord":
        """Create a persistence record from a document version."""

        return cls(
            id=version.id,
            workspace_id=version.workspace_id,
            document_id=version.document_id,
            version_number=version.version_number,
            media_type=version.media_type.value,
            content=version.content,
            content_sha256=version.content_sha256,
            status=version.status.value,
            chunking_strategy=version.chunking_strategy,
            chunking_version=version.chunking_version,
            tokenizer_encoding=version.tokenizer_encoding,
            embedding_provider=version.embedding_provider,
            embedding_model=version.embedding_model,
            embedding_dimensions=version.embedding_dimensions,
            knowledge_collection=version.knowledge_collection,
            knowledge_vector_name=version.knowledge_vector_name,
            embedding_input_tokens=version.embedding_input_tokens,
            embedding_estimated_cost_usd=(version.embedding_estimated_cost_usd),
            embedding_pricing_catalog_version=(version.embedding_pricing_catalog_version),
            chunk_count=version.chunk_count,
            indexed_at=version.indexed_at,
            last_error_code=version.last_error_code,
            created_at=version.created_at,
            updated_at=version.updated_at,
        )

    def to_domain(self) -> DocumentVersion:
        """Map the persistence record to a document version."""

        return DocumentVersion(
            id=self.id,
            workspace_id=self.workspace_id,
            document_id=self.document_id,
            version_number=self.version_number,
            media_type=DocumentMediaType(self.media_type),
            content=self.content,
            content_sha256=self.content_sha256,
            status=DocumentVersionStatus(self.status),
            chunking_strategy=self.chunking_strategy,
            chunking_version=self.chunking_version,
            tokenizer_encoding=self.tokenizer_encoding,
            embedding_provider=self.embedding_provider,
            embedding_model=self.embedding_model,
            embedding_dimensions=self.embedding_dimensions,
            knowledge_collection=self.knowledge_collection,
            knowledge_vector_name=self.knowledge_vector_name,
            embedding_input_tokens=self.embedding_input_tokens,
            embedding_estimated_cost_usd=(self.embedding_estimated_cost_usd),
            embedding_pricing_catalog_version=(self.embedding_pricing_catalog_version),
            chunk_count=self.chunk_count,
            indexed_at=self.indexed_at,
            last_error_code=self.last_error_code,
            created_at=self.created_at,
            updated_at=self.updated_at,
        )


class DocumentChunkRecord(Base):
    """Persisted authoritative content for one deterministic chunk."""

    __tablename__ = "knowledge_document_chunks"

    id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True),
        primary_key=True,
    )
    workspace_id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True),
        nullable=False,
    )
    document_id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True),
        nullable=False,
    )
    document_version_id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True),
        nullable=False,
    )
    ordinal: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )
    section_path: Mapped[list[str]] = mapped_column(
        JSONB,
        nullable=False,
    )
    content: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )
    content_sha256: Mapped[str] = mapped_column(
        String(_DOCUMENT_CONTENT_HASH_LENGTH),
        nullable=False,
    )
    token_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )
    chunking_strategy: Mapped[str] = mapped_column(
        String(DOCUMENT_INDEX_IDENTIFIER_MAX_LENGTH),
        nullable=False,
    )
    chunking_version: Mapped[str] = mapped_column(
        String(DOCUMENT_INDEX_IDENTIFIER_MAX_LENGTH),
        nullable=False,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
    )

    __table_args__ = (
        ForeignKeyConstraint(
            [
                "workspace_id",
                "document_id",
                "document_version_id",
            ],
            [
                "knowledge_document_versions.workspace_id",
                "knowledge_document_versions.document_id",
                "knowledge_document_versions.id",
            ],
            name="fk_knowledge_document_chunks_version",
            ondelete="RESTRICT",
        ),
        UniqueConstraint(
            "document_version_id",
            "ordinal",
            name=("uq_knowledge_document_chunks_version_ordinal"),
        ),
        CheckConstraint(
            "ordinal >= 0",
            name="document_chunk_ordinal_non_negative",
        ),
        CheckConstraint(
            (
                "jsonb_typeof(section_path) = 'array' "
                f"AND jsonb_array_length(section_path) "
                f"<= {DOCUMENT_CHUNK_SECTION_DEPTH_MAX}"
            ),
            name="document_chunk_section_path",
        ),
        CheckConstraint(
            "content ~ '[^[:space:]]'",
            name="document_chunk_content_non_whitespace",
        ),
        CheckConstraint(
            "content_sha256 ~ '^[0-9a-f]{64}$'",
            name="document_chunk_content_sha256",
        ),
        CheckConstraint(
            "token_count > 0",
            name="document_chunk_token_count_positive",
        ),
        CheckConstraint(
            _required_identifier_sql(
                "chunking_strategy",
                maximum_length=DOCUMENT_INDEX_IDENTIFIER_MAX_LENGTH,
            ),
            name="document_chunk_chunking_strategy_format",
        ),
        CheckConstraint(
            _required_identifier_sql(
                "chunking_version",
                maximum_length=DOCUMENT_INDEX_IDENTIFIER_MAX_LENGTH,
            ),
            name="document_chunk_chunking_version_format",
        ),
        Index(
            "ix_knowledge_document_chunks_workspace_version_id",
            "workspace_id",
            "document_version_id",
            "id",
        ),
    )

    @classmethod
    def from_domain(
        cls,
        chunk: DocumentChunk,
    ) -> "DocumentChunkRecord":
        """Create a persistence record from a document chunk."""

        return cls(
            id=chunk.id,
            workspace_id=chunk.workspace_id,
            document_id=chunk.document_id,
            document_version_id=chunk.document_version_id,
            ordinal=chunk.ordinal,
            section_path=list(chunk.section_path),
            content=chunk.content,
            content_sha256=chunk.content_sha256,
            token_count=chunk.token_count,
            chunking_strategy=chunk.chunking_strategy,
            chunking_version=chunk.chunking_version,
            created_at=chunk.created_at,
        )

    def to_domain(self) -> DocumentChunk:
        """Map the persistence record to a document chunk."""

        return DocumentChunk(
            id=self.id,
            workspace_id=self.workspace_id,
            document_id=self.document_id,
            document_version_id=self.document_version_id,
            ordinal=self.ordinal,
            section_path=tuple(self.section_path),
            content=self.content,
            content_sha256=self.content_sha256,
            token_count=self.token_count,
            chunking_strategy=self.chunking_strategy,
            chunking_version=self.chunking_version,
            created_at=self.created_at,
        )
