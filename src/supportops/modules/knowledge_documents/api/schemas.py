"""HTTP schemas for versioned knowledge documents."""

from datetime import datetime
from decimal import Decimal
from typing import Self
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator

from supportops.modules.knowledge_documents.application.results import (
    CreateDocumentResult,
)
from supportops.modules.knowledge_documents.domain.content import (
    normalize_document_content,
)
from supportops.modules.knowledge_documents.domain.models import (
    DOCUMENT_EXTERNAL_REFERENCE_MAX_LENGTH,
    DOCUMENT_TITLE_MAX_LENGTH,
    Document,
    DocumentMediaType,
    DocumentVersion,
)


class DocumentCreateRequest(BaseModel):
    """Payload accepted when creating a document and version one."""

    model_config = ConfigDict(extra="forbid")

    title: str = Field(
        min_length=1,
        max_length=DOCUMENT_TITLE_MAX_LENGTH,
    )
    external_reference: str | None = Field(
        default=None,
        min_length=1,
        max_length=DOCUMENT_EXTERNAL_REFERENCE_MAX_LENGTH,
    )
    media_type: DocumentMediaType
    content: str = Field(min_length=1)

    @field_validator("title", mode="before")
    @classmethod
    def normalize_title(cls, value: object) -> object:
        """Trim surrounding whitespace from the human-authored title."""

        if isinstance(value, str):
            return value.strip()
        return value

    @field_validator("external_reference")
    @classmethod
    def validate_external_reference(
        cls,
        value: str | None,
    ) -> str | None:
        """Reject noncanonical upstream references."""

        if value is not None and value != value.strip():
            raise ValueError("External reference must not contain surrounding whitespace.")
        return value

    @field_validator("content")
    @classmethod
    def validate_content(cls, value: str) -> str:
        """Reject source content that normalizes to no meaningful text."""

        normalize_document_content(value)
        return value


class DocumentVersionCreateRequest(BaseModel):
    """Payload accepted when adding an immutable document version."""

    model_config = ConfigDict(extra="forbid")

    media_type: DocumentMediaType
    content: str = Field(min_length=1)

    @field_validator("content")
    @classmethod
    def validate_content(cls, value: str) -> str:
        """Reject source content that normalizes to no meaningful text."""

        normalize_document_content(value)
        return value


class DocumentResponse(BaseModel):
    """Stable document metadata without source content."""

    id: UUID
    workspace_id: UUID
    title: str
    external_reference: str | None
    active_version_id: UUID | None
    created_at: datetime
    updated_at: datetime

    @classmethod
    def from_domain(cls, document: Document) -> Self:
        """Create an API representation from a document entity."""

        return cls(
            id=document.id,
            workspace_id=document.workspace_id,
            title=document.title,
            external_reference=document.external_reference,
            active_version_id=document.active_version_id,
            created_at=document.created_at,
            updated_at=document.updated_at,
        )


class DocumentVersionResponse(BaseModel):
    """Version metadata without authoritative source content."""

    id: UUID
    workspace_id: UUID
    document_id: UUID
    version_number: int
    media_type: str
    content_sha256: str
    status: str
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

    @classmethod
    def from_domain(cls, version: DocumentVersion) -> Self:
        """Create a metadata response from a version entity."""

        return cls(
            id=version.id,
            workspace_id=version.workspace_id,
            document_id=version.document_id,
            version_number=version.version_number,
            media_type=version.media_type.value,
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


class DocumentVersionDetailResponse(DocumentVersionResponse):
    """Version metadata plus authoritative normalized source content."""

    content: str

    @classmethod
    def from_domain(cls, version: DocumentVersion) -> Self:
        """Create a version-detail response from a domain entity."""

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


class DocumentCreateResponse(BaseModel):
    """Created document and its atomically created first version."""

    document: DocumentResponse
    version: DocumentVersionResponse

    @classmethod
    def from_result(
        cls,
        result: CreateDocumentResult,
    ) -> Self:
        """Create an HTTP response from the application result."""

        return cls(
            document=DocumentResponse.from_domain(result.document),
            version=DocumentVersionResponse.from_domain(result.version),
        )


class DocumentListResponse(BaseModel):
    """One bounded page of workspace documents."""

    items: list[DocumentResponse]
    next_cursor: str | None


class DocumentVersionListResponse(BaseModel):
    """One bounded page of immutable document versions."""

    items: list[DocumentVersionResponse]
    next_cursor: str | None
