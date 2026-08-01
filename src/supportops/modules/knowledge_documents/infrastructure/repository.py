"""PostgreSQL repositories for versioned knowledge documents."""

from collections.abc import Sequence
from datetime import datetime
from uuid import UUID

from sqlalchemy import and_, func, literal, or_, select, tuple_
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from supportops.infrastructure.postgresql.errors import get_constraint_name
from supportops.modules.knowledge_documents.domain.models import (
    Document,
    DocumentChunk,
    DocumentVersion,
)
from supportops.modules.knowledge_documents.domain.repositories import (
    DocumentChunkConflictError,
    DocumentChunkRepository,
    DocumentExternalReferenceConflictError,
    DocumentRepository,
    DocumentVersionContentConflictError,
    DocumentVersionNumberConflictError,
    DocumentVersionRepository,
)
from supportops.modules.knowledge_documents.infrastructure.models import (
    DocumentChunkRecord,
    DocumentRecord,
    DocumentVersionRecord,
)

_DOCUMENT_EXTERNAL_REFERENCE_CONSTRAINT = "uq_knowledge_documents_workspace_external_reference"
_DOCUMENT_VERSION_CONTENT_CONSTRAINT = "uq_knowledge_document_versions_document_content_sha256"
_DOCUMENT_VERSION_NUMBER_CONSTRAINT = "uq_knowledge_document_versions_document_version_number"
_DOCUMENT_CHUNK_PRIMARY_KEY_CONSTRAINT = "pk_knowledge_document_chunks"
_DOCUMENT_CHUNK_ORDINAL_CONSTRAINT = "uq_knowledge_document_chunks_version_ordinal"


class SqlAlchemyDocumentRepository(DocumentRepository):
    """Persist workspace-scoped knowledge-document identities."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def add(self, document: Document) -> None:
        """Add and flush a document inside the active transaction."""

        self._session.add(DocumentRecord.from_domain(document))
        try:
            await self._session.flush()
        except IntegrityError as error:
            self._translate_external_reference_conflict(error)
            raise

    async def update(self, document: Document) -> None:
        """Persist document metadata and its active-version pointer."""

        record = await self._load_record(
            workspace_id=document.workspace_id,
            document_id=document.id,
            for_update=True,
        )
        if record is None:
            raise LookupError("Knowledge document does not exist.")
        if record.created_at != document.created_at:
            raise ValueError("Document created_at is immutable.")

        record.title = document.title
        record.external_reference = document.external_reference
        record.active_version_id = document.active_version_id
        record.updated_at = document.updated_at

        try:
            await self._session.flush()
        except IntegrityError as error:
            self._translate_external_reference_conflict(error)
            raise

    async def get(
        self,
        *,
        workspace_id: UUID,
        document_id: UUID,
    ) -> Document | None:
        """Return a document only through its workspace boundary."""

        record = await self._load_record(
            workspace_id=workspace_id,
            document_id=document_id,
            for_update=False,
        )
        return None if record is None else record.to_domain()

    async def get_for_update(
        self,
        *,
        workspace_id: UUID,
        document_id: UUID,
    ) -> Document | None:
        """Return and lock a document for a short transaction."""

        record = await self._load_record(
            workspace_id=workspace_id,
            document_id=document_id,
            for_update=True,
        )
        return None if record is None else record.to_domain()

    async def list(
        self,
        workspace_id: UUID,
        *,
        limit: int,
        after_created_at: datetime | None = None,
        after_document_id: UUID | None = None,
    ) -> list[Document]:
        """List documents in deterministic descending order."""

        if limit < 1:
            raise ValueError("Document list limit must be positive.")
        if (after_created_at is None) != (after_document_id is None):
            raise ValueError("Document pagination position requires both timestamp and ID.")

        statement = select(DocumentRecord).where(DocumentRecord.workspace_id == workspace_id)
        if after_created_at is not None:
            assert after_document_id is not None
            statement = statement.where(
                tuple_(
                    DocumentRecord.created_at,
                    DocumentRecord.id,
                )
                < tuple_(
                    literal(after_created_at),
                    literal(after_document_id),
                )
            )

        statement = statement.order_by(
            DocumentRecord.created_at.desc(),
            DocumentRecord.id.desc(),
        ).limit(limit)
        result = await self._session.execute(statement)
        return [record.to_domain() for record in result.scalars().all()]

    async def _load_record(
        self,
        *,
        workspace_id: UUID,
        document_id: UUID,
        for_update: bool,
    ) -> DocumentRecord | None:
        statement = select(DocumentRecord).where(
            DocumentRecord.workspace_id == workspace_id,
            DocumentRecord.id == document_id,
        )
        if for_update:
            statement = statement.with_for_update()

        result = await self._session.execute(statement)
        return result.scalar_one_or_none()

    @staticmethod
    def _translate_external_reference_conflict(
        error: IntegrityError,
    ) -> None:
        if get_constraint_name(error) == _DOCUMENT_EXTERNAL_REFERENCE_CONSTRAINT:
            raise DocumentExternalReferenceConflictError(
                "Document external reference already exists in the workspace."
            ) from error


class SqlAlchemyDocumentVersionRepository(DocumentVersionRepository):
    """Persist immutable source versions and mutable indexing state."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def add(self, version: DocumentVersion) -> None:
        """Add and flush a document version in the active transaction."""

        self._session.add(DocumentVersionRecord.from_domain(version))
        try:
            await self._session.flush()
        except IntegrityError as error:
            constraint_name = get_constraint_name(error)
            if constraint_name == _DOCUMENT_VERSION_CONTENT_CONSTRAINT:
                raise DocumentVersionContentConflictError(
                    "Document content already exists for this document."
                ) from error
            if constraint_name == _DOCUMENT_VERSION_NUMBER_CONSTRAINT:
                raise DocumentVersionNumberConflictError(
                    "Document version number already exists."
                ) from error
            raise

    async def update(self, version: DocumentVersion) -> None:
        """Persist indexing lifecycle state without rewriting source data."""

        record = await self._load_record(
            workspace_id=version.workspace_id,
            document_id=version.document_id,
            document_version_id=version.id,
            for_update=True,
        )
        if record is None:
            raise LookupError("Knowledge document version does not exist.")

        persisted = record.to_domain()
        if (
            persisted.version_number != version.version_number
            or persisted.media_type is not version.media_type
            or persisted.content != version.content
            or persisted.content_sha256 != version.content_sha256
            or persisted.created_at != version.created_at
        ):
            raise ValueError("Immutable document version fields do not match persisted state.")
        if persisted.index_profile is not None and persisted.index_profile != version.index_profile:
            raise ValueError("Persisted document version index profile is immutable.")

        record.status = version.status.value
        record.chunking_strategy = version.chunking_strategy
        record.chunking_version = version.chunking_version
        record.tokenizer_encoding = version.tokenizer_encoding
        record.embedding_provider = version.embedding_provider
        record.embedding_model = version.embedding_model
        record.embedding_dimensions = version.embedding_dimensions
        record.knowledge_collection = version.knowledge_collection
        record.knowledge_vector_name = version.knowledge_vector_name
        record.embedding_input_tokens = version.embedding_input_tokens
        record.embedding_estimated_cost_usd = version.embedding_estimated_cost_usd
        record.embedding_pricing_catalog_version = version.embedding_pricing_catalog_version
        record.chunk_count = version.chunk_count
        record.indexed_at = version.indexed_at
        record.last_error_code = version.last_error_code
        record.updated_at = version.updated_at
        await self._session.flush()

    async def get(
        self,
        *,
        workspace_id: UUID,
        document_id: UUID,
        document_version_id: UUID,
    ) -> DocumentVersion | None:
        """Return one version through workspace and document ownership."""

        record = await self._load_record(
            workspace_id=workspace_id,
            document_id=document_id,
            document_version_id=document_version_id,
            for_update=False,
        )
        return None if record is None else record.to_domain()

    async def get_for_update(
        self,
        *,
        workspace_id: UUID,
        document_id: UUID,
        document_version_id: UUID,
    ) -> DocumentVersion | None:
        """Return and lock one owned version."""

        record = await self._load_record(
            workspace_id=workspace_id,
            document_id=document_id,
            document_version_id=document_version_id,
            for_update=True,
        )
        return None if record is None else record.to_domain()

    async def get_by_content_hash(
        self,
        *,
        workspace_id: UUID,
        document_id: UUID,
        content_sha256: str,
    ) -> DocumentVersion | None:
        """Return a version with the same normalized content hash."""

        statement = select(DocumentVersionRecord).where(
            DocumentVersionRecord.workspace_id == workspace_id,
            DocumentVersionRecord.document_id == document_id,
            DocumentVersionRecord.content_sha256 == content_sha256,
        )
        result = await self._session.execute(statement)
        record = result.scalar_one_or_none()
        return None if record is None else record.to_domain()

    async def next_version_number(
        self,
        *,
        workspace_id: UUID,
        document_id: UUID,
    ) -> int:
        """Return the next number while the owning document is locked."""

        statement = select(
            func.coalesce(
                func.max(DocumentVersionRecord.version_number),
                0,
            )
            + 1
        ).where(
            DocumentVersionRecord.workspace_id == workspace_id,
            DocumentVersionRecord.document_id == document_id,
        )
        result = await self._session.execute(statement)
        return int(result.scalar_one())

    async def list(
        self,
        *,
        workspace_id: UUID,
        document_id: UUID,
        limit: int,
        after_version_number: int | None = None,
        after_document_version_id: UUID | None = None,
    ) -> list[DocumentVersion]:
        """List owned versions in deterministic descending order."""

        if limit < 1:
            raise ValueError("Document version list limit must be positive.")
        if (after_version_number is None) != (after_document_version_id is None):
            raise ValueError(
                "Document version pagination position requires both version number and ID."
            )
        if after_version_number is not None and after_version_number < 1:
            raise ValueError("Document version pagination number must be positive.")

        statement = select(DocumentVersionRecord).where(
            DocumentVersionRecord.workspace_id == workspace_id,
            DocumentVersionRecord.document_id == document_id,
        )
        if after_version_number is not None:
            assert after_document_version_id is not None
            statement = statement.where(
                tuple_(
                    DocumentVersionRecord.version_number,
                    DocumentVersionRecord.id,
                )
                < tuple_(
                    literal(after_version_number),
                    literal(after_document_version_id),
                )
            )

        statement = statement.order_by(
            DocumentVersionRecord.version_number.desc(),
            DocumentVersionRecord.id.desc(),
        ).limit(limit)
        result = await self._session.execute(statement)
        return [record.to_domain() for record in result.scalars().all()]

    async def _load_record(
        self,
        *,
        workspace_id: UUID,
        document_id: UUID,
        document_version_id: UUID,
        for_update: bool,
    ) -> DocumentVersionRecord | None:
        statement = select(DocumentVersionRecord).where(
            DocumentVersionRecord.workspace_id == workspace_id,
            DocumentVersionRecord.document_id == document_id,
            DocumentVersionRecord.id == document_version_id,
        )
        if for_update:
            statement = statement.with_for_update()

        result = await self._session.execute(statement)
        return result.scalar_one_or_none()


class SqlAlchemyDocumentChunkRepository(DocumentChunkRepository):
    """Persist and verify deterministic authoritative chunks."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def add_many(
        self,
        chunks: Sequence[DocumentChunk],
    ) -> None:
        """Persist missing chunks or verify an identical safe rerun."""

        if not chunks:
            return

        first = chunks[0]
        chunks_by_id = {chunk.id: chunk for chunk in chunks}
        chunks_by_ordinal = {chunk.ordinal: chunk for chunk in chunks}
        if len(chunks_by_id) != len(chunks):
            raise DocumentChunkConflictError("Chunk batch contains duplicate identifiers.")
        if len(chunks_by_ordinal) != len(chunks):
            raise DocumentChunkConflictError("Chunk batch contains duplicate ordinals.")

        for chunk in chunks:
            if (
                chunk.workspace_id != first.workspace_id
                or chunk.document_id != first.document_id
                or chunk.document_version_id != first.document_version_id
                or chunk.chunking_strategy != first.chunking_strategy
                or chunk.chunking_version != first.chunking_version
            ):
                raise ValueError("Chunk batch must belong to one version and profile.")

        statement = select(DocumentChunkRecord).where(
            or_(
                DocumentChunkRecord.id.in_(tuple(chunks_by_id)),
                and_(
                    DocumentChunkRecord.document_version_id == first.document_version_id,
                    DocumentChunkRecord.ordinal.in_(tuple(chunks_by_ordinal)),
                ),
            )
        )
        result = await self._session.execute(statement)
        existing_ids: set[UUID] = set()

        for record in result.scalars().all():
            candidate_by_id = chunks_by_id.get(record.id)
            candidate_by_ordinal = (
                chunks_by_ordinal.get(record.ordinal)
                if record.document_version_id == first.document_version_id
                else None
            )
            if (
                candidate_by_id is not None
                and candidate_by_ordinal is not None
                and candidate_by_id.id != candidate_by_ordinal.id
            ):
                raise DocumentChunkConflictError(
                    "Persisted chunk identity conflicts with its ordinal."
                )

            candidate = candidate_by_id or candidate_by_ordinal
            if candidate is None or record.to_domain() != candidate:
                raise DocumentChunkConflictError(
                    "Persisted chunk does not match the deterministic rerun."
                )
            existing_ids.add(candidate.id)

        missing_records = [
            DocumentChunkRecord.from_domain(chunk)
            for chunk in chunks
            if chunk.id not in existing_ids
        ]
        if not missing_records:
            return

        self._session.add_all(missing_records)
        try:
            await self._session.flush()
        except IntegrityError as error:
            if get_constraint_name(error) in {
                _DOCUMENT_CHUNK_PRIMARY_KEY_CONSTRAINT,
                _DOCUMENT_CHUNK_ORDINAL_CONSTRAINT,
            }:
                raise DocumentChunkConflictError(
                    "Persisted chunk conflicts with existing deterministic chunk state."
                ) from error
            raise

    async def list_by_version(
        self,
        *,
        workspace_id: UUID,
        document_id: UUID,
        document_version_id: UUID,
    ) -> list[DocumentChunk]:
        """Return authoritative chunks ordered by ordinal."""

        statement = (
            select(DocumentChunkRecord)
            .where(
                DocumentChunkRecord.workspace_id == workspace_id,
                DocumentChunkRecord.document_id == document_id,
                DocumentChunkRecord.document_version_id == document_version_id,
            )
            .order_by(DocumentChunkRecord.ordinal.asc())
        )
        result = await self._session.execute(statement)
        return [record.to_domain() for record in result.scalars().all()]

    async def count_by_version(
        self,
        *,
        workspace_id: UUID,
        document_id: UUID,
        document_version_id: UUID,
    ) -> int:
        """Return the authoritative chunk count for one version."""

        statement = select(func.count(DocumentChunkRecord.id)).where(
            DocumentChunkRecord.workspace_id == workspace_id,
            DocumentChunkRecord.document_id == document_id,
            DocumentChunkRecord.document_version_id == document_version_id,
        )
        result = await self._session.execute(statement)
        return int(result.scalar_one())
