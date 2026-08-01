"""Versioned knowledge-document repository contracts."""

from collections.abc import Sequence
from datetime import datetime
from typing import Protocol
from uuid import UUID

from supportops.modules.knowledge_documents.domain.models import (
    Document,
    DocumentChunk,
    DocumentVersion,
)


class DocumentExternalReferenceConflictError(Exception):
    """Raised when an external reference exists in one workspace."""


class DocumentVersionContentConflictError(Exception):
    """Raised when normalized content exists for one document."""


class DocumentVersionNumberConflictError(Exception):
    """Raised when concurrent version allocation violates uniqueness."""


class DocumentChunkConflictError(Exception):
    """Raised when chunk persistence finds incompatible content."""


class DocumentRepository(Protocol):
    """Workspace-scoped persistence operations for documents."""

    async def add(
        self,
        document: Document,
    ) -> None:
        """Add a document to the active transaction."""

        ...

    async def update(
        self,
        document: Document,
    ) -> None:
        """Persist replacement document state in the transaction."""

        ...

    async def get(
        self,
        *,
        workspace_id: UUID,
        document_id: UUID,
    ) -> Document | None:
        """Return a document only through its workspace boundary."""

        ...

    async def get_for_update(
        self,
        *,
        workspace_id: UUID,
        document_id: UUID,
    ) -> Document | None:
        """Return and lock a document for a short transaction."""

        ...

    async def list(
        self,
        workspace_id: UUID,
        *,
        limit: int,
        after_created_at: datetime | None = None,
        after_document_id: UUID | None = None,
    ) -> Sequence[Document]:
        """List documents in deterministic descending order."""

        ...


class DocumentVersionRepository(Protocol):
    """Persistence operations for immutable document versions."""

    async def add(
        self,
        version: DocumentVersion,
    ) -> None:
        """Add a document version to the active transaction."""

        ...

    async def update(
        self,
        version: DocumentVersion,
    ) -> None:
        """Persist lifecycle metadata in the active transaction."""

        ...

    async def get(
        self,
        *,
        workspace_id: UUID,
        document_id: UUID,
        document_version_id: UUID,
    ) -> DocumentVersion | None:
        """Return one version through workspace and document ownership."""

        ...

    async def get_for_update(
        self,
        *,
        workspace_id: UUID,
        document_id: UUID,
        document_version_id: UUID,
    ) -> DocumentVersion | None:
        """Return and lock one owned version."""

        ...

    async def get_by_content_hash(
        self,
        *,
        workspace_id: UUID,
        document_id: UUID,
        content_sha256: str,
    ) -> DocumentVersion | None:
        """Return a version with the same normalized content hash."""

        ...

    async def next_version_number(
        self,
        *,
        workspace_id: UUID,
        document_id: UUID,
    ) -> int:
        """Return the next number while the document row is locked."""

        ...

    async def list(
        self,
        *,
        workspace_id: UUID,
        document_id: UUID,
        limit: int,
        after_version_number: int | None = None,
        after_document_version_id: UUID | None = None,
    ) -> Sequence[DocumentVersion]:
        """List owned versions in deterministic descending order."""

        ...


class DocumentChunkRepository(Protocol):
    """Persistence operations for authoritative immutable chunks."""

    async def add_many(
        self,
        chunks: Sequence[DocumentChunk],
    ) -> None:
        """Add deterministic chunks to the active transaction."""

        ...

    async def list_by_version(
        self,
        *,
        workspace_id: UUID,
        document_id: UUID,
        document_version_id: UUID,
    ) -> Sequence[DocumentChunk]:
        """Return chunks ordered by ordinal ascending."""

        ...

    async def count_by_version(
        self,
        *,
        workspace_id: UUID,
        document_id: UUID,
        document_version_id: UUID,
    ) -> int:
        """Return the authoritative chunk count for one version."""

        ...
