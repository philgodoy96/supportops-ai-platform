"""Workspace-scoped versioned knowledge-document use cases."""

from collections.abc import Sequence
from datetime import datetime
from uuid import UUID

from supportops.core.transactions import TransactionManager
from supportops.modules.knowledge_documents.application.errors import (
    DocumentExternalReferenceConflictApplicationError,
    DocumentNotFoundError,
    DocumentVersionContentConflictApplicationError,
    DocumentVersionNotFoundError,
    DocumentVersionNotReadyError,
    DocumentVersionNumberConflictApplicationError,
)
from supportops.modules.knowledge_documents.application.results import (
    CreateDocumentResult,
)
from supportops.modules.knowledge_documents.domain.content import (
    compute_content_sha256,
    normalize_document_content,
)
from supportops.modules.knowledge_documents.domain.models import (
    Document,
    DocumentMediaType,
    DocumentVersion,
    DocumentVersionStatus,
)
from supportops.modules.knowledge_documents.domain.repositories import (
    DocumentExternalReferenceConflictError,
    DocumentRepository,
    DocumentVersionContentConflictError,
    DocumentVersionNumberConflictError,
    DocumentVersionRepository,
)
from supportops.modules.workspaces.application.errors import (
    WorkspaceNotFoundError,
)
from supportops.modules.workspaces.domain.repositories import (
    WorkspaceRepository,
)


class CreateDocument:
    """Create a document and its first immutable version atomically."""

    def __init__(
        self,
        *,
        workspace_repository: WorkspaceRepository,
        document_repository: DocumentRepository,
        version_repository: DocumentVersionRepository,
        transaction_manager: TransactionManager,
    ) -> None:
        self._workspace_repository = workspace_repository
        self._document_repository = document_repository
        self._version_repository = version_repository
        self._transaction_manager = transaction_manager

    async def execute(
        self,
        *,
        workspace_id: UUID,
        title: str,
        media_type: DocumentMediaType,
        content: str,
        external_reference: str | None = None,
    ) -> CreateDocumentResult:
        """Persist one document and version one without external calls."""

        document = Document.create(
            workspace_id=workspace_id,
            title=title,
            external_reference=external_reference,
        )
        version = DocumentVersion.create_pending(
            workspace_id=workspace_id,
            document_id=document.id,
            version_number=1,
            media_type=media_type,
            content=content,
            now=document.created_at,
        )

        try:
            async with self._transaction_manager.transaction():
                if not await self._workspace_repository.exists(workspace_id):
                    raise WorkspaceNotFoundError("Workspace was not found.")

                await self._document_repository.add(document)
                await self._version_repository.add(version)
        except DocumentExternalReferenceConflictError as error:
            raise DocumentExternalReferenceConflictApplicationError(
                "Document external reference already exists in the workspace."
            ) from error
        except DocumentVersionContentConflictError as error:
            raise DocumentVersionContentConflictApplicationError(
                "Document content already exists for this document."
            ) from error
        except DocumentVersionNumberConflictError as error:
            raise DocumentVersionNumberConflictApplicationError(
                "Document version number already exists."
            ) from error

        return CreateDocumentResult(
            document=document,
            version=version,
        )


class GetDocument:
    """Retrieve a document through its workspace boundary."""

    def __init__(
        self,
        *,
        repository: DocumentRepository,
    ) -> None:
        self._repository = repository

    async def execute(
        self,
        *,
        workspace_id: UUID,
        document_id: UUID,
    ) -> Document:
        """Return a scoped document or raise a stable not-found error."""

        document = await self._repository.get(
            workspace_id=workspace_id,
            document_id=document_id,
        )
        if document is None:
            raise DocumentNotFoundError("Document was not found.")

        return document


class ListDocuments:
    """List documents belonging to one workspace."""

    def __init__(
        self,
        *,
        workspace_repository: WorkspaceRepository,
        document_repository: DocumentRepository,
    ) -> None:
        self._workspace_repository = workspace_repository
        self._document_repository = document_repository

    async def execute(
        self,
        *,
        workspace_id: UUID,
        limit: int,
        after_created_at: datetime | None = None,
        after_document_id: UUID | None = None,
    ) -> Sequence[Document]:
        """Return one deterministic workspace-scoped page."""

        if not await self._workspace_repository.exists(workspace_id):
            raise WorkspaceNotFoundError("Workspace was not found.")

        return await self._document_repository.list(
            workspace_id,
            limit=limit,
            after_created_at=after_created_at,
            after_document_id=after_document_id,
        )


class CreateDocumentVersion:
    """Create the next immutable version under a document row lock."""

    def __init__(
        self,
        *,
        document_repository: DocumentRepository,
        version_repository: DocumentVersionRepository,
        transaction_manager: TransactionManager,
    ) -> None:
        self._document_repository = document_repository
        self._version_repository = version_repository
        self._transaction_manager = transaction_manager

    async def execute(
        self,
        *,
        workspace_id: UUID,
        document_id: UUID,
        media_type: DocumentMediaType,
        content: str,
    ) -> DocumentVersion:
        """Create the next pending version atomically."""

        normalized_content = normalize_document_content(content)
        content_sha256 = compute_content_sha256(normalized_content)

        try:
            async with self._transaction_manager.transaction():
                document = await self._document_repository.get_for_update(
                    workspace_id=workspace_id,
                    document_id=document_id,
                )
                if document is None:
                    raise DocumentNotFoundError("Document was not found.")

                existing = await self._version_repository.get_by_content_hash(
                    workspace_id=workspace_id,
                    document_id=document_id,
                    content_sha256=content_sha256,
                )
                if existing is not None:
                    raise (
                        DocumentVersionContentConflictApplicationError(
                            "Document content already exists for this document."
                        )
                    )

                version_number = await self._version_repository.next_version_number(
                    workspace_id=workspace_id,
                    document_id=document_id,
                )
                version = DocumentVersion.create_pending(
                    workspace_id=workspace_id,
                    document_id=document_id,
                    version_number=version_number,
                    media_type=media_type,
                    content=normalized_content,
                )
                await self._version_repository.add(version)
        except DocumentVersionContentConflictError as error:
            raise DocumentVersionContentConflictApplicationError(
                "Document content already exists for this document."
            ) from error
        except DocumentVersionNumberConflictError as error:
            raise DocumentVersionNumberConflictApplicationError(
                "Document version number already exists."
            ) from error

        return version


class GetDocumentVersion:
    """Retrieve one version through document and workspace ownership."""

    def __init__(
        self,
        *,
        document_repository: DocumentRepository,
        version_repository: DocumentVersionRepository,
    ) -> None:
        self._document_repository = document_repository
        self._version_repository = version_repository

    async def execute(
        self,
        *,
        workspace_id: UUID,
        document_id: UUID,
        document_version_id: UUID,
    ) -> DocumentVersion:
        """Return a scoped version or raise a stable not-found error."""

        document = await self._document_repository.get(
            workspace_id=workspace_id,
            document_id=document_id,
        )
        if document is None:
            raise DocumentNotFoundError("Document was not found.")

        version = await self._version_repository.get(
            workspace_id=workspace_id,
            document_id=document_id,
            document_version_id=document_version_id,
        )
        if version is None:
            raise DocumentVersionNotFoundError("Document version was not found.")

        return version


class ListDocumentVersions:
    """List immutable versions belonging to one scoped document."""

    def __init__(
        self,
        *,
        document_repository: DocumentRepository,
        version_repository: DocumentVersionRepository,
    ) -> None:
        self._document_repository = document_repository
        self._version_repository = version_repository

    async def execute(
        self,
        *,
        workspace_id: UUID,
        document_id: UUID,
        limit: int,
        after_version_number: int | None = None,
        after_document_version_id: UUID | None = None,
    ) -> Sequence[DocumentVersion]:
        """Return one deterministic page of scoped versions."""

        document = await self._document_repository.get(
            workspace_id=workspace_id,
            document_id=document_id,
        )
        if document is None:
            raise DocumentNotFoundError("Document was not found.")

        return await self._version_repository.list(
            workspace_id=workspace_id,
            document_id=document_id,
            limit=limit,
            after_version_number=after_version_number,
            after_document_version_id=after_document_version_id,
        )


class ActivateDocumentVersion:
    """Point a document at an explicitly selected ready version."""

    def __init__(
        self,
        *,
        document_repository: DocumentRepository,
        version_repository: DocumentVersionRepository,
        transaction_manager: TransactionManager,
    ) -> None:
        self._document_repository = document_repository
        self._version_repository = version_repository
        self._transaction_manager = transaction_manager

    async def execute(
        self,
        *,
        workspace_id: UUID,
        document_id: UUID,
        document_version_id: UUID,
    ) -> Document:
        """Activate a ready owned version transactionally."""

        async with self._transaction_manager.transaction():
            document = await self._document_repository.get_for_update(
                workspace_id=workspace_id,
                document_id=document_id,
            )
            if document is None:
                raise DocumentNotFoundError("Document was not found.")

            version = await self._version_repository.get_for_update(
                workspace_id=workspace_id,
                document_id=document_id,
                document_version_id=document_version_id,
            )
            if version is None:
                raise DocumentVersionNotFoundError("Document version was not found.")
            if version.status is not DocumentVersionStatus.READY:
                raise DocumentVersionNotReadyError("Document version is not ready for activation.")

            activated = document.activate_version(version)
            if activated is not document:
                await self._document_repository.update(activated)

        return activated
