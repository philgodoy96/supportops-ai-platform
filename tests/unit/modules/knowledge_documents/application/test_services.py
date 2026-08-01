"""Unit tests for versioned knowledge-document application services."""

from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from uuid import UUID

import pytest

from supportops.modules.knowledge_documents.application.errors import (
    DocumentExternalReferenceConflictApplicationError,
    DocumentNotFoundError,
    DocumentVersionContentConflictApplicationError,
    DocumentVersionNotFoundError,
    DocumentVersionNotReadyError,
    DocumentVersionNumberConflictApplicationError,
)
from supportops.modules.knowledge_documents.application.services import (
    ActivateDocumentVersion,
    CreateDocument,
    CreateDocumentVersion,
    GetDocument,
    GetDocumentVersion,
    ListDocuments,
    ListDocumentVersions,
)
from supportops.modules.knowledge_documents.domain.models import (
    Document,
    DocumentMediaType,
    DocumentVersion,
    KnowledgeIndexProfile,
)
from supportops.modules.knowledge_documents.domain.repositories import (
    DocumentExternalReferenceConflictError,
    DocumentVersionContentConflictError,
    DocumentVersionNumberConflictError,
)
from supportops.modules.workspaces.application.errors import (
    WorkspaceNotFoundError,
)
from supportops.modules.workspaces.domain.models import Workspace

_WORKSPACE_ID = UUID("032c8c87-57cc-4d14-bfbd-04968b4e8cd4")
_OTHER_WORKSPACE_ID = UUID("4aefba3b-b57e-47d1-889e-bb28762fa1ed")
_DOCUMENT_ID = UUID("276046a2-28ec-4cb1-8bb6-a2ff70f9064b")
_VERSION_ID = UUID("09036916-84cf-4a58-bdf4-09bc52716ec5")
_OTHER_VERSION_ID = UUID("8ac21e0c-f869-4d84-a9e1-37d8074c9e54")
_TIMESTAMP = datetime(
    2026,
    8,
    1,
    22,
    0,
    tzinfo=UTC,
)


class FakeTransactionManager:
    """Record transaction completion and rollback."""

    def __init__(self) -> None:
        self.entered = False
        self.completed = False
        self.rolled_back = False

    @asynccontextmanager
    async def transaction(self) -> AsyncIterator[None]:
        """Provide one observable fake transaction."""

        self.entered = True
        try:
            yield
        except Exception:
            self.rolled_back = True
            raise
        else:
            self.completed = True


class FakeWorkspaceRepository:
    """In-memory workspace repository fake."""

    def __init__(
        self,
        *,
        workspace_exists: bool = True,
    ) -> None:
        self.workspace_exists = workspace_exists
        self.requested_workspace_id: UUID | None = None

    async def add(self, workspace: Workspace) -> None:
        """Reject unsupported writes in these service tests."""

        raise AssertionError("add must not be called")

    async def get(
        self,
        workspace_id: UUID,
    ) -> Workspace | None:
        """Reject unsupported reads in these service tests."""

        raise AssertionError("get must not be called")

    async def exists(self, workspace_id: UUID) -> bool:
        """Return the configured workspace existence result."""

        self.requested_workspace_id = workspace_id
        return self.workspace_exists


class FakeDocumentRepository:
    """In-memory workspace-scoped document repository fake."""

    def __init__(self) -> None:
        self.documents: dict[tuple[UUID, UUID], Document] = {}
        self.added_document: Document | None = None
        self.updated_document: Document | None = None
        self.external_reference_conflict = False
        self.list_result: Sequence[Document] = ()
        self.list_arguments: (
            tuple[
                UUID,
                int,
                datetime | None,
                UUID | None,
            ]
            | None
        ) = None
        self.lock_arguments: tuple[UUID, UUID] | None = None

    async def add(self, document: Document) -> None:
        """Add a document or raise the configured conflict."""

        if self.external_reference_conflict:
            raise DocumentExternalReferenceConflictError("duplicate external reference")

        self.added_document = document
        self.documents[(document.workspace_id, document.id)] = document

    async def update(self, document: Document) -> None:
        """Replace one stored document."""

        self.updated_document = document
        self.documents[(document.workspace_id, document.id)] = document

    async def get(
        self,
        *,
        workspace_id: UUID,
        document_id: UUID,
    ) -> Document | None:
        """Return one scoped document."""

        return self.documents.get((workspace_id, document_id))

    async def get_for_update(
        self,
        *,
        workspace_id: UUID,
        document_id: UUID,
    ) -> Document | None:
        """Return one scoped document and record the lock request."""

        self.lock_arguments = (
            workspace_id,
            document_id,
        )
        return self.documents.get((workspace_id, document_id))

    async def list(
        self,
        workspace_id: UUID,
        *,
        limit: int,
        after_created_at: datetime | None = None,
        after_document_id: UUID | None = None,
    ) -> Sequence[Document]:
        """Return the configured deterministic page."""

        self.list_arguments = (
            workspace_id,
            limit,
            after_created_at,
            after_document_id,
        )
        return self.list_result


class FakeDocumentVersionRepository:
    """In-memory document-version repository fake."""

    def __init__(self) -> None:
        self.versions: dict[
            tuple[UUID, UUID, UUID],
            DocumentVersion,
        ] = {}
        self.added_version: DocumentVersion | None = None
        self.content_conflict = False
        self.number_conflict = False
        self.next_number = 1
        self.list_result: Sequence[DocumentVersion] = ()
        self.list_arguments: (
            tuple[
                UUID,
                UUID,
                int,
                int | None,
                UUID | None,
            ]
            | None
        ) = None
        self.content_hash_arguments: tuple[UUID, UUID, str] | None = None
        self.lock_arguments: tuple[UUID, UUID, UUID] | None = None

    async def add(
        self,
        version: DocumentVersion,
    ) -> None:
        """Add a version or raise one configured conflict."""

        if self.content_conflict:
            raise DocumentVersionContentConflictError("duplicate content")
        if self.number_conflict:
            raise DocumentVersionNumberConflictError("duplicate version number")

        self.added_version = version
        self.versions[
            (
                version.workspace_id,
                version.document_id,
                version.id,
            )
        ] = version

    async def update(
        self,
        version: DocumentVersion,
    ) -> None:
        """Replace one stored version."""

        self.versions[
            (
                version.workspace_id,
                version.document_id,
                version.id,
            )
        ] = version

    async def get(
        self,
        *,
        workspace_id: UUID,
        document_id: UUID,
        document_version_id: UUID,
    ) -> DocumentVersion | None:
        """Return one scoped version."""

        return self.versions.get(
            (
                workspace_id,
                document_id,
                document_version_id,
            )
        )

    async def get_for_update(
        self,
        *,
        workspace_id: UUID,
        document_id: UUID,
        document_version_id: UUID,
    ) -> DocumentVersion | None:
        """Return one scoped version and record the lock request."""

        self.lock_arguments = (
            workspace_id,
            document_id,
            document_version_id,
        )
        return await self.get(
            workspace_id=workspace_id,
            document_id=document_id,
            document_version_id=document_version_id,
        )

    async def get_by_content_hash(
        self,
        *,
        workspace_id: UUID,
        document_id: UUID,
        content_sha256: str,
    ) -> DocumentVersion | None:
        """Return a matching normalized content hash."""

        self.content_hash_arguments = (
            workspace_id,
            document_id,
            content_sha256,
        )
        for version in self.versions.values():
            if (
                version.workspace_id == workspace_id
                and version.document_id == document_id
                and version.content_sha256 == content_sha256
            ):
                return version

        return None

    async def next_version_number(
        self,
        *,
        workspace_id: UUID,
        document_id: UUID,
    ) -> int:
        """Return the configured next version number."""

        return self.next_number

    async def list(
        self,
        *,
        workspace_id: UUID,
        document_id: UUID,
        limit: int,
        after_version_number: int | None = None,
        after_document_version_id: UUID | None = None,
    ) -> Sequence[DocumentVersion]:
        """Return the configured deterministic page."""

        self.list_arguments = (
            workspace_id,
            document_id,
            limit,
            after_version_number,
            after_document_version_id,
        )
        return self.list_result


def create_document(
    *,
    workspace_id: UUID = _WORKSPACE_ID,
) -> Document:
    """Create one deterministic document."""

    return Document.create(
        document_id=_DOCUMENT_ID,
        workspace_id=workspace_id,
        title="Database Incident Runbook",
        external_reference="runbook-database-incidents",
        now=_TIMESTAMP,
    )


def create_pending_version(
    *,
    workspace_id: UUID = _WORKSPACE_ID,
    document_id: UUID = _DOCUMENT_ID,
    version_id: UUID = _VERSION_ID,
    version_number: int = 1,
    content: str = ("# Database incidents\nRestart the connection pool.\n"),
) -> DocumentVersion:
    """Create one deterministic pending version."""

    return DocumentVersion.create_pending(
        document_version_id=version_id,
        workspace_id=workspace_id,
        document_id=document_id,
        version_number=version_number,
        media_type=DocumentMediaType.TEXT_MARKDOWN,
        content=content,
        now=_TIMESTAMP,
    )


def create_ready_version() -> DocumentVersion:
    """Create one ready version eligible for activation."""

    profiled = create_pending_version().bind_index_profile(
        KnowledgeIndexProfile(
            chunking_strategy="markdown-token",
            chunking_version="v1",
            tokenizer_encoding="cl100k_base",
            embedding_provider="mock",
            embedding_model="mock-hashing-embedding-v1",
            embedding_dimensions=64,
            knowledge_collection=("supportops-knowledge-mock-v1"),
            knowledge_vector_name="dense",
        ),
        now=_TIMESTAMP,
    )
    return profiled.mark_ready(
        chunk_count=1,
        embedding_input_tokens=18,
        embedding_estimated_cost_usd=Decimal("0"),
        embedding_pricing_catalog_version=("supportops-embedding-pricing-2026-08-01"),
        indexed_at=_TIMESTAMP + timedelta(minutes=1),
    )


async def test_create_document_persists_document_and_first_version_atomically() -> None:
    workspace_repository = FakeWorkspaceRepository()
    document_repository = FakeDocumentRepository()
    version_repository = FakeDocumentVersionRepository()
    transaction_manager = FakeTransactionManager()
    service = CreateDocument(
        workspace_repository=workspace_repository,
        document_repository=document_repository,
        version_repository=version_repository,
        transaction_manager=transaction_manager,
    )

    result = await service.execute(
        workspace_id=_WORKSPACE_ID,
        title="  Database Incident Runbook  ",
        external_reference=("  runbook-database-incidents  "),
        media_type=DocumentMediaType.TEXT_MARKDOWN,
        content=("\ufeff# Database incidents\r\nRestart the connection pool.\r"),
    )

    assert workspace_repository.requested_workspace_id == _WORKSPACE_ID
    assert document_repository.added_document == result.document
    assert version_repository.added_version == result.version
    assert result.version.document_id == result.document.id
    assert result.version.workspace_id == _WORKSPACE_ID
    assert result.version.version_number == 1
    assert result.version.created_at == result.document.created_at
    assert result.version.content == ("# Database incidents\nRestart the connection pool.\n")
    assert transaction_manager.completed


async def test_create_document_rolls_back_when_workspace_is_missing() -> None:
    workspace_repository = FakeWorkspaceRepository(workspace_exists=False)
    document_repository = FakeDocumentRepository()
    version_repository = FakeDocumentVersionRepository()
    transaction_manager = FakeTransactionManager()
    service = CreateDocument(
        workspace_repository=workspace_repository,
        document_repository=document_repository,
        version_repository=version_repository,
        transaction_manager=transaction_manager,
    )

    with pytest.raises(
        WorkspaceNotFoundError,
        match=r"Workspace was not found\.",
    ):
        await service.execute(
            workspace_id=_WORKSPACE_ID,
            title="Database Incident Runbook",
            media_type=DocumentMediaType.TEXT_MARKDOWN,
            content="# Database incidents\n",
        )

    assert document_repository.added_document is None
    assert version_repository.added_version is None
    assert transaction_manager.rolled_back


async def test_create_document_translates_external_reference_conflict() -> None:
    workspace_repository = FakeWorkspaceRepository()
    document_repository = FakeDocumentRepository()
    document_repository.external_reference_conflict = True
    version_repository = FakeDocumentVersionRepository()
    transaction_manager = FakeTransactionManager()
    service = CreateDocument(
        workspace_repository=workspace_repository,
        document_repository=document_repository,
        version_repository=version_repository,
        transaction_manager=transaction_manager,
    )

    with pytest.raises(
        DocumentExternalReferenceConflictApplicationError,
        match=(
            r"Document external reference already exists "
            r"in the workspace\."
        ),
    ):
        await service.execute(
            workspace_id=_WORKSPACE_ID,
            title="Database Incident Runbook",
            external_reference="runbook-database-incidents",
            media_type=DocumentMediaType.TEXT_MARKDOWN,
            content="# Database incidents\n",
        )

    assert transaction_manager.rolled_back


async def test_get_document_requires_matching_workspace() -> None:
    document = create_document()
    repository = FakeDocumentRepository()
    repository.documents[(document.workspace_id, document.id)] = document
    service = GetDocument(repository=repository)

    assert (
        await service.execute(
            workspace_id=_WORKSPACE_ID,
            document_id=_DOCUMENT_ID,
        )
        == document
    )

    with pytest.raises(
        DocumentNotFoundError,
        match=r"Document was not found\.",
    ):
        await service.execute(
            workspace_id=_OTHER_WORKSPACE_ID,
            document_id=_DOCUMENT_ID,
        )


async def test_list_documents_checks_workspace_and_forwards_keyset() -> None:
    document = create_document()
    workspace_repository = FakeWorkspaceRepository()
    document_repository = FakeDocumentRepository()
    document_repository.list_result = (document,)
    service = ListDocuments(
        workspace_repository=workspace_repository,
        document_repository=document_repository,
    )

    result = await service.execute(
        workspace_id=_WORKSPACE_ID,
        limit=21,
        after_created_at=_TIMESTAMP,
        after_document_id=_DOCUMENT_ID,
    )

    assert result == (document,)
    assert document_repository.list_arguments == (
        _WORKSPACE_ID,
        21,
        _TIMESTAMP,
        _DOCUMENT_ID,
    )


async def test_create_version_locks_document_and_assigns_next_number() -> None:
    document = create_document()
    document_repository = FakeDocumentRepository()
    document_repository.documents[(document.workspace_id, document.id)] = document
    version_repository = FakeDocumentVersionRepository()
    version_repository.next_number = 4
    transaction_manager = FakeTransactionManager()
    service = CreateDocumentVersion(
        document_repository=document_repository,
        version_repository=version_repository,
        transaction_manager=transaction_manager,
    )

    version = await service.execute(
        workspace_id=_WORKSPACE_ID,
        document_id=_DOCUMENT_ID,
        media_type=DocumentMediaType.TEXT_MARKDOWN,
        content=("\ufeff# Database incidents\r\nEscalate after two failed restarts.\r"),
    )

    assert document_repository.lock_arguments == (
        _WORKSPACE_ID,
        _DOCUMENT_ID,
    )
    assert version.version_number == 4
    assert version.content == ("# Database incidents\nEscalate after two failed restarts.\n")
    assert version_repository.added_version == version
    assert transaction_manager.completed


async def test_create_version_rejects_duplicate_normalized_content() -> None:
    document = create_document()
    existing = create_pending_version()
    document_repository = FakeDocumentRepository()
    document_repository.documents[(document.workspace_id, document.id)] = document
    version_repository = FakeDocumentVersionRepository()
    version_repository.versions[
        (
            existing.workspace_id,
            existing.document_id,
            existing.id,
        )
    ] = existing
    transaction_manager = FakeTransactionManager()
    service = CreateDocumentVersion(
        document_repository=document_repository,
        version_repository=version_repository,
        transaction_manager=transaction_manager,
    )

    with pytest.raises(
        DocumentVersionContentConflictApplicationError,
        match=(r"Document content already exists for this document\."),
    ):
        await service.execute(
            workspace_id=_WORKSPACE_ID,
            document_id=_DOCUMENT_ID,
            media_type=DocumentMediaType.TEXT_MARKDOWN,
            content=("# Database incidents\r\nRestart the connection pool.\r\n"),
        )

    assert version_repository.added_version is None
    assert transaction_manager.rolled_back


async def test_create_version_translates_database_number_conflict() -> None:
    document = create_document()
    document_repository = FakeDocumentRepository()
    document_repository.documents[(document.workspace_id, document.id)] = document
    version_repository = FakeDocumentVersionRepository()
    version_repository.number_conflict = True
    transaction_manager = FakeTransactionManager()
    service = CreateDocumentVersion(
        document_repository=document_repository,
        version_repository=version_repository,
        transaction_manager=transaction_manager,
    )

    with pytest.raises(
        DocumentVersionNumberConflictApplicationError,
        match=r"Document version number already exists\.",
    ):
        await service.execute(
            workspace_id=_WORKSPACE_ID,
            document_id=_DOCUMENT_ID,
            media_type=DocumentMediaType.TEXT_MARKDOWN,
            content="# New version\nUpdated procedure.\n",
        )

    assert transaction_manager.rolled_back


async def test_create_version_hides_cross_workspace_document() -> None:
    document = create_document()
    document_repository = FakeDocumentRepository()
    document_repository.documents[(document.workspace_id, document.id)] = document
    version_repository = FakeDocumentVersionRepository()
    transaction_manager = FakeTransactionManager()
    service = CreateDocumentVersion(
        document_repository=document_repository,
        version_repository=version_repository,
        transaction_manager=transaction_manager,
    )

    with pytest.raises(
        DocumentNotFoundError,
        match=r"Document was not found\.",
    ):
        await service.execute(
            workspace_id=_OTHER_WORKSPACE_ID,
            document_id=_DOCUMENT_ID,
            media_type=DocumentMediaType.TEXT_MARKDOWN,
            content="# Cross-workspace version\n",
        )

    assert version_repository.added_version is None
    assert transaction_manager.rolled_back


async def test_get_version_distinguishes_document_and_version_not_found() -> None:
    document = create_document()
    version = create_pending_version()
    document_repository = FakeDocumentRepository()
    version_repository = FakeDocumentVersionRepository()
    service = GetDocumentVersion(
        document_repository=document_repository,
        version_repository=version_repository,
    )

    with pytest.raises(
        DocumentNotFoundError,
        match=r"Document was not found\.",
    ):
        await service.execute(
            workspace_id=_WORKSPACE_ID,
            document_id=_DOCUMENT_ID,
            document_version_id=_VERSION_ID,
        )

    document_repository.documents[(document.workspace_id, document.id)] = document
    with pytest.raises(
        DocumentVersionNotFoundError,
        match=r"Document version was not found\.",
    ):
        await service.execute(
            workspace_id=_WORKSPACE_ID,
            document_id=_DOCUMENT_ID,
            document_version_id=_OTHER_VERSION_ID,
        )

    version_repository.versions[
        (
            version.workspace_id,
            version.document_id,
            version.id,
        )
    ] = version
    assert (
        await service.execute(
            workspace_id=_WORKSPACE_ID,
            document_id=_DOCUMENT_ID,
            document_version_id=_VERSION_ID,
        )
        == version
    )


async def test_list_versions_checks_document_and_forwards_keyset() -> None:
    document = create_document()
    version = create_pending_version(version_number=3)
    document_repository = FakeDocumentRepository()
    document_repository.documents[(document.workspace_id, document.id)] = document
    version_repository = FakeDocumentVersionRepository()
    version_repository.list_result = (version,)
    service = ListDocumentVersions(
        document_repository=document_repository,
        version_repository=version_repository,
    )

    result = await service.execute(
        workspace_id=_WORKSPACE_ID,
        document_id=_DOCUMENT_ID,
        limit=21,
        after_version_number=4,
        after_document_version_id=_OTHER_VERSION_ID,
    )

    assert result == (version,)
    assert version_repository.list_arguments == (
        _WORKSPACE_ID,
        _DOCUMENT_ID,
        21,
        4,
        _OTHER_VERSION_ID,
    )


async def test_activation_updates_document_to_ready_owned_version() -> None:
    document = create_document()
    version = create_ready_version()
    document_repository = FakeDocumentRepository()
    document_repository.documents[(document.workspace_id, document.id)] = document
    version_repository = FakeDocumentVersionRepository()
    version_repository.versions[
        (
            version.workspace_id,
            version.document_id,
            version.id,
        )
    ] = version
    transaction_manager = FakeTransactionManager()
    service = ActivateDocumentVersion(
        document_repository=document_repository,
        version_repository=version_repository,
        transaction_manager=transaction_manager,
    )

    activated = await service.execute(
        workspace_id=_WORKSPACE_ID,
        document_id=_DOCUMENT_ID,
        document_version_id=_VERSION_ID,
    )

    assert activated.active_version_id == _VERSION_ID
    assert document_repository.updated_document == activated
    assert document_repository.lock_arguments == (
        _WORKSPACE_ID,
        _DOCUMENT_ID,
    )
    assert version_repository.lock_arguments == (
        _WORKSPACE_ID,
        _DOCUMENT_ID,
        _VERSION_ID,
    )
    assert transaction_manager.completed


async def test_activation_is_idempotent_for_current_version() -> None:
    version = create_ready_version()
    document = create_document().activate_version(
        version,
        now=_TIMESTAMP + timedelta(minutes=2),
    )
    document_repository = FakeDocumentRepository()
    document_repository.documents[(document.workspace_id, document.id)] = document
    version_repository = FakeDocumentVersionRepository()
    version_repository.versions[
        (
            version.workspace_id,
            version.document_id,
            version.id,
        )
    ] = version
    transaction_manager = FakeTransactionManager()
    service = ActivateDocumentVersion(
        document_repository=document_repository,
        version_repository=version_repository,
        transaction_manager=transaction_manager,
    )

    result = await service.execute(
        workspace_id=_WORKSPACE_ID,
        document_id=_DOCUMENT_ID,
        document_version_id=_VERSION_ID,
    )

    assert result is document
    assert document_repository.updated_document is None
    assert transaction_manager.completed


async def test_activation_rejects_pending_version() -> None:
    document = create_document()
    version = create_pending_version()
    document_repository = FakeDocumentRepository()
    document_repository.documents[(document.workspace_id, document.id)] = document
    version_repository = FakeDocumentVersionRepository()
    version_repository.versions[
        (
            version.workspace_id,
            version.document_id,
            version.id,
        )
    ] = version
    transaction_manager = FakeTransactionManager()
    service = ActivateDocumentVersion(
        document_repository=document_repository,
        version_repository=version_repository,
        transaction_manager=transaction_manager,
    )

    with pytest.raises(
        DocumentVersionNotReadyError,
        match=(r"Document version is not ready for activation\."),
    ):
        await service.execute(
            workspace_id=_WORKSPACE_ID,
            document_id=_DOCUMENT_ID,
            document_version_id=_VERSION_ID,
        )

    assert document_repository.updated_document is None
    assert transaction_manager.rolled_back


async def test_activation_hides_cross_workspace_document() -> None:
    document = create_document()
    version = create_ready_version()
    document_repository = FakeDocumentRepository()
    document_repository.documents[(document.workspace_id, document.id)] = document
    version_repository = FakeDocumentVersionRepository()
    version_repository.versions[
        (
            version.workspace_id,
            version.document_id,
            version.id,
        )
    ] = version
    transaction_manager = FakeTransactionManager()
    service = ActivateDocumentVersion(
        document_repository=document_repository,
        version_repository=version_repository,
        transaction_manager=transaction_manager,
    )

    with pytest.raises(
        DocumentNotFoundError,
        match=r"Document was not found\.",
    ):
        await service.execute(
            workspace_id=_OTHER_WORKSPACE_ID,
            document_id=_DOCUMENT_ID,
            document_version_id=_VERSION_ID,
        )

    assert document_repository.updated_document is None
    assert transaction_manager.rolled_back
