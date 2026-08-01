"""FastAPI dependencies for knowledge-document use cases."""

from typing import Annotated

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from supportops.api.dependencies import get_postgresql_session
from supportops.infrastructure.postgresql.transaction import (
    SqlAlchemyTransactionManager,
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
from supportops.modules.knowledge_documents.infrastructure.repository import (
    SqlAlchemyDocumentRepository,
    SqlAlchemyDocumentVersionRepository,
)
from supportops.modules.workspaces.infrastructure.repository import (
    SqlAlchemyWorkspaceRepository,
)

PostgresqlSessionDependency = Annotated[
    AsyncSession,
    Depends(get_postgresql_session),
]


def get_create_document(
    session: PostgresqlSessionDependency,
) -> CreateDocument:
    """Construct the create-document use case."""

    return CreateDocument(
        workspace_repository=SqlAlchemyWorkspaceRepository(session),
        document_repository=SqlAlchemyDocumentRepository(session),
        version_repository=SqlAlchemyDocumentVersionRepository(session),
        transaction_manager=SqlAlchemyTransactionManager(session),
    )


def get_get_document(
    session: PostgresqlSessionDependency,
) -> GetDocument:
    """Construct the get-document use case."""

    return GetDocument(
        repository=SqlAlchemyDocumentRepository(session),
    )


def get_list_documents(
    session: PostgresqlSessionDependency,
) -> ListDocuments:
    """Construct the list-documents use case."""

    return ListDocuments(
        workspace_repository=SqlAlchemyWorkspaceRepository(session),
        document_repository=SqlAlchemyDocumentRepository(session),
    )


def get_create_document_version(
    session: PostgresqlSessionDependency,
) -> CreateDocumentVersion:
    """Construct the create-document-version use case."""

    return CreateDocumentVersion(
        document_repository=SqlAlchemyDocumentRepository(session),
        version_repository=SqlAlchemyDocumentVersionRepository(session),
        transaction_manager=SqlAlchemyTransactionManager(session),
    )


def get_get_document_version(
    session: PostgresqlSessionDependency,
) -> GetDocumentVersion:
    """Construct the get-document-version use case."""

    return GetDocumentVersion(
        document_repository=SqlAlchemyDocumentRepository(session),
        version_repository=SqlAlchemyDocumentVersionRepository(session),
    )


def get_list_document_versions(
    session: PostgresqlSessionDependency,
) -> ListDocumentVersions:
    """Construct the list-document-versions use case."""

    return ListDocumentVersions(
        document_repository=SqlAlchemyDocumentRepository(session),
        version_repository=SqlAlchemyDocumentVersionRepository(session),
    )


def get_activate_document_version(
    session: PostgresqlSessionDependency,
) -> ActivateDocumentVersion:
    """Construct the activate-document-version use case."""

    return ActivateDocumentVersion(
        document_repository=SqlAlchemyDocumentRepository(session),
        version_repository=SqlAlchemyDocumentVersionRepository(session),
        transaction_manager=SqlAlchemyTransactionManager(session),
    )


CreateDocumentDependency = Annotated[
    CreateDocument,
    Depends(get_create_document),
]
GetDocumentDependency = Annotated[
    GetDocument,
    Depends(get_get_document),
]
ListDocumentsDependency = Annotated[
    ListDocuments,
    Depends(get_list_documents),
]
CreateDocumentVersionDependency = Annotated[
    CreateDocumentVersion,
    Depends(get_create_document_version),
]
GetDocumentVersionDependency = Annotated[
    GetDocumentVersion,
    Depends(get_get_document_version),
]
ListDocumentVersionsDependency = Annotated[
    ListDocumentVersions,
    Depends(get_list_document_versions),
]
ActivateDocumentVersionDependency = Annotated[
    ActivateDocumentVersion,
    Depends(get_activate_document_version),
]
