"""Workspace-scoped versioned knowledge-document HTTP routes."""

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Query, status

from supportops.modules.knowledge_documents.api.dependencies import (
    ActivateDocumentVersionDependency,
    CreateDocumentDependency,
    CreateDocumentVersionDependency,
    GetDocumentDependency,
    GetDocumentVersionDependency,
    ListDocumentsDependency,
    ListDocumentVersionsDependency,
)
from supportops.modules.knowledge_documents.api.pagination import (
    decode_document_cursor,
    decode_document_version_cursor,
    encode_document_cursor,
    encode_document_version_cursor,
)
from supportops.modules.knowledge_documents.api.schemas import (
    DocumentCreateRequest,
    DocumentCreateResponse,
    DocumentListResponse,
    DocumentResponse,
    DocumentVersionCreateRequest,
    DocumentVersionDetailResponse,
    DocumentVersionListResponse,
    DocumentVersionResponse,
)

_DEFAULT_PAGE_SIZE = 20
_MAX_PAGE_SIZE = 100

router = APIRouter(
    prefix="/workspaces/{workspace_id}/documents",
    tags=["knowledge documents"],
)


@router.post(
    "",
    response_model=DocumentCreateResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_document(
    workspace_id: UUID,
    request: DocumentCreateRequest,
    service: CreateDocumentDependency,
) -> DocumentCreateResponse:
    """Create a document and its first immutable pending version."""

    result = await service.execute(
        workspace_id=workspace_id,
        title=request.title,
        external_reference=request.external_reference,
        media_type=request.media_type,
        content=request.content,
    )
    return DocumentCreateResponse.from_result(result)


@router.get(
    "",
    response_model=DocumentListResponse,
)
async def list_documents(
    workspace_id: UUID,
    service: ListDocumentsDependency,
    page_size: Annotated[
        int,
        Query(
            ge=1,
            le=_MAX_PAGE_SIZE,
        ),
    ] = _DEFAULT_PAGE_SIZE,
    cursor: str | None = None,
) -> DocumentListResponse:
    """List one deterministic page of workspace documents."""

    position = decode_document_cursor(cursor) if cursor is not None else None
    documents = await service.execute(
        workspace_id=workspace_id,
        limit=page_size + 1,
        after_created_at=(position.created_at if position is not None else None),
        after_document_id=(position.document_id if position is not None else None),
    )

    has_next_page = len(documents) > page_size
    page = list(documents[:page_size])
    next_cursor = None
    if has_next_page and page:
        last_document = page[-1]
        next_cursor = encode_document_cursor(
            created_at=last_document.created_at,
            document_id=last_document.id,
        )

    return DocumentListResponse(
        items=[DocumentResponse.from_domain(document) for document in page],
        next_cursor=next_cursor,
    )


@router.get(
    "/{document_id}",
    response_model=DocumentResponse,
)
async def get_document(
    workspace_id: UUID,
    document_id: UUID,
    service: GetDocumentDependency,
) -> DocumentResponse:
    """Retrieve document metadata through its workspace boundary."""

    document = await service.execute(
        workspace_id=workspace_id,
        document_id=document_id,
    )
    return DocumentResponse.from_domain(document)


@router.post(
    "/{document_id}/versions",
    response_model=DocumentVersionResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_document_version(
    workspace_id: UUID,
    document_id: UUID,
    request: DocumentVersionCreateRequest,
    service: CreateDocumentVersionDependency,
) -> DocumentVersionResponse:
    """Create the next immutable pending version."""

    version = await service.execute(
        workspace_id=workspace_id,
        document_id=document_id,
        media_type=request.media_type,
        content=request.content,
    )
    return DocumentVersionResponse.from_domain(version)


@router.get(
    "/{document_id}/versions",
    response_model=DocumentVersionListResponse,
)
async def list_document_versions(
    workspace_id: UUID,
    document_id: UUID,
    service: ListDocumentVersionsDependency,
    page_size: Annotated[
        int,
        Query(
            ge=1,
            le=_MAX_PAGE_SIZE,
        ),
    ] = _DEFAULT_PAGE_SIZE,
    cursor: str | None = None,
) -> DocumentVersionListResponse:
    """List one deterministic page of immutable document versions."""

    position = decode_document_version_cursor(cursor) if cursor is not None else None
    versions = await service.execute(
        workspace_id=workspace_id,
        document_id=document_id,
        limit=page_size + 1,
        after_version_number=(position.version_number if position is not None else None),
        after_document_version_id=(position.document_version_id if position is not None else None),
    )

    has_next_page = len(versions) > page_size
    page = list(versions[:page_size])
    next_cursor = None
    if has_next_page and page:
        last_version = page[-1]
        next_cursor = encode_document_version_cursor(
            version_number=last_version.version_number,
            document_version_id=last_version.id,
        )

    return DocumentVersionListResponse(
        items=[DocumentVersionResponse.from_domain(version) for version in page],
        next_cursor=next_cursor,
    )


@router.get(
    "/{document_id}/versions/{document_version_id}",
    response_model=DocumentVersionDetailResponse,
)
async def get_document_version(
    workspace_id: UUID,
    document_id: UUID,
    document_version_id: UUID,
    service: GetDocumentVersionDependency,
) -> DocumentVersionDetailResponse:
    """Retrieve metadata and source content for one owned version."""

    version = await service.execute(
        workspace_id=workspace_id,
        document_id=document_id,
        document_version_id=document_version_id,
    )
    return DocumentVersionDetailResponse.from_domain(version)


@router.post(
    "/{document_id}/versions/{document_version_id}/activate",
    response_model=DocumentResponse,
)
async def activate_document_version(
    workspace_id: UUID,
    document_id: UUID,
    document_version_id: UUID,
    service: ActivateDocumentVersionDependency,
) -> DocumentResponse:
    """Activate an explicitly selected ready document version."""

    document = await service.execute(
        workspace_id=workspace_id,
        document_id=document_id,
        document_version_id=document_version_id,
    )
    return DocumentResponse.from_domain(document)
