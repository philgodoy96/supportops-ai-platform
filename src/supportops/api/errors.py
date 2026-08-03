"""Stable HTTP responses for expected application errors."""

from collections.abc import Callable, Coroutine
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from supportops.ai.embeddings.errors import EmbeddingError
from supportops.core.request_context import get_request_context
from supportops.knowledge_index.vector_store.contracts import (
    KnowledgeVectorStoreError,
)
from supportops.modules.agent_runs.application.errors import AgentRunNotFoundError
from supportops.modules.approvals.api.pagination import (
    InvalidApprovalPaginationCursor,
)
from supportops.modules.approvals.application.errors import (
    ApprovalDecisionConflictError,
    ApprovalRequestExpiredError,
    ApprovalRunStateConflictError,
    ApprovalToolCallStateConflictError,
)
from supportops.modules.approvals.application.errors import (
    ApprovalRequestNotFoundError as ApprovalDecisionNotFoundError,
)
from supportops.modules.approvals.application.queries import (
    ApprovalRequestNotFoundError,
)
from supportops.modules.controlled_support_inspection.application.errors import (
    ControlledSupportInspectionInconsistentError,
    ControlledSupportInspectionNotFoundError,
    UnsupportedAgentRunInspectionError,
)
from supportops.modules.knowledge_documents.api.pagination import (
    InvalidKnowledgePaginationCursorError,
)
from supportops.modules.knowledge_documents.application.errors import (
    DocumentExternalReferenceConflictApplicationError,
    DocumentNotFoundError,
    DocumentVersionContentConflictApplicationError,
    DocumentVersionNotFoundError,
    DocumentVersionNotReadyError,
    DocumentVersionNumberConflictApplicationError,
)
from supportops.modules.ticket_classifications.application.errors import (
    TicketClassificationNotFoundError,
)
from supportops.modules.ticket_classifications.application.pagination import (
    InvalidClassificationPaginationCursorError,
)
from supportops.modules.tickets.api.pagination import InvalidPaginationCursorError
from supportops.modules.tickets.application.errors import (
    TicketExternalReferenceConflictApplicationError,
    TicketNotFoundError,
)
from supportops.modules.workspaces.application.errors import (
    WorkspaceNotFoundError,
    WorkspaceSlugConflictApplicationError,
)


class ErrorDetail(BaseModel):
    """Machine-readable expected error details."""

    code: str
    message: str
    request_id: str


class ErrorResponse(BaseModel):
    """Stable expected error response envelope."""

    error: ErrorDetail


ErrorHandler = Callable[
    [Request, Exception],
    Coroutine[Any, Any, JSONResponse],
]


def register_error_handlers(app: FastAPI) -> None:
    """Register handlers for expected application errors."""

    app.add_exception_handler(
        WorkspaceNotFoundError,
        _workspace_not_found_handler,
    )
    app.add_exception_handler(
        WorkspaceSlugConflictApplicationError,
        _workspace_slug_conflict_handler,
    )
    app.add_exception_handler(
        TicketNotFoundError,
        _ticket_not_found_handler,
    )
    app.add_exception_handler(
        TicketExternalReferenceConflictApplicationError,
        _ticket_external_reference_conflict_handler,
    )
    app.add_exception_handler(
        AgentRunNotFoundError,
        _agent_run_not_found_handler,
    )
    app.add_exception_handler(
        InvalidPaginationCursorError,
        _invalid_pagination_cursor_handler,
    )
    app.add_exception_handler(
        TicketClassificationNotFoundError,
        _ticket_classification_not_found_handler,
    )
    app.add_exception_handler(
        InvalidClassificationPaginationCursorError,
        _invalid_classification_pagination_cursor_handler,
    )
    app.add_exception_handler(
        DocumentNotFoundError,
        _document_not_found_handler,
    )
    app.add_exception_handler(
        DocumentVersionNotFoundError,
        _document_version_not_found_handler,
    )
    app.add_exception_handler(
        DocumentExternalReferenceConflictApplicationError,
        _document_external_reference_conflict_handler,
    )
    app.add_exception_handler(
        DocumentVersionContentConflictApplicationError,
        _document_version_content_conflict_handler,
    )
    app.add_exception_handler(
        DocumentVersionNumberConflictApplicationError,
        _document_version_number_conflict_handler,
    )
    app.add_exception_handler(
        DocumentVersionNotReadyError,
        _document_version_not_ready_handler,
    )
    app.add_exception_handler(
        InvalidKnowledgePaginationCursorError,
        _invalid_knowledge_pagination_cursor_handler,
    )
    app.add_exception_handler(
        EmbeddingError,
        _embedding_error_handler,
    )
    app.add_exception_handler(
        KnowledgeVectorStoreError,
        _knowledge_vector_store_error_handler,
    )
    app.add_exception_handler(
        ControlledSupportInspectionNotFoundError,
        _controlled_support_inspection_not_found_handler,
    )
    app.add_exception_handler(
        UnsupportedAgentRunInspectionError,
        _unsupported_agent_run_inspection_handler,
    )
    app.add_exception_handler(
        ControlledSupportInspectionInconsistentError,
        _controlled_support_inspection_inconsistent_handler,
    )
    app.add_exception_handler(
        ApprovalRequestNotFoundError,
        _approval_request_not_found_handler,
    )
    app.add_exception_handler(
        ApprovalDecisionNotFoundError,
        _approval_request_not_found_handler,
    )
    app.add_exception_handler(
        ApprovalDecisionConflictError,
        _approval_decision_conflict_handler,
    )
    app.add_exception_handler(
        ApprovalRequestExpiredError,
        _approval_request_expired_handler,
    )
    app.add_exception_handler(
        ApprovalRunStateConflictError,
        _approval_run_state_conflict_handler,
    )
    app.add_exception_handler(
        ApprovalToolCallStateConflictError,
        _approval_tool_call_state_conflict_handler,
    )
    app.add_exception_handler(
        InvalidApprovalPaginationCursor,
        _invalid_approval_pagination_cursor_handler,
    )


async def _workspace_not_found_handler(
    request: Request,
    error: Exception,
) -> JSONResponse:
    del request
    del error

    return _expected_error_response(
        status_code=404,
        code="workspace_not_found",
        message="Workspace was not found.",
    )


async def _workspace_slug_conflict_handler(
    request: Request,
    error: Exception,
) -> JSONResponse:
    del request
    del error

    return _expected_error_response(
        status_code=409,
        code="workspace_slug_conflict",
        message="Workspace slug is already in use.",
    )


async def _ticket_not_found_handler(
    request: Request,
    error: Exception,
) -> JSONResponse:
    del request
    del error

    return _expected_error_response(
        status_code=404,
        code="ticket_not_found",
        message="Ticket was not found.",
    )


async def _ticket_external_reference_conflict_handler(
    request: Request,
    error: Exception,
) -> JSONResponse:
    del request
    del error

    return _expected_error_response(
        status_code=409,
        code="ticket_external_reference_conflict",
        message=("Ticket external reference already exists in the workspace."),
    )


async def _agent_run_not_found_handler(
    request: Request,
    error: Exception,
) -> JSONResponse:
    del request
    del error

    return _expected_error_response(
        status_code=404,
        code="agent_run_not_found",
        message="AgentRun was not found.",
    )


async def _invalid_pagination_cursor_handler(
    request: Request,
    error: Exception,
) -> JSONResponse:
    del request
    del error

    return _expected_error_response(
        status_code=400,
        code="invalid_pagination_cursor",
        message="Pagination cursor is invalid.",
    )


async def _ticket_classification_not_found_handler(
    request: Request,
    error: Exception,
) -> JSONResponse:
    del request
    del error

    return _expected_error_response(
        status_code=404,
        code="ticket_classification_not_found",
        message="Ticket classification was not found.",
    )


async def _invalid_classification_pagination_cursor_handler(
    request: Request,
    error: Exception,
) -> JSONResponse:
    del request
    del error

    return _expected_error_response(
        status_code=400,
        code="invalid_pagination_cursor",
        message="Pagination cursor is invalid.",
    )


async def _document_not_found_handler(
    request: Request,
    error: Exception,
) -> JSONResponse:
    del request
    del error

    return _expected_error_response(
        status_code=404,
        code="document_not_found",
        message="Document was not found.",
    )


async def _document_version_not_found_handler(
    request: Request,
    error: Exception,
) -> JSONResponse:
    del request
    del error

    return _expected_error_response(
        status_code=404,
        code="document_version_not_found",
        message="Document version was not found.",
    )


async def _document_external_reference_conflict_handler(
    request: Request,
    error: Exception,
) -> JSONResponse:
    del request
    del error

    return _expected_error_response(
        status_code=409,
        code="document_external_reference_conflict",
        message="Document external reference already exists in the workspace.",
    )


async def _document_version_content_conflict_handler(
    request: Request,
    error: Exception,
) -> JSONResponse:
    del request
    del error

    return _expected_error_response(
        status_code=409,
        code="document_version_content_conflict",
        message="Document content already exists for this document.",
    )


async def _document_version_number_conflict_handler(
    request: Request,
    error: Exception,
) -> JSONResponse:
    del request
    del error

    return _expected_error_response(
        status_code=409,
        code="document_version_number_conflict",
        message="Document version number already exists.",
    )


async def _document_version_not_ready_handler(
    request: Request,
    error: Exception,
) -> JSONResponse:
    del request
    del error

    return _expected_error_response(
        status_code=409,
        code="document_version_not_ready",
        message="Document version is not ready for activation.",
    )


async def _invalid_knowledge_pagination_cursor_handler(
    request: Request,
    error: Exception,
) -> JSONResponse:
    del request
    del error

    return _expected_error_response(
        status_code=400,
        code="invalid_pagination_cursor",
        message="Pagination cursor is invalid.",
    )


async def _embedding_error_handler(
    request: Request,
    error: Exception,
) -> JSONResponse:
    del request
    del error

    return _expected_error_response(
        status_code=503,
        code="knowledge_retrieval_unavailable",
        message="Knowledge retrieval is temporarily unavailable.",
    )


async def _knowledge_vector_store_error_handler(
    request: Request,
    error: Exception,
) -> JSONResponse:
    del request
    del error

    return _expected_error_response(
        status_code=503,
        code="knowledge_retrieval_unavailable",
        message="Knowledge retrieval is temporarily unavailable.",
    )


async def _controlled_support_inspection_not_found_handler(
    request: Request,
    error: Exception,
) -> JSONResponse:
    del request
    del error

    return _expected_error_response(
        status_code=404,
        code="controlled_support_inspection_not_found",
        message="The controlled support inspection was not found.",
    )


async def _unsupported_agent_run_inspection_handler(
    request: Request,
    error: Exception,
) -> JSONResponse:
    del request
    del error

    return _expected_error_response(
        status_code=409,
        code="unsupported_agent_run_inspection",
        message="The AgentRun workflow does not support this inspection view.",
    )


async def _controlled_support_inspection_inconsistent_handler(
    request: Request,
    error: Exception,
) -> JSONResponse:
    del request
    del error

    return _expected_error_response(
        status_code=500,
        code="controlled_support_inspection_inconsistent",
        message="The controlled support inspection data is internally inconsistent.",
    )


async def _approval_request_not_found_handler(
    request: Request,
    error: Exception,
) -> JSONResponse:
    del request
    del error

    return _expected_error_response(
        status_code=404,
        code="approval_request_not_found",
        message="Approval request was not found.",
    )


async def _approval_decision_conflict_handler(
    request: Request,
    error: Exception,
) -> JSONResponse:
    del request
    del error

    return _expected_error_response(
        status_code=409,
        code="approval_decision_conflict",
        message=("The approval request already has a conflicting decision."),
    )


async def _approval_request_expired_handler(
    request: Request,
    error: Exception,
) -> JSONResponse:
    del request
    del error

    return _expected_error_response(
        status_code=409,
        code="approval_request_expired",
        message="The approval request has expired.",
    )


async def _approval_run_state_conflict_handler(
    request: Request,
    error: Exception,
) -> JSONResponse:
    del request
    del error

    return _expected_error_response(
        status_code=409,
        code="approval_run_state_conflict",
        message=("The AgentRun is not waiting for this approval decision."),
    )


async def _approval_tool_call_state_conflict_handler(
    request: Request,
    error: Exception,
) -> JSONResponse:
    del request
    del error

    return _expected_error_response(
        status_code=409,
        code="approval_tool_call_state_conflict",
        message=("The proposed sensitive tool call is inconsistent."),
    )


async def _invalid_approval_pagination_cursor_handler(
    request: Request,
    error: Exception,
) -> JSONResponse:
    del request
    del error

    return _expected_error_response(
        status_code=400,
        code="invalid_pagination_cursor",
        message="Approval pagination cursor is invalid.",
    )


def _expected_error_response(
    *,
    status_code: int,
    code: str,
    message: str,
) -> JSONResponse:
    context = get_request_context()

    request_id = str(context.request_id) if context is not None else "unavailable"

    response = ErrorResponse(
        error=ErrorDetail(
            code=code,
            message=message,
            request_id=request_id,
        )
    )

    return JSONResponse(
        status_code=status_code,
        content=response.model_dump(),
    )
