"""Unit tests for the controlled search_knowledge tool."""

from hashlib import sha256
from uuid import UUID

import pytest

from supportops.agent_tools.application.execution import (
    ToolExecutionContext,
)
from supportops.agent_tools.domain.contracts import (
    ToolFailurePolicy,
    ToolSafetyLevel,
)
from supportops.agent_tools.domain.errors import (
    ToolDependencyUnavailableError,
)
from supportops.agent_tools.tools.search_knowledge import (
    SEARCH_KNOWLEDGE_TOOL_NAME,
    SEARCH_KNOWLEDGE_TOOL_VERSION,
    SearchKnowledgeInput,
    SearchKnowledgeOutput,
    SearchKnowledgeToolHandler,
    create_search_knowledge_binding,
    project_search_knowledge_safe_input,
    project_search_knowledge_safe_output,
)
from supportops.ai.embeddings.errors import (
    EmbeddingTimeoutError,
)
from supportops.knowledge_retrieval.contracts import (
    KnowledgeCitation,
    KnowledgeEvidence,
    KnowledgeSearchRequest,
    KnowledgeSearchResult,
)
from supportops.modules.knowledge_documents.domain.models import (
    DocumentMediaType,
)

_WORKSPACE_ID = UUID("10000000-0000-4000-8000-000000000001")
_TICKET_ID = UUID("20000000-0000-4000-8000-000000000002")
_AGENT_RUN_ID = UUID("30000000-0000-4000-8000-000000000003")
_ATTEMPT_ID = UUID("40000000-0000-4000-8000-000000000004")
_DOCUMENT_ID = UUID("50000000-0000-4000-8000-000000000005")
_DOCUMENT_VERSION_ID = UUID("60000000-0000-4000-8000-000000000006")
_CHUNK_ID = UUID("70000000-0000-4000-8000-000000000007")


class StubKnowledgeSearch:
    """Deterministic knowledge-search service test double."""

    def __init__(
        self,
        result: KnowledgeSearchResult | Exception,
    ) -> None:
        self._result = result
        self.requests: list[KnowledgeSearchRequest] = []

    async def execute(
        self,
        request: KnowledgeSearchRequest,
    ) -> KnowledgeSearchResult:
        self.requests.append(request)

        if isinstance(self._result, Exception):
            raise self._result

        return self._result


def _context() -> ToolExecutionContext:
    return ToolExecutionContext(
        workspace_id=_WORKSPACE_ID,
        ticket_id=_TICKET_ID,
        agent_run_id=_AGENT_RUN_ID,
        agent_run_attempt_id=_ATTEMPT_ID,
    )


def _arguments() -> SearchKnowledgeInput:
    return SearchKnowledgeInput(
        query="account access reset",
        top_k=5,
        document_ids=(_DOCUMENT_ID,),
    )


def _result() -> KnowledgeSearchResult:
    request = KnowledgeSearchRequest(
        workspace_id=_WORKSPACE_ID,
        query="account access reset",
        top_k=5,
        document_ids=(_DOCUMENT_ID,),
    )
    content = "Verify customer identity before starting the account-access recovery procedure."

    return KnowledgeSearchResult(
        request=request,
        searched_version_count=1,
        evidence=(
            KnowledgeEvidence(
                rank=1,
                score=0.91,
                content=content,
                content_sha256=sha256(content.encode("utf-8")).hexdigest(),
                token_count=14,
                citation=KnowledgeCitation(
                    workspace_id=_WORKSPACE_ID,
                    document_id=_DOCUMENT_ID,
                    document_title=("Account access recovery"),
                    document_external_reference=("runbook-account-access"),
                    document_version_id=(_DOCUMENT_VERSION_ID),
                    version_number=1,
                    chunk_id=_CHUNK_ID,
                    ordinal=0,
                    section_path=("Account access recovery",),
                    media_type=(DocumentMediaType.TEXT_MARKDOWN),
                ),
            ),
        ),
    )


async def test_executes_workspace_scoped_retrieval() -> None:
    service = StubKnowledgeSearch(_result())
    handler = SearchKnowledgeToolHandler(service)

    output = await handler.execute(
        _context(),
        _arguments(),
    )

    assert len(service.requests) == 1

    request = service.requests[0]

    assert request.workspace_id == _WORKSPACE_ID
    assert request.query == "account access reset"
    assert request.top_k == 5
    assert request.document_ids == (_DOCUMENT_ID,)

    assert isinstance(output, SearchKnowledgeOutput)
    assert output.searched_version_count == 1
    assert len(output.evidence) == 1

    evidence = output.evidence[0]

    assert evidence.rank == 1
    assert evidence.score == 0.91
    assert evidence.document_id == _DOCUMENT_ID
    assert evidence.document_version_id == _DOCUMENT_VERSION_ID
    assert evidence.chunk_id == _CHUNK_ID
    assert evidence.content_sha256 == sha256(evidence.content.encode("utf-8")).hexdigest()
    assert "Verify customer identity" in evidence.content


async def test_retrieval_query_identity_is_deterministic() -> None:
    first_service = StubKnowledgeSearch(_result())
    second_service = StubKnowledgeSearch(_result())

    first_output = await SearchKnowledgeToolHandler(first_service).execute(
        _context(),
        _arguments(),
    )
    second_output = await SearchKnowledgeToolHandler(second_service).execute(
        _context(),
        _arguments(),
    )

    assert isinstance(
        first_output,
        SearchKnowledgeOutput,
    )
    assert isinstance(
        second_output,
        SearchKnowledgeOutput,
    )
    assert first_output.retrieval_query_id == second_output.retrieval_query_id


async def test_query_change_changes_retrieval_identity() -> None:
    service = StubKnowledgeSearch(_result())
    handler = SearchKnowledgeToolHandler(service)

    first_output = await handler.execute(
        _context(),
        _arguments(),
    )
    second_output = await handler.execute(
        _context(),
        SearchKnowledgeInput(
            query="billing refund procedure",
            top_k=5,
            document_ids=(_DOCUMENT_ID,),
        ),
    )

    assert isinstance(
        first_output,
        SearchKnowledgeOutput,
    )
    assert isinstance(
        second_output,
        SearchKnowledgeOutput,
    )
    assert first_output.retrieval_query_id != second_output.retrieval_query_id


def test_input_normalizes_document_order() -> None:
    second_document_id = UUID("00000000-0000-4000-8000-000000000001")

    arguments = SearchKnowledgeInput(
        query="account access reset",
        top_k=5,
        document_ids=(
            _DOCUMENT_ID,
            second_document_id,
        ),
    )

    assert arguments.document_ids == (
        second_document_id,
        _DOCUMENT_ID,
    )


def test_input_rejects_duplicate_documents() -> None:
    with pytest.raises(
        ValueError,
        match="must not contain duplicates",
    ):
        SearchKnowledgeInput(
            query="account access reset",
            top_k=5,
            document_ids=(
                _DOCUMENT_ID,
                _DOCUMENT_ID,
            ),
        )


def test_safe_input_omits_raw_query() -> None:
    projection = project_search_knowledge_safe_input(_arguments())

    assert "query" not in projection
    assert projection["query_length"] == 20
    assert projection["query_sha256"] == (
        "77ede1652621f829c7961ea6d68c958f3775d7816de6d327518bd640b2265b83"
    )
    assert projection["top_k"] == 5
    assert projection["document_ids"] == [str(_DOCUMENT_ID)]


async def test_safe_output_omits_content() -> None:
    service = StubKnowledgeSearch(_result())
    output = await SearchKnowledgeToolHandler(service).execute(
        _context(),
        _arguments(),
    )

    assert isinstance(output, SearchKnowledgeOutput)

    projection = project_search_knowledge_safe_output(output)
    serialized_projection = str(projection)

    assert projection["result_count"] == 1
    assert projection["searched_version_count"] == 1
    assert str(_CHUNK_ID) in serialized_projection

    evidence_projection = projection["evidence"]
    assert isinstance(evidence_projection, list)
    assert len(evidence_projection) == 1
    first_evidence = evidence_projection[0]
    assert isinstance(first_evidence, dict)
    assert "content" not in first_evidence
    assert "document_title" not in first_evidence
    assert "Verify customer identity" not in serialized_projection


async def test_normalizes_embedding_dependency_failure() -> None:
    service = StubKnowledgeSearch(EmbeddingTimeoutError())
    handler = SearchKnowledgeToolHandler(service)

    with pytest.raises(
        ToolDependencyUnavailableError,
        match="dependency is unavailable",
    ):
        await handler.execute(
            _context(),
            _arguments(),
        )


def test_binding_has_exact_read_only_policy() -> None:
    service = StubKnowledgeSearch(_result())

    binding = create_search_knowledge_binding(service=service)

    assert binding.definition.name == (SEARCH_KNOWLEDGE_TOOL_NAME)
    assert binding.definition.version == (SEARCH_KNOWLEDGE_TOOL_VERSION)
    assert binding.definition.input_schema is (SearchKnowledgeInput)
    assert binding.definition.output_schema is (SearchKnowledgeOutput)
    assert binding.definition.safety_level is (ToolSafetyLevel.READ_ONLY)
    assert binding.definition.failure_policy is (ToolFailurePolicy.RETRY_AGENT_RUN)
    assert binding.definition.timeout_seconds == 15


async def test_handler_requires_trusted_context() -> None:
    handler = SearchKnowledgeToolHandler(StubKnowledgeSearch(_result()))

    with pytest.raises(
        TypeError,
        match="requires ToolExecutionContext",
    ):
        await handler.execute(
            object(),
            _arguments(),
        )
