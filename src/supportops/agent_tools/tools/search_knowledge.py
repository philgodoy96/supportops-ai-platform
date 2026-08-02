"""Controlled workspace-scoped knowledge-search tool."""

import hashlib
import json
from collections.abc import Mapping
from typing import Annotated, Protocol
from uuid import UUID, uuid5

from pydantic import (
    Field,
    JsonValue,
    StringConstraints,
    field_validator,
)

from supportops.agent_tools.application.bindings import (
    ExecutableToolBinding,
)
from supportops.agent_tools.application.execution import (
    ToolExecutionContext,
)
from supportops.agent_tools.domain.contracts import (
    StrictToolSchema,
    ToolAuditPolicy,
    ToolDefinition,
    ToolFailurePolicy,
    ToolSafetyLevel,
)
from supportops.agent_tools.domain.errors import (
    ToolDependencyUnavailableError,
)
from supportops.ai.embeddings.errors import EmbeddingError
from supportops.knowledge_index.vector_store.contracts import (
    KnowledgeVectorStoreError,
)
from supportops.knowledge_retrieval.contracts import (
    KnowledgeSearchRequest,
    KnowledgeSearchResult,
)

SEARCH_KNOWLEDGE_TOOL_NAME = "search_knowledge"
SEARCH_KNOWLEDGE_TOOL_VERSION = 1

SEARCH_KNOWLEDGE_MAX_RESULTS = 5
SEARCH_KNOWLEDGE_MAX_DOCUMENT_FILTERS = 20
SEARCH_KNOWLEDGE_MAX_QUERY_LENGTH = 2_000
SEARCH_KNOWLEDGE_MAX_CONTENT_LENGTH = 12_000

_RETRIEVAL_QUERY_NAMESPACE = UUID("764f4485-29bb-48b4-bc69-96570d524f7b")

QueryText = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        min_length=1,
        max_length=SEARCH_KNOWLEDGE_MAX_QUERY_LENGTH,
    ),
]
EvidenceContent = Annotated[
    str,
    StringConstraints(
        min_length=1,
        max_length=SEARCH_KNOWLEDGE_MAX_CONTENT_LENGTH,
    ),
]
BoundedText = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        min_length=1,
        max_length=255,
    ),
]
ContentSha256 = Annotated[
    str,
    StringConstraints(
        pattern=r"^[0-9a-f]{64}$",
    ),
]


class SearchKnowledgeInput(StrictToolSchema):
    """Strict model-visible knowledge-search arguments."""

    query: QueryText
    top_k: Annotated[
        int,
        Field(
            strict=True,
            ge=1,
            le=SEARCH_KNOWLEDGE_MAX_RESULTS,
        ),
    ]
    document_ids: Annotated[
        tuple[UUID, ...] | None,
        Field(max_length=SEARCH_KNOWLEDGE_MAX_DOCUMENT_FILTERS),
    ]

    @field_validator("document_ids")
    @classmethod
    def normalize_document_ids(
        cls,
        value: tuple[UUID, ...] | None,
    ) -> tuple[UUID, ...] | None:
        """Reject duplicates and establish deterministic ordering."""

        if value is None:
            return None

        if len(set(value)) != len(value):
            raise ValueError("document_ids must not contain duplicates.")

        return tuple(
            sorted(
                value,
                key=str,
            )
        )


class SearchKnowledgeEvidence(StrictToolSchema):
    """One authoritative retrieved evidence item."""

    rank: Annotated[
        int,
        Field(
            strict=True,
            ge=1,
            le=SEARCH_KNOWLEDGE_MAX_RESULTS,
        ),
    ]
    score: float
    content: EvidenceContent
    content_sha256: ContentSha256
    token_count: Annotated[
        int,
        Field(
            strict=True,
            ge=1,
        ),
    ]
    document_id: UUID
    document_title: BoundedText
    document_external_reference: BoundedText | None
    document_version_id: UUID
    version_number: Annotated[
        int,
        Field(
            strict=True,
            ge=1,
        ),
    ]
    chunk_id: UUID
    chunk_ordinal: Annotated[
        int,
        Field(
            strict=True,
            ge=0,
        ),
    ]
    section_path: tuple[str, ...]
    media_type: BoundedText


class SearchKnowledgeOutput(StrictToolSchema):
    """Bounded authoritative output returned to the graph."""

    retrieval_query_id: UUID
    searched_version_count: Annotated[
        int,
        Field(
            strict=True,
            ge=0,
        ),
    ]
    evidence: Annotated[
        tuple[SearchKnowledgeEvidence, ...],
        Field(max_length=SEARCH_KNOWLEDGE_MAX_RESULTS),
    ]


class KnowledgeSearchService(Protocol):
    """Application service consumed by the controlled tool."""

    async def execute(
        self,
        request: KnowledgeSearchRequest,
    ) -> KnowledgeSearchResult:
        """Retrieve authoritative workspace-scoped evidence."""

        ...


class SearchKnowledgeToolHandler:
    """Execute controlled knowledge retrieval through the existing service."""

    def __init__(
        self,
        service: KnowledgeSearchService,
    ) -> None:
        self._service = service

    async def execute(
        self,
        context: object,
        arguments: StrictToolSchema,
    ) -> object:
        """Search authoritative knowledge within the trusted workspace."""

        if not isinstance(context, ToolExecutionContext):
            raise TypeError("search_knowledge requires ToolExecutionContext.")

        if not isinstance(arguments, SearchKnowledgeInput):
            raise TypeError("search_knowledge requires SearchKnowledgeInput.")

        request = KnowledgeSearchRequest(
            workspace_id=context.workspace_id,
            query=arguments.query,
            top_k=arguments.top_k,
            document_ids=arguments.document_ids or (),
        )

        try:
            result = await self._service.execute(request)
        except (
            EmbeddingError,
            KnowledgeVectorStoreError,
        ) as exc:
            raise ToolDependencyUnavailableError() from exc

        return SearchKnowledgeOutput(
            retrieval_query_id=_create_retrieval_query_id(
                context=context,
                arguments=arguments,
            ),
            searched_version_count=(result.searched_version_count),
            evidence=tuple(
                SearchKnowledgeEvidence(
                    rank=evidence.rank,
                    score=evidence.score,
                    content=evidence.content,
                    content_sha256=evidence.content_sha256,
                    token_count=evidence.token_count,
                    document_id=(evidence.citation.document_id),
                    document_title=(evidence.citation.document_title),
                    document_external_reference=(evidence.citation.document_external_reference),
                    document_version_id=(evidence.citation.document_version_id),
                    version_number=(evidence.citation.version_number),
                    chunk_id=evidence.citation.chunk_id,
                    chunk_ordinal=evidence.citation.ordinal,
                    section_path=(evidence.citation.section_path),
                    media_type=(evidence.citation.media_type.value),
                )
                for evidence in result.evidence
            ),
        )


def create_search_knowledge_binding(
    *,
    service: KnowledgeSearchService,
    timeout_seconds: float = 15,
) -> ExecutableToolBinding:
    """Create the immutable search_knowledge runtime binding."""

    return ExecutableToolBinding(
        definition=ToolDefinition(
            name=SEARCH_KNOWLEDGE_TOOL_NAME,
            version=SEARCH_KNOWLEDGE_TOOL_VERSION,
            description=(
                "Search active workspace-scoped support runbooks and return authoritative evidence."
            ),
            input_schema=SearchKnowledgeInput,
            output_schema=SearchKnowledgeOutput,
            safety_level=ToolSafetyLevel.READ_ONLY,
            timeout_seconds=timeout_seconds,
            failure_policy=(ToolFailurePolicy.RETRY_AGENT_RUN),
            audit_policy=ToolAuditPolicy.SAFE_PROJECTION,
        ),
        handler=SearchKnowledgeToolHandler(service),
        safe_input_projector=(project_search_knowledge_safe_input),
        safe_output_projector=(project_search_knowledge_safe_output),
    )


def project_search_knowledge_safe_input(
    value: StrictToolSchema,
) -> Mapping[str, JsonValue]:
    """Project arguments without storing the raw query text."""

    if not isinstance(value, SearchKnowledgeInput):
        raise TypeError("Expected SearchKnowledgeInput.")

    encoded_query = value.query.encode("utf-8")

    return {
        "query_sha256": hashlib.sha256(encoded_query).hexdigest(),
        "query_length": len(value.query),
        "top_k": value.top_k,
        "document_ids": (
            None
            if value.document_ids is None
            else [str(document_id) for document_id in value.document_ids]
        ),
    }


def project_search_knowledge_safe_output(
    value: StrictToolSchema,
) -> Mapping[str, JsonValue]:
    """Project provenance without storing retrieved content."""

    if not isinstance(value, SearchKnowledgeOutput):
        raise TypeError("Expected SearchKnowledgeOutput.")

    return {
        "retrieval_query_id": str(value.retrieval_query_id),
        "searched_version_count": (value.searched_version_count),
        "result_count": len(value.evidence),
        "evidence": [
            {
                "rank": evidence.rank,
                "score": evidence.score,
                "document_id": str(evidence.document_id),
                "document_version_id": str(evidence.document_version_id),
                "chunk_id": str(evidence.chunk_id),
                "chunk_ordinal": (evidence.chunk_ordinal),
                "content_sha256": (evidence.content_sha256),
            }
            for evidence in value.evidence
        ],
    }


def _create_retrieval_query_id(
    *,
    context: ToolExecutionContext,
    arguments: SearchKnowledgeInput,
) -> UUID:
    payload = json.dumps(
        {
            "agent_run_attempt_id": str(context.agent_run_attempt_id),
            "document_ids": (
                None
                if arguments.document_ids is None
                else [str(document_id) for document_id in arguments.document_ids]
            ),
            "query": arguments.query,
            "top_k": arguments.top_k,
            "workspace_id": str(context.workspace_id),
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )

    return uuid5(
        _RETRIEVAL_QUERY_NAMESPACE,
        payload,
    )
