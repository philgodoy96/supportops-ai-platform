"""HTTP schemas for workspace-scoped semantic knowledge search."""

from typing import Self
from uuid import UUID

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
)

from supportops.knowledge_retrieval.contracts import (
    DEFAULT_KNOWLEDGE_SEARCH_TOP_K,
    MAX_KNOWLEDGE_SEARCH_DOCUMENT_FILTERS,
    MAX_KNOWLEDGE_SEARCH_QUERY_LENGTH,
    MAX_KNOWLEDGE_SEARCH_TOP_K,
    KnowledgeCitation,
    KnowledgeEvidence,
    KnowledgeSearchRequest,
    KnowledgeSearchResult,
)


class KnowledgeSearchRequestBody(BaseModel):
    """Semantic knowledge query accepted by the HTTP API."""

    model_config = ConfigDict(extra="forbid")

    query: str = Field(
        min_length=1,
        max_length=MAX_KNOWLEDGE_SEARCH_QUERY_LENGTH,
    )
    top_k: int = Field(
        default=DEFAULT_KNOWLEDGE_SEARCH_TOP_K,
        ge=1,
        le=MAX_KNOWLEDGE_SEARCH_TOP_K,
    )
    document_ids: list[UUID] = Field(
        default_factory=list,
        max_length=MAX_KNOWLEDGE_SEARCH_DOCUMENT_FILTERS,
    )

    @field_validator(
        "query",
        mode="before",
    )
    @classmethod
    def normalize_query(
        cls,
        value: object,
    ) -> object:
        """Trim surrounding whitespace before length validation."""

        if isinstance(value, str):
            return value.strip()

        return value

    @field_validator("document_ids")
    @classmethod
    def validate_document_ids(
        cls,
        value: list[UUID],
    ) -> list[UUID]:
        """Reject ambiguous duplicate document filters."""

        if len(set(value)) != len(value):
            raise ValueError("document_ids must not contain duplicates.")

        return value

    def to_domain(
        self,
        *,
        workspace_id: UUID,
    ) -> KnowledgeSearchRequest:
        """Create the provider-independent application request."""

        return KnowledgeSearchRequest(
            workspace_id=workspace_id,
            query=self.query,
            top_k=self.top_k,
            document_ids=tuple(self.document_ids),
        )


class KnowledgeCitationResponse(BaseModel):
    """Stable citation metadata for one evidence item."""

    workspace_id: UUID
    document_id: UUID
    document_title: str
    document_external_reference: str | None
    document_version_id: UUID
    version_number: int
    chunk_id: UUID
    ordinal: int
    section_path: list[str]
    media_type: str

    @classmethod
    def from_domain(
        cls,
        citation: KnowledgeCitation,
    ) -> Self:
        """Create an HTTP citation from application evidence."""

        return cls(
            workspace_id=citation.workspace_id,
            document_id=citation.document_id,
            document_title=citation.document_title,
            document_external_reference=(citation.document_external_reference),
            document_version_id=(citation.document_version_id),
            version_number=citation.version_number,
            chunk_id=citation.chunk_id,
            ordinal=citation.ordinal,
            section_path=list(citation.section_path),
            media_type=citation.media_type.value,
        )


class KnowledgeEvidenceResponse(BaseModel):
    """Ranked authoritative content returned by semantic search."""

    rank: int
    score: float
    content: str
    content_sha256: str
    token_count: int
    citation: KnowledgeCitationResponse

    @classmethod
    def from_domain(
        cls,
        evidence: KnowledgeEvidence,
    ) -> Self:
        """Create an HTTP evidence representation."""

        return cls(
            rank=evidence.rank,
            score=evidence.score,
            content=evidence.content,
            content_sha256=(evidence.content_sha256),
            token_count=evidence.token_count,
            citation=(KnowledgeCitationResponse.from_domain(evidence.citation)),
        )


class KnowledgeSearchResponse(BaseModel):
    """Semantic retrieval response without generated answers."""

    workspace_id: UUID
    query: str
    top_k: int
    document_ids: list[UUID]
    searched_version_count: int
    evidence: list[KnowledgeEvidenceResponse]

    @classmethod
    def from_domain(
        cls,
        result: KnowledgeSearchResult,
    ) -> Self:
        """Create the stable HTTP response from a retrieval result."""

        return cls(
            workspace_id=(result.request.workspace_id),
            query=result.request.query,
            top_k=result.request.top_k,
            document_ids=list(result.request.document_ids),
            searched_version_count=(result.searched_version_count),
            evidence=[KnowledgeEvidenceResponse.from_domain(item) for item in result.evidence],
        )
