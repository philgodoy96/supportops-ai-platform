"""Strict schemas for persisted controlled-tool audit projections."""

from collections.abc import Mapping
from typing import (
    Annotated,
    Literal,
    Self,
)
from uuid import UUID

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    StrictBool,
    StringConstraints,
    ValidationError,
    model_validator,
)

from supportops.agent_tools.tools.search_knowledge import (
    SEARCH_KNOWLEDGE_MAX_RESULTS,
)
from supportops.agent_tools.tools.service_status import (
    ServiceOperationalStatus,
)

ContentSha256 = Annotated[
    str,
    StringConstraints(
        pattern=r"^[0-9a-f]{64}$",
    ),
]
ServiceName = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        to_lower=True,
        min_length=1,
        max_length=64,
        pattern=r"^[a-z][a-z0-9_-]*$",
    ),
]
IncidentReference = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        min_length=1,
        max_length=128,
    ),
]


class PersistedToolAuditOutputError(RuntimeError):
    """Raised when a safe persisted tool output is malformed."""

    error_code = "persisted_tool_audit_output_invalid"
    retryable = False


class PersistedSearchKnowledgeEvidence(BaseModel):
    """One evidence identity persisted without authoritative content."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        allow_inf_nan=False,
    )

    rank: Annotated[
        int,
        Field(
            strict=True,
            ge=1,
            le=SEARCH_KNOWLEDGE_MAX_RESULTS,
        ),
    ]
    score: float
    document_id: UUID
    document_version_id: UUID
    chunk_id: UUID
    chunk_ordinal: Annotated[
        int,
        Field(
            strict=True,
            ge=0,
        ),
    ]
    content_sha256: ContentSha256


class PersistedSearchKnowledgeOutput(BaseModel):
    """Safe persisted output for one knowledge-search tool call."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        allow_inf_nan=False,
    )

    retrieval_query_id: UUID
    searched_version_count: Annotated[
        int,
        Field(
            strict=True,
            ge=0,
        ),
    ]
    result_count: Annotated[
        int,
        Field(
            strict=True,
            ge=0,
            le=SEARCH_KNOWLEDGE_MAX_RESULTS,
        ),
    ]
    evidence: Annotated[
        tuple[PersistedSearchKnowledgeEvidence, ...],
        Field(
            max_length=SEARCH_KNOWLEDGE_MAX_RESULTS,
        ),
    ]

    @model_validator(mode="after")
    def validate_evidence_projection(
        self,
    ) -> Self:
        """Require deterministic rank, score and chunk identities."""

        if self.result_count != len(self.evidence):
            raise ValueError("Search audit result_count must match evidence.")

        expected_ranks = tuple(
            range(
                1,
                len(self.evidence) + 1,
            )
        )
        actual_ranks = tuple(item.rank for item in self.evidence)

        if actual_ranks != expected_ranks:
            raise ValueError("Search audit evidence ranks must be contiguous and one-based.")

        scores = tuple(item.score for item in self.evidence)

        if scores != tuple(
            sorted(
                scores,
                reverse=True,
            )
        ):
            raise ValueError("Search audit evidence must be ordered by descending score.")

        chunk_ids = tuple(item.chunk_id for item in self.evidence)

        if len(set(chunk_ids)) != len(chunk_ids):
            raise ValueError("Search audit evidence must not contain duplicate chunks.")

        return self


class PersistedServiceStatusOutput(BaseModel):
    """Safe persisted output for one service-status lookup."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    service_name: ServiceName
    status: ServiceOperationalStatus
    incident_reference: IncidentReference | None
    has_incident: StrictBool
    source: Literal["deterministic_catalog"]

    @model_validator(mode="after")
    def validate_incident_projection(
        self,
    ) -> Self:
        """Require the derived incident flag to remain consistent."""

        if self.has_incident != (self.incident_reference is not None):
            raise ValueError("Service-status audit incident fields conflict.")

        return self


def parse_persisted_search_knowledge_output(
    value: Mapping[str, JsonValue],
) -> PersistedSearchKnowledgeOutput:
    """Parse one safe persisted knowledge-search output."""

    try:
        return PersistedSearchKnowledgeOutput.model_validate(dict(value))
    except ValidationError as exc:
        raise PersistedToolAuditOutputError(
            "The persisted knowledge-search audit output is invalid."
        ) from exc


def parse_persisted_service_status_output(
    value: Mapping[str, JsonValue],
) -> PersistedServiceStatusOutput:
    """Parse one safe persisted service-status output."""

    try:
        return PersistedServiceStatusOutput.model_validate(dict(value))
    except ValidationError as exc:
        raise PersistedToolAuditOutputError(
            "The persisted service-status audit output is invalid."
        ) from exc
