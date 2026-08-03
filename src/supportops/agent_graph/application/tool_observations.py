"""Durable reconstruction of controlled tool observations."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import isfinite
from typing import (
    Annotated,
    Literal,
    cast,
)
from uuid import UUID

from pydantic import (
    Field,
    JsonValue,
    StringConstraints,
)

from supportops.agent_graph.application.tool_audit_schemas import (
    PersistedSearchKnowledgeEvidence,
    PersistedSearchKnowledgeOutput,
    PersistedServiceStatusOutput,
    PersistedToolAuditOutputError,
    parse_persisted_search_knowledge_output,
    parse_persisted_service_status_output,
)
from supportops.agent_graph.domain.state import (
    ControlledSupportGraphStateSnapshot,
)
from supportops.agent_tools.application.execution import (
    ToolExecutionContext,
)
from supportops.agent_tools.application.queries import (
    AgentToolCallLookup,
    AgentToolCallQueryRepository,
)
from supportops.agent_tools.domain.audit import (
    AgentToolCall,
    AgentToolCallStatus,
)
from supportops.agent_tools.domain.contracts import (
    StrictToolSchema,
)
from supportops.agent_tools.tools.search_knowledge import (
    SEARCH_KNOWLEDGE_MAX_RESULTS,
    SEARCH_KNOWLEDGE_TOOL_NAME,
    SEARCH_KNOWLEDGE_TOOL_VERSION,
)
from supportops.agent_tools.tools.service_status import (
    LOOKUP_SERVICE_STATUS_TOOL_NAME,
    LOOKUP_SERVICE_STATUS_TOOL_VERSION,
)
from supportops.core.transactions import TransactionManager
from supportops.knowledge_retrieval.contracts import (
    KnowledgeChunkHydrator,
)
from supportops.modules.knowledge_documents.domain.models import (
    DocumentChunk,
)

ContentSha256 = Annotated[
    str,
    StringConstraints(
        pattern=r"^[0-9a-f]{64}$",
    ),
]


class ControlledToolObservationError(RuntimeError):
    """Raised when durable tool state cannot be reconstructed."""

    error_code = "tool_observation_reconstruction_failed"
    retryable = False


class ReconstructedKnowledgeEvidence(StrictToolSchema):
    """Authoritative content reconstructed for recommendation context."""

    rank: Annotated[
        int,
        Field(
            strict=True,
            ge=1,
            le=SEARCH_KNOWLEDGE_MAX_RESULTS,
        ),
    ]
    score: float
    content: str
    content_sha256: ContentSha256
    token_count: Annotated[
        int,
        Field(
            strict=True,
            ge=1,
        ),
    ]
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
    section_path: tuple[str, ...]


class ReconstructedSearchKnowledgeOutput(StrictToolSchema):
    """Prompt-ready output reconstructed from audit and PostgreSQL."""

    retrieval_query_id: UUID
    searched_version_count: Annotated[
        int,
        Field(
            strict=True,
            ge=0,
        ),
    ]
    evidence: Annotated[
        tuple[ReconstructedKnowledgeEvidence, ...],
        Field(
            max_length=SEARCH_KNOWLEDGE_MAX_RESULTS,
        ),
    ]


class SearchKnowledgePromptObservation(StrictToolSchema):
    """One prompt-ready successful knowledge-search observation."""

    sequence: Annotated[
        int,
        Field(
            strict=True,
            ge=1,
        ),
    ]
    tool_name: Literal["search_knowledge"]
    tool_version: Literal[1]
    status: Literal["succeeded"]
    output: ReconstructedSearchKnowledgeOutput


class ServiceStatusPromptObservation(StrictToolSchema):
    """One prompt-ready successful service-status observation."""

    sequence: Annotated[
        int,
        Field(
            strict=True,
            ge=1,
        ),
    ]
    tool_name: Literal["lookup_service_status"]
    tool_version: Literal[1]
    status: Literal["succeeded"]
    output: PersistedServiceStatusOutput


type PromptToolObservation = SearchKnowledgePromptObservation | ServiceStatusPromptObservation


@dataclass(frozen=True, slots=True)
class RecommendationCitationSource:
    """Durable evidence provenance used to create final citations."""

    retrieval_query_id: UUID
    retrieval_rank: int
    retrieval_score: float
    document_id: UUID
    document_version_id: UUID
    chunk_id: UUID

    def __post_init__(self) -> None:
        identifiers = (
            self.retrieval_query_id,
            self.document_id,
            self.document_version_id,
            self.chunk_id,
        )

        if not all(isinstance(identifier, UUID) for identifier in identifiers):
            raise TypeError("Citation-source identifiers must be UUID values.")

        if type(self.retrieval_rank) is not int:
            raise TypeError("retrieval_rank must be an integer.")

        if self.retrieval_rank < 0:
            raise ValueError("retrieval_rank must be non-negative.")

        if not isfinite(self.retrieval_score):
            raise ValueError("retrieval_score must be finite.")


@dataclass(frozen=True, slots=True)
class ControlledToolObservationBundle:
    """Reconstructed observations and deterministic citation sources."""

    observations: tuple[PromptToolObservation, ...]
    citation_sources: tuple[
        RecommendationCitationSource,
        ...,
    ]

    def to_prompt_observations(
        self,
    ) -> tuple[Mapping[str, JsonValue], ...]:
        """Return fresh JSON-compatible prompt projections."""

        return tuple(
            cast(
                Mapping[str, JsonValue],
                observation.model_dump(mode="json"),
            )
            for observation in self.observations
        )


class ControlledToolObservationAssembler:
    """Reconstruct durable observations without process memory."""

    def __init__(
        self,
        *,
        transaction_manager: TransactionManager,
        tool_call_repository: AgentToolCallQueryRepository,
        chunk_hydrator: KnowledgeChunkHydrator,
    ) -> None:
        self._transaction_manager = transaction_manager
        self._tool_call_repository = tool_call_repository
        self._chunk_hydrator = chunk_hydrator

    async def assemble(
        self,
        *,
        state: ControlledSupportGraphStateSnapshot,
        context: ToolExecutionContext,
    ) -> ControlledToolObservationBundle:
        """Load and reconstruct every checkpointed tool outcome."""

        _validate_context_ownership(
            state=state,
            context=context,
        )

        if state.current_error_code is not None:
            raise ControlledToolObservationError(
                "Tool observations cannot be reconstructed after a graph error."
            )

        if state.tool_call_count == 0:
            return ControlledToolObservationBundle(
                observations=(),
                citation_sources=(),
            )

        async with self._transaction_manager.transaction():
            audits = await self._load_ordered_audits(
                state=state,
                context=context,
            )
            parsed_outputs = _parse_successful_outputs(audits)
            requested_chunk_ids = _collect_requested_chunk_ids(parsed_outputs)
            chunks = (
                tuple(
                    await self._chunk_hydrator.hydrate(
                        workspace_id=context.workspace_id,
                        chunk_ids=requested_chunk_ids,
                    )
                )
                if requested_chunk_ids
                else ()
            )

        chunks_by_id = _index_authoritative_chunks(
            requested_chunk_ids=requested_chunk_ids,
            chunks=chunks,
        )

        return _build_observation_bundle(
            audits=audits,
            parsed_outputs=parsed_outputs,
            chunks_by_id=chunks_by_id,
        )

    async def _load_ordered_audits(
        self,
        *,
        state: ControlledSupportGraphStateSnapshot,
        context: ToolExecutionContext,
    ) -> tuple[AgentToolCall, ...]:
        audits: list[AgentToolCall] = []

        for sequence in range(
            1,
            state.tool_call_count + 1,
        ):
            audit = await self._tool_call_repository.get_by_proposal_attempt_sequence(
                AgentToolCallLookup(
                    workspace_id=context.workspace_id,
                    ticket_id=context.ticket_id,
                    agent_run_id=context.agent_run_id,
                    proposed_by_agent_run_attempt_id=(context.agent_run_attempt_id),
                    sequence=sequence,
                )
            )

            if audit is None:
                raise ControlledToolObservationError("A checkpointed tool-call audit is missing.")

            expected_index = sequence - 1

            if audit.id != state.tool_call_ids[expected_index]:
                raise ControlledToolObservationError(
                    "A persisted tool-call ID does not match checkpoint state."
                )

            if audit.input_fingerprint != state.seen_tool_call_fingerprints[expected_index]:
                raise ControlledToolObservationError(
                    "A persisted tool-call fingerprint does not match checkpoint state."
                )

            audits.append(audit)

        return tuple(audits)


def _parse_successful_outputs(
    audits: tuple[AgentToolCall, ...],
) -> tuple[
    PersistedSearchKnowledgeOutput | PersistedServiceStatusOutput,
    ...,
]:
    outputs: list[PersistedSearchKnowledgeOutput | PersistedServiceStatusOutput] = []

    for audit in audits:
        if audit.status is not AgentToolCallStatus.SUCCEEDED:
            raise ControlledToolObservationError(
                "Prompt observations require successful persisted tool calls."
            )

        if audit.safe_output is None:
            raise ControlledToolObservationError(
                "A successful persisted tool call is missing safe output."
            )

        try:
            if (
                audit.tool_name == SEARCH_KNOWLEDGE_TOOL_NAME
                and audit.tool_version == SEARCH_KNOWLEDGE_TOOL_VERSION
            ):
                outputs.append(parse_persisted_search_knowledge_output(audit.safe_output))
                continue

            if (
                audit.tool_name == LOOKUP_SERVICE_STATUS_TOOL_NAME
                and audit.tool_version == LOOKUP_SERVICE_STATUS_TOOL_VERSION
            ):
                outputs.append(parse_persisted_service_status_output(audit.safe_output))
                continue
        except PersistedToolAuditOutputError as exc:
            raise ControlledToolObservationError(
                "A persisted tool-call output cannot be reconstructed."
            ) from exc

        raise ControlledToolObservationError(
            "A persisted audit references an unsupported controlled tool identity."
        )

    return tuple(outputs)


def _collect_requested_chunk_ids(
    outputs: tuple[
        PersistedSearchKnowledgeOutput | PersistedServiceStatusOutput,
        ...,
    ],
) -> tuple[UUID, ...]:
    ordered_ids: list[UUID] = []
    known_ids: set[UUID] = set()

    for output in outputs:
        if not isinstance(
            output,
            PersistedSearchKnowledgeOutput,
        ):
            continue

        for evidence in output.evidence:
            if evidence.chunk_id in known_ids:
                continue

            ordered_ids.append(evidence.chunk_id)
            known_ids.add(evidence.chunk_id)

    return tuple(ordered_ids)


def _index_authoritative_chunks(
    *,
    requested_chunk_ids: tuple[UUID, ...],
    chunks: Sequence[DocumentChunk],
) -> dict[UUID, DocumentChunk]:
    chunks_by_id: dict[UUID, DocumentChunk] = {}

    for chunk in chunks:
        if chunk.id in chunks_by_id:
            raise ControlledToolObservationError("The chunk hydrator returned duplicate chunks.")

        chunks_by_id[chunk.id] = chunk

    requested_ids = set(requested_chunk_ids)
    returned_ids = set(chunks_by_id)

    if returned_ids != requested_ids:
        raise ControlledToolObservationError(
            "The chunk hydrator did not return the exact requested authoritative chunk set."
        )

    return chunks_by_id


def _build_observation_bundle(
    *,
    audits: tuple[AgentToolCall, ...],
    parsed_outputs: tuple[
        PersistedSearchKnowledgeOutput | PersistedServiceStatusOutput,
        ...,
    ],
    chunks_by_id: Mapping[UUID, DocumentChunk],
) -> ControlledToolObservationBundle:
    observations: list[PromptToolObservation] = []
    citation_sources: list[RecommendationCitationSource] = []
    cited_chunk_ids: set[UUID] = set()

    for audit, output in zip(
        audits,
        parsed_outputs,
        strict=True,
    ):
        if isinstance(
            output,
            PersistedSearchKnowledgeOutput,
        ):
            evidence_items = tuple(
                _reconstruct_evidence(
                    persisted=evidence,
                    chunk=chunks_by_id[evidence.chunk_id],
                )
                for evidence in output.evidence
            )
            observations.append(
                SearchKnowledgePromptObservation(
                    sequence=audit.sequence,
                    tool_name=SEARCH_KNOWLEDGE_TOOL_NAME,
                    tool_version=(SEARCH_KNOWLEDGE_TOOL_VERSION),
                    status="succeeded",
                    output=(
                        ReconstructedSearchKnowledgeOutput(
                            retrieval_query_id=(output.retrieval_query_id),
                            searched_version_count=(output.searched_version_count),
                            evidence=evidence_items,
                        )
                    ),
                )
            )

            for evidence in output.evidence:
                if evidence.chunk_id in cited_chunk_ids:
                    continue

                citation_sources.append(
                    RecommendationCitationSource(
                        retrieval_query_id=(output.retrieval_query_id),
                        retrieval_rank=(evidence.rank - 1),
                        retrieval_score=evidence.score,
                        document_id=evidence.document_id,
                        document_version_id=(evidence.document_version_id),
                        chunk_id=evidence.chunk_id,
                    )
                )
                cited_chunk_ids.add(evidence.chunk_id)

            continue

        observations.append(
            ServiceStatusPromptObservation(
                sequence=audit.sequence,
                tool_name=(LOOKUP_SERVICE_STATUS_TOOL_NAME),
                tool_version=(LOOKUP_SERVICE_STATUS_TOOL_VERSION),
                status="succeeded",
                output=output,
            )
        )

    return ControlledToolObservationBundle(
        observations=tuple(observations),
        citation_sources=tuple(citation_sources),
    )


def _reconstruct_evidence(
    *,
    persisted: PersistedSearchKnowledgeEvidence,
    chunk: DocumentChunk,
) -> ReconstructedKnowledgeEvidence:
    ownership_values = (
        (
            chunk.document_id,
            persisted.document_id,
            "document",
        ),
        (
            chunk.document_version_id,
            persisted.document_version_id,
            "document version",
        ),
        (
            chunk.id,
            persisted.chunk_id,
            "chunk",
        ),
        (
            chunk.ordinal,
            persisted.chunk_ordinal,
            "chunk ordinal",
        ),
        (
            chunk.content_sha256,
            persisted.content_sha256,
            "content hash",
        ),
    )

    for actual, expected, resource_name in ownership_values:
        if actual != expected:
            raise ControlledToolObservationError(
                f"Authoritative {resource_name} does not match the persisted retrieval audit."
            )

    return ReconstructedKnowledgeEvidence(
        rank=persisted.rank,
        score=persisted.score,
        content=chunk.content,
        content_sha256=chunk.content_sha256,
        token_count=chunk.token_count,
        document_id=chunk.document_id,
        document_version_id=chunk.document_version_id,
        chunk_id=chunk.id,
        chunk_ordinal=chunk.ordinal,
        section_path=chunk.section_path,
    )


def _validate_context_ownership(
    *,
    state: ControlledSupportGraphStateSnapshot,
    context: ToolExecutionContext,
) -> None:
    ownership_values = (
        (
            context.workspace_id,
            state.workspace_id,
            "workspace",
        ),
        (
            context.ticket_id,
            state.ticket_id,
            "ticket",
        ),
        (
            context.agent_run_id,
            state.agent_run_id,
            "AgentRun",
        ),
    )

    for actual, expected, resource_name in ownership_values:
        if actual != expected:
            raise ControlledToolObservationError(
                f"Observation {resource_name} ownership does not match graph state."
            )
