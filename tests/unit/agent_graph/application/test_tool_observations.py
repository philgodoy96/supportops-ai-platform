"""Unit tests for durable controlled-tool observations."""

from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import cast
from uuid import UUID

import pytest

from supportops.agent_graph.application.tool_observations import (
    ControlledToolObservationAssembler,
    ControlledToolObservationError,
    SearchKnowledgePromptObservation,
    ServiceStatusPromptObservation,
)
from supportops.agent_graph.domain.state import (
    CONTROLLED_SUPPORT_GRAPH_VERSION,
    CONTROLLED_SUPPORT_STATE_SCHEMA_VERSION,
    CONTROLLED_SUPPORT_WORKFLOW_NAME,
    CONTROLLED_SUPPORT_WORKFLOW_VERSION,
    ControlledSupportGraphStateSnapshot,
)
from supportops.agent_tools.application.execution import (
    ToolExecutionContext,
)
from supportops.agent_tools.application.queries import (
    AgentToolCallLookup,
)
from supportops.agent_tools.domain.audit import (
    AgentToolCall,
    AgentToolCallStatus,
)
from supportops.agent_tools.domain.contracts import (
    ToolSafetyLevel,
)
from supportops.ai.schemas.ticket_classification import (
    TicketCategory,
    TicketIntent,
    TicketSentiment,
    TicketUrgency,
)
from supportops.modules.knowledge_documents.domain.models import (
    DocumentChunk,
)

_WORKSPACE_ID = UUID("10000000-0000-4000-8000-000000000001")
_TICKET_ID = UUID("20000000-0000-4000-8000-000000000002")
_AGENT_RUN_ID = UUID("30000000-0000-4000-8000-000000000003")
_ATTEMPT_ID = UUID("40000000-0000-4000-8000-000000000004")
_CLASSIFICATION_ID = UUID("50000000-0000-4000-8000-000000000005")
_TOOL_CALL_ID = UUID("60000000-0000-4000-8000-000000000006")
_RETRIEVAL_QUERY_ID = UUID("70000000-0000-4000-8000-000000000007")
_DOCUMENT_ID = UUID("80000000-0000-4000-8000-000000000008")
_DOCUMENT_VERSION_ID = UUID("90000000-0000-4000-8000-000000000009")
_CHUNK_ID = UUID("a0000000-0000-4000-8000-000000000010")

_NOW = datetime(
    2026,
    8,
    2,
    18,
    0,
    tzinfo=UTC,
)
_CONTENT = "Verify the customer identity before starting the documented account recovery procedure."
_CONTENT_HASH = "b8cb74c71f6ed9ce17f85e3a671ded964b03e93cf1de69efca02064d6e475906"


class RecordingTransactionManager:
    """Record one non-nested read transaction."""

    def __init__(self) -> None:
        self.active = False
        self.enter_count = 0

    @asynccontextmanager
    async def transaction(
        self,
    ) -> AsyncIterator[None]:
        assert self.active is False

        self.active = True
        self.enter_count += 1

        try:
            yield
        finally:
            self.active = False


class StubToolCallRepository:
    """Return configured audits by exact sequence."""

    def __init__(
        self,
        *,
        transaction_manager: RecordingTransactionManager,
        audits: dict[int, AgentToolCall],
    ) -> None:
        self._transaction_manager = transaction_manager
        self._audits = audits
        self.queries: list[AgentToolCallLookup] = []

    async def get_by_proposal_attempt_sequence(
        self,
        query: AgentToolCallLookup,
    ) -> AgentToolCall | None:
        assert self._transaction_manager.active is True
        self.queries.append(query)

        return self._audits.get(query.sequence)

    async def get_sensitive_by_identity(
        self,
        query: object,
    ) -> AgentToolCall | None:
        del query
        return None


@dataclass(frozen=True, slots=True)
class StubChunk:
    """Authoritative chunk projection required by the assembler."""

    id: UUID
    workspace_id: UUID
    document_id: UUID
    document_version_id: UUID
    ordinal: int
    content: str
    content_sha256: str
    token_count: int
    section_path: tuple[str, ...]


class StubChunkHydrator:
    """Return configured chunks inside the read transaction."""

    def __init__(
        self,
        *,
        transaction_manager: RecordingTransactionManager,
        chunks: tuple[StubChunk, ...],
    ) -> None:
        self._transaction_manager = transaction_manager
        self._chunks = chunks
        self.calls: list[tuple[UUID, tuple[UUID, ...]]] = []

    async def hydrate(
        self,
        *,
        workspace_id: UUID,
        chunk_ids: tuple[UUID, ...],
    ) -> Sequence[DocumentChunk]:
        assert self._transaction_manager.active is True
        self.calls.append(
            (
                workspace_id,
                chunk_ids,
            )
        )

        return tuple(cast(DocumentChunk, chunk) for chunk in self._chunks)


def _context() -> ToolExecutionContext:
    return ToolExecutionContext(
        workspace_id=_WORKSPACE_ID,
        ticket_id=_TICKET_ID,
        agent_run_id=_AGENT_RUN_ID,
        agent_run_attempt_id=_ATTEMPT_ID,
    )


def _state(
    *,
    audit: AgentToolCall | None,
    retrieval_query_ids: tuple[UUID, ...] = (),
    retrieved_chunk_ids: tuple[UUID, ...] = (),
    service_status_ids: tuple[UUID, ...] = (),
) -> ControlledSupportGraphStateSnapshot:
    tool_call_count = 0 if audit is None else 1

    return ControlledSupportGraphStateSnapshot(
        state_schema_version=(CONTROLLED_SUPPORT_STATE_SCHEMA_VERSION),
        workflow_name=CONTROLLED_SUPPORT_WORKFLOW_NAME,
        workflow_version=(CONTROLLED_SUPPORT_WORKFLOW_VERSION),
        graph_version=CONTROLLED_SUPPORT_GRAPH_VERSION,
        workspace_id=_WORKSPACE_ID,
        ticket_id=_TICKET_ID,
        agent_run_id=_AGENT_RUN_ID,
        classification_id=_CLASSIFICATION_ID,
        classification_category=(TicketCategory.ACCOUNT_ACCESS),
        classification_intent=(TicketIntent.REQUEST_ACCESS),
        classification_urgency=TicketUrgency.NORMAL,
        classification_sentiment=(TicketSentiment.NEUTRAL),
        classification_requires_human_review=False,
        classification_summary=("The customer needs documented account recovery guidance."),
        graph_step_count=0,
        decision_turn_count=tool_call_count,
        tool_call_count=tool_call_count,
        seen_tool_call_fingerprints=(() if audit is None else (audit.input_fingerprint,)),
        tool_call_ids=(() if audit is None else (audit.id,)),
        retrieval_query_ids=retrieval_query_ids,
        retrieved_chunk_ids=retrieved_chunk_ids,
        service_status_tool_call_ids=(service_status_ids),
        analysis_completion=None,
        recommendation_invocation_id=None,
        recommendation_id=None,
        current_error_code=None,
    )


def _search_audit() -> AgentToolCall:
    return AgentToolCall.create_terminal(
        tool_call_id=_TOOL_CALL_ID,
        workspace_id=_WORKSPACE_ID,
        ticket_id=_TICKET_ID,
        agent_run_id=_AGENT_RUN_ID,
        agent_run_attempt_id=_ATTEMPT_ID,
        sequence=1,
        provider_tool_call_id="provider-tool-call-1",
        tool_name="search_knowledge",
        tool_version=1,
        safety_level=ToolSafetyLevel.READ_ONLY,
        status=AgentToolCallStatus.SUCCEEDED,
        input_fingerprint="a" * 64,
        safe_input={
            "query_sha256": "b" * 64,
            "query_length": 20,
            "top_k": 5,
            "document_ids": None,
        },
        safe_output={
            "retrieval_query_id": str(_RETRIEVAL_QUERY_ID),
            "searched_version_count": 1,
            "result_count": 1,
            "evidence": [
                {
                    "rank": 1,
                    "score": 0.91,
                    "document_id": str(_DOCUMENT_ID),
                    "document_version_id": str(_DOCUMENT_VERSION_ID),
                    "chunk_id": str(_CHUNK_ID),
                    "chunk_ordinal": 0,
                    "content_sha256": _CONTENT_HASH,
                }
            ],
        },
        latency_ms=25,
        error_code=None,
        started_at=_NOW,
        finished_at=_NOW,
    )


def _status_audit() -> AgentToolCall:
    return AgentToolCall.create_terminal(
        tool_call_id=_TOOL_CALL_ID,
        workspace_id=_WORKSPACE_ID,
        ticket_id=_TICKET_ID,
        agent_run_id=_AGENT_RUN_ID,
        agent_run_attempt_id=_ATTEMPT_ID,
        sequence=1,
        provider_tool_call_id="provider-tool-call-1",
        tool_name="lookup_service_status",
        tool_version=1,
        safety_level=ToolSafetyLevel.READ_ONLY,
        status=AgentToolCallStatus.SUCCEEDED,
        input_fingerprint="c" * 64,
        safe_input={
            "service_name": "payments-api",
        },
        safe_output={
            "service_name": "payments-api",
            "status": "degraded",
            "incident_reference": "incident-local-001",
            "has_incident": True,
            "source": "deterministic_catalog",
        },
        latency_ms=1,
        error_code=None,
        started_at=_NOW,
        finished_at=_NOW,
    )


def _chunk(
    *,
    content_hash: str = _CONTENT_HASH,
) -> StubChunk:
    return StubChunk(
        id=_CHUNK_ID,
        workspace_id=_WORKSPACE_ID,
        document_id=_DOCUMENT_ID,
        document_version_id=_DOCUMENT_VERSION_ID,
        ordinal=0,
        content=_CONTENT,
        content_sha256=content_hash,
        token_count=13,
        section_path=("Account recovery",),
    )


def _assembler(
    *,
    audits: dict[int, AgentToolCall],
    chunks: tuple[StubChunk, ...],
) -> tuple[
    ControlledToolObservationAssembler,
    RecordingTransactionManager,
    StubToolCallRepository,
    StubChunkHydrator,
]:
    transaction_manager = RecordingTransactionManager()
    repository = StubToolCallRepository(
        transaction_manager=transaction_manager,
        audits=audits,
    )
    hydrator = StubChunkHydrator(
        transaction_manager=transaction_manager,
        chunks=chunks,
    )

    return (
        ControlledToolObservationAssembler(
            transaction_manager=transaction_manager,
            tool_call_repository=repository,
            chunk_hydrator=hydrator,
        ),
        transaction_manager,
        repository,
        hydrator,
    )


async def test_reconstructs_authoritative_knowledge_observation() -> None:
    audit = _search_audit()
    assembler, transaction_manager, repository, hydrator = _assembler(
        audits={
            1: audit,
        },
        chunks=(_chunk(),),
    )

    bundle = await assembler.assemble(
        state=_state(
            audit=audit,
            retrieval_query_ids=(_RETRIEVAL_QUERY_ID,),
            retrieved_chunk_ids=(_CHUNK_ID,),
        ),
        context=_context(),
    )

    assert transaction_manager.enter_count == 1
    assert len(repository.queries) == 1
    assert hydrator.calls == [
        (
            _WORKSPACE_ID,
            (_CHUNK_ID,),
        )
    ]

    assert len(bundle.observations) == 1
    observation = bundle.observations[0]

    assert isinstance(
        observation,
        SearchKnowledgePromptObservation,
    )
    assert observation.output.evidence[0].content == (_CONTENT)
    assert observation.output.evidence[0].content_sha256 == _CONTENT_HASH

    assert len(bundle.citation_sources) == 1
    citation = bundle.citation_sources[0]

    assert citation.retrieval_query_id == (_RETRIEVAL_QUERY_ID)
    assert citation.retrieval_rank == 0
    assert citation.retrieval_score == 0.91
    assert citation.chunk_id == _CHUNK_ID

    prompt_projection = bundle.to_prompt_observations()[0]

    assert prompt_projection["tool_name"] == ("search_knowledge")
    assert _CONTENT in str(prompt_projection)


async def test_reconstructs_service_status_without_hydration() -> None:
    audit = _status_audit()
    assembler, transaction_manager, _, hydrator = _assembler(
        audits={
            1: audit,
        },
        chunks=(),
    )

    bundle = await assembler.assemble(
        state=_state(
            audit=audit,
            service_status_ids=(audit.id,),
        ),
        context=_context(),
    )

    assert transaction_manager.enter_count == 1
    assert hydrator.calls == []
    assert bundle.citation_sources == ()

    observation = bundle.observations[0]

    assert isinstance(
        observation,
        ServiceStatusPromptObservation,
    )
    assert observation.output.service_name == ("payments-api")
    assert observation.output.status.value == ("degraded")


async def test_empty_tool_history_requires_no_transaction() -> None:
    assembler, transaction_manager, _, hydrator = _assembler(
        audits={},
        chunks=(),
    )

    bundle = await assembler.assemble(
        state=_state(audit=None),
        context=_context(),
    )

    assert bundle.observations == ()
    assert bundle.citation_sources == ()
    assert transaction_manager.enter_count == 0
    assert hydrator.calls == []


async def test_rejects_missing_checkpointed_audit() -> None:
    audit = _search_audit()
    assembler, _, _, _ = _assembler(
        audits={},
        chunks=(),
    )

    with pytest.raises(
        ControlledToolObservationError,
        match="audit is missing",
    ):
        await assembler.assemble(
            state=_state(
                audit=audit,
                retrieval_query_ids=(_RETRIEVAL_QUERY_ID,),
                retrieved_chunk_ids=(_CHUNK_ID,),
            ),
            context=_context(),
        )


async def test_rejects_checkpoint_audit_id_mismatch() -> None:
    audit = _search_audit()
    mismatched_state_audit = AgentToolCall.create_terminal(
        tool_call_id=UUID("b0000000-0000-4000-8000-000000000011"),
        workspace_id=audit.workspace_id,
        ticket_id=audit.ticket_id,
        agent_run_id=audit.agent_run_id,
        agent_run_attempt_id=audit.proposed_by_agent_run_attempt_id,
        sequence=audit.sequence,
        provider_tool_call_id=audit.provider_tool_call_id,
        tool_name=audit.tool_name,
        tool_version=audit.tool_version,
        safety_level=audit.safety_level,
        status=audit.status,
        input_fingerprint=audit.input_fingerprint,
        safe_input=audit.safe_input,
        safe_output=audit.safe_output,
        latency_ms=audit.latency_ms or 0,
        error_code=audit.error_code,
        started_at=audit.execution_started_at or audit.proposed_at,
        finished_at=audit.finished_at or audit.proposed_at,
    )
    assembler, _, _, _ = _assembler(
        audits={
            1: audit,
        },
        chunks=(_chunk(),),
    )

    with pytest.raises(
        ControlledToolObservationError,
        match="ID does not match",
    ):
        await assembler.assemble(
            state=_state(
                audit=mismatched_state_audit,
                retrieval_query_ids=(_RETRIEVAL_QUERY_ID,),
                retrieved_chunk_ids=(_CHUNK_ID,),
            ),
            context=_context(),
        )


async def test_rejects_missing_authoritative_chunk() -> None:
    audit = _search_audit()
    assembler, _, _, _ = _assembler(
        audits={
            1: audit,
        },
        chunks=(),
    )

    with pytest.raises(
        ControlledToolObservationError,
        match="exact requested",
    ):
        await assembler.assemble(
            state=_state(
                audit=audit,
                retrieval_query_ids=(_RETRIEVAL_QUERY_ID,),
                retrieved_chunk_ids=(_CHUNK_ID,),
            ),
            context=_context(),
        )


async def test_rejects_authoritative_hash_mismatch() -> None:
    audit = _search_audit()
    assembler, _, _, _ = _assembler(
        audits={
            1: audit,
        },
        chunks=(_chunk(content_hash="f" * 64),),
    )

    with pytest.raises(
        ControlledToolObservationError,
        match="content hash",
    ):
        await assembler.assemble(
            state=_state(
                audit=audit,
                retrieval_query_ids=(_RETRIEVAL_QUERY_ID,),
                retrieved_chunk_ids=(_CHUNK_ID,),
            ),
            context=_context(),
        )


async def test_context_ownership_must_match_state() -> None:
    audit = _search_audit()
    assembler, transaction_manager, _, _ = _assembler(
        audits={
            1: audit,
        },
        chunks=(_chunk(),),
    )
    context = ToolExecutionContext(
        workspace_id=UUID("c0000000-0000-4000-8000-000000000012"),
        ticket_id=_TICKET_ID,
        agent_run_id=_AGENT_RUN_ID,
        agent_run_attempt_id=_ATTEMPT_ID,
    )

    with pytest.raises(
        ControlledToolObservationError,
        match="workspace ownership",
    ):
        await assembler.assemble(
            state=_state(
                audit=audit,
                retrieval_query_ids=(_RETRIEVAL_QUERY_ID,),
                retrieved_chunk_ids=(_CHUNK_ID,),
            ),
            context=context,
        )

    assert transaction_manager.enter_count == 0
