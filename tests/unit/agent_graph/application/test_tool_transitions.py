"""Unit tests for persisted tool-call graph transitions."""

from datetime import UTC, datetime
from uuid import UUID

import pytest

from supportops.agent_graph.application.tool_transitions import (
    ControlledSupportToolStateError,
    ToolStateTransitionConflictError,
    record_persisted_tool_call,
)
from supportops.agent_graph.application.transitions import (
    attach_classification,
    reserve_decision_turn,
)
from supportops.agent_graph.domain.state import (
    ControlledSupportGraphStateSnapshot,
    create_initial_controlled_support_state,
    validate_controlled_support_state,
)
from supportops.agent_tools.domain.audit import (
    AgentToolCall,
    AgentToolCallStatus,
)
from supportops.agent_tools.domain.contracts import (
    ToolSafetyLevel,
)
from supportops.ai.schemas.ticket_classification import (
    TICKET_CLASSIFICATION_SCHEMA_VERSION,
    TicketCategory,
    TicketIntent,
    TicketSentiment,
    TicketUrgency,
)
from supportops.modules.ticket_classifications.domain.models import (
    TicketClassification,
)

_WORKSPACE_ID = UUID("10000000-0000-4000-8000-000000000001")
_TICKET_ID = UUID("20000000-0000-4000-8000-000000000002")
_AGENT_RUN_ID = UUID("30000000-0000-4000-8000-000000000003")
_ATTEMPT_ID = UUID("40000000-0000-4000-8000-000000000004")
_CLASSIFICATION_ID = UUID("50000000-0000-4000-8000-000000000005")
_CLASSIFICATION_INVOCATION_ID = UUID("60000000-0000-4000-8000-000000000006")
_TOOL_CALL_ID = UUID("70000000-0000-4000-8000-000000000007")
_RETRIEVAL_QUERY_ID = UUID("80000000-0000-4000-8000-000000000008")
_FIRST_CHUNK_ID = UUID("90000000-0000-4000-8000-000000000009")
_SECOND_CHUNK_ID = UUID("a0000000-0000-4000-8000-000000000010")

_STARTED_AT = datetime(
    2026,
    8,
    2,
    18,
    0,
    tzinfo=UTC,
)
_FINISHED_AT = _STARTED_AT


def _classification() -> TicketClassification:
    return TicketClassification.create(
        classification_id=_CLASSIFICATION_ID,
        workspace_id=_WORKSPACE_ID,
        ticket_id=_TICKET_ID,
        agent_run_id=_AGENT_RUN_ID,
        accepted_llm_invocation_id=(_CLASSIFICATION_INVOCATION_ID),
        category=TicketCategory.ACCOUNT_ACCESS,
        intent=TicketIntent.REQUEST_ACCESS,
        urgency=TicketUrgency.NORMAL,
        sentiment=TicketSentiment.NEUTRAL,
        requires_human_review=False,
        summary=("The customer needs account-access guidance."),
        schema_version=(TICKET_CLASSIFICATION_SCHEMA_VERSION),
        prompt_id="ticket-classification",
        prompt_version=1,
        prompt_content_hash="a" * 64,
        provider="mock",
        model="mock-model",
    )


def _classified_state(
    *,
    reserve_decision: bool,
) -> ControlledSupportGraphStateSnapshot:
    state = validate_controlled_support_state(
        create_initial_controlled_support_state(
            workspace_id=_WORKSPACE_ID,
            ticket_id=_TICKET_ID,
            agent_run_id=_AGENT_RUN_ID,
        )
    )
    state = attach_classification(
        state,
        _classification(),
    )

    if reserve_decision:
        return reserve_decision_turn(state)

    return state


def _search_audit(
    *,
    status: AgentToolCallStatus = (AgentToolCallStatus.SUCCEEDED),
    error_code: str | None = None,
) -> AgentToolCall:
    return AgentToolCall.create_terminal(
        tool_call_id=_TOOL_CALL_ID,
        workspace_id=_WORKSPACE_ID,
        ticket_id=_TICKET_ID,
        agent_run_id=_AGENT_RUN_ID,
        agent_run_attempt_id=_ATTEMPT_ID,
        sequence=1,
        provider_tool_call_id="provider-call-1",
        tool_name="search_knowledge",
        tool_version=1,
        safety_level=ToolSafetyLevel.READ_ONLY,
        status=status,
        input_fingerprint="b" * 64,
        safe_input={
            "query_sha256": "c" * 64,
            "query_length": 20,
            "top_k": 5,
            "document_ids": None,
        },
        safe_output=(
            {
                "retrieval_query_id": str(_RETRIEVAL_QUERY_ID),
                "searched_version_count": 1,
                "result_count": 2,
                "evidence": [
                    {
                        "rank": 1,
                        "score": 0.91,
                        "document_id": str(UUID("b0000000-0000-4000-8000-000000000011")),
                        "document_version_id": str(UUID("c0000000-0000-4000-8000-000000000012")),
                        "chunk_id": str(_FIRST_CHUNK_ID),
                        "chunk_ordinal": 0,
                        "content_sha256": "d" * 64,
                    },
                    {
                        "rank": 2,
                        "score": 0.81,
                        "document_id": str(UUID("b0000000-0000-4000-8000-000000000011")),
                        "document_version_id": str(UUID("c0000000-0000-4000-8000-000000000012")),
                        "chunk_id": str(_SECOND_CHUNK_ID),
                        "chunk_ordinal": 1,
                        "content_sha256": "e" * 64,
                    },
                ],
            }
            if status is AgentToolCallStatus.SUCCEEDED
            else None
        ),
        latency_ms=25,
        error_code=error_code,
        started_at=_STARTED_AT,
        finished_at=_FINISHED_AT,
    )


def _service_status_audit() -> AgentToolCall:
    return AgentToolCall.create_terminal(
        tool_call_id=_TOOL_CALL_ID,
        workspace_id=_WORKSPACE_ID,
        ticket_id=_TICKET_ID,
        agent_run_id=_AGENT_RUN_ID,
        agent_run_attempt_id=_ATTEMPT_ID,
        sequence=1,
        provider_tool_call_id="provider-call-1",
        tool_name="lookup_service_status",
        tool_version=1,
        safety_level=ToolSafetyLevel.READ_ONLY,
        status=AgentToolCallStatus.SUCCEEDED,
        input_fingerprint="f" * 64,
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
        started_at=_STARTED_AT,
        finished_at=_FINISHED_AT,
    )


def test_records_successful_knowledge_search() -> None:
    audit = _search_audit()

    state = record_persisted_tool_call(
        _classified_state(
            reserve_decision=True,
        ),
        audit,
        expected_attempt_id=_ATTEMPT_ID,
    )

    assert state.decision_turn_count == 1
    assert state.tool_call_count == 1
    assert state.tool_call_ids == (audit.id,)
    assert state.seen_tool_call_fingerprints == (audit.input_fingerprint,)
    assert state.retrieval_query_ids == (_RETRIEVAL_QUERY_ID,)
    assert state.retrieved_chunk_ids == (
        _FIRST_CHUNK_ID,
        _SECOND_CHUNK_ID,
    )
    assert state.current_error_code is None


def test_recovers_missing_decision_counter() -> None:
    state = record_persisted_tool_call(
        _classified_state(
            reserve_decision=False,
        ),
        _search_audit(),
        expected_attempt_id=_ATTEMPT_ID,
    )

    assert state.decision_turn_count == 1
    assert state.tool_call_count == 1


def test_records_service_status_reference() -> None:
    audit = _service_status_audit()

    state = record_persisted_tool_call(
        _classified_state(
            reserve_decision=True,
        ),
        audit,
        expected_attempt_id=_ATTEMPT_ID,
    )

    assert state.service_status_tool_call_ids == (audit.id,)
    assert state.retrieval_query_ids == ()


def test_unsuccessful_tool_records_stable_error() -> None:
    audit = _search_audit(
        status=AgentToolCallStatus.TIMED_OUT,
        error_code="tool_timeout",
    )

    state = record_persisted_tool_call(
        _classified_state(
            reserve_decision=True,
        ),
        audit,
        expected_attempt_id=_ATTEMPT_ID,
    )

    assert state.tool_call_count == 1
    assert state.tool_call_ids == (audit.id,)
    assert state.current_error_code == "tool_timeout"


def test_exact_transition_replay_is_idempotent() -> None:
    audit = _search_audit()
    state = record_persisted_tool_call(
        _classified_state(
            reserve_decision=True,
        ),
        audit,
        expected_attempt_id=_ATTEMPT_ID,
    )

    replayed = record_persisted_tool_call(
        state,
        audit,
        expected_attempt_id=_ATTEMPT_ID,
    )

    assert replayed is state


def test_rejects_sequence_conflict() -> None:
    audit = _search_audit()
    state = record_persisted_tool_call(
        _classified_state(
            reserve_decision=True,
        ),
        audit,
        expected_attempt_id=_ATTEMPT_ID,
    )
    conflicting = AgentToolCall.create_terminal(
        tool_call_id=UUID("d0000000-0000-4000-8000-000000000013"),
        workspace_id=audit.workspace_id,
        ticket_id=audit.ticket_id,
        agent_run_id=audit.agent_run_id,
        agent_run_attempt_id=audit.proposed_by_agent_run_attempt_id,
        sequence=1,
        provider_tool_call_id="provider-call-2",
        tool_name=audit.tool_name,
        tool_version=audit.tool_version,
        safety_level=audit.safety_level,
        status=audit.status,
        input_fingerprint="1" * 64,
        safe_input=audit.safe_input,
        safe_output=audit.safe_output,
        latency_ms=audit.latency_ms or 0,
        error_code=audit.error_code,
        started_at=audit.execution_started_at or audit.proposed_at,
        finished_at=audit.finished_at or audit.proposed_at,
    )

    with pytest.raises(
        ToolStateTransitionConflictError,
        match="conflicts with checkpoint state",
    ):
        record_persisted_tool_call(
            state,
            conflicting,
            expected_attempt_id=_ATTEMPT_ID,
        )


def test_rejects_attempt_ownership_mismatch() -> None:
    with pytest.raises(
        ControlledSupportToolStateError,
        match="AgentRunAttempt ownership",
    ):
        record_persisted_tool_call(
            _classified_state(
                reserve_decision=True,
            ),
            _search_audit(),
            expected_attempt_id=UUID("e0000000-0000-4000-8000-000000000014"),
        )


def test_rejects_malformed_safe_output() -> None:
    audit = _search_audit()
    malformed = AgentToolCall.create_terminal(
        tool_call_id=audit.id,
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
        safe_output={
            "result_count": 1,
        },
        latency_ms=audit.latency_ms or 0,
        error_code=audit.error_code,
        started_at=audit.execution_started_at or audit.proposed_at,
        finished_at=audit.finished_at or audit.proposed_at,
    )

    with pytest.raises(
        ControlledSupportToolStateError,
        match="incompatible with graph state",
    ):
        record_persisted_tool_call(
            _classified_state(
                reserve_decision=True,
            ),
            malformed,
            expected_attempt_id=_ATTEMPT_ID,
        )


def test_rejects_non_contiguous_search_evidence_ranks() -> None:
    audit = _search_audit()
    assert audit.safe_output is not None
    original_evidence = audit.safe_output["evidence"]
    assert isinstance(original_evidence, list)
    non_contiguous = AgentToolCall.create_terminal(
        tool_call_id=audit.id,
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
        safe_output={
            **dict(audit.safe_output),
            "result_count": 1,
            "evidence": [original_evidence[1]],
        },
        latency_ms=audit.latency_ms or 0,
        error_code=audit.error_code,
        started_at=audit.execution_started_at or audit.proposed_at,
        finished_at=audit.finished_at or audit.proposed_at,
    )

    with pytest.raises(
        ControlledSupportToolStateError,
        match="knowledge-search audit output",
    ):
        record_persisted_tool_call(
            _classified_state(
                reserve_decision=True,
            ),
            non_contiguous,
            expected_attempt_id=_ATTEMPT_ID,
        )


def test_rejects_duplicate_search_evidence_chunks() -> None:
    audit = _search_audit()
    assert audit.safe_output is not None
    original_evidence = audit.safe_output["evidence"]
    assert isinstance(original_evidence, list)
    first_item = original_evidence[0]
    second_item = original_evidence[1]
    assert isinstance(first_item, dict)
    assert isinstance(second_item, dict)
    first_evidence = dict(first_item)
    second_evidence = dict(second_item)
    second_evidence["chunk_id"] = first_evidence["chunk_id"]
    duplicate_chunks = AgentToolCall.create_terminal(
        tool_call_id=audit.id,
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
        safe_output={
            **dict(audit.safe_output),
            "result_count": 2,
            "evidence": [
                first_evidence,
                second_evidence,
            ],
        },
        latency_ms=audit.latency_ms or 0,
        error_code=audit.error_code,
        started_at=audit.execution_started_at or audit.proposed_at,
        finished_at=audit.finished_at or audit.proposed_at,
    )

    with pytest.raises(
        ControlledSupportToolStateError,
        match="knowledge-search audit output",
    ):
        record_persisted_tool_call(
            _classified_state(
                reserve_decision=True,
            ),
            duplicate_chunks,
            expected_attempt_id=_ATTEMPT_ID,
        )
