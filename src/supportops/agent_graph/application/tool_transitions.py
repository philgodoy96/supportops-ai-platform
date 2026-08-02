"""Validated tool-outcome transitions for controlled graph state."""

from collections.abc import Mapping
from typing import Any
from uuid import UUID

from pydantic import JsonValue

from supportops.agent_graph.application.tool_audit_schemas import (
    PersistedSearchKnowledgeOutput,
    PersistedServiceStatusOutput,
    PersistedToolAuditOutputError,
    parse_persisted_search_knowledge_output,
    parse_persisted_service_status_output,
)
from supportops.agent_graph.domain.routing import (
    CONTROLLED_SUPPORT_RUNTIME_LIMITS,
    ControlledSupportRuntimeLimits,
    reserve_next_decision_turn,
    reserve_next_tool_call,
)
from supportops.agent_graph.domain.state import (
    ControlledSupportGraphStateSnapshot,
)
from supportops.agent_tools.domain.audit import (
    AgentToolCall,
    AgentToolCallStatus,
)
from supportops.agent_tools.tools.search_knowledge import (
    SEARCH_KNOWLEDGE_TOOL_NAME,
    SEARCH_KNOWLEDGE_TOOL_VERSION,
)
from supportops.agent_tools.tools.service_status import (
    LOOKUP_SERVICE_STATUS_TOOL_NAME,
    LOOKUP_SERVICE_STATUS_TOOL_VERSION,
)


class ControlledSupportToolStateError(RuntimeError):
    """Base failure for invalid persisted tool-state projection."""

    error_code = "tool_state_transition_invalid"
    retryable = False


class ToolStateTransitionConflictError(ControlledSupportToolStateError):
    """Raised when persisted tool data conflicts with checkpoint state."""

    error_code = "tool_state_transition_conflict"

    def __init__(
        self,
        message: str,
    ) -> None:
        super().__init__(message)


def record_persisted_tool_call(
    state: ControlledSupportGraphStateSnapshot,
    audit: AgentToolCall,
    *,
    expected_attempt_id: UUID,
    limits: ControlledSupportRuntimeLimits = (CONTROLLED_SUPPORT_RUNTIME_LIMITS),
) -> ControlledSupportGraphStateSnapshot:
    """Project one terminal PostgreSQL audit into graph state."""

    _validate_runtime_ownership(
        state=state,
        audit=audit,
        expected_attempt_id=expected_attempt_id,
    )

    replay = _resolve_existing_sequence_replay(
        state=state,
        audit=audit,
    )

    if replay is not None:
        return replay

    _require_active_state(state)

    working_state = state

    if working_state.decision_turn_count == working_state.tool_call_count:
        working_state = _replace_state(
            working_state,
            decision_turn_count=reserve_next_decision_turn(
                working_state,
                limits=limits,
            ),
        )
    elif working_state.decision_turn_count != working_state.tool_call_count + 1:
        raise ControlledSupportToolStateError(
            "A persisted tool call requires either a settled "
            "checkpoint or exactly one unresolved decision."
        )

    expected_sequence = reserve_next_tool_call(
        working_state,
        limits=limits,
    )

    if audit.sequence != expected_sequence:
        raise ToolStateTransitionConflictError(
            "The persisted tool-call sequence does not match the next graph sequence."
        )

    updates: dict[str, object] = {
        "tool_call_count": expected_sequence,
        "tool_call_ids": (
            *working_state.tool_call_ids,
            audit.id,
        ),
        "seen_tool_call_fingerprints": (
            *working_state.seen_tool_call_fingerprints,
            audit.input_fingerprint,
        ),
    }

    if audit.status is AgentToolCallStatus.SUCCEEDED:
        updates.update(
            _successful_tool_projection(
                state=working_state,
                audit=audit,
            )
        )
    else:
        if audit.error_code is None:
            raise ControlledSupportToolStateError(
                "An unsuccessful persisted tool call requires a stable error code."
            )

        updates["current_error_code"] = audit.error_code

    return _replace_state(
        working_state,
        **updates,
    )


def _successful_tool_projection(
    *,
    state: ControlledSupportGraphStateSnapshot,
    audit: AgentToolCall,
) -> dict[str, object]:
    if audit.safe_output is None:
        raise ControlledSupportToolStateError(
            "A successful persisted tool call requires safe output."
        )

    if (
        audit.tool_name == SEARCH_KNOWLEDGE_TOOL_NAME
        and audit.tool_version == SEARCH_KNOWLEDGE_TOOL_VERSION
    ):
        output = _parse_search_output(audit.safe_output)

        if output.retrieval_query_id in state.retrieval_query_ids:
            raise ToolStateTransitionConflictError(
                "The retrieval query is already represented by another tool-call sequence."
            )

        retrieved_chunk_ids = list(state.retrieved_chunk_ids)
        known_chunk_ids = set(retrieved_chunk_ids)

        for evidence in output.evidence:
            if evidence.chunk_id in known_chunk_ids:
                continue

            retrieved_chunk_ids.append(evidence.chunk_id)
            known_chunk_ids.add(evidence.chunk_id)

        return {
            "retrieval_query_ids": (
                *state.retrieval_query_ids,
                output.retrieval_query_id,
            ),
            "retrieved_chunk_ids": tuple(retrieved_chunk_ids),
        }

    if (
        audit.tool_name == LOOKUP_SERVICE_STATUS_TOOL_NAME
        and audit.tool_version == LOOKUP_SERVICE_STATUS_TOOL_VERSION
    ):
        _parse_service_status_output(audit.safe_output)

        return {
            "service_status_tool_call_ids": (
                *state.service_status_tool_call_ids,
                audit.id,
            ),
        }

    raise ControlledSupportToolStateError(
        "The persisted audit references an unsupported controlled tool identity."
    )


def _resolve_existing_sequence_replay(
    *,
    state: ControlledSupportGraphStateSnapshot,
    audit: AgentToolCall,
) -> ControlledSupportGraphStateSnapshot | None:
    if audit.sequence > state.tool_call_count:
        return None

    index = audit.sequence - 1

    if (
        state.tool_call_ids[index] == audit.id
        and state.seen_tool_call_fingerprints[index] == audit.input_fingerprint
    ):
        return state

    raise ToolStateTransitionConflictError(
        "The persisted tool-call sequence conflicts with checkpoint state."
    )


def _validate_runtime_ownership(
    *,
    state: ControlledSupportGraphStateSnapshot,
    audit: AgentToolCall,
    expected_attempt_id: UUID,
) -> None:
    if not isinstance(expected_attempt_id, UUID):
        raise TypeError("expected_attempt_id must be a UUID.")

    ownership_values = (
        (
            audit.workspace_id,
            state.workspace_id,
            "workspace",
        ),
        (
            audit.ticket_id,
            state.ticket_id,
            "ticket",
        ),
        (
            audit.agent_run_id,
            state.agent_run_id,
            "AgentRun",
        ),
        (
            audit.agent_run_attempt_id,
            expected_attempt_id,
            "AgentRunAttempt",
        ),
    )

    for actual, expected, resource_name in ownership_values:
        if actual != expected:
            raise ControlledSupportToolStateError(
                f"Tool-call {resource_name} ownership does not match graph runtime."
            )


def _require_active_state(
    state: ControlledSupportGraphStateSnapshot,
) -> None:
    if state.classification_id is None:
        raise ControlledSupportToolStateError(
            "Tool-call projection requires a persisted classification."
        )

    if state.analysis_completion is not None:
        raise ControlledSupportToolStateError(
            "Tool-call projection cannot occur after terminal analysis."
        )

    if state.current_error_code is not None:
        raise ControlledSupportToolStateError(
            "Tool-call projection cannot continue after a graph error."
        )

    if state.recommendation_id is not None:
        raise ControlledSupportToolStateError(
            "Tool-call projection cannot continue after recommendation persistence."
        )


def _parse_search_output(
    value: Mapping[str, JsonValue],
) -> PersistedSearchKnowledgeOutput:
    try:
        return parse_persisted_search_knowledge_output(value)
    except PersistedToolAuditOutputError as exc:
        raise ControlledSupportToolStateError(
            "The persisted knowledge-search audit output is incompatible with graph state."
        ) from exc


def _parse_service_status_output(
    value: Mapping[str, JsonValue],
) -> PersistedServiceStatusOutput:
    try:
        return parse_persisted_service_status_output(value)
    except PersistedToolAuditOutputError as exc:
        raise ControlledSupportToolStateError(
            "The persisted service-status audit output is incompatible with graph state."
        ) from exc


def _replace_state(
    state: ControlledSupportGraphStateSnapshot,
    **updates: object,
) -> ControlledSupportGraphStateSnapshot:
    payload: dict[str, Any] = state.model_dump(mode="python")
    payload.update(updates)

    return ControlledSupportGraphStateSnapshot.model_validate(payload)
