"""Deterministic application identities for observability traces."""

from __future__ import annotations

import re
from dataclasses import dataclass
from uuid import UUID

from supportops.observability.models import Metadata, TraceAttributes

_SAFE_EXTERNAL_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")


@dataclass(frozen=True, slots=True)
class TraceIdentity:
    """Provider-independent identity for one logical trace."""

    trace_seed: str
    trace_name: str
    session_id: str | None = None
    tags: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _require_non_blank("trace_seed", self.trace_seed)
        _require_non_blank("trace_name", self.trace_name)

        if self.session_id is not None:
            _require_non_blank("session_id", self.session_id)

        for tag in self.tags:
            _require_non_blank("tag", tag)

        if len(set(self.tags)) != len(self.tags):
            raise ValueError("trace identity tags must be unique")

    def to_trace_attributes(
        self,
        *,
        metadata: Metadata | None = None,
    ) -> TraceAttributes:
        """Convert the identity into application-owned trace attributes."""

        return TraceAttributes(
            trace_seed=self.trace_seed,
            name=self.trace_name,
            session_id=self.session_id,
            metadata={} if metadata is None else metadata,
            tags=self.tags,
        )


def agent_run_trace_identity(
    *,
    agent_run_id: UUID,
    ticket_id: UUID,
) -> TraceIdentity:
    """Return the stable identity for one durable AgentRun."""

    return TraceIdentity(
        trace_seed=f"agent-run:{agent_run_id}",
        trace_name="agent-run",
        session_id=ticket_session_id(ticket_id),
        tags=("supportops", "agent-run"),
    )


def semantic_search_trace_identity(
    *,
    request_id: str,
) -> TraceIdentity:
    """Return the identity for one independent semantic-search request."""

    normalized_request_id = _normalize_external_identifier(
        "request_id",
        request_id,
    )

    return TraceIdentity(
        trace_seed=f"semantic-search:{normalized_request_id}",
        trace_name="semantic-search",
        tags=("supportops", "semantic-search"),
    )


def knowledge_index_trace_identity(
    *,
    execution_id: UUID | str,
) -> TraceIdentity:
    """Return the identity for one knowledge-indexing command execution."""

    normalized_execution_id = _normalize_external_identifier(
        "execution_id",
        str(execution_id),
    )

    return TraceIdentity(
        trace_seed=f"knowledge-index:{normalized_execution_id}",
        trace_name="knowledge-index",
        tags=("supportops", "knowledge-index"),
    )


def ticket_session_id(ticket_id: UUID) -> str:
    """Return the stable session identity for one ticket."""

    return f"ticket:{ticket_id}"


def _normalize_external_identifier(
    field_name: str,
    value: str,
) -> str:
    normalized = value.strip()

    if not normalized:
        raise ValueError(f"{field_name} must not be blank")

    if _SAFE_EXTERNAL_IDENTIFIER.fullmatch(normalized) is None:
        raise ValueError(f"{field_name} must contain only safe identifier characters")

    return normalized


def _require_non_blank(field_name: str, value: str) -> None:
    if not value.strip():
        raise ValueError(f"{field_name} must not be blank")
