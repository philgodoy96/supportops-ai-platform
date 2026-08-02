"""Fenced persistence contracts for terminal controlled tool calls."""

from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import StrEnum
from typing import Protocol
from uuid import UUID

from supportops.agent_tools.domain.audit import AgentToolCall


class AgentToolCallPersistenceResult(StrEnum):
    """Outcome of one fenced tool-call persistence operation."""

    APPLIED = "applied"
    ALREADY_RECORDED = "already_recorded"
    LEASE_LOST = "lease_lost"


@dataclass(frozen=True, slots=True)
class PersistAgentToolCallCommand:
    """Persist one terminal tool call while the lease remains valid."""

    workspace_id: UUID
    ticket_id: UUID
    agent_run_id: UUID
    agent_run_attempt_id: UUID
    lease_token: UUID
    persisted_at: datetime
    tool_call: AgentToolCall

    def __post_init__(self) -> None:
        _validate_utc_timestamp(
            self.persisted_at,
            field_name="persisted_at",
        )
        _validate_tool_call_ownership(
            command=self,
            tool_call=self.tool_call,
        )

        if self.persisted_at < self.tool_call.finished_at:
            raise ValueError("persisted_at must not precede the tool-call completion timestamp.")


class AgentToolCallExecutionRepository(Protocol):
    """Atomic persistence boundary for one terminal tool call."""

    async def persist_fenced(
        self,
        command: PersistAgentToolCallCommand,
    ) -> AgentToolCallPersistenceResult:
        """Persist one terminal audit record under lease fencing."""

        ...


def _validate_tool_call_ownership(
    *,
    command: PersistAgentToolCallCommand,
    tool_call: AgentToolCall,
) -> None:
    ownership_values = (
        (
            tool_call.workspace_id,
            command.workspace_id,
            "workspace",
        ),
        (
            tool_call.ticket_id,
            command.ticket_id,
            "ticket",
        ),
        (
            tool_call.agent_run_id,
            command.agent_run_id,
            "AgentRun",
        ),
        (
            tool_call.agent_run_attempt_id,
            command.agent_run_attempt_id,
            "AgentRunAttempt",
        ),
    )

    for actual, expected, resource_name in ownership_values:
        if actual != expected:
            raise ValueError(
                f"Tool-call {resource_name} ownership does not match the persistence command."
            )


def _validate_utc_timestamp(
    value: datetime,
    *,
    field_name: str,
) -> None:
    if value.tzinfo is None or value.utcoffset() != timedelta(0):
        raise ValueError(f"{field_name} must be a UTC-aware timestamp.")
