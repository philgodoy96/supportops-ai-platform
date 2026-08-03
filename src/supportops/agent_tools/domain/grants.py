"""Immutable authorization grants for sensitive tool execution."""

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta
from types import MappingProxyType
from uuid import UUID, uuid4

from pydantic import JsonValue

from supportops.agent_tools.domain.audit import (
    AgentToolCall,
    AgentToolCallStatus,
)
from supportops.agent_tools.domain.contracts import (
    ToolSafetyLevel,
)
from supportops.modules.approvals.domain.models import (
    ApprovalRequest,
    ApprovalRequestStatus,
)

SENSITIVE_EXECUTION_GRANTED_INPUT_MAX_BYTES = 8192


@dataclass(frozen=True, slots=True)
class SensitiveExecutionGrant:
    """Application-owned authorization for one sensitive execution."""

    id: UUID
    workspace_id: UUID
    ticket_id: UUID
    agent_run_id: UUID
    executed_by_agent_run_attempt_id: UUID
    approval_request_id: UUID
    agent_tool_call_id: UUID
    tool_name: str
    tool_version: int
    safety_level: ToolSafetyLevel
    input_fingerprint: str
    granted_input: Mapping[str, JsonValue]
    decision_actor_reference: str
    decision_request_id: UUID
    decision_correlation_id: UUID
    approved_at: datetime
    created_at: datetime

    def __post_init__(self) -> None:
        for field_name in (
            "id",
            "workspace_id",
            "ticket_id",
            "agent_run_id",
            "executed_by_agent_run_attempt_id",
            "approval_request_id",
            "agent_tool_call_id",
            "decision_request_id",
            "decision_correlation_id",
        ):
            if not isinstance(getattr(self, field_name), UUID):
                raise TypeError(f"{field_name} must be a UUID.")

        if self.safety_level is not ToolSafetyLevel.SENSITIVE_WRITE:
            raise ValueError(
                "Sensitive execution grants require sensitive_write.",
            )
        if not self.tool_name or self.tool_name != self.tool_name.strip():
            raise ValueError("tool_name must be normalized.")
        if self.tool_version < 1:
            raise ValueError("tool_version must be positive.")
        if (
            len(self.input_fingerprint) != 64
            or self.input_fingerprint.lower() != self.input_fingerprint
        ):
            raise ValueError(
                "input_fingerprint must be lowercase SHA-256.",
            )
        if not self.decision_actor_reference:
            raise ValueError(
                "decision_actor_reference is required.",
            )
        if self.decision_actor_reference != self.decision_actor_reference.strip():
            raise ValueError(
                "decision_actor_reference must be normalized.",
            )
        _validate_utc(self.approved_at, "approved_at")
        _validate_utc(self.created_at, "created_at")
        if self.created_at < self.approved_at:
            raise ValueError(
                "created_at must not precede approved_at.",
            )

        frozen_input = MappingProxyType(dict(self.granted_input))
        object.__setattr__(self, "granted_input", frozen_input)

    @classmethod
    def create(
        cls,
        *,
        approval_request: ApprovalRequest,
        tool_call: AgentToolCall,
        executed_by_agent_run_attempt_id: UUID,
        created_at: datetime,
        grant_id: UUID | None = None,
    ) -> "SensitiveExecutionGrant":
        """Create a grant only from one matching approved decision."""

        if approval_request.status is not ApprovalRequestStatus.APPROVED:
            raise ValueError(
                "Only approved requests may create execution grants.",
            )
        if tool_call.status is not AgentToolCallStatus.PENDING_APPROVAL:
            raise ValueError(
                "Execution grants require pending_approval tool calls.",
            )
        if tool_call.safety_level is not ToolSafetyLevel.SENSITIVE_WRITE:
            raise ValueError(
                "Execution grants require sensitive_write tool calls.",
            )

        matches = (
            approval_request.workspace_id == tool_call.workspace_id,
            approval_request.ticket_id == tool_call.ticket_id,
            approval_request.agent_run_id == tool_call.agent_run_id,
            approval_request.agent_tool_call_id == tool_call.id,
            approval_request.tool_name == tool_call.tool_name,
            approval_request.tool_version == tool_call.tool_version,
            approval_request.input_fingerprint == tool_call.input_fingerprint,
            dict(approval_request.proposed_input) == dict(tool_call.safe_input),
        )
        if not all(matches):
            raise ValueError(
                "ApprovalRequest and AgentToolCall must match.",
            )

        if (
            approval_request.decision_actor_reference is None
            or approval_request.decision_request_id is None
            or approval_request.decision_correlation_id is None
            or approval_request.decided_at is None
        ):
            raise ValueError(
                "Approved requests require complete decision metadata.",
            )

        return cls(
            id=grant_id or uuid4(),
            workspace_id=approval_request.workspace_id,
            ticket_id=approval_request.ticket_id,
            agent_run_id=approval_request.agent_run_id,
            executed_by_agent_run_attempt_id=(executed_by_agent_run_attempt_id),
            approval_request_id=approval_request.id,
            agent_tool_call_id=tool_call.id,
            tool_name=tool_call.tool_name,
            tool_version=tool_call.tool_version,
            safety_level=tool_call.safety_level,
            input_fingerprint=tool_call.input_fingerprint,
            granted_input=dict(tool_call.safe_input),
            decision_actor_reference=(approval_request.decision_actor_reference),
            decision_request_id=(approval_request.decision_request_id),
            decision_correlation_id=(approval_request.decision_correlation_id),
            approved_at=approval_request.decided_at,
            created_at=created_at,
        )

    def matches_authorization(
        self,
        candidate: "SensitiveExecutionGrant",
    ) -> bool:
        """Compare immutable authorization identity and content."""

        return (
            self.workspace_id == candidate.workspace_id
            and self.ticket_id == candidate.ticket_id
            and self.agent_run_id == candidate.agent_run_id
            and self.executed_by_agent_run_attempt_id == candidate.executed_by_agent_run_attempt_id
            and self.approval_request_id == candidate.approval_request_id
            and self.agent_tool_call_id == candidate.agent_tool_call_id
            and self.tool_name == candidate.tool_name
            and self.tool_version == candidate.tool_version
            and self.safety_level is candidate.safety_level
            and self.input_fingerprint == candidate.input_fingerprint
            and dict(self.granted_input) == dict(candidate.granted_input)
            and self.decision_actor_reference == candidate.decision_actor_reference
            and self.decision_request_id == candidate.decision_request_id
            and self.decision_correlation_id == candidate.decision_correlation_id
            and self.approved_at == candidate.approved_at
        )


def _validate_utc(value: datetime, field_name: str) -> None:
    if value.tzinfo is None or value.utcoffset() != timedelta(0):
        raise ValueError(f"{field_name} must be UTC-aware.")
