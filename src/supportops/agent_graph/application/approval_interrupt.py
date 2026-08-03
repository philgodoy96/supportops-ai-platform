"""Safe LangGraph interrupt payload for durable human approval."""

from collections.abc import Mapping
from typing import Annotated
from uuid import UUID

from langgraph.types import interrupt
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    StringConstraints,
)

from supportops.agent_tools.domain.audit import AgentToolCall
from supportops.modules.approvals.domain.models import ApprovalRequest

ApprovalToolName = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        min_length=1,
        max_length=64,
        pattern=r"^[a-z][a-z0-9_]*$",
    ),
]
ApprovalReason = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        min_length=1,
        max_length=1000,
    ),
]


class ApprovalInterruptPayload(BaseModel):
    """Bounded JSON-compatible value exposed by graph interruption."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    approval_request_id: UUID
    agent_tool_call_id: UUID
    agent_run_id: UUID
    ticket_id: UUID
    tool_name: ApprovalToolName
    tool_version: Annotated[int, Field(strict=True, ge=1)]
    proposed_input: dict[str, JsonValue]
    request_reason: ApprovalReason
    expires_at: str

    @classmethod
    def from_records(
        cls,
        *,
        tool_call: AgentToolCall,
        approval_request: ApprovalRequest,
    ) -> "ApprovalInterruptPayload":
        """Create a safe payload from matching durable records."""

        ownership_matches = (
            approval_request.agent_tool_call_id == tool_call.id
            and approval_request.workspace_id == tool_call.workspace_id
            and approval_request.ticket_id == tool_call.ticket_id
            and approval_request.agent_run_id == tool_call.agent_run_id
            and approval_request.tool_name == tool_call.tool_name
            and approval_request.tool_version == tool_call.tool_version
            and approval_request.safety_level is tool_call.safety_level
            and approval_request.input_fingerprint == tool_call.input_fingerprint
            and dict(approval_request.proposed_input) == dict(tool_call.safe_input)
        )
        if not ownership_matches:
            raise ValueError(
                "Approval interrupt records must describe the same sensitive proposal.",
            )

        return cls(
            approval_request_id=approval_request.id,
            agent_tool_call_id=tool_call.id,
            agent_run_id=tool_call.agent_run_id,
            ticket_id=tool_call.ticket_id,
            tool_name=tool_call.tool_name,
            tool_version=tool_call.tool_version,
            proposed_input=dict(tool_call.safe_input),
            request_reason=approval_request.request_reason,
            expires_at=approval_request.expires_at.isoformat(),
        )

    def to_interrupt_value(self) -> dict[str, JsonValue]:
        """Return the exact safe value checkpointed by LangGraph."""

        return self.model_dump(mode="json")


def parse_approval_interrupt_payload(
    value: Mapping[str, object],
) -> ApprovalInterruptPayload:
    """Validate one framework-returned interrupt value."""

    return ApprovalInterruptPayload.model_validate(dict(value))


def interrupt_for_approval(
    payload: ApprovalInterruptPayload,
) -> object:
    """Pause the graph after durable approval state exists."""

    return interrupt(payload.to_interrupt_value())
