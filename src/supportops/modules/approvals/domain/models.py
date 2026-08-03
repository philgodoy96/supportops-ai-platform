"""Immutable domain model for application-owned approval requests."""

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from types import MappingProxyType
from typing import cast
from uuid import UUID, uuid4

from pydantic import JsonValue, TypeAdapter, ValidationError

from supportops.agent_tools.domain.audit import (
    AGENT_TOOL_CALL_NAME_MAX_LENGTH,
    AgentToolCall,
    AgentToolCallStatus,
)
from supportops.agent_tools.domain.contracts import ToolSafetyLevel

APPROVAL_REQUEST_REASON_MAX_LENGTH = 1000
APPROVAL_DECISION_ACTOR_MAX_LENGTH = 255
APPROVAL_DECISION_COMMENT_MAX_LENGTH = 2000
APPROVAL_PROPOSED_INPUT_MAX_BYTES = 8192
APPROVAL_EXPIRATION_ACTOR_REFERENCE = "system:approval-expiration"

_SHA256_HEXADECIMAL_LENGTH = 64
_STABLE_TOOL_NAME_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")
_JSON_OBJECT_ADAPTER = TypeAdapter(dict[str, JsonValue])


class ApprovalRequestStatus(StrEnum):
    """Lifecycle outcomes persisted for approval requests."""

    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXPIRED = "expired"


@dataclass(frozen=True, slots=True)
class ApprovalRequest:
    """One application-owned durable approval request."""

    id: UUID
    workspace_id: UUID
    ticket_id: UUID
    agent_run_id: UUID
    agent_tool_call_id: UUID
    requested_by_llm_invocation_id: UUID
    status: ApprovalRequestStatus
    tool_name: str
    tool_version: int
    safety_level: ToolSafetyLevel
    input_fingerprint: str
    proposed_input: Mapping[str, JsonValue]
    request_reason: str
    expires_at: datetime
    decision_actor_reference: str | None
    decision_comment: str | None
    decision_request_id: UUID | None
    decision_correlation_id: UUID | None
    decided_at: datetime | None
    created_at: datetime
    updated_at: datetime

    def __post_init__(self) -> None:
        _validate_uuid(self.id, field_name="id")
        _validate_uuid(self.workspace_id, field_name="workspace_id")
        _validate_uuid(self.ticket_id, field_name="ticket_id")
        _validate_uuid(self.agent_run_id, field_name="agent_run_id")
        _validate_uuid(
            self.agent_tool_call_id,
            field_name="agent_tool_call_id",
        )
        _validate_uuid(
            self.requested_by_llm_invocation_id,
            field_name="requested_by_llm_invocation_id",
        )

        if not isinstance(self.status, ApprovalRequestStatus):
            raise ValueError("status must be a supported ApprovalRequestStatus.")

        _validate_tool_name(self.tool_name)

        if self.tool_version < 1:
            raise ValueError("tool_version must be positive.")

        if self.safety_level is not ToolSafetyLevel.SENSITIVE_WRITE:
            raise ValueError("Approval requests require sensitive_write safety.")

        _validate_sha256_hash(
            self.input_fingerprint,
            field_name="input_fingerprint",
        )

        proposed_input = _validate_safe_json_object(
            self.proposed_input,
            field_name="proposed_input",
            maximum_bytes=APPROVAL_PROPOSED_INPUT_MAX_BYTES,
        )
        object.__setattr__(self, "proposed_input", proposed_input)

        _validate_bounded_text(
            self.request_reason,
            field_name="request_reason",
            maximum_length=APPROVAL_REQUEST_REASON_MAX_LENGTH,
        )

        _validate_utc_timestamp(self.created_at, field_name="created_at")
        _validate_utc_timestamp(self.updated_at, field_name="updated_at")
        _validate_utc_timestamp(self.expires_at, field_name="expires_at")

        if self.expires_at <= self.created_at:
            raise ValueError("expires_at must be strictly after created_at.")

        if self.updated_at < self.created_at:
            raise ValueError("updated_at must not precede created_at.")

        if self.decided_at is not None:
            _validate_utc_timestamp(self.decided_at, field_name="decided_at")
            if self.decided_at < self.created_at:
                raise ValueError("decided_at must not precede created_at.")

        if self.decision_request_id is not None:
            _validate_uuid(
                self.decision_request_id,
                field_name="decision_request_id",
            )

        if self.decision_correlation_id is not None:
            _validate_uuid(
                self.decision_correlation_id,
                field_name="decision_correlation_id",
            )

        if self.decision_actor_reference is not None:
            _validate_bounded_text(
                self.decision_actor_reference,
                field_name="decision_actor_reference",
                maximum_length=APPROVAL_DECISION_ACTOR_MAX_LENGTH,
            )

        if self.decision_comment is not None:
            _validate_bounded_text(
                self.decision_comment,
                field_name="decision_comment",
                maximum_length=APPROVAL_DECISION_COMMENT_MAX_LENGTH,
            )

        _validate_decision_state(self)

    @classmethod
    def create_pending(
        cls,
        *,
        tool_call: AgentToolCall,
        requested_by_llm_invocation_id: UUID,
        request_reason: str,
        expires_at: datetime,
        approval_request_id: UUID | None = None,
        now: datetime | None = None,
    ) -> "ApprovalRequest":
        """Create one pending approval from a sensitive tool proposal."""

        if tool_call.status is not AgentToolCallStatus.PENDING_APPROVAL:
            raise ValueError(
                "Approval requests require a pending_approval AgentToolCall.",
            )

        if tool_call.safety_level is not ToolSafetyLevel.SENSITIVE_WRITE:
            raise ValueError("Approval requests require sensitive_write safety.")

        _validate_uuid(
            requested_by_llm_invocation_id,
            field_name="requested_by_llm_invocation_id",
        )

        created_at = now or datetime.now(UTC)
        stripped_reason = request_reason.strip()

        return cls(
            id=approval_request_id or uuid4(),
            workspace_id=tool_call.workspace_id,
            ticket_id=tool_call.ticket_id,
            agent_run_id=tool_call.agent_run_id,
            agent_tool_call_id=tool_call.id,
            requested_by_llm_invocation_id=(requested_by_llm_invocation_id),
            status=ApprovalRequestStatus.PENDING,
            tool_name=tool_call.tool_name,
            tool_version=tool_call.tool_version,
            safety_level=tool_call.safety_level,
            input_fingerprint=tool_call.input_fingerprint,
            proposed_input=tool_call.safe_input,
            request_reason=stripped_reason,
            expires_at=expires_at,
            decision_actor_reference=None,
            decision_comment=None,
            decision_request_id=None,
            decision_correlation_id=None,
            decided_at=None,
            created_at=created_at,
            updated_at=created_at,
        )

    @property
    def is_terminal(self) -> bool:
        """Return whether the approval decision is immutable."""

        return self.status in {
            ApprovalRequestStatus.APPROVED,
            ApprovalRequestStatus.REJECTED,
            ApprovalRequestStatus.EXPIRED,
        }

    def matches_pending_proposal(
        self,
        candidate: "ApprovalRequest",
    ) -> bool:
        """Compare immutable pending proposal identity and content."""

        if self.status is not ApprovalRequestStatus.PENDING:
            return False

        if candidate.status is not ApprovalRequestStatus.PENDING:
            return False

        return (
            self.workspace_id == candidate.workspace_id
            and self.ticket_id == candidate.ticket_id
            and self.agent_run_id == candidate.agent_run_id
            and self.agent_tool_call_id == candidate.agent_tool_call_id
            and self.requested_by_llm_invocation_id == candidate.requested_by_llm_invocation_id
            and self.tool_name == candidate.tool_name
            and self.tool_version == candidate.tool_version
            and self.safety_level is candidate.safety_level
            and self.input_fingerprint == candidate.input_fingerprint
            and dict(self.proposed_input) == dict(candidate.proposed_input)
            and self.request_reason == candidate.request_reason
            and self.expires_at == candidate.expires_at
        )

    def approve(
        self,
        *,
        actor_reference: str,
        comment: str | None,
        request_id: UUID,
        correlation_id: UUID,
        decided_at: datetime,
    ) -> "ApprovalRequest":
        """Transition one pending request to an approved decision."""

        if self.status is not ApprovalRequestStatus.PENDING:
            raise ValueError("Only pending approval requests can be approved.")

        _validate_utc_timestamp(decided_at, field_name="decided_at")
        if decided_at < self.created_at:
            raise ValueError("decided_at must not precede created_at.")
        if decided_at >= self.expires_at:
            raise ValueError("decided_at must precede expires_at.")

        _validate_uuid(request_id, field_name="request_id")
        _validate_uuid(correlation_id, field_name="correlation_id")
        _validate_bounded_text(
            actor_reference,
            field_name="actor_reference",
            maximum_length=APPROVAL_DECISION_ACTOR_MAX_LENGTH,
        )
        if comment is not None:
            _validate_bounded_text(
                comment,
                field_name="comment",
                maximum_length=APPROVAL_DECISION_COMMENT_MAX_LENGTH,
            )

        return replace(
            self,
            status=ApprovalRequestStatus.APPROVED,
            decision_actor_reference=actor_reference,
            decision_comment=comment,
            decision_request_id=request_id,
            decision_correlation_id=correlation_id,
            decided_at=decided_at,
            updated_at=decided_at,
        )

    def reject(
        self,
        *,
        actor_reference: str,
        comment: str,
        request_id: UUID,
        correlation_id: UUID,
        decided_at: datetime,
    ) -> "ApprovalRequest":
        """Transition one pending request to a rejected decision."""

        if self.status is not ApprovalRequestStatus.PENDING:
            raise ValueError("Only pending approval requests can be rejected.")

        _validate_utc_timestamp(decided_at, field_name="decided_at")
        if decided_at < self.created_at:
            raise ValueError("decided_at must not precede created_at.")
        if decided_at >= self.expires_at:
            raise ValueError("decided_at must precede expires_at.")

        _validate_uuid(request_id, field_name="request_id")
        _validate_uuid(correlation_id, field_name="correlation_id")
        _validate_bounded_text(
            actor_reference,
            field_name="actor_reference",
            maximum_length=APPROVAL_DECISION_ACTOR_MAX_LENGTH,
        )
        _validate_bounded_text(
            comment,
            field_name="comment",
            maximum_length=APPROVAL_DECISION_COMMENT_MAX_LENGTH,
        )

        return replace(
            self,
            status=ApprovalRequestStatus.REJECTED,
            decision_actor_reference=actor_reference,
            decision_comment=comment,
            decision_request_id=request_id,
            decision_correlation_id=correlation_id,
            decided_at=decided_at,
            updated_at=decided_at,
        )

    def expire(
        self,
        *,
        decided_at: datetime,
    ) -> "ApprovalRequest":
        """Transition one overdue pending request to expiration."""

        if self.status is not ApprovalRequestStatus.PENDING:
            raise ValueError("Only pending approval requests can expire.")

        _validate_utc_timestamp(decided_at, field_name="decided_at")
        if decided_at < self.created_at:
            raise ValueError("decided_at must not precede created_at.")
        if decided_at < self.expires_at:
            raise ValueError("decided_at must not precede expires_at.")

        return replace(
            self,
            status=ApprovalRequestStatus.EXPIRED,
            decision_actor_reference=APPROVAL_EXPIRATION_ACTOR_REFERENCE,
            decision_comment=None,
            decision_request_id=None,
            decision_correlation_id=None,
            decided_at=decided_at,
            updated_at=decided_at,
        )


def _validate_decision_state(
    approval_request: ApprovalRequest,
) -> None:
    if approval_request.status is ApprovalRequestStatus.PENDING:
        if (
            approval_request.decision_actor_reference is not None
            or approval_request.decision_comment is not None
            or approval_request.decision_request_id is not None
            or approval_request.decision_correlation_id is not None
            or approval_request.decided_at is not None
        ):
            raise ValueError("Pending approvals cannot define decision fields.")
        return

    if approval_request.status is ApprovalRequestStatus.APPROVED:
        if approval_request.decision_actor_reference is None:
            raise ValueError("Approved approvals require decision_actor_reference.")
        if approval_request.decision_request_id is None:
            raise ValueError("Approved approvals require decision_request_id.")
        if approval_request.decision_correlation_id is None:
            raise ValueError("Approved approvals require decision_correlation_id.")
        if approval_request.decided_at is None:
            raise ValueError("Approved approvals require decided_at.")
        return

    if approval_request.status is ApprovalRequestStatus.REJECTED:
        if approval_request.decision_actor_reference is None:
            raise ValueError("Rejected approvals require decision_actor_reference.")
        if approval_request.decision_comment is None:
            raise ValueError("Rejected approvals require decision_comment.")
        if approval_request.decision_request_id is None:
            raise ValueError("Rejected approvals require decision_request_id.")
        if approval_request.decision_correlation_id is None:
            raise ValueError("Rejected approvals require decision_correlation_id.")
        if approval_request.decided_at is None:
            raise ValueError("Rejected approvals require decided_at.")
        return

    if approval_request.status is ApprovalRequestStatus.EXPIRED:
        if approval_request.decision_actor_reference != APPROVAL_EXPIRATION_ACTOR_REFERENCE:
            raise ValueError(
                "Expired approvals require the system expiration actor.",
            )
        if approval_request.decision_comment is not None:
            raise ValueError("Expired approvals cannot define decision_comment.")
        if approval_request.decision_request_id is not None:
            raise ValueError("Expired approvals cannot define decision_request_id.")
        if approval_request.decision_correlation_id is not None:
            raise ValueError(
                "Expired approvals cannot define decision_correlation_id.",
            )
        if approval_request.decided_at is None:
            raise ValueError("Expired approvals require decided_at.")
        return

    raise ValueError("status must be a supported ApprovalRequestStatus.")


def _validate_uuid(value: object, *, field_name: str) -> None:
    if not isinstance(value, UUID):
        raise TypeError(f"{field_name} must be a UUID.")


def _validate_tool_name(value: str) -> None:
    if not value:
        raise ValueError("tool_name is required.")

    if value != value.strip():
        raise ValueError("tool_name must not contain surrounding whitespace.")

    if len(value) > AGENT_TOOL_CALL_NAME_MAX_LENGTH:
        raise ValueError("tool_name exceeds the maximum length.")

    if _STABLE_TOOL_NAME_PATTERN.fullmatch(value) is None:
        raise ValueError("tool_name must use stable lowercase snake case.")


def _validate_bounded_text(
    value: str,
    *,
    field_name: str,
    maximum_length: int,
) -> None:
    if not value:
        raise ValueError(f"{field_name} is required.")

    if value != value.strip():
        raise ValueError(f"{field_name} must not contain surrounding whitespace.")

    if len(value) > maximum_length:
        raise ValueError(f"{field_name} exceeds the maximum length.")


def _validate_safe_json_object(
    value: Mapping[str, JsonValue],
    *,
    field_name: str,
    maximum_bytes: int,
) -> Mapping[str, JsonValue]:
    try:
        validated = _JSON_OBJECT_ADAPTER.validate_python(
            dict(value),
            strict=True,
        )
        canonical_json = json.dumps(
            validated,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (
        TypeError,
        ValueError,
        ValidationError,
    ) as exc:
        raise ValueError(f"{field_name} must be a JSON-compatible object.") from exc

    if len(canonical_json.encode("utf-8")) > maximum_bytes:
        raise ValueError(f"{field_name} exceeds the supported size.")

    defensive_copy = cast(
        dict[str, JsonValue],
        json.loads(canonical_json),
    )

    return MappingProxyType(defensive_copy)


def _validate_sha256_hash(value: str, *, field_name: str) -> None:
    if len(value) != _SHA256_HEXADECIMAL_LENGTH:
        raise ValueError(f"{field_name} must be a lowercase SHA-256 hash.")

    if any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{field_name} must be a lowercase SHA-256 hash.")


def _validate_utc_timestamp(value: datetime, *, field_name: str) -> None:
    if value.tzinfo is None or value.utcoffset() != timedelta(0):
        raise ValueError(f"{field_name} must be a UTC-aware timestamp.")
