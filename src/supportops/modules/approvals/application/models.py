"""Application commands and results for durable approval decisions."""

from dataclasses import dataclass
from datetime import datetime, timedelta
from uuid import UUID

from supportops.modules.approvals.domain.models import (
    APPROVAL_DECISION_ACTOR_MAX_LENGTH,
    APPROVAL_DECISION_COMMENT_MAX_LENGTH,
    ApprovalRequest,
)


@dataclass(frozen=True, slots=True)
class ApproveApprovalRequestCommand:
    """Values required to approve one workspace-scoped request."""

    workspace_id: UUID
    approval_request_id: UUID
    actor_reference: str
    comment: str | None
    request_id: UUID
    correlation_id: UUID
    decided_at: datetime

    def __post_init__(self) -> None:
        _validate_uuid(self.workspace_id, field_name="workspace_id")
        _validate_uuid(
            self.approval_request_id,
            field_name="approval_request_id",
        )
        _validate_uuid(self.request_id, field_name="request_id")
        _validate_uuid(
            self.correlation_id,
            field_name="correlation_id",
        )
        _validate_required_text(
            self.actor_reference,
            field_name="actor_reference",
            maximum_length=APPROVAL_DECISION_ACTOR_MAX_LENGTH,
        )
        _validate_optional_text(
            self.comment,
            field_name="comment",
            maximum_length=APPROVAL_DECISION_COMMENT_MAX_LENGTH,
        )
        _validate_utc_timestamp(
            self.decided_at,
            field_name="decided_at",
        )


@dataclass(frozen=True, slots=True)
class RejectApprovalRequestCommand:
    """Values required to reject one workspace-scoped request."""

    workspace_id: UUID
    approval_request_id: UUID
    actor_reference: str
    comment: str
    request_id: UUID
    correlation_id: UUID
    decided_at: datetime

    def __post_init__(self) -> None:
        _validate_uuid(self.workspace_id, field_name="workspace_id")
        _validate_uuid(
            self.approval_request_id,
            field_name="approval_request_id",
        )
        _validate_uuid(self.request_id, field_name="request_id")
        _validate_uuid(
            self.correlation_id,
            field_name="correlation_id",
        )
        _validate_required_text(
            self.actor_reference,
            field_name="actor_reference",
            maximum_length=APPROVAL_DECISION_ACTOR_MAX_LENGTH,
        )
        _validate_required_text(
            self.comment,
            field_name="comment",
            maximum_length=APPROVAL_DECISION_COMMENT_MAX_LENGTH,
        )
        _validate_utc_timestamp(
            self.decided_at,
            field_name="decided_at",
        )


@dataclass(frozen=True, slots=True)
class ExpirePendingApprovalRequestsCommand:
    """Bounded expiration work requested by one worker cycle."""

    now: datetime
    batch_size: int

    def __post_init__(self) -> None:
        _validate_utc_timestamp(self.now, field_name="now")
        if self.batch_size < 1:
            raise ValueError("batch_size must be at least one.")


@dataclass(frozen=True, slots=True)
class ApprovalDecisionResult:
    """Persisted terminal decision returned to an application caller."""

    approval_request: ApprovalRequest
    idempotent: bool


@dataclass(frozen=True, slots=True)
class ApprovalExpirationBatchResult:
    """Bounded expiration activity completed by one worker cycle."""

    approval_request_ids: tuple[UUID, ...]

    @property
    def expired_count(self) -> int:
        """Return the number of requests expired by the batch."""

        return len(self.approval_request_ids)


def _validate_uuid(value: UUID, *, field_name: str) -> None:
    if not isinstance(value, UUID):
        raise TypeError(f"{field_name} must be a UUID.")


def _validate_required_text(
    value: str,
    *,
    field_name: str,
    maximum_length: int,
) -> None:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string.")
    if not value:
        raise ValueError(f"{field_name} is required.")
    if value != value.strip():
        raise ValueError(
            f"{field_name} must not contain surrounding whitespace.",
        )
    if len(value) > maximum_length:
        raise ValueError(f"{field_name} exceeds the maximum length.")


def _validate_optional_text(
    value: str | None,
    *,
    field_name: str,
    maximum_length: int,
) -> None:
    if value is None:
        return
    _validate_required_text(
        value,
        field_name=field_name,
        maximum_length=maximum_length,
    )


def _validate_utc_timestamp(
    value: datetime,
    *,
    field_name: str,
) -> None:
    if not isinstance(value, datetime):
        raise TypeError(f"{field_name} must be a datetime.")
    if value.tzinfo is None or value.utcoffset() != timedelta(0):
        raise ValueError(f"{field_name} must be a UTC-aware timestamp.")
