"""Unit tests for approval application commands and results."""

from datetime import UTC, datetime
from uuid import UUID, uuid4

import pytest

from supportops.modules.approvals.application.models import (
    ApprovalExpirationBatchResult,
    ApproveApprovalRequestCommand,
    ExpirePendingApprovalRequestsCommand,
    RejectApprovalRequestCommand,
)
from supportops.modules.approvals.domain.models import (
    APPROVAL_DECISION_ACTOR_MAX_LENGTH,
    APPROVAL_DECISION_COMMENT_MAX_LENGTH,
)

_NOW = datetime(2026, 8, 2, 22, 45, tzinfo=UTC)
_WORKSPACE_ID = UUID("11111111-1111-4111-8111-111111111111")
_APPROVAL_REQUEST_ID = UUID("22222222-2222-4222-8222-222222222222")
_REQUEST_ID = UUID("33333333-3333-4333-8333-333333333333")
_CORRELATION_ID = UUID("44444444-4444-4444-8444-444444444444")


def test_approve_command_accepts_optional_comment() -> None:
    command = ApproveApprovalRequestCommand(
        workspace_id=_WORKSPACE_ID,
        approval_request_id=_APPROVAL_REQUEST_ID,
        actor_reference="operator:alice",
        comment=None,
        request_id=_REQUEST_ID,
        correlation_id=_CORRELATION_ID,
        decided_at=_NOW,
    )

    assert command.comment is None


def test_reject_command_requires_comment() -> None:
    with pytest.raises(ValueError, match="comment is required"):
        RejectApprovalRequestCommand(
            workspace_id=_WORKSPACE_ID,
            approval_request_id=_APPROVAL_REQUEST_ID,
            actor_reference="operator:alice",
            comment="",
            request_id=_REQUEST_ID,
            correlation_id=_CORRELATION_ID,
            decided_at=_NOW,
        )


def test_commands_reject_actor_surrounding_whitespace() -> None:
    with pytest.raises(ValueError, match="surrounding whitespace"):
        ApproveApprovalRequestCommand(
            workspace_id=_WORKSPACE_ID,
            approval_request_id=_APPROVAL_REQUEST_ID,
            actor_reference=" operator:alice",
            comment=None,
            request_id=_REQUEST_ID,
            correlation_id=_CORRELATION_ID,
            decided_at=_NOW,
        )

    with pytest.raises(ValueError, match="surrounding whitespace"):
        RejectApprovalRequestCommand(
            workspace_id=_WORKSPACE_ID,
            approval_request_id=_APPROVAL_REQUEST_ID,
            actor_reference="operator:alice ",
            comment="Escalation is not required.",
            request_id=_REQUEST_ID,
            correlation_id=_CORRELATION_ID,
            decided_at=_NOW,
        )


def test_commands_reject_empty_actor() -> None:
    with pytest.raises(ValueError, match="actor_reference is required"):
        ApproveApprovalRequestCommand(
            workspace_id=_WORKSPACE_ID,
            approval_request_id=_APPROVAL_REQUEST_ID,
            actor_reference="",
            comment=None,
            request_id=_REQUEST_ID,
            correlation_id=_CORRELATION_ID,
            decided_at=_NOW,
        )

    with pytest.raises(ValueError, match="actor_reference is required"):
        RejectApprovalRequestCommand(
            workspace_id=_WORKSPACE_ID,
            approval_request_id=_APPROVAL_REQUEST_ID,
            actor_reference="",
            comment="Escalation is not required.",
            request_id=_REQUEST_ID,
            correlation_id=_CORRELATION_ID,
            decided_at=_NOW,
        )


def test_commands_reject_actor_above_maximum_length() -> None:
    too_long = "a" * (APPROVAL_DECISION_ACTOR_MAX_LENGTH + 1)

    with pytest.raises(ValueError, match="maximum length"):
        ApproveApprovalRequestCommand(
            workspace_id=_WORKSPACE_ID,
            approval_request_id=_APPROVAL_REQUEST_ID,
            actor_reference=too_long,
            comment=None,
            request_id=_REQUEST_ID,
            correlation_id=_CORRELATION_ID,
            decided_at=_NOW,
        )

    with pytest.raises(ValueError, match="maximum length"):
        RejectApprovalRequestCommand(
            workspace_id=_WORKSPACE_ID,
            approval_request_id=_APPROVAL_REQUEST_ID,
            actor_reference=too_long,
            comment="Escalation is not required.",
            request_id=_REQUEST_ID,
            correlation_id=_CORRELATION_ID,
            decided_at=_NOW,
        )


def test_approve_command_rejects_comment_surrounding_whitespace() -> None:
    with pytest.raises(ValueError, match="surrounding whitespace"):
        ApproveApprovalRequestCommand(
            workspace_id=_WORKSPACE_ID,
            approval_request_id=_APPROVAL_REQUEST_ID,
            actor_reference="operator:alice",
            comment=" Looks good.",
            request_id=_REQUEST_ID,
            correlation_id=_CORRELATION_ID,
            decided_at=_NOW,
        )

    with pytest.raises(ValueError, match="surrounding whitespace"):
        ApproveApprovalRequestCommand(
            workspace_id=_WORKSPACE_ID,
            approval_request_id=_APPROVAL_REQUEST_ID,
            actor_reference="operator:alice",
            comment="Looks good. ",
            request_id=_REQUEST_ID,
            correlation_id=_CORRELATION_ID,
            decided_at=_NOW,
        )


def test_reject_command_rejects_comment_surrounding_whitespace() -> None:
    with pytest.raises(ValueError, match="surrounding whitespace"):
        RejectApprovalRequestCommand(
            workspace_id=_WORKSPACE_ID,
            approval_request_id=_APPROVAL_REQUEST_ID,
            actor_reference="operator:alice",
            comment=" Not warranted.",
            request_id=_REQUEST_ID,
            correlation_id=_CORRELATION_ID,
            decided_at=_NOW,
        )


def test_reject_command_rejects_comment_above_maximum_length() -> None:
    with pytest.raises(ValueError, match="maximum length"):
        RejectApprovalRequestCommand(
            workspace_id=_WORKSPACE_ID,
            approval_request_id=_APPROVAL_REQUEST_ID,
            actor_reference="operator:alice",
            comment="x" * (APPROVAL_DECISION_COMMENT_MAX_LENGTH + 1),
            request_id=_REQUEST_ID,
            correlation_id=_CORRELATION_ID,
            decided_at=_NOW,
        )


@pytest.mark.parametrize(
    "field_name",
    [
        "workspace_id",
        "approval_request_id",
        "request_id",
        "correlation_id",
    ],
)
def test_commands_reject_non_uuid_identifiers(field_name: str) -> None:
    approve_values: dict[str, object] = {
        "workspace_id": _WORKSPACE_ID,
        "approval_request_id": _APPROVAL_REQUEST_ID,
        "actor_reference": "operator:alice",
        "comment": None,
        "request_id": _REQUEST_ID,
        "correlation_id": _CORRELATION_ID,
        "decided_at": _NOW,
    }
    approve_values[field_name] = "not-a-uuid"
    with pytest.raises(TypeError, match=field_name):
        ApproveApprovalRequestCommand(**approve_values)  # type: ignore[arg-type]

    reject_values: dict[str, object] = {
        "workspace_id": _WORKSPACE_ID,
        "approval_request_id": _APPROVAL_REQUEST_ID,
        "actor_reference": "operator:alice",
        "comment": "Escalation is not required.",
        "request_id": _REQUEST_ID,
        "correlation_id": _CORRELATION_ID,
        "decided_at": _NOW,
    }
    reject_values[field_name] = "not-a-uuid"
    with pytest.raises(TypeError, match=field_name):
        RejectApprovalRequestCommand(**reject_values)  # type: ignore[arg-type]


def test_commands_reject_non_utc_timestamps() -> None:
    with pytest.raises(ValueError, match="UTC-aware"):
        ApproveApprovalRequestCommand(
            workspace_id=_WORKSPACE_ID,
            approval_request_id=_APPROVAL_REQUEST_ID,
            actor_reference="operator:alice",
            comment=None,
            request_id=_REQUEST_ID,
            correlation_id=_CORRELATION_ID,
            decided_at=datetime(2026, 8, 2, 22, 45),
        )

    with pytest.raises(ValueError, match="UTC-aware"):
        RejectApprovalRequestCommand(
            workspace_id=_WORKSPACE_ID,
            approval_request_id=_APPROVAL_REQUEST_ID,
            actor_reference="operator:alice",
            comment="Escalation is not required.",
            request_id=_REQUEST_ID,
            correlation_id=_CORRELATION_ID,
            decided_at=datetime(2026, 8, 2, 22, 45),
        )

    with pytest.raises(ValueError, match="UTC-aware"):
        ExpirePendingApprovalRequestsCommand(
            now=datetime(2026, 8, 2, 22, 45),
            batch_size=1,
        )


def test_expiration_command_requires_positive_batch_size() -> None:
    with pytest.raises(ValueError, match="at least one"):
        ExpirePendingApprovalRequestsCommand(
            now=_NOW,
            batch_size=0,
        )

    with pytest.raises(ValueError, match="at least one"):
        ExpirePendingApprovalRequestsCommand(
            now=_NOW,
            batch_size=-1,
        )


def test_expiration_command_does_not_enforce_settings_upper_bound() -> None:
    command = ExpirePendingApprovalRequestsCommand(
        now=_NOW,
        batch_size=1001,
    )

    assert command.batch_size == 1001


def test_expiration_batch_result_exposes_expired_count() -> None:
    first = uuid4()
    second = uuid4()
    result = ApprovalExpirationBatchResult(
        approval_request_ids=(first, second),
    )

    assert result.expired_count == 2
    assert (
        ApprovalExpirationBatchResult(
            approval_request_ids=(),
        ).expired_count
        == 0
    )
