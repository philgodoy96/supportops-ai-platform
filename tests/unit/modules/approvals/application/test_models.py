"""Unit tests for approval application commands and results."""

from datetime import UTC, datetime
from uuid import uuid4

import pytest

from supportops.modules.approvals.application.models import (
    ApproveApprovalRequestCommand,
    ExpirePendingApprovalRequestsCommand,
    RejectApprovalRequestCommand,
)

_NOW = datetime(2026, 8, 2, 22, 45, tzinfo=UTC)


def test_approve_command_accepts_optional_comment() -> None:
    command = ApproveApprovalRequestCommand(
        workspace_id=uuid4(),
        approval_request_id=uuid4(),
        actor_reference="operator:alice",
        comment=None,
        request_id=uuid4(),
        correlation_id=uuid4(),
        decided_at=_NOW,
    )

    assert command.comment is None


def test_reject_command_requires_comment() -> None:
    with pytest.raises(ValueError, match="comment is required"):
        RejectApprovalRequestCommand(
            workspace_id=uuid4(),
            approval_request_id=uuid4(),
            actor_reference="operator:alice",
            comment="",
            request_id=uuid4(),
            correlation_id=uuid4(),
            decided_at=_NOW,
        )


def test_commands_reject_non_utc_timestamps() -> None:
    with pytest.raises(ValueError, match="UTC-aware"):
        ApproveApprovalRequestCommand(
            workspace_id=uuid4(),
            approval_request_id=uuid4(),
            actor_reference="operator:alice",
            comment=None,
            request_id=uuid4(),
            correlation_id=uuid4(),
            decided_at=datetime(2026, 8, 2, 22, 45),
        )


def test_expiration_command_requires_positive_batch_size() -> None:
    with pytest.raises(ValueError, match="at least one"):
        ExpirePendingApprovalRequestsCommand(
            now=_NOW,
            batch_size=0,
        )
