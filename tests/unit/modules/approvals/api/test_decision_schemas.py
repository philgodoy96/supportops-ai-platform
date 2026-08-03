"""Unit tests for approval decision HTTP schemas."""

from uuid import uuid4

import pytest
from pydantic import ValidationError

from supportops.modules.approvals.api.schemas import (
    ApproveApprovalRequestBody,
    RejectApprovalRequestBody,
)
from supportops.modules.approvals.domain.models import (
    APPROVAL_DECISION_ACTOR_MAX_LENGTH,
    APPROVAL_DECISION_COMMENT_MAX_LENGTH,
)


def test_approve_body_accepts_asserted_actor_and_request_id() -> None:
    body = ApproveApprovalRequestBody(
        actor_reference="operator:alice",
        decision_request_id=uuid4(),
        comment="Approved after operational review.",
    )

    assert body.actor_reference == "operator:alice"
    assert body.comment == "Approved after operational review."


def test_approve_body_allows_omitted_comment() -> None:
    body = ApproveApprovalRequestBody(
        actor_reference="operator:alice",
        decision_request_id=uuid4(),
    )

    assert body.comment is None


def test_reject_body_requires_comment() -> None:
    with pytest.raises(ValidationError):
        RejectApprovalRequestBody(
            actor_reference="operator:alice",
            decision_request_id=uuid4(),
            comment="",
        )


def test_actor_reference_is_normalized_and_bounded() -> None:
    with pytest.raises(ValidationError):
        ApproveApprovalRequestBody(
            actor_reference=" operator:alice ",
            decision_request_id=uuid4(),
        )


def test_decision_body_rejects_extra_fields() -> None:
    with pytest.raises(ValidationError):
        ApproveApprovalRequestBody.model_validate(
            {
                "actor_reference": "operator:alice",
                "decision_request_id": str(uuid4()),
                "unexpected": "value",
            },
        )


def test_approve_body_rejects_missing_actor() -> None:
    with pytest.raises(ValidationError):
        ApproveApprovalRequestBody.model_validate(
            {
                "decision_request_id": str(uuid4()),
            },
        )


def test_approve_body_rejects_whitespace_actor() -> None:
    with pytest.raises(ValidationError):
        ApproveApprovalRequestBody(
            actor_reference="   ",
            decision_request_id=uuid4(),
        )


def test_approve_body_rejects_overlong_actor() -> None:
    with pytest.raises(ValidationError):
        ApproveApprovalRequestBody(
            actor_reference="a" * (APPROVAL_DECISION_ACTOR_MAX_LENGTH + 1),
            decision_request_id=uuid4(),
        )


def test_approve_body_rejects_malformed_request_uuid() -> None:
    with pytest.raises(ValidationError):
        ApproveApprovalRequestBody.model_validate(
            {
                "actor_reference": "operator:alice",
                "decision_request_id": "not-a-uuid",
            },
        )


def test_reject_body_rejects_missing_comment() -> None:
    with pytest.raises(ValidationError):
        RejectApprovalRequestBody.model_validate(
            {
                "actor_reference": "operator:alice",
                "decision_request_id": str(uuid4()),
            },
        )


def test_reject_body_rejects_blank_comment() -> None:
    with pytest.raises(ValidationError):
        RejectApprovalRequestBody(
            actor_reference="operator:alice",
            decision_request_id=uuid4(),
            comment="   ",
        )


def test_reject_body_rejects_overlong_comment() -> None:
    with pytest.raises(ValidationError):
        RejectApprovalRequestBody(
            actor_reference="operator:alice",
            decision_request_id=uuid4(),
            comment="c" * (APPROVAL_DECISION_COMMENT_MAX_LENGTH + 1),
        )


@pytest.mark.parametrize(
    "field_name",
    [
        "decided_at",
        "correlation_id",
        "status",
    ],
)
def test_decision_body_rejects_client_supplied_server_fields(
    field_name: str,
) -> None:
    payload = {
        "actor_reference": "operator:alice",
        "decision_request_id": str(uuid4()),
        field_name: "client-supplied",
    }

    with pytest.raises(ValidationError):
        ApproveApprovalRequestBody.model_validate(payload)
