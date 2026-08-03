"""Unit tests for approval decision HTTP schemas."""

from uuid import uuid4

import pytest
from pydantic import ValidationError

from supportops.modules.approvals.api.schemas import (
    ApproveApprovalRequestBody,
    RejectApprovalRequestBody,
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
