from decimal import Decimal

import pytest
from pydantic import BaseModel, ConfigDict, ValidationError

from supportops.evaluation.contracts.predictions import (
    EvaluationPredictionEnvelope,
    EvaluationPredictionStatus,
)


class ClassificationPayload(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    category: str
    urgency: str


def test_succeeded_prediction_requires_typed_payload() -> None:
    prediction = EvaluationPredictionEnvelope[ClassificationPayload](
        case_id="case-001",
        status=EvaluationPredictionStatus.SUCCEEDED,
        payload=ClassificationPayload(category="security", urgency="critical"),
        latency_ms=120,
        input_tokens=100,
        output_tokens=20,
        estimated_cost_usd=Decimal("0.0012"),
    )

    assert prediction.payload is not None
    assert prediction.payload.category == "security"
    assert prediction.error_code is None
    assert prediction.canonical_payload()["estimated_cost_usd"] == "0.0012"


def test_failed_prediction_requires_error_code_and_no_payload() -> None:
    prediction = EvaluationPredictionEnvelope[ClassificationPayload](
        case_id="case-002",
        status=EvaluationPredictionStatus.FAILED,
        error_code="provider_timeout",
    )

    assert prediction.payload is None
    assert prediction.content_hash() == prediction.content_hash()


@pytest.mark.parametrize(
    "values",
    [
        {
            "status": EvaluationPredictionStatus.SUCCEEDED,
            "payload": None,
            "error_code": None,
        },
        {
            "status": EvaluationPredictionStatus.SUCCEEDED,
            "payload": ClassificationPayload(
                category="security",
                urgency="critical",
            ),
            "error_code": "unexpected",
        },
        {
            "status": EvaluationPredictionStatus.FAILED,
            "payload": ClassificationPayload(
                category="security",
                urgency="critical",
            ),
            "error_code": "provider_timeout",
        },
        {
            "status": EvaluationPredictionStatus.FAILED,
            "payload": None,
            "error_code": None,
        },
    ],
)
def test_prediction_rejects_inconsistent_status_contract(
    values: dict[str, object],
) -> None:
    with pytest.raises(ValidationError):
        EvaluationPredictionEnvelope[ClassificationPayload].model_validate(
            {"case_id": "case-invalid", **values}
        )
