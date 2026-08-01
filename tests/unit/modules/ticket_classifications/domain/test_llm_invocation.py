"""Unit tests for durable logical LLM invocation records."""

from dataclasses import FrozenInstanceError, replace
from datetime import UTC, datetime
from decimal import Decimal
from typing import cast
from uuid import uuid4

import pytest

from supportops.ai.gateway.errors import LLMErrorCode
from supportops.ai.gateway.results import LLMInvocationStatus
from supportops.modules.ticket_classifications.domain.models import (
    LLMInvocation,
)

_NOW = datetime(
    2026,
    8,
    1,
    17,
    0,
    tzinfo=UTC,
)
_PROMPT_HASH = "b" * 64


def _invocation() -> LLMInvocation:
    return LLMInvocation.create(
        workspace_id=uuid4(),
        ticket_id=uuid4(),
        agent_run_id=uuid4(),
        agent_run_attempt_id=uuid4(),
        invocation_sequence=1,
        status=LLMInvocationStatus.SUCCEEDED,
        provider="openai",
        model="gpt-5-nano",
        provider_request_id="req_test_1",
        prompt_id="ticket-classification",
        prompt_version=1,
        prompt_content_hash=_PROMPT_HASH,
        schema_version="ticket-classification-v1",
        input_tokens=100,
        cached_input_tokens=20,
        output_tokens=25,
        reasoning_tokens=5,
        total_tokens=125,
        pricing_catalog_version=("supportops-pricing-2026-08-01"),
        pricing_found=True,
        estimated_input_cost_usd=Decimal(
            "0.000004000000",
        ),
        estimated_cached_input_cost_usd=Decimal(
            "0.000000100000",
        ),
        estimated_output_cost_usd=Decimal(
            "0.000010000000",
        ),
        estimated_total_cost_usd=Decimal(
            "0.000014100000",
        ),
        latency_ms=125,
        error_code=None,
        now=_NOW,
    )


def test_create_preserves_invocation_provenance() -> None:
    invocation = _invocation()

    assert invocation.invocation_sequence == 1
    assert invocation.status is LLMInvocationStatus.SUCCEEDED
    assert invocation.provider == "openai"
    assert invocation.model == "gpt-5-nano"
    assert invocation.provider_request_id == "req_test_1"
    assert invocation.prompt_id == "ticket-classification"
    assert invocation.prompt_version == 1
    assert invocation.prompt_content_hash == _PROMPT_HASH
    assert invocation.schema_version == ("ticket-classification-v1")
    assert invocation.input_tokens == 100
    assert invocation.cached_input_tokens == 20
    assert invocation.output_tokens == 25
    assert invocation.reasoning_tokens == 5
    assert invocation.total_tokens == 125
    assert invocation.pricing_found is True
    assert invocation.estimated_total_cost_usd == Decimal(
        "0.000014100000",
    )
    assert invocation.latency_ms == 125
    assert invocation.error_code is None
    assert invocation.created_at == _NOW

    with pytest.raises(FrozenInstanceError):
        invocation.status = LLMInvocationStatus.PROVIDER_FAILED  # type: ignore[misc]


def test_accepts_failed_invocation_with_error_code() -> None:
    invocation = replace(
        _invocation(),
        status=LLMInvocationStatus.TIMED_OUT,
        error_code=LLMErrorCode.TIMEOUT,
    )

    assert invocation.status is LLMInvocationStatus.TIMED_OUT
    assert invocation.error_code is LLMErrorCode.TIMEOUT


def test_successful_invocation_rejects_error_code() -> None:
    with pytest.raises(
        ValueError,
        match=("Successful invocations cannot define an error_code"),
    ):
        replace(
            _invocation(),
            error_code=LLMErrorCode.TIMEOUT,
        )


def test_failed_invocation_requires_error_code() -> None:
    with pytest.raises(
        ValueError,
        match="Failed invocations require an error_code",
    ):
        replace(
            _invocation(),
            status=LLMInvocationStatus.PROVIDER_FAILED,
            error_code=None,
        )


def test_rejects_raw_status_string() -> None:
    with pytest.raises(
        ValueError,
        match="supported LLMInvocationStatus",
    ):
        replace(
            _invocation(),
            status=cast(
                LLMInvocationStatus,
                "succeeded",
            ),
        )


def test_rejects_raw_error_code_string() -> None:
    with pytest.raises(
        ValueError,
        match="supported LLMErrorCode",
    ):
        replace(
            _invocation(),
            status=LLMInvocationStatus.TIMED_OUT,
            error_code=cast(
                LLMErrorCode,
                "llm_timeout",
            ),
        )


@pytest.mark.parametrize(
    "invocation_sequence",
    [
        0,
        -1,
    ],
)
def test_rejects_non_positive_invocation_sequence(
    invocation_sequence: int,
) -> None:
    with pytest.raises(
        ValueError,
        match="invocation_sequence must be positive",
    ):
        replace(
            _invocation(),
            invocation_sequence=invocation_sequence,
        )


def test_rejects_cached_tokens_greater_than_input_tokens() -> None:
    with pytest.raises(
        ValueError,
        match=("cached_input_tokens cannot exceed input_tokens"),
    ):
        replace(
            _invocation(),
            input_tokens=10,
            cached_input_tokens=11,
            total_tokens=35,
        )


def test_rejects_reasoning_tokens_greater_than_output_tokens() -> None:
    with pytest.raises(
        ValueError,
        match="reasoning_tokens cannot exceed output_tokens",
    ):
        replace(
            _invocation(),
            output_tokens=4,
            reasoning_tokens=5,
            total_tokens=104,
        )


def test_rejects_inconsistent_total_tokens() -> None:
    with pytest.raises(
        ValueError,
        match=("total_tokens must equal input_tokens plus output_tokens"),
    ):
        replace(
            _invocation(),
            total_tokens=999,
        )


def test_unknown_pricing_requires_null_costs() -> None:
    with pytest.raises(
        ValueError,
        match="Unknown pricing cannot define cost components",
    ):
        replace(
            _invocation(),
            pricing_found=False,
        )


def test_unknown_pricing_accepts_null_costs() -> None:
    invocation = replace(
        _invocation(),
        pricing_found=False,
        estimated_input_cost_usd=None,
        estimated_cached_input_cost_usd=None,
        estimated_output_cost_usd=None,
        estimated_total_cost_usd=None,
    )

    assert invocation.pricing_found is False
    assert invocation.estimated_total_cost_usd is None


def test_known_mock_pricing_accepts_explicit_zero_cost() -> None:
    invocation = replace(
        _invocation(),
        provider="mock",
        model="mock-ticket-classifier-v1",
        estimated_input_cost_usd=Decimal(
            "0.000000000000",
        ),
        estimated_cached_input_cost_usd=Decimal(
            "0.000000000000",
        ),
        estimated_output_cost_usd=Decimal(
            "0.000000000000",
        ),
        estimated_total_cost_usd=Decimal(
            "0.000000000000",
        ),
    )

    assert invocation.pricing_found is True
    assert invocation.estimated_total_cost_usd == Decimal(
        "0.000000000000",
    )


def test_total_cost_must_equal_components() -> None:
    with pytest.raises(
        ValueError,
        match="must equal the stored cost components",
    ):
        replace(
            _invocation(),
            estimated_total_cost_usd=Decimal(
                "1.000000000000",
            ),
        )


def test_partial_cost_requires_unknown_total() -> None:
    invocation = replace(
        _invocation(),
        estimated_input_cost_usd=None,
        estimated_cached_input_cost_usd=None,
        estimated_total_cost_usd=None,
    )

    assert invocation.estimated_output_cost_usd is not None
    assert invocation.estimated_total_cost_usd is None


@pytest.mark.parametrize(
    "invalid_cost",
    [
        Decimal("-0.000000000001"),
        Decimal("NaN"),
        Decimal("Infinity"),
        Decimal("-Infinity"),
        Decimal("0.0000000000001"),
        Decimal("100000000.000000000000"),
    ],
)
def test_rejects_invalid_monetary_values(
    invalid_cost: Decimal,
) -> None:
    with pytest.raises(ValueError):
        replace(
            _invocation(),
            estimated_input_cost_usd=invalid_cost,
        )


def test_rejects_non_decimal_monetary_value() -> None:
    with pytest.raises(
        TypeError,
        match="must be a Decimal",
    ):
        replace(
            _invocation(),
            estimated_input_cost_usd=cast(
                Decimal,
                0.1,
            ),
        )


def test_rejects_negative_latency() -> None:
    with pytest.raises(
        ValueError,
        match="latency_ms must be non-negative",
    ):
        replace(
            _invocation(),
            latency_ms=-1,
        )


def test_rejects_non_boolean_pricing_found() -> None:
    with pytest.raises(
        ValueError,
        match="pricing_found must be a boolean",
    ):
        replace(
            _invocation(),
            pricing_found=cast(bool, 1),
        )


def test_rejects_non_utc_timestamp() -> None:
    with pytest.raises(
        ValueError,
        match="created_at must be a UTC-aware timestamp",
    ):
        replace(
            _invocation(),
            created_at=datetime(
                2026,
                8,
                1,
                17,
                0,
            ),
        )
