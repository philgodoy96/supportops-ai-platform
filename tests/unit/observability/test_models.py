"""Unit tests for application-owned observability models."""

from datetime import UTC, datetime
from decimal import Decimal

import pytest

from supportops.observability.models import (
    CostDetails,
    EventObservation,
    ObservabilityCaptureMode,
    ObservabilityProvider,
    ObservationAttributes,
    ObservationType,
    PricingStatus,
    TraceAttributes,
    UsageDetails,
)


def test_observability_provider_values_are_stable() -> None:
    assert ObservabilityProvider.NOOP.value == "noop"
    assert ObservabilityProvider.LANGFUSE.value == "langfuse"


def test_capture_mode_does_not_offer_unrestricted_raw_capture() -> None:
    assert {mode.value for mode in ObservabilityCaptureMode} == {
        "metadata_only",
        "redacted_content",
    }


def test_observation_type_values_match_application_taxonomy() -> None:
    assert {observation_type.value for observation_type in ObservationType} == {
        "agent",
        "span",
        "generation",
        "embedding",
        "retriever",
        "tool",
        "chain",
        "evaluator",
        "event",
    }


def test_usage_details_accept_mutually_exclusive_buckets() -> None:
    usage = UsageDetails(
        input_tokens=80,
        cached_input_tokens=20,
        output_tokens=30,
        reasoning_tokens=10,
        total_tokens=140,
    )

    assert usage.total_tokens == 140


def test_usage_details_accept_total_without_breakdown() -> None:
    usage = UsageDetails(total_tokens=42)

    assert usage.total_tokens == 42


def test_usage_details_reject_negative_values() -> None:
    with pytest.raises(ValueError, match="input_tokens must be non-negative"):
        UsageDetails(input_tokens=-1)


def test_usage_details_reject_mismatched_total() -> None:
    with pytest.raises(ValueError, match="total_tokens must equal"):
        UsageDetails(
            input_tokens=10,
            output_tokens=5,
            total_tokens=20,
        )


def test_unknown_pricing_omits_cost_values() -> None:
    cost = CostDetails(
        pricing_status=PricingStatus.UNKNOWN,
        pricing_catalog_version=None,
    )

    assert cost.total_cost is None


def test_unknown_pricing_rejects_fabricated_zero_cost() -> None:
    with pytest.raises(ValueError, match="unknown pricing must not contain"):
        CostDetails(
            pricing_status=PricingStatus.UNKNOWN,
            total_cost=Decimal("0"),
        )


def test_known_mock_pricing_accepts_explicit_zero_cost() -> None:
    cost = CostDetails(
        pricing_status=PricingStatus.KNOWN,
        input_cost=Decimal("0"),
        output_cost=Decimal("0"),
        total_cost=Decimal("0"),
        pricing_catalog_version="catalog-v1",
    )

    assert cost.total_cost == Decimal("0")


def test_cost_details_reject_mismatched_total() -> None:
    with pytest.raises(ValueError, match="total_cost must equal"):
        CostDetails(
            pricing_status=PricingStatus.KNOWN,
            input_cost=Decimal("0.01"),
            output_cost=Decimal("0.02"),
            total_cost=Decimal("0.04"),
            pricing_catalog_version="catalog-v1",
        )


def test_trace_attributes_require_non_blank_identity() -> None:
    with pytest.raises(ValueError, match="trace_id must not be blank"):
        TraceAttributes(
            trace_id=" ",
            name="ticket-processing",
        )


def test_trace_attributes_reject_duplicate_tags() -> None:
    with pytest.raises(ValueError, match="trace tags must be unique"):
        TraceAttributes(
            trace_id="trace-id",
            name="ticket-processing",
            tags=("supportops", "supportops"),
        )


def test_observation_attributes_require_non_blank_name() -> None:
    with pytest.raises(ValueError, match="name must not be blank"):
        ObservationAttributes(
            name=" ",
            observation_type=ObservationType.SPAN,
        )


def test_event_timestamp_must_be_timezone_aware() -> None:
    with pytest.raises(ValueError, match="occurred_at must be timezone-aware"):
        EventObservation(
            name="workflow_paused",
            occurred_at=datetime(2026, 8, 3, 12, 0, 0),
        )


def test_event_accepts_authoritative_timezone_aware_timestamp() -> None:
    event = EventObservation(
        name="workflow_paused",
        occurred_at=datetime(2026, 8, 3, 12, 0, 0, tzinfo=UTC),
    )

    assert event.occurred_at is not None
    assert event.occurred_at.tzinfo is UTC
