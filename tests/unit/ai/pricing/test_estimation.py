"""Unit tests for Decimal-based LLM cost estimation."""

from datetime import date
from decimal import Decimal

import pytest

from supportops.ai.gateway.contracts import LLMTokenUsage
from supportops.ai.pricing.catalog import (
    ModelPricing,
    PricingCatalog,
)
from supportops.ai.pricing.estimation import (
    LLMCostEstimate,
    estimate_llm_cost,
)


def _catalog() -> PricingCatalog:
    return PricingCatalog(
        version="test-catalog-v1",
        entries=(
            ModelPricing(
                provider="test-provider",
                model="test-model",
                effective_from=date(2026, 8, 1),
                input_cost_per_million_tokens=Decimal("1.00"),
                cached_input_cost_per_million_tokens=Decimal(
                    "0.10",
                ),
                output_cost_per_million_tokens=Decimal("2.00"),
            ),
        ),
    )


def test_calculates_input_cached_output_and_total_cost() -> None:
    estimate = estimate_llm_cost(
        provider="test-provider",
        model="test-model",
        usage=LLMTokenUsage(
            input_tokens=1_000_000,
            cached_input_tokens=250_000,
            output_tokens=500_000,
            reasoning_tokens=100_000,
            total_tokens=1_500_000,
        ),
        catalog=_catalog(),
    )

    assert estimate.pricing_found is True
    assert estimate.pricing_catalog_version == "test-catalog-v1"
    assert estimate.estimated_input_cost_usd == Decimal("0.750000000000")
    assert estimate.estimated_cached_input_cost_usd == Decimal("0.025000000000")
    assert estimate.estimated_output_cost_usd == Decimal("1.000000000000")
    assert estimate.estimated_total_cost_usd == Decimal("1.775000000000")


def test_reasoning_tokens_are_not_charged_twice() -> None:
    without_reasoning = estimate_llm_cost(
        provider="test-provider",
        model="test-model",
        usage=LLMTokenUsage(
            input_tokens=100,
            cached_input_tokens=0,
            output_tokens=40,
            reasoning_tokens=0,
            total_tokens=140,
        ),
        catalog=_catalog(),
    )
    with_reasoning = estimate_llm_cost(
        provider="test-provider",
        model="test-model",
        usage=LLMTokenUsage(
            input_tokens=100,
            cached_input_tokens=0,
            output_tokens=40,
            reasoning_tokens=30,
            total_tokens=140,
        ),
        catalog=_catalog(),
    )

    assert with_reasoning.estimated_output_cost_usd == without_reasoning.estimated_output_cost_usd
    assert with_reasoning.estimated_total_cost_usd == without_reasoning.estimated_total_cost_usd


def test_explicit_zero_tokens_produce_explicit_zero_costs() -> None:
    estimate = estimate_llm_cost(
        provider="test-provider",
        model="test-model",
        usage=LLMTokenUsage(
            input_tokens=0,
            cached_input_tokens=0,
            output_tokens=0,
            reasoning_tokens=0,
            total_tokens=0,
        ),
        catalog=_catalog(),
    )

    assert estimate.estimated_input_cost_usd == Decimal("0.000000000000")
    assert estimate.estimated_cached_input_cost_usd == Decimal("0.000000000000")
    assert estimate.estimated_output_cost_usd == Decimal("0.000000000000")
    assert estimate.estimated_total_cost_usd == Decimal("0.000000000000")


def test_unknown_usage_remains_unknown() -> None:
    estimate = estimate_llm_cost(
        provider="test-provider",
        model="test-model",
        usage=None,
        catalog=_catalog(),
    )

    assert estimate.pricing_found is True
    assert estimate.estimated_input_cost_usd is None
    assert estimate.estimated_cached_input_cost_usd is None
    assert estimate.estimated_output_cost_usd is None
    assert estimate.estimated_total_cost_usd is None


def test_missing_cached_input_prevents_input_and_total_estimate() -> None:
    estimate = estimate_llm_cost(
        provider="test-provider",
        model="test-model",
        usage=LLMTokenUsage(
            input_tokens=100,
            cached_input_tokens=None,
            output_tokens=20,
            total_tokens=120,
        ),
        catalog=_catalog(),
    )

    assert estimate.estimated_input_cost_usd is None
    assert estimate.estimated_cached_input_cost_usd is None
    assert estimate.estimated_output_cost_usd == Decimal("0.000040000000")
    assert estimate.estimated_total_cost_usd is None


def test_missing_output_tokens_prevents_total_estimate() -> None:
    estimate = estimate_llm_cost(
        provider="test-provider",
        model="test-model",
        usage=LLMTokenUsage(
            input_tokens=100,
            cached_input_tokens=0,
            output_tokens=None,
            total_tokens=None,
        ),
        catalog=_catalog(),
    )

    assert estimate.estimated_input_cost_usd == Decimal("0.000100000000")
    assert estimate.estimated_cached_input_cost_usd == Decimal("0.000000000000")
    assert estimate.estimated_output_cost_usd is None
    assert estimate.estimated_total_cost_usd is None


def test_unknown_model_preserves_usage_but_returns_null_costs() -> None:
    usage = LLMTokenUsage(
        input_tokens=100,
        cached_input_tokens=0,
        output_tokens=20,
        total_tokens=120,
    )

    estimate = estimate_llm_cost(
        provider="test-provider",
        model="unknown-model",
        usage=usage,
        catalog=_catalog(),
    )

    assert usage.input_tokens == 100
    assert estimate.pricing_found is False
    assert estimate.pricing_catalog_version == "test-catalog-v1"
    assert estimate.estimated_input_cost_usd is None
    assert estimate.estimated_cached_input_cost_usd is None
    assert estimate.estimated_output_cost_usd is None
    assert estimate.estimated_total_cost_usd is None


def test_unknown_provider_does_not_reuse_another_provider_price() -> None:
    estimate = estimate_llm_cost(
        provider="another-provider",
        model="test-model",
        usage=LLMTokenUsage(
            input_tokens=100,
            cached_input_tokens=0,
            output_tokens=20,
            total_tokens=120,
        ),
        catalog=_catalog(),
    )

    assert estimate.pricing_found is False
    assert estimate.estimated_total_cost_usd is None


def test_small_costs_use_deterministic_decimal_rounding() -> None:
    estimate = estimate_llm_cost(
        provider="test-provider",
        model="test-model",
        usage=LLMTokenUsage(
            input_tokens=1,
            cached_input_tokens=0,
            output_tokens=1,
            total_tokens=2,
        ),
        catalog=_catalog(),
    )

    assert estimate.estimated_input_cost_usd == Decimal("0.000001000000")
    assert estimate.estimated_cached_input_cost_usd == Decimal("0.000000000000")
    assert estimate.estimated_output_cost_usd == Decimal("0.000002000000")
    assert estimate.estimated_total_cost_usd == Decimal("0.000003000000")


def test_estimated_total_must_equal_components() -> None:
    with pytest.raises(
        ValueError,
        match="must equal the stored cost components",
    ):
        LLMCostEstimate(
            pricing_catalog_version="test-catalog-v1",
            pricing_found=True,
            estimated_input_cost_usd=Decimal("1.00"),
            estimated_cached_input_cost_usd=Decimal("0.10"),
            estimated_output_cost_usd=Decimal("2.00"),
            estimated_total_cost_usd=Decimal("9.99"),
        )


def test_unknown_pricing_cannot_define_zero_cost() -> None:
    with pytest.raises(
        ValueError,
        match="Unknown pricing cannot define cost components",
    ):
        LLMCostEstimate(
            pricing_catalog_version="test-catalog-v1",
            pricing_found=False,
            estimated_input_cost_usd=Decimal("0"),
            estimated_cached_input_cost_usd=None,
            estimated_output_cost_usd=None,
            estimated_total_cost_usd=None,
        )
