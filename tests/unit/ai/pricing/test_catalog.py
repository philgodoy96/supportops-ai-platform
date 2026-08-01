"""Unit tests for the immutable versioned pricing catalog."""

from datetime import date
from decimal import Decimal

import pytest

from supportops.ai.pricing.catalog import (
    DEFAULT_PRICING_CATALOG,
    PRICING_CATALOG_VERSION,
    DuplicateModelPricingError,
    ModelPricing,
    PricingCatalog,
)


def _pricing(
    *,
    provider: str = "test-provider",
    model: str = "test-model",
) -> ModelPricing:
    return ModelPricing(
        provider=provider,
        model=model,
        effective_from=date(2026, 8, 1),
        input_cost_per_million_tokens=Decimal("1.25"),
        cached_input_cost_per_million_tokens=Decimal("0.125"),
        output_cost_per_million_tokens=Decimal("10.00"),
    )


def test_returns_exact_provider_and_model_pricing() -> None:
    pricing = _pricing()
    catalog = PricingCatalog(
        version="test-catalog-v1",
        entries=(pricing,),
    )

    result = catalog.find(
        provider="test-provider",
        model="test-model",
    )

    assert result is pricing


def test_lookup_requires_exact_provider_and_model() -> None:
    catalog = PricingCatalog(
        version="test-catalog-v1",
        entries=(_pricing(),),
    )

    assert (
        catalog.find(
            provider="another-provider",
            model="test-model",
        )
        is None
    )
    assert (
        catalog.find(
            provider="test-provider",
            model="another-model",
        )
        is None
    )


def test_rejects_duplicate_provider_and_model() -> None:
    with pytest.raises(
        DuplicateModelPricingError,
        match="test-provider/test-model",
    ):
        PricingCatalog(
            version="test-catalog-v1",
            entries=(
                _pricing(),
                _pricing(),
            ),
        )


def test_allows_same_model_for_different_providers() -> None:
    catalog = PricingCatalog(
        version="test-catalog-v1",
        entries=(
            _pricing(provider="first-provider"),
            _pricing(provider="second-provider"),
        ),
    )

    assert len(catalog) == 2


@pytest.mark.parametrize(
    "invalid_price",
    [
        Decimal("-0.01"),
        Decimal("NaN"),
        Decimal("Infinity"),
        Decimal("-Infinity"),
    ],
)
def test_rejects_invalid_prices(
    invalid_price: Decimal,
) -> None:
    with pytest.raises(ValueError):
        ModelPricing(
            provider="test-provider",
            model="test-model",
            effective_from=date(2026, 8, 1),
            input_cost_per_million_tokens=invalid_price,
            cached_input_cost_per_million_tokens=Decimal("0"),
            output_cost_per_million_tokens=Decimal("0"),
        )


def test_requires_decimal_prices() -> None:
    with pytest.raises(TypeError, match="must be a Decimal"):
        ModelPricing(
            provider="test-provider",
            model="test-model",
            effective_from=date(2026, 8, 1),
            input_cost_per_million_tokens=1.25,  # type: ignore[arg-type]
            cached_input_cost_per_million_tokens=Decimal("0"),
            output_cost_per_million_tokens=Decimal("0"),
        )


def test_default_catalog_has_explicit_version() -> None:
    assert DEFAULT_PRICING_CATALOG.version == (PRICING_CATALOG_VERSION)


def test_default_catalog_contains_openai_gpt_5_nano() -> None:
    pricing = DEFAULT_PRICING_CATALOG.find(
        provider="openai",
        model="gpt-5-nano",
    )

    assert pricing is not None
    assert pricing.input_cost_per_million_tokens == Decimal("0.05")
    assert pricing.cached_input_cost_per_million_tokens == Decimal("0.005")
    assert pricing.output_cost_per_million_tokens == Decimal("0.40")


def test_default_catalog_contains_explicit_zero_cost_mock() -> None:
    pricing = DEFAULT_PRICING_CATALOG.find(
        provider="mock",
        model="mock-ticket-classifier-v1",
    )

    assert pricing is not None
    assert pricing.input_cost_per_million_tokens == Decimal("0")
    assert pricing.cached_input_cost_per_million_tokens == Decimal("0")
    assert pricing.output_cost_per_million_tokens == Decimal("0")
