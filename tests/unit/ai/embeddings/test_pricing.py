"""Unit tests for embedding pricing and cost estimation."""

from datetime import date
from decimal import Decimal
from typing import cast

import pytest

from supportops.ai.embeddings.pricing import (
    DEFAULT_EMBEDDING_PRICING_CATALOG,
    EMBEDDING_PRICING_CATALOG_VERSION,
    DuplicateEmbeddingModelPricingError,
    EmbeddingModelPricing,
    EmbeddingPricingCatalog,
    estimate_embedding_cost,
)


def create_pricing(
    *,
    provider: str = "test-provider",
    model: str = "test-model",
    price: Decimal = Decimal("0.02"),
) -> EmbeddingModelPricing:
    """Create one deterministic embedding price."""

    return EmbeddingModelPricing(
        provider=provider,
        model=model,
        effective_from=date(2026, 8, 1),
        input_cost_per_million_tokens=price,
    )


def test_catalog_returns_exact_provider_and_model_price() -> None:
    pricing = create_pricing()
    catalog = EmbeddingPricingCatalog(
        version="test-embedding-pricing-v1",
        entries=(pricing,),
    )

    assert (
        catalog.find(
            provider="test-provider",
            model="test-model",
        )
        is pricing
    )
    assert (
        catalog.find(
            provider="other-provider",
            model="test-model",
        )
        is None
    )


def test_catalog_rejects_duplicate_provider_and_model() -> None:
    with pytest.raises(
        DuplicateEmbeddingModelPricingError,
        match="test-provider/test-model",
    ):
        EmbeddingPricingCatalog(
            version="test-embedding-pricing-v1",
            entries=(
                create_pricing(),
                create_pricing(),
            ),
        )


def test_catalog_allows_same_model_for_different_providers() -> None:
    catalog = EmbeddingPricingCatalog(
        version="test-embedding-pricing-v1",
        entries=(
            create_pricing(provider="first-provider"),
            create_pricing(provider="second-provider"),
        ),
    )

    assert len(catalog) == 2


def test_default_catalog_registers_mock_and_openai_models() -> None:
    mock_pricing = DEFAULT_EMBEDDING_PRICING_CATALOG.find(
        provider="mock",
        model="mock-hashing-embedding-v1",
    )
    openai_pricing = DEFAULT_EMBEDDING_PRICING_CATALOG.find(
        provider="openai",
        model="text-embedding-3-small",
    )

    assert DEFAULT_EMBEDDING_PRICING_CATALOG.version == EMBEDDING_PRICING_CATALOG_VERSION
    assert mock_pricing is not None
    assert mock_pricing.input_cost_per_million_tokens == Decimal("0")
    assert openai_pricing is not None
    assert openai_pricing.input_cost_per_million_tokens == Decimal("0.02")


def test_estimates_openai_input_token_cost() -> None:
    estimate = estimate_embedding_cost(
        provider="openai",
        model="text-embedding-3-small",
        input_tokens=500_000,
        catalog=DEFAULT_EMBEDDING_PRICING_CATALOG,
    )

    assert estimate.pricing_found
    assert estimate.estimated_cost_usd == Decimal("0.010000000000")
    assert estimate.pricing_catalog_version == (EMBEDDING_PRICING_CATALOG_VERSION)


def test_estimates_small_usage_with_database_scale() -> None:
    estimate = estimate_embedding_cost(
        provider="openai",
        model="text-embedding-3-small",
        input_tokens=18,
        catalog=DEFAULT_EMBEDDING_PRICING_CATALOG,
    )

    assert estimate.estimated_cost_usd == Decimal("0.000000360000")


def test_mock_embedding_cost_is_zero_not_unknown() -> None:
    estimate = estimate_embedding_cost(
        provider="mock",
        model="mock-hashing-embedding-v1",
        input_tokens=1_000,
        catalog=DEFAULT_EMBEDDING_PRICING_CATALOG,
    )

    assert estimate.pricing_found
    assert estimate.estimated_cost_usd == Decimal("0")


def test_unknown_model_preserves_unknown_cost() -> None:
    estimate = estimate_embedding_cost(
        provider="openai",
        model="unknown-embedding-model",
        input_tokens=100,
        catalog=DEFAULT_EMBEDDING_PRICING_CATALOG,
    )

    assert not estimate.pricing_found
    assert estimate.estimated_cost_usd is None
    assert estimate.pricing_catalog_version == (EMBEDDING_PRICING_CATALOG_VERSION)


def test_known_price_with_missing_usage_preserves_unknown_cost() -> None:
    estimate = estimate_embedding_cost(
        provider="openai",
        model="text-embedding-3-small",
        input_tokens=None,
        catalog=DEFAULT_EMBEDDING_PRICING_CATALOG,
    )

    assert estimate.pricing_found
    assert estimate.estimated_cost_usd is None


@pytest.mark.parametrize(
    "invalid_price",
    [
        Decimal("-0.01"),
        Decimal("NaN"),
        Decimal("Infinity"),
        Decimal("-Infinity"),
    ],
)
def test_pricing_rejects_invalid_decimal_amounts(
    invalid_price: Decimal,
) -> None:
    with pytest.raises(ValueError):
        create_pricing(price=invalid_price)


def test_pricing_requires_decimal_amount() -> None:
    with pytest.raises(
        TypeError,
        match="must be a Decimal",
    ):
        create_pricing(
            price=cast(Decimal, 0.02),
        )


def test_cost_estimation_rejects_negative_usage() -> None:
    with pytest.raises(
        ValueError,
        match="input_tokens must be non-negative",
    ):
        estimate_embedding_cost(
            provider="openai",
            model="text-embedding-3-small",
            input_tokens=-1,
            catalog=DEFAULT_EMBEDDING_PRICING_CATALOG,
        )
