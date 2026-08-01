"""Deterministic Decimal-based estimation of LLM token costs."""

from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal

from supportops.ai.gateway.contracts import LLMTokenUsage
from supportops.ai.pricing.catalog import (
    DEFAULT_PRICING_CATALOG,
    ModelPricing,
    PricingCatalog,
)

_TOKENS_PER_MILLION = Decimal("1000000")
_USD_QUANTUM = Decimal("0.000000000001")


@dataclass(frozen=True, slots=True)
class LLMCostEstimate:
    """Estimated USD costs calculated from one pricing catalog version."""

    pricing_catalog_version: str
    pricing_found: bool
    estimated_input_cost_usd: Decimal | None
    estimated_cached_input_cost_usd: Decimal | None
    estimated_output_cost_usd: Decimal | None
    estimated_total_cost_usd: Decimal | None

    def __post_init__(self) -> None:
        _validate_required_text(
            self.pricing_catalog_version,
            field_name="pricing_catalog_version",
        )

        components = (
            self.estimated_input_cost_usd,
            self.estimated_cached_input_cost_usd,
            self.estimated_output_cost_usd,
        )

        for component in components:
            if component is not None and component < 0:
                raise ValueError(
                    "Estimated cost components must be non-negative.",
                )

        if not self.pricing_found:
            if any(component is not None for component in components):
                raise ValueError(
                    "Unknown pricing cannot define cost components.",
                )

            if self.estimated_total_cost_usd is not None:
                raise ValueError(
                    "Unknown pricing cannot define a total cost.",
                )

            return

        if all(component is not None for component in components):
            expected_total = _quantize_usd(
                sum(
                    (component for component in components if component is not None),
                    start=Decimal("0"),
                ),
            )

            if self.estimated_total_cost_usd != expected_total:
                raise ValueError(
                    "estimated_total_cost_usd must equal the stored cost components.",
                )
        elif self.estimated_total_cost_usd is not None:
            raise ValueError(
                "Total cost must remain unknown when any required component is unknown.",
            )


def estimate_llm_cost(
    *,
    provider: str,
    model: str,
    usage: LLMTokenUsage | None,
    catalog: PricingCatalog = DEFAULT_PRICING_CATALOG,
) -> LLMCostEstimate:
    """Estimate one invocation cost without inventing missing usage."""

    pricing = catalog.find(
        provider=provider,
        model=model,
    )

    if pricing is None:
        return _unknown_pricing_estimate(
            catalog=catalog,
        )

    if usage is None:
        return _known_pricing_with_unknown_usage(
            catalog=catalog,
        )

    (
        input_cost,
        cached_input_cost,
    ) = _estimate_input_costs(
        usage=usage,
        pricing=pricing,
    )

    output_cost = _estimate_component(
        tokens=usage.output_tokens,
        price_per_million_tokens=(pricing.output_cost_per_million_tokens),
    )

    component_costs = (
        input_cost,
        cached_input_cost,
        output_cost,
    )

    total_cost = None

    if all(component_cost is not None for component_cost in component_costs):
        total_cost = _quantize_usd(
            sum(
                (
                    component_cost
                    for component_cost in component_costs
                    if component_cost is not None
                ),
                start=Decimal("0"),
            ),
        )

    return LLMCostEstimate(
        pricing_catalog_version=catalog.version,
        pricing_found=True,
        estimated_input_cost_usd=input_cost,
        estimated_cached_input_cost_usd=cached_input_cost,
        estimated_output_cost_usd=output_cost,
        estimated_total_cost_usd=total_cost,
    )


def _estimate_input_costs(
    *,
    usage: LLMTokenUsage,
    pricing: ModelPricing,
) -> tuple[Decimal | None, Decimal | None]:
    if usage.input_tokens is None or usage.cached_input_tokens is None:
        return (
            None,
            None,
        )

    uncached_input_tokens = usage.input_tokens - usage.cached_input_tokens

    input_cost = _estimate_component(
        tokens=uncached_input_tokens,
        price_per_million_tokens=(pricing.input_cost_per_million_tokens),
    )
    cached_input_cost = _estimate_component(
        tokens=usage.cached_input_tokens,
        price_per_million_tokens=(pricing.cached_input_cost_per_million_tokens),
    )

    return (
        input_cost,
        cached_input_cost,
    )


def _estimate_component(
    *,
    tokens: int | None,
    price_per_million_tokens: Decimal,
) -> Decimal | None:
    if tokens is None:
        return None

    return _quantize_usd(
        Decimal(tokens) * price_per_million_tokens / _TOKENS_PER_MILLION,
    )


def _unknown_pricing_estimate(
    *,
    catalog: PricingCatalog,
) -> LLMCostEstimate:
    return LLMCostEstimate(
        pricing_catalog_version=catalog.version,
        pricing_found=False,
        estimated_input_cost_usd=None,
        estimated_cached_input_cost_usd=None,
        estimated_output_cost_usd=None,
        estimated_total_cost_usd=None,
    )


def _known_pricing_with_unknown_usage(
    *,
    catalog: PricingCatalog,
) -> LLMCostEstimate:
    return LLMCostEstimate(
        pricing_catalog_version=catalog.version,
        pricing_found=True,
        estimated_input_cost_usd=None,
        estimated_cached_input_cost_usd=None,
        estimated_output_cost_usd=None,
        estimated_total_cost_usd=None,
    )


def _quantize_usd(
    value: Decimal,
) -> Decimal:
    return value.quantize(
        _USD_QUANTUM,
        rounding=ROUND_HALF_UP,
    )


def _validate_required_text(
    value: str,
    *,
    field_name: str,
) -> None:
    if not value:
        raise ValueError(f"{field_name} is required.")

    if value != value.strip():
        raise ValueError(
            f"{field_name} must not contain surrounding whitespace.",
        )
