"""Immutable versioned pricing catalog for supported LLM models."""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from types import MappingProxyType

PRICING_CATALOG_VERSION = "supportops-pricing-2026-08-01"


@dataclass(frozen=True, slots=True)
class ModelPricing:
    """Per-million-token prices for one provider and model."""

    provider: str
    model: str
    effective_from: date
    input_cost_per_million_tokens: Decimal
    cached_input_cost_per_million_tokens: Decimal
    output_cost_per_million_tokens: Decimal

    def __post_init__(self) -> None:
        _validate_required_text(
            self.provider,
            field_name="provider",
        )
        _validate_required_text(
            self.model,
            field_name="model",
        )

        prices = {
            "input_cost_per_million_tokens": (self.input_cost_per_million_tokens),
            "cached_input_cost_per_million_tokens": (self.cached_input_cost_per_million_tokens),
            "output_cost_per_million_tokens": (self.output_cost_per_million_tokens),
        }

        for field_name, value in prices.items():
            _validate_price(
                value,
                field_name=field_name,
            )


class DuplicateModelPricingError(ValueError):
    """Raised when one provider and model are registered twice."""


class PricingCatalog:
    """Immutable model-pricing lookup with an explicit catalog version."""

    __slots__ = (
        "_entries",
        "_version",
    )

    _entries: Mapping[tuple[str, str], ModelPricing]
    _version: str

    def __init__(
        self,
        *,
        version: str,
        entries: Iterable[ModelPricing],
    ) -> None:
        _validate_required_text(
            version,
            field_name="version",
        )

        entries_by_key: dict[
            tuple[str, str],
            ModelPricing,
        ] = {}

        for entry in entries:
            key = (
                entry.provider,
                entry.model,
            )

            if key in entries_by_key:
                raise DuplicateModelPricingError(
                    f"Duplicate model pricing entry: {entry.provider}/{entry.model}.",
                )

            entries_by_key[key] = entry

        self._version = version
        self._entries = MappingProxyType(entries_by_key)

    @property
    def version(self) -> str:
        """Return the immutable pricing catalog version."""

        return self._version

    def find(
        self,
        *,
        provider: str,
        model: str,
    ) -> ModelPricing | None:
        """Return pricing when the exact provider and model are known."""

        return self._entries.get(
            (
                provider,
                model,
            ),
        )

    def __len__(self) -> int:
        """Return the number of model-pricing entries."""

        return len(self._entries)


def _validate_price(
    value: Decimal,
    *,
    field_name: str,
) -> None:
    if not isinstance(value, Decimal):
        raise TypeError(
            f"{field_name} must be a Decimal.",
        )

    if not value.is_finite():
        raise ValueError(
            f"{field_name} must be finite.",
        )

    if value < 0:
        raise ValueError(
            f"{field_name} must be non-negative.",
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


DEFAULT_PRICING_CATALOG = PricingCatalog(
    version=PRICING_CATALOG_VERSION,
    entries=(
        ModelPricing(
            provider="mock",
            model="mock-ticket-classifier-v1",
            effective_from=date(2026, 8, 1),
            input_cost_per_million_tokens=Decimal("0"),
            cached_input_cost_per_million_tokens=Decimal("0"),
            output_cost_per_million_tokens=Decimal("0"),
        ),
        ModelPricing(
            provider="openai",
            model="gpt-5-nano",
            effective_from=date(2026, 8, 1),
            input_cost_per_million_tokens=Decimal("0.05"),
            cached_input_cost_per_million_tokens=Decimal("0.005"),
            output_cost_per_million_tokens=Decimal("0.40"),
        ),
    ),
)
