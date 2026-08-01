"""Versioned pricing and deterministic embedding cost estimation."""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import date
from decimal import ROUND_HALF_UP, Decimal
from types import MappingProxyType

EMBEDDING_PRICING_CATALOG_VERSION = "supportops-embedding-pricing-2026-08-01"

_TOKENS_PER_MILLION = Decimal("1000000")
_USD_QUANTUM = Decimal("0.000000000001")


@dataclass(frozen=True, slots=True)
class EmbeddingModelPricing:
    """Per-million-input-token price for one embedding model."""

    provider: str
    model: str
    effective_from: date
    input_cost_per_million_tokens: Decimal

    def __post_init__(self) -> None:
        _validate_required_text(
            self.provider,
            field_name="provider",
        )
        _validate_required_text(
            self.model,
            field_name="model",
        )
        _validate_decimal_amount(
            self.input_cost_per_million_tokens,
            field_name="input_cost_per_million_tokens",
        )


class DuplicateEmbeddingModelPricingError(ValueError):
    """Raised when one provider and model are registered twice."""


class EmbeddingPricingCatalog:
    """Immutable exact-match embedding pricing lookup."""

    __slots__ = (
        "_entries",
        "_version",
    )

    _entries: Mapping[
        tuple[str, str],
        EmbeddingModelPricing,
    ]
    _version: str

    def __init__(
        self,
        *,
        version: str,
        entries: Iterable[EmbeddingModelPricing],
    ) -> None:
        _validate_required_text(
            version,
            field_name="version",
        )

        entries_by_key: dict[
            tuple[str, str],
            EmbeddingModelPricing,
        ] = {}

        for entry in entries:
            key = (
                entry.provider,
                entry.model,
            )
            if key in entries_by_key:
                raise DuplicateEmbeddingModelPricingError(
                    f"Duplicate embedding model pricing entry: {entry.provider}/{entry.model}."
                )

            entries_by_key[key] = entry

        self._version = version
        self._entries = MappingProxyType(entries_by_key)

    @property
    def version(self) -> str:
        """Return the immutable catalog version."""

        return self._version

    def find(
        self,
        *,
        provider: str,
        model: str,
    ) -> EmbeddingModelPricing | None:
        """Return an exact provider-and-model price."""

        return self._entries.get(
            (
                provider,
                model,
            )
        )

    def __len__(self) -> int:
        """Return the number of registered prices."""

        return len(self._entries)


@dataclass(frozen=True, slots=True)
class EmbeddingCostEstimate:
    """Estimated USD cost from one immutable pricing catalog."""

    pricing_catalog_version: str
    pricing_found: bool
    estimated_cost_usd: Decimal | None

    def __post_init__(self) -> None:
        _validate_required_text(
            self.pricing_catalog_version,
            field_name="pricing_catalog_version",
        )

        if not self.pricing_found:
            if self.estimated_cost_usd is not None:
                raise ValueError("Unknown embedding pricing cannot define a cost.")
            return

        if self.estimated_cost_usd is not None:
            _validate_decimal_amount(
                self.estimated_cost_usd,
                field_name="estimated_cost_usd",
            )


def estimate_embedding_cost(
    *,
    provider: str,
    model: str,
    input_tokens: int | None,
    catalog: EmbeddingPricingCatalog,
) -> EmbeddingCostEstimate:
    """Estimate embedding cost without inventing price or usage."""

    if input_tokens is not None and input_tokens < 0:
        raise ValueError("input_tokens must be non-negative when reported.")

    pricing = catalog.find(
        provider=provider,
        model=model,
    )
    if pricing is None:
        return EmbeddingCostEstimate(
            pricing_catalog_version=catalog.version,
            pricing_found=False,
            estimated_cost_usd=None,
        )

    if input_tokens is None:
        return EmbeddingCostEstimate(
            pricing_catalog_version=catalog.version,
            pricing_found=True,
            estimated_cost_usd=None,
        )

    estimated_cost = _quantize_usd(
        Decimal(input_tokens) * pricing.input_cost_per_million_tokens / _TOKENS_PER_MILLION
    )

    return EmbeddingCostEstimate(
        pricing_catalog_version=catalog.version,
        pricing_found=True,
        estimated_cost_usd=estimated_cost,
    )


def _quantize_usd(
    value: Decimal,
) -> Decimal:
    return value.quantize(
        _USD_QUANTUM,
        rounding=ROUND_HALF_UP,
    )


def _validate_decimal_amount(
    value: Decimal,
    *,
    field_name: str,
) -> None:
    if not isinstance(value, Decimal):
        raise TypeError(f"{field_name} must be a Decimal.")
    if not value.is_finite():
        raise ValueError(f"{field_name} must be finite.")
    if value < 0:
        raise ValueError(f"{field_name} must be non-negative.")


def _validate_required_text(
    value: str,
    *,
    field_name: str,
) -> None:
    if not value:
        raise ValueError(f"{field_name} is required.")
    if value != value.strip():
        raise ValueError(f"{field_name} must not contain surrounding whitespace.")


DEFAULT_EMBEDDING_PRICING_CATALOG = EmbeddingPricingCatalog(
    version=EMBEDDING_PRICING_CATALOG_VERSION,
    entries=(
        EmbeddingModelPricing(
            provider="mock",
            model="mock-hashing-embedding-v1",
            effective_from=date(2026, 8, 1),
            input_cost_per_million_tokens=Decimal("0"),
        ),
        EmbeddingModelPricing(
            provider="openai",
            model="text-embedding-3-small",
            effective_from=date(2026, 8, 1),
            input_cost_per_million_tokens=Decimal("0.02"),
        ),
    ),
)
