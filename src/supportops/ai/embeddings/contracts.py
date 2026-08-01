"""Provider-independent contracts for text embedding operations."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from math import isfinite
from types import MappingProxyType
from typing import Protocol

type EmbeddingVector = tuple[float, ...]


class EmbeddingOperation(StrEnum):
    """Application-owned operations that require text embeddings."""

    KNOWLEDGE_INDEXING = "knowledge_indexing"
    KNOWLEDGE_QUERY = "knowledge_query"


@dataclass(frozen=True, slots=True)
class EmbeddingTokenUsage:
    """Provider-reported embedding token usage."""

    input_tokens: int | None = None
    total_tokens: int | None = None

    def __post_init__(self) -> None:
        token_fields = {
            "input_tokens": self.input_tokens,
            "total_tokens": self.total_tokens,
        }

        for field_name, value in token_fields.items():
            if value is not None and value < 0:
                raise ValueError(f"{field_name} must be non-negative when reported.")

        if (
            self.input_tokens is not None
            and self.total_tokens is not None
            and self.total_tokens != self.input_tokens
        ):
            raise ValueError("Embedding total_tokens must equal input_tokens.")


@dataclass(frozen=True, slots=True)
class EmbeddingRequest:
    """Application-owned batch supplied to an embedding provider."""

    operation: EmbeddingOperation
    model: str
    inputs: tuple[str, ...]
    dimensions: int
    timeout_seconds: float
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_required_text(
            self.model,
            field_name="model",
        )

        if self.dimensions <= 0:
            raise ValueError("dimensions must be positive.")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive.")

        normalized_inputs = tuple(self.inputs)
        if not normalized_inputs:
            raise ValueError("Embedding requests require at least one input.")

        for index, value in enumerate(normalized_inputs):
            if not isinstance(value, str):
                raise TypeError(f"inputs[{index}] must be a string.")
            if not value.strip():
                raise ValueError(f"inputs[{index}] must contain meaningful text.")

        normalized_metadata = dict(self.metadata)
        for key, value in normalized_metadata.items():
            _validate_required_text(
                key,
                field_name="metadata key",
            )
            _validate_required_text(
                value,
                field_name=f"metadata[{key!r}]",
            )

        object.__setattr__(
            self,
            "inputs",
            normalized_inputs,
        )
        object.__setattr__(
            self,
            "metadata",
            MappingProxyType(normalized_metadata),
        )


@dataclass(frozen=True, slots=True)
class EmbeddingProviderResponse:
    """Successful embedding result without SDK-specific objects."""

    embeddings: tuple[EmbeddingVector, ...]
    provider: str
    model: str
    dimensions: int
    usage: EmbeddingTokenUsage | None = None
    provider_request_id: str | None = None

    def __post_init__(self) -> None:
        _validate_required_text(
            self.provider,
            field_name="provider",
        )
        _validate_required_text(
            self.model,
            field_name="model",
        )
        _validate_optional_text(
            self.provider_request_id,
            field_name="provider_request_id",
        )

        if self.dimensions <= 0:
            raise ValueError("dimensions must be positive.")

        normalized_embeddings = tuple(
            _normalize_vector(
                vector,
                dimensions=self.dimensions,
                vector_index=index,
            )
            for index, vector in enumerate(self.embeddings)
        )
        if not normalized_embeddings:
            raise ValueError("Embedding responses require at least one vector.")

        object.__setattr__(
            self,
            "embeddings",
            normalized_embeddings,
        )


class EmbeddingProvider(Protocol):
    """Asynchronous adapter for one configured embedding provider."""

    @property
    def provider_name(self) -> str:
        """Return the stable provider identity used for provenance."""
        ...

    async def embed(
        self,
        request: EmbeddingRequest,
    ) -> EmbeddingProviderResponse:
        """Embed an ordered text batch or raise an owned error."""
        ...

    async def close(self) -> None:
        """Release provider-owned process resources."""
        ...


def _normalize_vector(
    vector: Sequence[float],
    *,
    dimensions: int,
    vector_index: int,
) -> EmbeddingVector:
    if len(vector) != dimensions:
        raise ValueError(
            f"embeddings[{vector_index}] must contain exactly {dimensions} coordinates."
        )

    normalized_coordinates: list[float] = []

    for coordinate_index, coordinate in enumerate(vector):
        if isinstance(coordinate, bool) or not isinstance(
            coordinate,
            (int, float),
        ):
            raise TypeError(f"embeddings[{vector_index}][{coordinate_index}] must be numeric.")

        normalized_coordinate = float(coordinate)
        if not isfinite(normalized_coordinate):
            raise ValueError(f"embeddings[{vector_index}][{coordinate_index}] must be finite.")

        normalized_coordinates.append(normalized_coordinate)

    return tuple(normalized_coordinates)


def _validate_required_text(
    value: str,
    *,
    field_name: str,
) -> None:
    if not value:
        raise ValueError(f"{field_name} is required.")
    if value != value.strip():
        raise ValueError(f"{field_name} must not contain surrounding whitespace.")


def _validate_optional_text(
    value: str | None,
    *,
    field_name: str,
) -> None:
    if value is not None:
        _validate_required_text(
            value,
            field_name=field_name,
        )
