"""Deterministic network-free mock embedding provider."""

import re
import unicodedata
from hashlib import sha256
from math import sqrt

from supportops.ai.embeddings.contracts import (
    EmbeddingProviderResponse,
    EmbeddingRequest,
    EmbeddingTokenUsage,
    EmbeddingVector,
)
from supportops.ai.embeddings.errors import (
    EmbeddingInvalidRequestError,
)

MOCK_EMBEDDING_PROVIDER_NAME = "mock"
MOCK_HASHING_EMBEDDING_MODEL = "mock-hashing-embedding-v1"
DEFAULT_MOCK_EMBEDDING_DIMENSIONS = 64

_LEXEME_PATTERN = re.compile(r"\w+", flags=re.UNICODE)


class MockEmbeddingProvider:
    """Create deterministic lexical vectors for local execution and tests.

    The produced vectors model lexical overlap only. They are not intended
    to represent semantic embedding quality.
    """

    def __init__(
        self,
        *,
        model: str = MOCK_HASHING_EMBEDDING_MODEL,
        dimensions: int = DEFAULT_MOCK_EMBEDDING_DIMENSIONS,
    ) -> None:
        _validate_required_text(
            model,
            field_name="model",
        )
        if dimensions <= 0:
            raise ValueError("dimensions must be positive.")

        self._model = model
        self._dimensions = dimensions
        self._invocation_count = 0
        self._closed = False

    @property
    def provider_name(self) -> str:
        """Return the explicit mock provider identity."""

        return MOCK_EMBEDDING_PROVIDER_NAME

    @property
    def model(self) -> str:
        """Return the configured mock embedding model."""

        return self._model

    @property
    def dimensions(self) -> int:
        """Return the configured vector dimensions."""

        return self._dimensions

    @property
    def invocation_count(self) -> int:
        """Return the number of logical provider requests."""

        return self._invocation_count

    async def embed(
        self,
        request: EmbeddingRequest,
    ) -> EmbeddingProviderResponse:
        """Return deterministic lexical-hashing vectors."""

        if self._closed:
            raise RuntimeError("Mock embedding provider is closed.")
        if request.model != self._model or request.dimensions != self._dimensions:
            raise EmbeddingInvalidRequestError()

        self._invocation_count += 1

        embeddings = tuple(
            _create_lexical_hashing_vector(
                text,
                dimensions=self._dimensions,
            )
            for text in request.inputs
        )
        input_tokens = sum(_mock_token_count(text) for text in request.inputs)

        return EmbeddingProviderResponse(
            embeddings=embeddings,
            provider=self.provider_name,
            model=self._model,
            dimensions=self._dimensions,
            usage=EmbeddingTokenUsage(
                input_tokens=input_tokens,
                total_tokens=input_tokens,
            ),
            provider_request_id=(f"mock-embedding-request-{self._invocation_count}"),
        )

    async def close(self) -> None:
        """Close the provider idempotently."""

        self._closed = True


def _create_lexical_hashing_vector(
    text: str,
    *,
    dimensions: int,
) -> EmbeddingVector:
    features = _lexical_features(text)
    coordinates = [0.0] * dimensions

    for feature in features:
        digest = sha256(feature.encode("utf-8")).digest()
        coordinate_index = (
            int.from_bytes(
                digest[:8],
                byteorder="big",
                signed=False,
            )
            % dimensions
        )
        direction = 1.0 if digest[8] & 1 else -1.0
        coordinates[coordinate_index] += direction

    norm = sqrt(sum(coordinate * coordinate for coordinate in coordinates))
    if norm == 0:
        fallback_digest = sha256(f"text:{text}".encode()).digest()
        fallback_index = (
            int.from_bytes(
                fallback_digest[:8],
                byteorder="big",
                signed=False,
            )
            % dimensions
        )
        coordinates[fallback_index] = 1.0
        norm = 1.0

    return tuple(coordinate / norm for coordinate in coordinates)


def _lexical_features(
    text: str,
) -> tuple[str, ...]:
    normalized = unicodedata.normalize(
        "NFKC",
        text,
    ).casefold()
    lexemes = _LEXEME_PATTERN.findall(normalized)

    if lexemes:
        return tuple(f"token:{lexeme}" for lexeme in lexemes)

    character_features = tuple(
        f"character:{character}" for character in normalized if not character.isspace()
    )
    if character_features:
        return character_features

    return (f"text:{normalized}",)


def _mock_token_count(
    text: str,
) -> int:
    return len(_lexical_features(text))


def _validate_required_text(
    value: str,
    *,
    field_name: str,
) -> None:
    if not value:
        raise ValueError(f"{field_name} is required.")
    if value != value.strip():
        raise ValueError(f"{field_name} must not contain surrounding whitespace.")
