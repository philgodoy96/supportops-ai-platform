"""Unit tests for deterministic mock embeddings."""

from math import isclose, sqrt

import pytest

from supportops.ai.embeddings.contracts import (
    EmbeddingOperation,
    EmbeddingRequest,
)
from supportops.ai.embeddings.errors import (
    EmbeddingInvalidRequestError,
)
from supportops.ai.embeddings.mock import (
    MOCK_EMBEDDING_PROVIDER_NAME,
    MOCK_HASHING_EMBEDDING_MODEL,
    MockEmbeddingProvider,
)


def create_request(
    *,
    inputs: tuple[str, ...] = (
        "database recovery procedure",
        "billing invoice refund",
    ),
    model: str = MOCK_HASHING_EMBEDDING_MODEL,
    dimensions: int = 64,
) -> EmbeddingRequest:
    """Create one deterministic mock request."""

    return EmbeddingRequest(
        operation=EmbeddingOperation.KNOWLEDGE_INDEXING,
        model=model,
        inputs=inputs,
        dimensions=dimensions,
        timeout_seconds=12,
    )


def dot_product(
    first: tuple[float, ...],
    second: tuple[float, ...],
) -> float:
    """Return the dot product of two fixed-dimensional vectors."""

    return sum(
        first_value * second_value
        for first_value, second_value in zip(
            first,
            second,
            strict=True,
        )
    )


async def test_returns_deterministic_normalized_vectors() -> None:
    provider = MockEmbeddingProvider(dimensions=64)
    request = create_request()

    first = await provider.embed(request)
    second = await provider.embed(request)

    assert first.embeddings == second.embeddings
    assert first.provider == MOCK_EMBEDDING_PROVIDER_NAME
    assert first.model == MOCK_HASHING_EMBEDDING_MODEL
    assert first.dimensions == 64
    assert len(first.embeddings) == 2

    for vector in first.embeddings:
        norm = sqrt(sum(coordinate * coordinate for coordinate in vector))
        assert isclose(
            norm,
            1.0,
            rel_tol=1e-12,
            abs_tol=1e-12,
        )


async def test_preserves_batch_order_and_reports_mock_usage() -> None:
    provider = MockEmbeddingProvider(dimensions=64)
    first = await provider.embed(
        create_request(
            inputs=(
                "database recovery",
                "billing invoice",
            )
        )
    )
    reversed_response = await provider.embed(
        create_request(
            inputs=(
                "billing invoice",
                "database recovery",
            )
        )
    )

    assert first.embeddings == tuple(reversed(reversed_response.embeddings))
    assert first.usage is not None
    assert first.usage.input_tokens == 4
    assert first.usage.total_tokens == 4


async def test_shared_lexemes_produce_greater_overlap() -> None:
    provider = MockEmbeddingProvider(dimensions=64)
    response = await provider.embed(
        create_request(
            inputs=(
                "database recovery procedure",
                "database recovery steps",
                "billing invoice refund",
            )
        )
    )
    anchor, related, unrelated = response.embeddings

    assert dot_product(anchor, related) > dot_product(
        anchor,
        unrelated,
    )


async def test_case_and_unicode_normalization_are_deterministic() -> None:
    provider = MockEmbeddingProvider(dimensions=64)
    response = await provider.embed(
        create_request(
            inputs=(
                "DATABASE Recovery",
                "database recovery",
            )
        )
    )

    assert response.embeddings[0] == response.embeddings[1]


async def test_supports_meaningful_punctuation_only_input() -> None:
    provider = MockEmbeddingProvider(dimensions=64)

    response = await provider.embed(create_request(inputs=("!!!",)))

    assert len(response.embeddings) == 1
    assert response.usage is not None
    assert response.usage.input_tokens == 3


async def test_request_ids_and_invocation_count_are_monotonic() -> None:
    provider = MockEmbeddingProvider(dimensions=64)

    first = await provider.embed(create_request())
    second = await provider.embed(create_request())

    assert first.provider_request_id == ("mock-embedding-request-1")
    assert second.provider_request_id == ("mock-embedding-request-2")
    assert provider.invocation_count == 2


@pytest.mark.parametrize(
    ("model", "dimensions"),
    [
        ("other-model", 64),
        (MOCK_HASHING_EMBEDDING_MODEL, 32),
    ],
)
async def test_rejects_incompatible_request_profile(
    model: str,
    dimensions: int,
) -> None:
    provider = MockEmbeddingProvider(dimensions=64)

    with pytest.raises(EmbeddingInvalidRequestError):
        await provider.embed(
            create_request(
                model=model,
                dimensions=dimensions,
            )
        )

    assert provider.invocation_count == 0


async def test_close_is_idempotent_and_prevents_new_requests() -> None:
    provider = MockEmbeddingProvider(dimensions=64)

    await provider.close()
    await provider.close()

    with pytest.raises(
        RuntimeError,
        match="provider is closed",
    ):
        await provider.embed(create_request())


@pytest.mark.parametrize(
    ("model", "dimensions"),
    [
        ("", 64),
        (" mock-model", 64),
        ("mock-model ", 64),
        ("mock-model", 0),
    ],
)
def test_rejects_invalid_provider_configuration(
    model: str,
    dimensions: int,
) -> None:
    with pytest.raises(ValueError):
        MockEmbeddingProvider(
            model=model,
            dimensions=dimensions,
        )
