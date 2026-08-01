"""Unit tests for provider-independent embedding contracts."""

from typing import cast

import pytest

from supportops.ai.embeddings.contracts import (
    EmbeddingOperation,
    EmbeddingProviderResponse,
    EmbeddingRequest,
    EmbeddingTokenUsage,
)


def create_request() -> EmbeddingRequest:
    """Create one valid deterministic embedding request."""

    return EmbeddingRequest(
        operation=EmbeddingOperation.KNOWLEDGE_INDEXING,
        model="text-embedding-3-small",
        inputs=(
            "  Preserve meaningful source whitespace.\n",
            "Second authoritative chunk.",
        ),
        dimensions=1536,
        timeout_seconds=12.0,
        metadata={
            "workspace_id": ("032c8c87-57cc-4d14-bfbd-04968b4e8cd4"),
        },
    )


def test_request_preserves_order_and_meaningful_whitespace() -> None:
    request = create_request()

    assert request.inputs == (
        "  Preserve meaningful source whitespace.\n",
        "Second authoritative chunk.",
    )
    assert request.operation is (EmbeddingOperation.KNOWLEDGE_INDEXING)
    assert request.dimensions == 1536


def test_request_metadata_is_immutable() -> None:
    request = create_request()

    with pytest.raises(TypeError):
        cast(
            dict[str, str],
            request.metadata,
        )["workspace_id"] = "replacement"


@pytest.mark.parametrize(
    "inputs",
    [
        (),
        ("",),
        ("   \n\t",),
    ],
)
def test_request_requires_meaningful_input_batch(
    inputs: tuple[str, ...],
) -> None:
    with pytest.raises(ValueError):
        EmbeddingRequest(
            operation=EmbeddingOperation.KNOWLEDGE_INDEXING,
            model="text-embedding-3-small",
            inputs=inputs,
            dimensions=1536,
            timeout_seconds=12.0,
        )


@pytest.mark.parametrize(
    ("dimensions", "timeout_seconds"),
    [
        (0, 12.0),
        (-1, 12.0),
        (1536, 0),
        (1536, -1),
    ],
)
def test_request_rejects_invalid_limits(
    dimensions: int,
    timeout_seconds: float,
) -> None:
    with pytest.raises(ValueError):
        EmbeddingRequest(
            operation=EmbeddingOperation.KNOWLEDGE_INDEXING,
            model="text-embedding-3-small",
            inputs=("Source text.",),
            dimensions=dimensions,
            timeout_seconds=timeout_seconds,
        )


def test_usage_accepts_provider_reported_input_total() -> None:
    usage = EmbeddingTokenUsage(
        input_tokens=42,
        total_tokens=42,
    )

    assert usage.input_tokens == 42
    assert usage.total_tokens == 42


@pytest.mark.parametrize(
    ("input_tokens", "total_tokens"),
    [
        (-1, None),
        (None, -1),
        (10, 11),
    ],
)
def test_usage_rejects_invalid_values(
    input_tokens: int | None,
    total_tokens: int | None,
) -> None:
    with pytest.raises(ValueError):
        EmbeddingTokenUsage(
            input_tokens=input_tokens,
            total_tokens=total_tokens,
        )


def test_response_preserves_vector_order_and_provenance() -> None:
    response = EmbeddingProviderResponse(
        embeddings=(
            (0.1, 0.2, 0.3),
            (0.4, 0.5, 0.6),
        ),
        provider="openai",
        model="text-embedding-3-small",
        dimensions=3,
        usage=EmbeddingTokenUsage(
            input_tokens=12,
            total_tokens=12,
        ),
        provider_request_id="req_embedding_123",
    )

    assert response.embeddings == (
        (0.1, 0.2, 0.3),
        (0.4, 0.5, 0.6),
    )
    assert response.provider == "openai"
    assert response.model == "text-embedding-3-small"
    assert response.provider_request_id == "req_embedding_123"


def test_response_normalizes_integer_coordinates_to_float() -> None:
    response = EmbeddingProviderResponse(
        embeddings=((1, 0, -1),),
        provider="mock",
        model="mock-hashing-embedding-v1",
        dimensions=3,
    )

    assert response.embeddings == ((1.0, 0.0, -1.0),)


@pytest.mark.parametrize(
    "embeddings",
    [
        (),
        ((0.1, 0.2),),
        ((0.1, float("nan"), 0.3),),
        ((0.1, float("inf"), 0.3),),
    ],
)
def test_response_rejects_invalid_vectors(
    embeddings: tuple[tuple[float, ...], ...],
) -> None:
    with pytest.raises(ValueError):
        EmbeddingProviderResponse(
            embeddings=embeddings,
            provider="openai",
            model="text-embedding-3-small",
            dimensions=3,
        )


def test_response_rejects_non_numeric_coordinate() -> None:
    with pytest.raises(
        TypeError,
        match="must be numeric",
    ):
        EmbeddingProviderResponse(
            embeddings=cast(
                tuple[tuple[float, ...], ...],
                (("not-a-number", 0.2, 0.3),),
            ),
            provider="openai",
            model="text-embedding-3-small",
            dimensions=3,
        )
