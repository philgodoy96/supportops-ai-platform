"""Application-owned observability decorator for embedding providers."""

from __future__ import annotations

from contextlib import AbstractContextManager
from time import perf_counter
from typing import Final

from supportops.ai.embeddings.contracts import (
    EmbeddingProvider,
    EmbeddingProviderResponse,
    EmbeddingRequest,
    EmbeddingTokenUsage,
)
from supportops.ai.embeddings.errors import (
    EmbeddingError,
    EmbeddingErrorCode,
)
from supportops.ai.embeddings.pricing import (
    DEFAULT_EMBEDDING_PRICING_CATALOG,
    EmbeddingCostEstimate,
    EmbeddingPricingCatalog,
    estimate_embedding_cost,
)
from supportops.observability.contracts import (
    ObservabilityClient,
    ObservationScope,
)
from supportops.observability.models import (
    CostDetails,
    JsonValue,
    ObservationAttributes,
    ObservationStatus,
    ObservationType,
    ObservationUpdate,
    PricingStatus,
    UsageDetails,
)
from supportops.observability.noop import NoOpObservabilityClient

_OBSERVATION_NAME: Final = "embedding.request"

_SAFE_REQUEST_METADATA_KEYS: Final = frozenset(
    {
        "workspace_id",
        "document_id",
        "document_version_id",
        "agent_run_id",
        "agent_run_attempt_id",
        "correlation_id",
    }
)

_OBSERVATION_METADATA_KEYS: Final = frozenset(
    {
        "operation",
        "provider",
        "model",
        "dimensions",
        "input_item_count",
        "batch_size",
        "workspace_id",
        "document_id",
        "document_version_id",
        "agent_run_id",
        "agent_run_attempt_id",
        "correlation_id",
        "output_item_count",
        "provider_request_id",
        "latency_ms",
        "pricing_catalog_version",
        "pricing_found",
        "error_code",
    }
)

_OBSERVATION_METADATA_PATHS: Final = frozenset((key,) for key in _OBSERVATION_METADATA_KEYS)


class ObservingEmbeddingProvider:
    """Decorate one embedding provider with fail-open observations."""

    def __init__(
        self,
        *,
        provider: EmbeddingProvider,
        observability_client: ObservabilityClient | None = None,
        pricing_catalog: EmbeddingPricingCatalog = (DEFAULT_EMBEDDING_PRICING_CATALOG),
    ) -> None:
        self._provider = provider
        self._observability_client = observability_client or NoOpObservabilityClient()
        self._pricing_catalog = pricing_catalog

    @property
    def provider_name(self) -> str:
        """Return the underlying provider identity."""

        return self._provider.provider_name

    @property
    def wrapped_provider(self) -> EmbeddingProvider:
        """Return the underlying process-owned provider."""

        return self._provider

    async def embed(
        self,
        request: EmbeddingRequest,
    ) -> EmbeddingProviderResponse:
        """Observe one real embedding-provider request."""

        observation = _SafeEmbeddingObservation(
            client=self._observability_client,
            attributes=ObservationAttributes(
                name=_OBSERVATION_NAME,
                observation_type=ObservationType.EMBEDDING,
                provider=self.provider_name,
                model=request.model,
                metadata=_start_metadata(
                    request=request,
                    provider=self.provider_name,
                ),
                metadata_paths=_OBSERVATION_METADATA_PATHS,
                input_data=None,
                input_paths=frozenset(),
                output_paths=frozenset(),
            ),
        )
        observation.start()

        started_at = perf_counter()

        try:
            response = await self._provider.embed(request)
        except EmbeddingError as error:
            observation.update(
                _error_update(
                    error=error,
                    latency_ms=_elapsed_milliseconds(started_at),
                )
            )
            raise
        except Exception:
            observation.update(
                ObservationUpdate(
                    status=ObservationStatus.ERROR,
                    metadata={
                        "latency_ms": _elapsed_milliseconds(started_at),
                        "error_code": (EmbeddingErrorCode.UNEXPECTED_PROVIDER_FAILURE.value),
                    },
                    error_code=(EmbeddingErrorCode.UNEXPECTED_PROVIDER_FAILURE.value),
                )
            )
            raise
        else:
            observation.update(
                _safe_success_update(
                    response=response,
                    latency_ms=_elapsed_milliseconds(started_at),
                    pricing_catalog=self._pricing_catalog,
                )
            )
            return response
        finally:
            observation.close()

    async def close(self) -> None:
        """Close the wrapped process-owned provider."""

        await self._provider.close()


class _SafeEmbeddingObservation:
    """Isolate observability failures from provider behavior."""

    def __init__(
        self,
        *,
        client: ObservabilityClient,
        attributes: ObservationAttributes,
    ) -> None:
        self._client = client
        self._attributes = attributes
        self._manager: AbstractContextManager[ObservationScope] | None = None
        self._scope: ObservationScope | None = None

    def start(self) -> None:
        try:
            self._manager = self._client.start_observation(self._attributes)
            self._scope = self._manager.__enter__()
        except Exception:
            self._manager = None
            self._scope = None

    def update(self, update: ObservationUpdate) -> None:
        if self._scope is None:
            return

        try:
            self._scope.update(update)
        except Exception:
            return

    def close(self) -> None:
        if self._manager is None:
            return

        try:
            self._manager.__exit__(None, None, None)
        except Exception:
            return
        finally:
            self._manager = None
            self._scope = None


def _start_metadata(
    *,
    request: EmbeddingRequest,
    provider: str,
) -> dict[str, JsonValue]:
    input_item_count = len(request.inputs)

    metadata: dict[str, JsonValue] = {
        "operation": request.operation.value,
        "provider": provider,
        "model": request.model,
        "dimensions": request.dimensions,
        "input_item_count": input_item_count,
        "batch_size": input_item_count,
    }

    for key in _SAFE_REQUEST_METADATA_KEYS:
        value = request.metadata.get(key)
        if value is not None:
            metadata[key] = value

    return metadata


def _safe_success_update(
    *,
    response: EmbeddingProviderResponse,
    latency_ms: int,
    pricing_catalog: EmbeddingPricingCatalog,
) -> ObservationUpdate:
    metadata: dict[str, JsonValue] = {
        "output_item_count": len(response.embeddings),
        "latency_ms": latency_ms,
    }

    if response.provider_request_id is not None:
        metadata["provider_request_id"] = response.provider_request_id

    try:
        input_tokens = _reported_input_tokens(response.usage)

        estimate = estimate_embedding_cost(
            provider=response.provider,
            model=response.model,
            input_tokens=input_tokens,
            catalog=pricing_catalog,
        )

        metadata["pricing_catalog_version"] = estimate.pricing_catalog_version
        metadata["pricing_found"] = estimate.pricing_found

        return ObservationUpdate(
            status=ObservationStatus.OK,
            metadata=metadata,
            usage=_usage_details(response.usage),
            cost=_cost_details(estimate),
        )
    except Exception:
        return ObservationUpdate(
            status=ObservationStatus.OK,
            metadata=metadata,
        )


def _error_update(
    *,
    error: EmbeddingError,
    latency_ms: int,
) -> ObservationUpdate:
    metadata: dict[str, JsonValue] = {
        "latency_ms": latency_ms,
        "error_code": error.error_code.value,
    }

    if error.provider_request_id is not None:
        metadata["provider_request_id"] = error.provider_request_id

    return ObservationUpdate(
        status=ObservationStatus.ERROR,
        metadata=metadata,
        error_code=error.error_code.value,
    )


def _reported_input_tokens(
    usage: EmbeddingTokenUsage | None,
) -> int | None:
    if usage is None:
        return None

    if usage.input_tokens is not None:
        return usage.input_tokens

    return usage.total_tokens


def _usage_details(
    usage: EmbeddingTokenUsage | None,
) -> UsageDetails | None:
    input_tokens = _reported_input_tokens(usage)

    if input_tokens is None:
        return None

    return UsageDetails(input_tokens=input_tokens)


def _cost_details(
    estimate: EmbeddingCostEstimate,
) -> CostDetails | None:
    if not estimate.pricing_found:
        return CostDetails(
            pricing_status=PricingStatus.UNKNOWN,
            pricing_catalog_version=(estimate.pricing_catalog_version),
        )

    if estimate.estimated_cost_usd is None:
        return None

    return CostDetails(
        pricing_status=PricingStatus.KNOWN,
        input_cost=estimate.estimated_cost_usd,
        pricing_catalog_version=(estimate.pricing_catalog_version),
    )


def _elapsed_milliseconds(started_at: float) -> int:
    return max(
        0,
        round((perf_counter() - started_at) * 1000),
    )
