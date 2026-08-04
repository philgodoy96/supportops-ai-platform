"""Unit tests for embedding-provider observability."""

from __future__ import annotations

from contextlib import AbstractContextManager
from decimal import Decimal
from types import TracebackType
from typing import Literal

import pytest

from supportops.ai.embeddings.contracts import (
    EmbeddingOperation,
    EmbeddingProviderResponse,
    EmbeddingRequest,
    EmbeddingTokenUsage,
)
from supportops.ai.embeddings.errors import (
    EmbeddingTimeoutError,
)
from supportops.ai.embeddings.observability import (
    ObservingEmbeddingProvider,
)
from supportops.observability.context import (
    ActiveObservationContext,
    current_observation_context,
    observation_context_scope,
)
from supportops.observability.contracts import TraceScope
from supportops.observability.models import (
    EventObservation,
    ObservabilityProvider,
    ObservationAttributes,
    ObservationStatus,
    ObservationType,
    ObservationUpdate,
    PricingStatus,
    TraceAttributes,
)


class FakeEmbeddingProvider:
    def __init__(
        self,
        *,
        provider_name: str = "mock",
        response: EmbeddingProviderResponse | None = None,
        error: Exception | None = None,
    ) -> None:
        self._provider_name = provider_name
        self._response = response
        self._error = error
        self.requests: list[EmbeddingRequest] = []
        self.parent_observation_names: list[str | None] = []
        self.close_calls = 0

    @property
    def provider_name(self) -> str:
        return self._provider_name

    async def embed(
        self,
        request: EmbeddingRequest,
    ) -> EmbeddingProviderResponse:
        self.requests.append(request)
        parent = current_observation_context()
        self.parent_observation_names.append(
            None if parent is None else parent.name,
        )

        if self._error is not None:
            raise self._error

        if self._response is None:
            raise AssertionError("A fake response was not configured.")

        return self._response

    async def close(self) -> None:
        self.close_calls += 1


class RecordingObservationScope:
    def __init__(
        self,
        *,
        attributes: ObservationAttributes,
        fail_update: bool = False,
    ) -> None:
        self.attributes = attributes
        self._fail_update = fail_update
        self.updates: list[ObservationUpdate] = []

    @property
    def observation_id(self) -> str | None:
        return "embedding-observation-1"

    def update(self, update: ObservationUpdate) -> None:
        if self._fail_update:
            raise RuntimeError("synthetic update failure")

        self.updates.append(update)

    def start_observation(
        self,
        attributes: ObservationAttributes,
    ) -> AbstractContextManager[RecordingObservationScope]:
        del attributes
        raise AssertionError("Nested observations are not expected.")

    def record_event(self, event: EventObservation) -> None:
        del event
        raise AssertionError("Events are not expected.")


class RecordingObservationManager(AbstractContextManager[RecordingObservationScope]):
    def __init__(
        self,
        *,
        scope: RecordingObservationScope,
        fail_enter: bool = False,
        fail_exit: bool = False,
    ) -> None:
        self._scope = scope
        self._fail_enter = fail_enter
        self._fail_exit = fail_exit
        self.exit_calls = 0
        self._context_manager = observation_context_scope(
            ActiveObservationContext(
                name=scope.attributes.name,
                observation_id=scope.observation_id,
            )
        )

    def __enter__(self) -> RecordingObservationScope:
        if self._fail_enter:
            raise RuntimeError("synthetic enter failure")

        self._context_manager.__enter__()
        return self._scope

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> Literal[False]:
        self.exit_calls += 1
        self._context_manager.__exit__(exc_type, exc, traceback)

        if self._fail_exit:
            raise RuntimeError("synthetic exit failure")

        return False


class RecordingObservabilityClient:
    def __init__(
        self,
        *,
        fail_start: bool = False,
        fail_enter: bool = False,
        fail_update: bool = False,
        fail_exit: bool = False,
    ) -> None:
        self._fail_start = fail_start
        self._fail_enter = fail_enter
        self._fail_update = fail_update
        self._fail_exit = fail_exit

        self.attributes: list[ObservationAttributes] = []
        self.scopes: list[RecordingObservationScope] = []
        self.managers: list[RecordingObservationManager] = []
        self.parent_observation_names: list[str | None] = []
        self.lifecycle: list[tuple[str, str]] = []

    @property
    def provider(self) -> ObservabilityProvider:
        return ObservabilityProvider.NOOP

    @property
    def enabled(self) -> bool:
        return True

    def start_trace(
        self,
        attributes: TraceAttributes,
    ) -> AbstractContextManager[TraceScope]:
        del attributes
        raise AssertionError("Embedding tracing must not create roots.")

    def start_observation(
        self,
        attributes: ObservationAttributes,
    ) -> AbstractContextManager[RecordingObservationScope]:
        if self._fail_start:
            raise RuntimeError("synthetic start failure")

        parent = current_observation_context()
        self.parent_observation_names.append(
            None if parent is None else parent.name,
        )
        self.lifecycle.append(("start", attributes.name))

        scope = RecordingObservationScope(
            attributes=attributes,
            fail_update=self._fail_update,
        )
        manager = RecordingObservationManager(
            scope=scope,
            fail_enter=self._fail_enter,
            fail_exit=self._fail_exit,
        )

        self.attributes.append(attributes)
        self.scopes.append(scope)
        self.managers.append(manager)

        return manager

    def record_event(self, event: EventObservation) -> None:
        del event
        raise AssertionError("Embedding tracing must not emit events.")

    def flush(self) -> None:
        return None

    def shutdown(self) -> None:
        return None


_DEFAULT_USAGE = EmbeddingTokenUsage(
    input_tokens=12,
    total_tokens=12,
)


def _request(
    operation: EmbeddingOperation = (EmbeddingOperation.KNOWLEDGE_INDEXING),
) -> EmbeddingRequest:
    return EmbeddingRequest(
        operation=operation,
        model="mock-hashing-embedding-v1",
        inputs=(
            "first synthetic input",
            "second synthetic input",
        ),
        dimensions=2,
        timeout_seconds=5.0,
        metadata={
            "workspace_id": "workspace-1",
            "document_version_id": "version-1",
            "unsafe_field": "must-not-be-exported",
        },
    )


def _response(
    *,
    provider: str = "mock",
    model: str = "mock-hashing-embedding-v1",
    usage: EmbeddingTokenUsage | None = _DEFAULT_USAGE,
    provider_request_id: str | None = "embedding-request-1",
) -> EmbeddingProviderResponse:
    return EmbeddingProviderResponse(
        embeddings=(
            (1.0, 0.0),
            (0.0, 1.0),
        ),
        provider=provider,
        model=model,
        dimensions=2,
        usage=usage,
        provider_request_id=provider_request_id,
    )


@pytest.mark.asyncio
async def test_records_one_embedding_observation_per_provider_call() -> None:
    provider = FakeEmbeddingProvider(response=_response())
    observability = RecordingObservabilityClient()
    wrapper = ObservingEmbeddingProvider(
        provider=provider,
        observability_client=observability,
    )

    result = await wrapper.embed(_request())

    assert result == _response()
    assert len(provider.requests) == 1
    assert len(observability.attributes) == 1
    assert len(observability.scopes) == 1
    assert len(observability.attributes) == len(provider.requests)

    attributes = observability.attributes[0]

    assert attributes.name == "embedding.request"
    assert attributes.observation_type is ObservationType.EMBEDDING
    assert attributes.provider == "mock"
    assert attributes.model == "mock-hashing-embedding-v1"
    assert attributes.input_data is None
    assert attributes.input_paths == frozenset()
    assert attributes.output_paths == frozenset()
    assert provider.parent_observation_names == ["embedding.request"]
    assert current_observation_context() is None
    assert observability.managers[0].exit_calls == 1


@pytest.mark.asyncio
async def test_exports_only_safe_embedding_metadata() -> None:
    provider = FakeEmbeddingProvider(response=_response())
    observability = RecordingObservabilityClient()
    wrapper = ObservingEmbeddingProvider(
        provider=provider,
        observability_client=observability,
    )

    await wrapper.embed(_request())

    metadata = observability.attributes[0].metadata

    assert metadata == {
        "operation": "knowledge_indexing",
        "provider": "mock",
        "model": "mock-hashing-embedding-v1",
        "dimensions": 2,
        "input_item_count": 2,
        "batch_size": 2,
        "workspace_id": "workspace-1",
        "document_version_id": "version-1",
    }

    assert "unsafe_field" not in metadata
    assert "inputs" not in metadata
    assert "embeddings" not in metadata
    assert "first synthetic input" not in repr(metadata)
    assert "second synthetic input" not in repr(metadata)


@pytest.mark.asyncio
async def test_maps_usage_and_known_zero_mock_cost() -> None:
    provider = FakeEmbeddingProvider(response=_response())
    observability = RecordingObservabilityClient()
    wrapper = ObservingEmbeddingProvider(
        provider=provider,
        observability_client=observability,
    )

    await wrapper.embed(_request())

    update = observability.scopes[0].updates[0]

    assert update.status is ObservationStatus.OK
    assert update.usage is not None
    assert update.usage.input_tokens == 12
    assert update.usage.total_tokens is None

    assert update.cost is not None
    assert update.cost.pricing_status is PricingStatus.KNOWN
    assert update.cost.input_cost == Decimal("0")
    assert update.cost.total_cost is None

    assert update.metadata["pricing_found"] is True
    assert update.metadata["provider_request_id"] == "embedding-request-1"
    assert update.metadata["output_item_count"] == 2
    assert isinstance(update.metadata["latency_ms"], int)
    assert "(1.0, 0.0)" not in repr(update.metadata)


@pytest.mark.asyncio
async def test_omits_provider_request_id_when_absent() -> None:
    provider = FakeEmbeddingProvider(
        response=_response(provider_request_id=None),
    )
    observability = RecordingObservabilityClient()
    wrapper = ObservingEmbeddingProvider(
        provider=provider,
        observability_client=observability,
    )

    await wrapper.embed(_request())

    update = observability.scopes[0].updates[0]

    assert "provider_request_id" not in update.metadata


@pytest.mark.asyncio
async def test_unknown_pricing_does_not_fabricate_cost() -> None:
    provider = FakeEmbeddingProvider(
        provider_name="unknown-provider",
        response=_response(
            provider="unknown-provider",
            model="unknown-model",
        ),
    )
    observability = RecordingObservabilityClient()
    wrapper = ObservingEmbeddingProvider(
        provider=provider,
        observability_client=observability,
    )

    request = EmbeddingRequest(
        operation=EmbeddingOperation.KNOWLEDGE_QUERY,
        model="unknown-model",
        inputs=("synthetic query",),
        dimensions=2,
        timeout_seconds=5.0,
    )

    await wrapper.embed(request)

    update = observability.scopes[0].updates[0]

    assert update.cost is not None
    assert update.cost.pricing_status is PricingStatus.UNKNOWN
    assert update.cost.input_cost is None
    assert update.cost.total_cost is None
    assert update.metadata["pricing_found"] is False


@pytest.mark.asyncio
async def test_missing_usage_fabricates_no_usage_or_cost() -> None:
    provider = FakeEmbeddingProvider(response=_response(usage=None))
    observability = RecordingObservabilityClient()
    wrapper = ObservingEmbeddingProvider(
        provider=provider,
        observability_client=observability,
    )

    await wrapper.embed(_request())

    update = observability.scopes[0].updates[0]

    assert update.usage is None
    assert update.cost is None
    assert update.metadata["pricing_found"] is True


@pytest.mark.asyncio
async def test_provider_error_is_observed_and_preserved() -> None:
    expected_error = EmbeddingTimeoutError(provider_request_id="embedding-timeout-1")
    provider = FakeEmbeddingProvider(error=expected_error)
    observability = RecordingObservabilityClient()
    wrapper = ObservingEmbeddingProvider(
        provider=provider,
        observability_client=observability,
    )

    with pytest.raises(EmbeddingTimeoutError) as exception_info:
        await wrapper.embed(_request())

    assert exception_info.value is expected_error
    assert current_observation_context() is None

    update = observability.scopes[0].updates[0]

    assert update.status is ObservationStatus.ERROR
    assert update.error_code == "embedding_timeout"
    assert update.metadata["error_code"] == "embedding_timeout"
    assert update.metadata["provider_request_id"] == "embedding-timeout-1"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure_mode",
    [
        "start",
        "enter",
        "update",
        "exit",
    ],
)
async def test_observability_failures_do_not_change_provider_result(
    failure_mode: str,
) -> None:
    provider = FakeEmbeddingProvider(response=_response())
    observability = RecordingObservabilityClient(
        fail_start=failure_mode == "start",
        fail_enter=failure_mode == "enter",
        fail_update=failure_mode == "update",
        fail_exit=failure_mode == "exit",
    )
    wrapper = ObservingEmbeddingProvider(
        provider=provider,
        observability_client=observability,
    )

    result = await wrapper.embed(_request())

    assert result == _response()
    assert len(provider.requests) == 1
    assert current_observation_context() is None


@pytest.mark.asyncio
async def test_query_operation_is_preserved_without_query_content() -> None:
    provider = FakeEmbeddingProvider(response=_response())
    observability = RecordingObservabilityClient()
    wrapper = ObservingEmbeddingProvider(
        provider=provider,
        observability_client=observability,
    )

    request = EmbeddingRequest(
        operation=EmbeddingOperation.KNOWLEDGE_QUERY,
        model="mock-hashing-embedding-v1",
        inputs=("private semantic query",),
        dimensions=2,
        timeout_seconds=5.0,
        metadata={"workspace_id": "workspace-1"},
    )

    await wrapper.embed(request)

    attributes = observability.attributes[0]

    assert attributes.metadata["operation"] == "knowledge_query"
    assert "private semantic query" not in repr(attributes.metadata)
    assert attributes.input_data is None


@pytest.mark.asyncio
async def test_close_is_delegated_to_underlying_provider() -> None:
    provider = FakeEmbeddingProvider(response=_response())
    wrapper = ObservingEmbeddingProvider(provider=provider)

    await wrapper.close()

    assert provider.close_calls == 1
    assert wrapper.wrapped_provider is provider


@pytest.mark.asyncio
async def test_observation_context_restores_parent_after_completion() -> None:
    provider = FakeEmbeddingProvider(response=_response())
    observability = RecordingObservabilityClient()
    wrapper = ObservingEmbeddingProvider(
        provider=provider,
        observability_client=observability,
    )

    with observation_context_scope(
        ActiveObservationContext(name="parent.boundary"),
    ):
        await wrapper.embed(_request())
        active = current_observation_context()
        assert active is not None
        assert active.name == "parent.boundary"

    assert observability.parent_observation_names == ["parent.boundary"]
    assert provider.parent_observation_names == ["embedding.request"]
    assert current_observation_context() is None
