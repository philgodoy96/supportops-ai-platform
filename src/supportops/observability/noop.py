"""No-op implementation of the application-owned observability contract."""

from __future__ import annotations

from contextlib import AbstractContextManager
from types import TracebackType
from typing import Literal
from uuid import uuid4

from supportops.observability.context import (
    ActiveObservationContext,
    ActiveTraceContext,
    observation_context_scope,
    trace_context_scope,
)
from supportops.observability.contracts import ObservationScope, TraceScope
from supportops.observability.identity import TraceIdentity
from supportops.observability.models import (
    EventObservation,
    ObservabilityProvider,
    ObservationAttributes,
    ObservationUpdate,
    TraceAttributes,
)


class NoOpObservabilityClient:
    """Process-owned observability client that never exports telemetry."""

    @property
    def provider(self) -> ObservabilityProvider:
        return ObservabilityProvider.NOOP

    @property
    def enabled(self) -> bool:
        return False

    def start_trace(
        self,
        attributes: TraceAttributes,
    ) -> AbstractContextManager[TraceScope]:
        return _NoOpTraceManager(attributes)

    def start_observation(
        self,
        attributes: ObservationAttributes,
    ) -> AbstractContextManager[ObservationScope]:
        return _NoOpObservationManager(attributes)

    def record_event(self, event: EventObservation) -> None:
        del event

    def record_trace_event(
        self,
        *,
        identity: TraceIdentity,
        event: EventObservation,
    ) -> None:
        del identity, event

    def flush(self) -> None:
        return None

    def shutdown(self) -> None:
        return None


class _NoOpTraceManager(AbstractContextManager[TraceScope]):
    def __init__(self, attributes: TraceAttributes) -> None:
        self._scope = _NoOpTraceScope(attributes)
        self._context_manager = trace_context_scope(
            ActiveTraceContext(
                trace_seed=attributes.trace_seed,
                session_id=attributes.session_id,
            )
        )

    def __enter__(self) -> TraceScope:
        self._context_manager.__enter__()
        return self._scope

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> Literal[False]:
        self._context_manager.__exit__(exc_type, exc, traceback)
        return False


class _NoOpObservationManager(AbstractContextManager[ObservationScope]):
    def __init__(self, attributes: ObservationAttributes) -> None:
        self._scope = _NoOpObservationScope(attributes)
        self._context_manager = observation_context_scope(
            ActiveObservationContext(
                name=attributes.name,
                observation_id=self._scope.observation_id,
            )
        )

    def __enter__(self) -> ObservationScope:
        self._context_manager.__enter__()
        return self._scope

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> Literal[False]:
        self._context_manager.__exit__(exc_type, exc, traceback)
        return False


class _NoOpTraceScope:
    def __init__(self, attributes: TraceAttributes) -> None:
        self._attributes = attributes

    @property
    def trace_seed(self) -> str:
        return self._attributes.trace_seed

    @property
    def trace_id(self) -> str | None:
        return None

    @property
    def session_id(self) -> str | None:
        return self._attributes.session_id

    def update(self, update: ObservationUpdate) -> None:
        del update

    def start_observation(
        self,
        attributes: ObservationAttributes,
    ) -> AbstractContextManager[ObservationScope]:
        return _NoOpObservationManager(attributes)

    def record_event(self, event: EventObservation) -> None:
        del event


class _NoOpObservationScope:
    def __init__(self, attributes: ObservationAttributes) -> None:
        self._attributes = attributes
        self._observation_id = uuid4().hex[:16]

    @property
    def observation_id(self) -> str | None:
        return self._observation_id

    def update(self, update: ObservationUpdate) -> None:
        del update

    def start_observation(
        self,
        attributes: ObservationAttributes,
    ) -> AbstractContextManager[ObservationScope]:
        return _NoOpObservationManager(attributes)

    def record_event(self, event: EventObservation) -> None:
        del event
