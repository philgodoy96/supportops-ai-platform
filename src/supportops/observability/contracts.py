"""Application-owned protocols for AI observability."""

from __future__ import annotations

from contextlib import AbstractContextManager
from typing import Protocol, runtime_checkable

from supportops.observability.identity import TraceIdentity
from supportops.observability.models import (
    EventObservation,
    ObservabilityProvider,
    ObservationAttributes,
    ObservationUpdate,
    TraceAttributes,
)


@runtime_checkable
class ObservationContainer(Protocol):
    """A scope that may contain child observations and events."""

    def start_observation(
        self,
        attributes: ObservationAttributes,
    ) -> AbstractContextManager[ObservationScope]:
        """Start a child observation."""

    def record_event(self, event: EventObservation) -> None:
        """Record one discrete event without opening a long-lived scope."""


@runtime_checkable
class TraceScope(ObservationContainer, Protocol):
    """Active logical trace scope."""

    @property
    def trace_seed(self) -> str:
        """Return the deterministic application trace seed."""

    @property
    def trace_id(self) -> str | None:
        """Return the backend-compatible trace identifier when available."""

    @property
    def session_id(self) -> str | None:
        """Return the optional application session identifier."""

    def update(self, update: ObservationUpdate) -> None:
        """Update the logical trace observation."""


@runtime_checkable
class ObservationScope(ObservationContainer, Protocol):
    """Active observation scope."""

    @property
    def observation_id(self) -> str | None:
        """Return the backend observation identifier when available."""

    def update(self, update: ObservationUpdate) -> None:
        """Apply a bounded observation update before scope completion."""


@runtime_checkable
class ObservabilityClient(ObservationContainer, Protocol):
    """Process-owned observability client contract."""

    @property
    def provider(self) -> ObservabilityProvider:
        """Return the configured observability provider."""

    @property
    def enabled(self) -> bool:
        """Return whether external telemetry export is enabled."""

    def start_trace(
        self,
        attributes: TraceAttributes,
    ) -> AbstractContextManager[TraceScope]:
        """Start or re-enter one logical trace."""

    def record_trace_event(
        self,
        *,
        identity: TraceIdentity,
        event: EventObservation,
    ) -> None:
        """Record a discrete event against a deterministic logical trace."""

    def flush(self) -> None:
        """Flush buffered telemetry when the process policy requires it."""

    def shutdown(self) -> None:
        """Release process-owned observability resources."""
