"""Task-local observability context propagation."""

from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ActiveTraceContext:
    """Current application trace identity and optional backend trace ID."""

    trace_seed: str
    trace_id: str | None = None
    session_id: str | None = None

    def __post_init__(self) -> None:
        _require_non_blank("trace_seed", self.trace_seed)

        if self.trace_id is not None:
            _require_non_blank("trace_id", self.trace_id)

        if self.session_id is not None:
            _require_non_blank("session_id", self.session_id)


@dataclass(frozen=True, slots=True)
class ActiveObservationContext:
    """Current observation identity within an active trace."""

    name: str
    observation_id: str | None = None

    def __post_init__(self) -> None:
        _require_non_blank("name", self.name)

        if self.observation_id is not None:
            _require_non_blank("observation_id", self.observation_id)


_CURRENT_TRACE_CONTEXT: ContextVar[ActiveTraceContext | None] = ContextVar(
    "supportops_observability_trace_context",
    default=None,
)

_CURRENT_OBSERVATION_CONTEXT: ContextVar[ActiveObservationContext | None] = ContextVar(
    "supportops_observability_observation_context",
    default=None,
)


def current_trace_context() -> ActiveTraceContext | None:
    """Return the task-local active trace context."""

    return _CURRENT_TRACE_CONTEXT.get()


def current_observation_context() -> ActiveObservationContext | None:
    """Return the task-local active observation context."""

    return _CURRENT_OBSERVATION_CONTEXT.get()


@contextmanager
def trace_context_scope(
    context: ActiveTraceContext,
) -> Generator[ActiveTraceContext, None, None]:
    """Bind trace context and restore the previous value in all outcomes."""

    token = _CURRENT_TRACE_CONTEXT.set(context)

    try:
        yield context
    finally:
        _CURRENT_TRACE_CONTEXT.reset(token)


@contextmanager
def observation_context_scope(
    context: ActiveObservationContext,
) -> Generator[ActiveObservationContext, None, None]:
    """Bind observation context and restore the previous value."""

    token = _CURRENT_OBSERVATION_CONTEXT.set(context)

    try:
        yield context
    finally:
        _CURRENT_OBSERVATION_CONTEXT.reset(token)


def _require_non_blank(field_name: str, value: str) -> None:
    if not value.strip():
        raise ValueError(f"{field_name} must not be blank")
