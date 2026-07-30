"""HTTP request correlation context primitives."""

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from uuid import UUID, uuid4


@dataclass(frozen=True, slots=True)
class RequestContext:
    """Trace identifiers associated with one HTTP request."""

    request_id: UUID
    correlation_id: UUID


_active_request_context: ContextVar[RequestContext | None] = ContextVar(
    "active_request_context",
    default=None,
)


def create_request_context(
    incoming_correlation_id: str | None = None,
) -> RequestContext:
    """Create server-owned request context from an optional correlation identifier."""

    request_id = uuid4()
    correlation_id = _parse_correlation_id(incoming_correlation_id) or request_id

    return RequestContext(
        request_id=request_id,
        correlation_id=correlation_id,
    )


def get_request_context() -> RequestContext | None:
    """Return the context bound to the current execution context, if present."""

    return _active_request_context.get()


@contextmanager
def request_context_scope(context: RequestContext) -> Iterator[None]:
    """Bind request context and restore the previous value on exit."""

    token = _active_request_context.set(context)

    try:
        yield
    finally:
        _active_request_context.reset(token)


def _parse_correlation_id(value: str | None) -> UUID | None:
    """Return a parsed UUID or None when an external value is invalid."""

    if value is None:
        return None

    try:
        return UUID(value)
    except ValueError:
        return None
