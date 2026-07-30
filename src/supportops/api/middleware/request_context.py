"""HTTP request context middleware."""

import logging
from time import perf_counter

from starlette.datastructures import Headers, MutableHeaders
from starlette.responses import PlainTextResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from supportops.core.request_context import (
    create_request_context,
    request_context_scope,
)

REQUEST_ID_HEADER = "X-Request-ID"
CORRELATION_ID_HEADER = "X-Correlation-ID"

_MAX_LOGGED_PATH_LENGTH = 512

logger = logging.getLogger(__name__)


class RequestContextMiddleware:
    """Bind trace identifiers to each HTTP request."""

    def __init__(self, app: ASGIApp) -> None:
        self._app = app

    async def __call__(
        self,
        scope: Scope,
        receive: Receive,
        send: Send,
    ) -> None:
        if scope["type"] != "http":
            await self._app(scope, receive, send)
            return

        incoming_correlation_id = Headers(scope=scope).get(
            CORRELATION_ID_HEADER,
        )
        context = create_request_context(incoming_correlation_id)

        started_at = perf_counter()
        status_code = 500
        response_started = False

        async def send_with_context(message: Message) -> None:
            nonlocal response_started
            nonlocal status_code

            if message["type"] == "http.response.start":
                response_started = True
                status_code = int(message["status"])

                response_headers = MutableHeaders(scope=message)
                response_headers[REQUEST_ID_HEADER] = str(context.request_id)
                response_headers[CORRELATION_ID_HEADER] = str(
                    context.correlation_id,
                )

            await send(message)

        with request_context_scope(context):
            try:
                await self._app(
                    scope,
                    receive,
                    send_with_context,
                )
            except Exception:
                if not response_started:
                    response = PlainTextResponse(
                        "Internal Server Error",
                        status_code=500,
                    )
                    await response(
                        scope,
                        receive,
                        send_with_context,
                    )

                raise
            finally:
                duration_ms = round(
                    (perf_counter() - started_at) * 1000,
                    3,
                )

                logger.info(
                    "http_request_completed",
                    extra={
                        "http_method": _resolve_http_method(scope),
                        "route_or_path": _resolve_route_or_path(scope),
                        "status_code": status_code,
                        "duration_ms": duration_ms,
                    },
                )


def _resolve_http_method(scope: Scope) -> str:
    """Return the HTTP method or a safe fallback."""

    method = scope.get("method")

    if isinstance(method, str):
        return method

    return "UNKNOWN"


def _resolve_route_or_path(scope: Scope) -> str:
    """Return the matched route template or a sanitized request path."""

    route = scope.get("route")
    route_path = getattr(route, "path", None)

    if isinstance(route_path, str):
        return route_path

    path = scope.get("path")

    if not isinstance(path, str):
        return "<unknown>"

    sanitized_path = "".join(character if character.isprintable() else "?" for character in path)

    return sanitized_path[:_MAX_LOGGED_PATH_LENGTH]
