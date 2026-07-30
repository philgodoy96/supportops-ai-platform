"""Unit tests for HTTP request context middleware."""

import json
import logging
from uuid import UUID

from fastapi import FastAPI, Response
from httpx import ASGITransport, AsyncClient

from supportops.api.middleware.request_context import (
    CORRELATION_ID_HEADER,
    REQUEST_ID_HEADER,
    RequestContextMiddleware,
)
from supportops.core.logging import JsonFormatter
from supportops.core.request_context import get_request_context
from supportops.core.settings import ApplicationEnvironment


class FormattedLogHandler(logging.Handler):
    """Capture log records after formatter execution."""

    def __init__(self) -> None:
        super().__init__()
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.messages.append(self.format(record))


def create_test_application() -> FastAPI:
    """Create an isolated application for middleware tests."""

    app = FastAPI()
    app.add_middleware(RequestContextMiddleware)

    @app.get("/context")
    async def get_context() -> dict[str, str]:
        context = get_request_context()

        assert context is not None

        return {
            "request_id": str(context.request_id),
            "correlation_id": str(context.correlation_id),
        }

    @app.get("/items/{item_id}")
    async def get_item(item_id: str) -> dict[str, str]:
        return {"item_id": item_id}

    @app.get("/spoofed-headers")
    async def get_spoofed_headers(
        response: Response,
    ) -> dict[str, str]:
        response.headers[REQUEST_ID_HEADER] = "00000000-0000-0000-0000-000000000001"
        response.headers[CORRELATION_ID_HEADER] = "00000000-0000-0000-0000-000000000002"

        return {"status": "ok"}

    @app.get("/failure")
    async def get_failure() -> None:
        raise RuntimeError("controlled middleware failure")

    return app


async def test_middleware_generates_request_and_correlation_ids() -> None:
    app = create_test_application()
    transport = ASGITransport(app=app)

    async with AsyncClient(
        transport=transport,
        base_url="http://test",
    ) as client:
        response = await client.get("/context")

    assert response.status_code == 200

    request_id = UUID(response.headers[REQUEST_ID_HEADER])
    correlation_id = UUID(response.headers[CORRELATION_ID_HEADER])

    assert request_id.version == 4
    assert correlation_id == request_id
    assert response.json() == {
        "request_id": str(request_id),
        "correlation_id": str(correlation_id),
    }
    assert get_request_context() is None


async def test_middleware_accepts_valid_correlation_id() -> None:
    incoming_correlation_id = UUID(
        "c11375a8-80e6-4aa2-838f-f342cfcb99ae",
    )
    app = create_test_application()
    transport = ASGITransport(app=app)

    async with AsyncClient(
        transport=transport,
        base_url="http://test",
    ) as client:
        response = await client.get(
            "/context",
            headers={
                CORRELATION_ID_HEADER: str(
                    incoming_correlation_id,
                ).upper(),
            },
        )

    request_id = UUID(response.headers[REQUEST_ID_HEADER])
    correlation_id = UUID(response.headers[CORRELATION_ID_HEADER])

    assert request_id != incoming_correlation_id
    assert correlation_id == incoming_correlation_id
    assert response.json()["correlation_id"] == str(
        incoming_correlation_id,
    )


async def test_middleware_rejects_invalid_correlation_id() -> None:
    app = create_test_application()
    transport = ASGITransport(app=app)

    async with AsyncClient(
        transport=transport,
        base_url="http://test",
    ) as client:
        response = await client.get(
            "/context",
            headers={
                CORRELATION_ID_HEADER: "untrusted-external-value",
            },
        )

    request_id = UUID(response.headers[REQUEST_ID_HEADER])
    correlation_id = UUID(response.headers[CORRELATION_ID_HEADER])

    assert correlation_id == request_id
    assert "untrusted-external-value" not in response.headers.values()


async def test_middleware_ignores_incoming_request_id() -> None:
    incoming_request_id = UUID(
        "065ad043-c2c3-4b4b-9234-fde33275ef57",
    )
    app = create_test_application()
    transport = ASGITransport(app=app)

    async with AsyncClient(
        transport=transport,
        base_url="http://test",
    ) as client:
        response = await client.get(
            "/context",
            headers={
                REQUEST_ID_HEADER: str(incoming_request_id),
            },
        )

    generated_request_id = UUID(
        response.headers[REQUEST_ID_HEADER],
    )

    assert generated_request_id.version == 4
    assert generated_request_id != incoming_request_id


async def test_middleware_generates_unique_request_ids() -> None:
    app = create_test_application()
    transport = ASGITransport(app=app)

    async with AsyncClient(
        transport=transport,
        base_url="http://test",
    ) as client:
        first_response = await client.get("/context")
        second_response = await client.get("/context")

    first_request_id = UUID(
        first_response.headers[REQUEST_ID_HEADER],
    )
    second_request_id = UUID(
        second_response.headers[REQUEST_ID_HEADER],
    )

    assert first_request_id != second_request_id


async def test_middleware_overrides_downstream_trace_headers() -> None:
    app = create_test_application()
    transport = ASGITransport(app=app)

    async with AsyncClient(
        transport=transport,
        base_url="http://test",
    ) as client:
        response = await client.get("/spoofed-headers")

    request_id = UUID(response.headers[REQUEST_ID_HEADER])
    correlation_id = UUID(response.headers[CORRELATION_ID_HEADER])

    assert request_id.version == 4
    assert correlation_id == request_id
    assert str(request_id) != ("00000000-0000-0000-0000-000000000001")
    assert str(correlation_id) != ("00000000-0000-0000-0000-000000000002")


async def test_middleware_emits_structured_completion_log() -> None:
    middleware_logger = logging.getLogger(
        "supportops.api.middleware.request_context",
    )
    original_level = middleware_logger.level
    original_propagate = middleware_logger.propagate

    handler = FormattedLogHandler()
    handler.setFormatter(
        JsonFormatter(ApplicationEnvironment.TEST),
    )

    middleware_logger.addHandler(handler)
    middleware_logger.setLevel(logging.INFO)
    middleware_logger.propagate = False

    try:
        app = create_test_application()
        transport = ASGITransport(app=app)

        async with AsyncClient(
            transport=transport,
            base_url="http://test",
        ) as client:
            response = await client.get("/items/item-123")
    finally:
        middleware_logger.removeHandler(handler)
        middleware_logger.setLevel(original_level)
        middleware_logger.propagate = original_propagate

    payloads = [json.loads(message) for message in handler.messages]

    assert len(payloads) == 1

    payload = payloads[0]

    assert payload["event"] == "http_request_completed"
    assert payload["request_id"] == response.headers[REQUEST_ID_HEADER]
    assert payload["correlation_id"] == (response.headers[CORRELATION_ID_HEADER])
    assert payload["http_method"] == "GET"
    assert payload["route_or_path"] == "/items/{item_id}"
    assert payload["status_code"] == 200
    assert payload["duration_ms"] >= 0
    assert "item-123" not in payload.values()


async def test_middleware_returns_trace_headers_for_unexpected_exception() -> None:
    app = create_test_application()
    transport = ASGITransport(
        app=app,
        raise_app_exceptions=False,
    )

    async with AsyncClient(
        transport=transport,
        base_url="http://test",
    ) as client:
        response = await client.get("/failure")

    request_id = UUID(response.headers[REQUEST_ID_HEADER])
    correlation_id = UUID(response.headers[CORRELATION_ID_HEADER])

    assert response.status_code == 500
    assert response.text == "Internal Server Error"
    assert request_id.version == 4
    assert correlation_id == request_id
    assert get_request_context() is None
