"""Unit tests for structured logging."""

import json
import logging
from uuid import UUID

from supportops.core.logging import JsonFormatter, configure_logging
from supportops.core.request_context import RequestContext, request_context_scope
from supportops.core.settings import ApplicationEnvironment, LogLevel


def test_json_formatter_includes_operational_context() -> None:
    formatter = JsonFormatter(ApplicationEnvironment.TEST)
    record = logging.LogRecord(
        name="supportops.test",
        level=logging.INFO,
        pathname=__file__,
        lineno=10,
        msg="dependency_check_completed",
        args=(),
        exc_info=None,
    )
    record.dependency = "postgresql"
    record.status = "healthy"

    payload = json.loads(formatter.format(record))

    assert payload["level"] == "INFO"
    assert payload["logger"] == "supportops.test"
    assert payload["environment"] == "test"
    assert payload["event"] == "dependency_check_completed"
    assert payload["dependency"] == "postgresql"
    assert payload["status"] == "healthy"
    assert "timestamp" in payload


def test_json_formatter_includes_exception_information() -> None:
    formatter = JsonFormatter(ApplicationEnvironment.TEST)

    try:
        raise RuntimeError("controlled failure")
    except RuntimeError:
        record = logging.LogRecord(
            name="supportops.test",
            level=logging.ERROR,
            pathname=__file__,
            lineno=20,
            msg="operation_failed",
            args=(),
            exc_info=__import__("sys").exc_info(),
        )

    payload = json.loads(formatter.format(record))

    assert payload["event"] == "operation_failed"
    assert "RuntimeError: controlled failure" in payload["exception"]


def test_configure_logging_replaces_root_handlers() -> None:
    root_logger = logging.getLogger()
    original_handlers = root_logger.handlers.copy()
    original_level = root_logger.level

    try:
        root_logger.addHandler(logging.NullHandler())

        configure_logging(
            environment=ApplicationEnvironment.TEST,
            log_level=LogLevel.WARNING,
        )

        assert len(root_logger.handlers) == 1
        assert isinstance(root_logger.handlers[0].formatter, JsonFormatter)
        assert root_logger.level == logging.WARNING
    finally:
        root_logger.handlers.clear()
        root_logger.handlers.extend(original_handlers)
        root_logger.setLevel(original_level)


def test_json_formatter_includes_request_context_identifiers() -> None:
    formatter = JsonFormatter(ApplicationEnvironment.TEST)
    request_id = UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")
    correlation_id = UUID("bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb")
    context = RequestContext(
        request_id=request_id,
        correlation_id=correlation_id,
    )
    record = logging.LogRecord(
        name="supportops.test",
        level=logging.INFO,
        pathname=__file__,
        lineno=30,
        msg="request_handled",
        args=(),
        exc_info=None,
    )

    with request_context_scope(context):
        payload = json.loads(formatter.format(record))

    assert payload["request_id"] == "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
    assert payload["correlation_id"] == "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"


def test_json_formatter_omits_request_identifiers_outside_context() -> None:
    formatter = JsonFormatter(ApplicationEnvironment.TEST)
    record = logging.LogRecord(
        name="supportops.test",
        level=logging.INFO,
        pathname=__file__,
        lineno=40,
        msg="startup_complete",
        args=(),
        exc_info=None,
    )

    payload = json.loads(formatter.format(record))

    assert "request_id" not in payload
    assert "correlation_id" not in payload


def test_json_formatter_ignores_spoofed_request_identifiers() -> None:
    formatter = JsonFormatter(ApplicationEnvironment.TEST)
    request_id = UUID("cccccccc-cccc-cccc-cccc-cccccccccccc")
    correlation_id = UUID("dddddddd-dddd-dddd-dddd-dddddddddddd")
    context = RequestContext(
        request_id=request_id,
        correlation_id=correlation_id,
    )
    record = logging.LogRecord(
        name="supportops.test",
        level=logging.INFO,
        pathname=__file__,
        lineno=50,
        msg="request_handled",
        args=(),
        exc_info=None,
    )
    record.request_id = "spoofed-request-id"
    record.correlation_id = "spoofed-correlation-id"

    with request_context_scope(context):
        payload = json.loads(formatter.format(record))

    assert payload["request_id"] == "cccccccc-cccc-cccc-cccc-cccccccccccc"
    assert payload["correlation_id"] == "dddddddd-dddd-dddd-dddd-dddddddddddd"
