"""Unit tests for structured logging."""

import json
import logging

from supportops.core.logging import JsonFormatter, configure_logging
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
