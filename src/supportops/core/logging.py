"""Structured JSON logging configuration."""

import json
import logging
from datetime import UTC, datetime
from typing import Any

from supportops.core.settings import ApplicationEnvironment, LogLevel


class JsonFormatter(logging.Formatter):
    """Format standard library log records as structured JSON."""

    _reserved_attributes = frozenset(
        {
            "args",
            "asctime",
            "created",
            "exc_info",
            "exc_text",
            "filename",
            "funcName",
            "levelname",
            "levelno",
            "lineno",
            "module",
            "msecs",
            "message",
            "msg",
            "name",
            "pathname",
            "process",
            "processName",
            "relativeCreated",
            "stack_info",
            "thread",
            "threadName",
            "taskName",
        }
    )

    def __init__(self, environment: ApplicationEnvironment) -> None:
        super().__init__()
        self._environment = environment

    def format(self, record: logging.LogRecord) -> str:
        """Return a serialized JSON representation of a log record."""

        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(
                record.created,
                tz=UTC,
            ).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "environment": self._environment.value,
            "event": record.getMessage(),
        }

        for attribute, value in record.__dict__.items():
            if attribute not in self._reserved_attributes and not attribute.startswith("_"):
                payload[attribute] = value

        if record.exc_info is not None:
            payload["exception"] = self.formatException(record.exc_info)

        return json.dumps(
            payload,
            default=str,
            ensure_ascii=False,
            separators=(",", ":"),
        )


def configure_logging(
    *,
    environment: ApplicationEnvironment,
    log_level: LogLevel,
) -> None:
    """Configure process-level structured JSON logging."""

    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter(environment))

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.addHandler(handler)
    root_logger.setLevel(log_level.value)
