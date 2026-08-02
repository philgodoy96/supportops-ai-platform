"""Fixtures for agent graph PostgreSQL integration tests."""

from __future__ import annotations

import asyncio
import sys
from collections.abc import Callable, Mapping

if sys.platform == "win32":

    def pytest_asyncio_loop_factories(
        config: object,
        item: object,
    ) -> Mapping[str, Callable[[], asyncio.AbstractEventLoop]]:
        """Force SelectorEventLoop so Psycopg can run under Windows asyncio."""

        del config, item
        return {"selector": asyncio.SelectorEventLoop}
