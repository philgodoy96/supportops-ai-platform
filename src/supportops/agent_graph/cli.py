"""Operator-owned command line interface for graph checkpoints."""

import argparse
import asyncio
import json
import sys
from collections.abc import Awaitable, Callable, Sequence
from typing import Protocol, TextIO

from pydantic import SecretStr, ValidationError

from supportops.agent_graph.infrastructure.checkpoints import (
    GraphCheckpointError,
    create_postgres_checkpoint_runtime,
)
from supportops.core.settings import Settings

_EXIT_SUCCESS = 0
_EXIT_RUNTIME_FAILURE = 1
_EXIT_USAGE_OR_CONFIGURATION_FAILURE = 2


class GraphCheckpointCLIError(ValueError):
    """Raised when checkpoint CLI execution is not configured safely."""


class GraphCheckpointCommandRuntime(Protocol):
    """Runtime behavior required by the checkpoint setup command."""

    async def setup(self) -> None:
        """Create or migrate framework-owned checkpoint tables."""
        ...

    async def close(self) -> None:
        """Release process-owned checkpoint resources."""
        ...


class GraphCheckpointRuntimeFactory(Protocol):
    """Construct one process-scoped checkpoint command runtime."""

    def __call__(
        self,
        *,
        database_url: SecretStr,
    ) -> Awaitable[GraphCheckpointCommandRuntime]:
        """Return one opened checkpoint runtime."""
        ...


type SettingsFactory = Callable[[], Settings]


def main() -> None:
    """Run the graph checkpoint operator CLI."""

    raise SystemExit(
        asyncio.run(
            run_cli(),
            loop_factory=_create_cli_event_loop,
        )
    )


def _create_cli_event_loop() -> asyncio.AbstractEventLoop:
    """Create a SelectorEventLoop (required by Psycopg on Windows)."""

    return asyncio.SelectorEventLoop()


async def run_cli(
    argv: Sequence[str] | None = None,
    *,
    stdout: TextIO = sys.stdout,
    stderr: TextIO = sys.stderr,
    settings_factory: SettingsFactory = Settings,
    runtime_factory: GraphCheckpointRuntimeFactory = (create_postgres_checkpoint_runtime),
) -> int:
    """Execute one explicit checkpoint operation."""

    parser = build_parser()
    arguments = parser.parse_args(argv)

    runtime: GraphCheckpointCommandRuntime | None = None
    output_payload: dict[str, object] | None = None
    exit_code = _EXIT_SUCCESS

    try:
        settings = settings_factory()
        checkpoint_database_url = settings.agent_graph_checkpoint_database_url

        if checkpoint_database_url is None:
            raise GraphCheckpointCLIError(
                "SUPPORTOPS_AGENT_GRAPH_CHECKPOINT_DATABASE_URL is required."
            )

        runtime = await runtime_factory(
            database_url=checkpoint_database_url,
        )

        if arguments.command != "setup":
            raise RuntimeError("Graph checkpoint parser produced an unsupported command.")

        await runtime.setup()

        output_payload = {
            "command": "setup",
            "owner": "langgraph-checkpoint-postgres",
            "status": "ready",
        }
    except (
        GraphCheckpointCLIError,
        ValidationError,
    ) as error:
        _write_expected_error(
            stderr=stderr,
            error=error,
        )
        exit_code = _EXIT_USAGE_OR_CONFIGURATION_FAILURE
    except GraphCheckpointError as error:
        _write_operational_error(
            stderr=stderr,
            error=error,
        )
        exit_code = _EXIT_RUNTIME_FAILURE
    except Exception:
        _write_runtime_error(
            stderr=stderr,
            message=("Graph checkpoint operation failed unexpectedly."),
        )
        exit_code = _EXIT_RUNTIME_FAILURE
    finally:
        if runtime is not None:
            try:
                await runtime.close()
            except GraphCheckpointError as error:
                if exit_code == _EXIT_SUCCESS:
                    _write_operational_error(
                        stderr=stderr,
                        error=error,
                    )
                    exit_code = _EXIT_RUNTIME_FAILURE
            except Exception:
                if exit_code == _EXIT_SUCCESS:
                    _write_runtime_error(
                        stderr=stderr,
                        message=("Graph checkpoint resources could not be closed safely."),
                    )
                    exit_code = _EXIT_RUNTIME_FAILURE

    if exit_code == _EXIT_SUCCESS and output_payload is not None:
        _write_json(
            stream=stdout,
            payload=output_payload,
        )

    return exit_code


def build_parser() -> argparse.ArgumentParser:
    """Build the stable checkpoint operator command surface."""

    parser = argparse.ArgumentParser(
        prog="supportops-graph-checkpoints",
        description=("Create and migrate framework-owned LangGraph checkpoint tables."),
    )

    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
    )

    subparsers.add_parser(
        "setup",
        help=("Create or migrate PostgreSQL checkpoint tables idempotently."),
    )

    return parser


def _write_expected_error(
    *,
    stderr: TextIO,
    error: Exception,
) -> None:
    stderr.write(f"graph_checkpoint_cli_error: {error}\n")


def _write_operational_error(
    *,
    stderr: TextIO,
    error: GraphCheckpointError,
) -> None:
    stderr.write(f"{error.error_code}: {error}\n")


def _write_runtime_error(
    *,
    stderr: TextIO,
    message: str,
) -> None:
    stderr.write(f"graph_checkpoint_runtime_failed: {message}\n")


def _write_json(
    *,
    stream: TextIO,
    payload: dict[str, object],
) -> None:
    stream.write(
        json.dumps(
            payload,
            sort_keys=True,
        )
    )
    stream.write("\n")


if __name__ == "__main__":
    main()
