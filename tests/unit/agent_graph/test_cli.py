"""Unit tests for the graph checkpoint operator CLI."""

import json
from io import StringIO

from pydantic import SecretStr

from supportops.agent_graph.cli import run_cli
from supportops.agent_graph.infrastructure.checkpoints import (
    GraphCheckpointSetupError,
    GraphCheckpointUnavailableError,
)
from supportops.core.settings import Settings

_CHECKPOINT_DATABASE_URL = (
    "postgresql://checkpoint-user:checkpoint-password@localhost:5432/supportops_checkpoints"
)


def create_settings(
    *,
    checkpoint_database_url: str | None = (_CHECKPOINT_DATABASE_URL),
) -> Settings:
    """Create valid local checkpoint command settings."""

    return Settings(
        postgresql_url=("postgresql+asyncpg://supportops:supportops@localhost:5432/supportops"),
        qdrant_url="http://localhost:6333",
        agent_graph_checkpoint_database_url=(checkpoint_database_url),
    )


class FakeRuntime:
    """Record command lifecycle without PostgreSQL access."""

    def __init__(self) -> None:
        self.setup_calls = 0
        self.close_calls = 0
        self.setup_error: Exception | None = None
        self.close_error: Exception | None = None

    async def setup(self) -> None:
        self.setup_calls += 1

        if self.setup_error is not None:
            raise self.setup_error

    async def close(self) -> None:
        self.close_calls += 1

        if self.close_error is not None:
            raise self.close_error


class FakeRuntimeFactory:
    """Return one configured fake command runtime."""

    def __init__(
        self,
        runtime: FakeRuntime,
    ) -> None:
        self.runtime = runtime
        self.database_urls: list[SecretStr] = []

    async def __call__(
        self,
        *,
        database_url: SecretStr,
    ) -> FakeRuntime:
        self.database_urls.append(database_url)
        return self.runtime


async def test_setup_runs_and_writes_safe_summary() -> None:
    settings = create_settings()
    runtime = FakeRuntime()
    factory = FakeRuntimeFactory(runtime)
    stdout = StringIO()
    stderr = StringIO()

    exit_code = await run_cli(
        ["setup"],
        stdout=stdout,
        stderr=stderr,
        settings_factory=lambda: settings,
        runtime_factory=factory,
    )

    assert exit_code == 0
    assert stderr.getvalue() == ""
    assert runtime.setup_calls == 1
    assert runtime.close_calls == 1
    assert len(factory.database_urls) == 1
    assert factory.database_urls[0].get_secret_value() == _CHECKPOINT_DATABASE_URL

    payload = json.loads(stdout.getvalue())

    assert payload == {
        "command": "setup",
        "owner": "langgraph-checkpoint-postgres",
        "status": "ready",
    }
    assert _CHECKPOINT_DATABASE_URL not in stdout.getvalue()


async def test_missing_checkpoint_url_is_configuration_error() -> None:
    settings = create_settings(
        checkpoint_database_url=None,
    )
    runtime = FakeRuntime()
    factory = FakeRuntimeFactory(runtime)
    stdout = StringIO()
    stderr = StringIO()

    exit_code = await run_cli(
        ["setup"],
        stdout=stdout,
        stderr=stderr,
        settings_factory=lambda: settings,
        runtime_factory=factory,
    )

    assert exit_code == 2
    assert stdout.getvalue() == ""
    assert stderr.getvalue() == (
        "graph_checkpoint_cli_error: SUPPORTOPS_AGENT_GRAPH_CHECKPOINT_DATABASE_URL is required.\n"
    )
    assert factory.database_urls == []
    assert runtime.setup_calls == 0
    assert runtime.close_calls == 0


async def test_setup_failure_is_safe_runtime_error() -> None:
    settings = create_settings()
    runtime = FakeRuntime()
    runtime.setup_error = GraphCheckpointSetupError()
    factory = FakeRuntimeFactory(runtime)
    stdout = StringIO()
    stderr = StringIO()

    exit_code = await run_cli(
        ["setup"],
        stdout=stdout,
        stderr=stderr,
        settings_factory=lambda: settings,
        runtime_factory=factory,
    )

    assert exit_code == 1
    assert stdout.getvalue() == ""
    assert stderr.getvalue() == (
        "graph_checkpoint_setup_failed: PostgreSQL graph checkpoint setup could not be completed.\n"
    )
    assert runtime.setup_calls == 1
    assert runtime.close_calls == 1
    assert _CHECKPOINT_DATABASE_URL not in stderr.getvalue()


async def test_runtime_creation_failure_is_safe() -> None:
    settings = create_settings()
    stdout = StringIO()
    stderr = StringIO()

    async def fail_runtime(
        *,
        database_url: SecretStr,
    ) -> FakeRuntime:
        del database_url
        raise GraphCheckpointUnavailableError()

    exit_code = await run_cli(
        ["setup"],
        stdout=stdout,
        stderr=stderr,
        settings_factory=lambda: settings,
        runtime_factory=fail_runtime,
    )

    assert exit_code == 1
    assert stdout.getvalue() == ""
    assert stderr.getvalue() == (
        "graph_checkpoint_unavailable: PostgreSQL graph checkpoint infrastructure is unavailable.\n"
    )


async def test_close_failure_prevents_success_summary() -> None:
    settings = create_settings()
    runtime = FakeRuntime()
    runtime.close_error = GraphCheckpointUnavailableError()
    factory = FakeRuntimeFactory(runtime)
    stdout = StringIO()
    stderr = StringIO()

    exit_code = await run_cli(
        ["setup"],
        stdout=stdout,
        stderr=stderr,
        settings_factory=lambda: settings,
        runtime_factory=factory,
    )

    assert exit_code == 1
    assert stdout.getvalue() == ""
    assert stderr.getvalue() == (
        "graph_checkpoint_unavailable: PostgreSQL graph checkpoint infrastructure is unavailable.\n"
    )
    assert runtime.setup_calls == 1
    assert runtime.close_calls == 1


async def test_unexpected_setup_failure_is_normalized() -> None:
    settings = create_settings()
    runtime = FakeRuntime()
    runtime.setup_error = RuntimeError(_CHECKPOINT_DATABASE_URL)
    factory = FakeRuntimeFactory(runtime)
    stdout = StringIO()
    stderr = StringIO()

    exit_code = await run_cli(
        ["setup"],
        stdout=stdout,
        stderr=stderr,
        settings_factory=lambda: settings,
        runtime_factory=factory,
    )

    assert exit_code == 1
    assert stdout.getvalue() == ""
    assert stderr.getvalue() == (
        "graph_checkpoint_runtime_failed: Graph checkpoint operation failed unexpectedly.\n"
    )
    assert _CHECKPOINT_DATABASE_URL not in stderr.getvalue()
    assert runtime.close_calls == 1
