"""Unit tests for the operator-controlled knowledge indexing CLI."""

import json
from contextlib import suppress
from datetime import UTC, datetime
from decimal import Decimal
from io import StringIO
from uuid import UUID

from supportops.ai.embeddings.errors import (
    EmbeddingTimeoutError,
)
from supportops.core.settings import (
    EmbeddingProviderName,
    Settings,
)
from supportops.knowledge_index.cli import (
    run_cli,
)
from supportops.knowledge_index.composition import (
    build_knowledge_index_profile,
)
from supportops.knowledge_index.indexing.results import (
    IndexDocumentVersionResult,
)
from supportops.modules.knowledge_documents.domain.models import (
    DocumentMediaType,
    DocumentVersion,
)

_WORKSPACE_ID = UUID("032c8c87-57cc-4d14-bfbd-04968b4e8cd4")
_DOCUMENT_ID = UUID("276046a2-28ec-4cb1-8bb6-a2ff70f9064b")
_VERSION_ID = UUID("09036916-84cf-4a58-bdf4-09bc52716ec5")
_INDEXED_AT = datetime(
    2026,
    8,
    2,
    2,
    0,
    tzinfo=UTC,
)


def create_settings(
    *,
    provider: EmbeddingProviderName = (EmbeddingProviderName.MOCK),
) -> Settings:
    """Create one valid CLI configuration."""

    if provider is EmbeddingProviderName.OPENAI:
        return Settings(
            postgresql_url=("postgresql+asyncpg://supportops:supportops@localhost:5432/supportops"),
            qdrant_url="http://localhost:6333",
            embedding_provider=provider,
            embedding_model=("text-embedding-3-small"),
            embedding_dimensions=1536,
            openai_api_key="test-openai-key",
        )

    return Settings(
        postgresql_url=("postgresql+asyncpg://supportops:supportops@localhost:5432/supportops"),
        qdrant_url="http://localhost:6333",
        embedding_provider=provider,
        embedding_model=("mock-hashing-embedding-v1"),
        embedding_dimensions=64,
    )


def create_ready_result(
    *,
    already_ready: bool = False,
) -> IndexDocumentVersionResult:
    """Create one successful indexing result."""

    settings = create_settings()
    profile = build_knowledge_index_profile(settings)
    pending = DocumentVersion.create_pending(
        document_version_id=_VERSION_ID,
        workspace_id=_WORKSPACE_ID,
        document_id=_DOCUMENT_ID,
        version_number=1,
        media_type=DocumentMediaType.TEXT_MARKDOWN,
        content=("# Recovery\n\nRestart the connection pool.\n"),
        now=_INDEXED_AT,
    )
    profiled = pending.bind_index_profile(
        profile,
        now=_INDEXED_AT,
    )
    ready = profiled.mark_ready(
        chunk_count=2,
        embedding_input_tokens=18,
        embedding_estimated_cost_usd=(Decimal("0")),
        embedding_pricing_catalog_version=("supportops-embedding-pricing-2026-08-01"),
        indexed_at=_INDEXED_AT,
    )

    return IndexDocumentVersionResult(
        version=ready,
        already_ready=already_ready,
    )


class FakeObservabilityClient:
    """Record observability shutdown for CLI lifecycle tests."""

    def __init__(self) -> None:
        self.shutdown_calls = 0
        self.shutdown_error: Exception | None = None

    def shutdown(self) -> None:
        self.shutdown_calls += 1
        if self.shutdown_error is not None:
            raise self.shutdown_error


class FakeRuntime:
    """Record CLI calls without infrastructure access."""

    def __init__(
        self,
        *,
        settings: Settings,
    ) -> None:
        self.index_profile = build_knowledge_index_profile(settings)
        self.observability_client = FakeObservabilityClient()
        self.ensure_calls = 0
        self.index_calls: list[tuple[UUID, UUID, UUID]] = []
        self.closed = 0
        self.ensure_error: Exception | None = None
        self.index_error: Exception | None = None
        self.close_error: Exception | None = None
        self.result = create_ready_result()

    async def ensure_collection(self) -> None:
        self.ensure_calls += 1
        if self.ensure_error is not None:
            raise self.ensure_error

    async def index_version(
        self,
        *,
        workspace_id: UUID,
        document_id: UUID,
        document_version_id: UUID,
    ) -> IndexDocumentVersionResult:
        self.index_calls.append(
            (
                workspace_id,
                document_id,
                document_version_id,
            )
        )
        if self.index_error is not None:
            raise self.index_error
        return self.result

    async def close(self) -> None:
        self.closed += 1
        with suppress(Exception):
            self.observability_client.shutdown()
        if self.close_error is not None:
            raise self.close_error


class FakeRuntimeFactory:
    """Create or return one configured fake runtime."""

    def __init__(
        self,
        runtime: FakeRuntime,
    ) -> None:
        self.runtime = runtime
        self.calls: list[Settings] = []

    async def __call__(
        self,
        *,
        settings: Settings,
    ) -> FakeRuntime:
        self.calls.append(settings)
        return self.runtime


async def test_ensure_collection_writes_stable_json_summary() -> None:
    settings = create_settings()
    runtime = FakeRuntime(settings=settings)
    factory = FakeRuntimeFactory(runtime)
    stdout = StringIO()
    stderr = StringIO()

    exit_code = await run_cli(
        ["ensure-collection"],
        stdout=stdout,
        stderr=stderr,
        settings_factory=lambda: settings,
        runtime_factory=factory,
    )

    assert exit_code == 0
    assert stderr.getvalue() == ""
    assert runtime.ensure_calls == 1
    assert runtime.index_calls == []
    assert runtime.closed == 1

    payload = json.loads(stdout.getvalue())
    assert payload == {
        "collection": ("supportops-knowledge-mock-v1"),
        "command": "ensure-collection",
        "dimensions": 64,
        "embedding_model": ("mock-hashing-embedding-v1"),
        "embedding_provider": "mock",
        "status": "compatible",
        "vector_name": "dense",
    }


async def test_index_version_passes_scoped_ids_and_writes_summary() -> None:
    settings = create_settings()
    runtime = FakeRuntime(settings=settings)
    factory = FakeRuntimeFactory(runtime)
    stdout = StringIO()
    stderr = StringIO()

    exit_code = await run_cli(
        [
            "index-version",
            "--workspace-id",
            str(_WORKSPACE_ID),
            "--document-id",
            str(_DOCUMENT_ID),
            "--document-version-id",
            str(_VERSION_ID),
        ],
        stdout=stdout,
        stderr=stderr,
        settings_factory=lambda: settings,
        runtime_factory=factory,
    )

    assert exit_code == 0
    assert stderr.getvalue() == ""
    assert runtime.index_calls == [
        (
            _WORKSPACE_ID,
            _DOCUMENT_ID,
            _VERSION_ID,
        )
    ]
    assert runtime.closed == 1

    payload = json.loads(stdout.getvalue())
    assert payload["command"] == "index-version"
    assert payload["workspace_id"] == str(_WORKSPACE_ID)
    assert payload["document_id"] == str(_DOCUMENT_ID)
    assert payload["document_version_id"] == str(_VERSION_ID)
    assert payload["status"] == "ready"
    assert payload["already_ready"] is False
    assert payload["chunk_count"] == 2
    assert payload["embedding_input_tokens"] == 18
    assert payload["embedding_estimated_cost_usd"] == "0"
    assert payload["indexed_at"] == (_INDEXED_AT.isoformat())
    assert "content" not in payload


async def test_openai_requires_explicit_permission_before_runtime() -> None:
    settings = create_settings(provider=EmbeddingProviderName.OPENAI)
    runtime = FakeRuntime(settings=settings)
    factory = FakeRuntimeFactory(runtime)
    stdout = StringIO()
    stderr = StringIO()

    exit_code = await run_cli(
        ["ensure-collection"],
        stdout=stdout,
        stderr=stderr,
        settings_factory=lambda: settings,
        runtime_factory=factory,
    )

    assert exit_code == 2
    assert stdout.getvalue() == ""
    assert stderr.getvalue() == (
        "indexing_cli_error: OpenAI knowledge indexing requires --allow-external-provider.\n"
    )
    assert factory.calls == []
    assert runtime.closed == 0


async def test_openai_runs_after_explicit_permission() -> None:
    settings = create_settings(provider=EmbeddingProviderName.OPENAI)
    runtime = FakeRuntime(settings=settings)
    factory = FakeRuntimeFactory(runtime)
    stdout = StringIO()
    stderr = StringIO()

    exit_code = await run_cli(
        [
            "ensure-collection",
            "--allow-external-provider",
        ],
        stdout=stdout,
        stderr=stderr,
        settings_factory=lambda: settings,
        runtime_factory=factory,
    )

    assert exit_code == 0
    assert stderr.getvalue() == ""
    assert len(factory.calls) == 1
    assert runtime.ensure_calls == 1
    assert runtime.closed == 1


async def test_mock_rejects_external_provider_flag() -> None:
    settings = create_settings()
    runtime = FakeRuntime(settings=settings)
    factory = FakeRuntimeFactory(runtime)
    stderr = StringIO()

    exit_code = await run_cli(
        [
            "ensure-collection",
            "--allow-external-provider",
        ],
        stdout=StringIO(),
        stderr=stderr,
        settings_factory=lambda: settings,
        runtime_factory=factory,
    )

    assert exit_code == 2
    assert "valid only when the OpenAI" in (stderr.getvalue())
    assert factory.calls == []
    assert runtime.closed == 0


async def test_owned_operational_error_returns_runtime_failure() -> None:
    settings = create_settings()
    runtime = FakeRuntime(settings=settings)
    runtime.index_error = EmbeddingTimeoutError()
    factory = FakeRuntimeFactory(runtime)
    stdout = StringIO()
    stderr = StringIO()

    exit_code = await run_cli(
        [
            "index-version",
            "--workspace-id",
            str(_WORKSPACE_ID),
            "--document-id",
            str(_DOCUMENT_ID),
            "--document-version-id",
            str(_VERSION_ID),
        ],
        stdout=stdout,
        stderr=stderr,
        settings_factory=lambda: settings,
        runtime_factory=factory,
    )

    assert exit_code == 1
    assert stdout.getvalue() == ""
    assert stderr.getvalue() == (
        "indexing_runtime_error: The embedding provider request exceeded its configured timeout.\n"
    )
    assert runtime.closed == 1


async def test_unexpected_error_is_sanitized() -> None:
    settings = create_settings()
    runtime = FakeRuntime(settings=settings)
    runtime.ensure_error = RuntimeError("database-password=secret")
    factory = FakeRuntimeFactory(runtime)
    stderr = StringIO()

    exit_code = await run_cli(
        ["ensure-collection"],
        stdout=StringIO(),
        stderr=stderr,
        settings_factory=lambda: settings,
        runtime_factory=factory,
    )

    assert exit_code == 1
    assert stderr.getvalue() == (
        "indexing_runtime_error: Knowledge indexing failed unexpectedly.\n"
    )
    assert "secret" not in stderr.getvalue()
    assert runtime.closed == 1


async def test_close_failure_changes_success_to_runtime_failure() -> None:
    settings = create_settings()
    runtime = FakeRuntime(settings=settings)
    runtime.close_error = RuntimeError("close failed")
    factory = FakeRuntimeFactory(runtime)
    stdout = StringIO()
    stderr = StringIO()

    exit_code = await run_cli(
        ["ensure-collection"],
        stdout=stdout,
        stderr=stderr,
        settings_factory=lambda: settings,
        runtime_factory=factory,
    )

    assert exit_code == 1
    assert runtime.ensure_calls == 1
    assert runtime.closed == 1
    assert runtime.observability_client.shutdown_calls == 1
    assert json.loads(stdout.getvalue())["status"] == "compatible"
    assert stderr.getvalue() == (
        "indexing_runtime_error: Knowledge indexing resources could not be closed safely.\n"
    )


async def test_observability_shutdown_on_success() -> None:
    settings = create_settings()
    runtime = FakeRuntime(settings=settings)
    factory = FakeRuntimeFactory(runtime)

    exit_code = await run_cli(
        ["ensure-collection"],
        stdout=StringIO(),
        stderr=StringIO(),
        settings_factory=lambda: settings,
        runtime_factory=factory,
    )

    assert exit_code == 0
    assert len(factory.calls) == 1
    assert runtime.closed == 1
    assert runtime.observability_client.shutdown_calls == 1


async def test_observability_shutdown_on_indexing_failure() -> None:
    settings = create_settings()
    runtime = FakeRuntime(settings=settings)
    runtime.index_error = EmbeddingTimeoutError()
    factory = FakeRuntimeFactory(runtime)

    exit_code = await run_cli(
        [
            "index-version",
            "--workspace-id",
            str(_WORKSPACE_ID),
            "--document-id",
            str(_DOCUMENT_ID),
            "--document-version-id",
            str(_VERSION_ID),
        ],
        stdout=StringIO(),
        stderr=StringIO(),
        settings_factory=lambda: settings,
        runtime_factory=factory,
    )

    assert exit_code == 1
    assert runtime.closed == 1
    assert runtime.observability_client.shutdown_calls == 1


async def test_observability_shutdown_failure_preserves_successful_exit_code() -> None:
    settings = create_settings()
    runtime = FakeRuntime(settings=settings)
    runtime.observability_client.shutdown_error = RuntimeError(
        "observability shutdown failed",
    )
    factory = FakeRuntimeFactory(runtime)
    stderr = StringIO()

    exit_code = await run_cli(
        ["ensure-collection"],
        stdout=StringIO(),
        stderr=stderr,
        settings_factory=lambda: settings,
        runtime_factory=factory,
    )

    assert exit_code == 0
    assert runtime.closed == 1
    assert runtime.observability_client.shutdown_calls == 1
    assert stderr.getvalue() == ""


async def test_observability_shutdown_failure_preserves_indexing_failure() -> None:
    settings = create_settings()
    runtime = FakeRuntime(settings=settings)
    runtime.index_error = EmbeddingTimeoutError()
    runtime.observability_client.shutdown_error = RuntimeError(
        "observability shutdown failed",
    )
    factory = FakeRuntimeFactory(runtime)
    stderr = StringIO()

    exit_code = await run_cli(
        [
            "index-version",
            "--workspace-id",
            str(_WORKSPACE_ID),
            "--document-id",
            str(_DOCUMENT_ID),
            "--document-version-id",
            str(_VERSION_ID),
        ],
        stdout=StringIO(),
        stderr=stderr,
        settings_factory=lambda: settings,
        runtime_factory=factory,
    )

    assert exit_code == 1
    assert runtime.closed == 1
    assert runtime.observability_client.shutdown_calls == 1
    assert stderr.getvalue() == (
        "indexing_runtime_error: The embedding provider request exceeded its configured timeout.\n"
    )
