"""Unit tests for the operator-controlled knowledge indexing CLI."""

from __future__ import annotations

import json
from collections.abc import Callable
from contextlib import AbstractContextManager, suppress
from datetime import UTC, datetime
from decimal import Decimal
from io import StringIO
from types import TracebackType
from typing import Literal, cast
from uuid import UUID

from supportops.ai.embeddings.errors import (
    EmbeddingTimeoutError,
)
from supportops.core.settings import (
    EmbeddingProviderName,
    Settings,
)
from supportops.knowledge_index.cli import (
    KnowledgeIndexCommandRuntime,
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
from supportops.observability.context import (
    ActiveTraceContext,
    current_trace_context,
    trace_context_scope,
)
from supportops.observability.contracts import ObservabilityClient
from supportops.observability.identity import (
    knowledge_index_trace_identity,
)
from supportops.observability.models import (
    EventObservation,
    ObservabilityProvider,
    ObservationAttributes,
    ObservationStatus,
    ObservationUpdate,
    TraceAttributes,
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

_FORBIDDEN_METADATA_TOKENS = (
    "content",
    "chunk_content",
    "chunk_preview",
    "embedding_vector",
    "lease_token",
    "execution_grant",
    "input_data",
    "output_data",
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


class RecordingTraceScope:
    def __init__(
        self,
        *,
        attributes: TraceAttributes,
        fail_update: bool = False,
    ) -> None:
        self.attributes = attributes
        self.fail_update = fail_update
        self.updates: list[ObservationUpdate] = []

    @property
    def trace_seed(self) -> str:
        return self.attributes.trace_seed

    @property
    def trace_id(self) -> str | None:
        return "trace-test"

    @property
    def session_id(self) -> str | None:
        return self.attributes.session_id

    def update(self, update: ObservationUpdate) -> None:
        if self.fail_update:
            raise RuntimeError("synthetic trace update failure")
        self.updates.append(update)

    def start_observation(
        self,
        attributes: ObservationAttributes,
    ) -> AbstractContextManager[object]:
        del attributes
        raise AssertionError("CLI tests must not start nested observations.")

    def record_event(self, event: EventObservation) -> None:
        del event
        raise AssertionError("CLI tests must not record events.")


class RecordingTraceManager(AbstractContextManager[RecordingTraceScope]):
    def __init__(
        self,
        *,
        scope: RecordingTraceScope,
        fail_enter: bool = False,
        fail_exit: bool = False,
        on_enter: Callable[[str], None] | None = None,
        on_exit: Callable[[str], None] | None = None,
    ) -> None:
        self.scope = scope
        self.fail_enter = fail_enter
        self.fail_exit = fail_exit
        self._on_enter = on_enter
        self._on_exit = on_exit
        self.entered = False
        self.exited = False
        self._context_manager = trace_context_scope(
            ActiveTraceContext(
                trace_seed=scope.attributes.trace_seed,
                session_id=scope.attributes.session_id,
            )
        )

    def __enter__(self) -> RecordingTraceScope:
        if self.fail_enter:
            raise RuntimeError("synthetic trace enter failure")
        self._context_manager.__enter__()
        self.entered = True
        if self._on_enter is not None:
            self._on_enter(self.scope.attributes.name)
        return self.scope

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> Literal[False]:
        if self._on_exit is not None:
            self._on_exit(self.scope.attributes.name)
        self._context_manager.__exit__(exc_type, exc, traceback)
        if self.fail_exit:
            raise RuntimeError("synthetic trace exit failure")
        self.exited = True
        return False


class RecordingObservabilityClient:
    """Record one process-owned observability client for CLI tests."""

    def __init__(
        self,
        *,
        fail_start: bool = False,
        fail_enter: bool = False,
        fail_update: bool = False,
        fail_exit: bool = False,
        shutdown_error: Exception | None = None,
    ) -> None:
        self._fail_start = fail_start
        self._fail_enter = fail_enter
        self._fail_update = fail_update
        self._fail_exit = fail_exit
        self.shutdown_error = shutdown_error
        self.shutdown_calls = 0
        self.trace_attributes: list[TraceAttributes] = []
        self.trace_scopes: list[RecordingTraceScope] = []
        self.trace_managers: list[RecordingTraceManager] = []
        self.lifecycle: list[tuple[str, str]] = []
        self.observation_start_calls = 0

    @property
    def provider(self) -> ObservabilityProvider:
        return ObservabilityProvider.NOOP

    @property
    def enabled(self) -> bool:
        return True

    def start_trace(
        self,
        attributes: TraceAttributes,
    ) -> AbstractContextManager[RecordingTraceScope]:
        if self._fail_start:
            raise RuntimeError("synthetic trace start failure")

        scope = RecordingTraceScope(
            attributes=attributes,
            fail_update=self._fail_update,
        )
        manager = RecordingTraceManager(
            scope=scope,
            fail_enter=self._fail_enter,
            fail_exit=self._fail_exit,
            on_enter=lambda name: self.lifecycle.append(("enter", name)),
            on_exit=lambda name: self.lifecycle.append(("exit", name)),
        )
        self.trace_attributes.append(attributes)
        self.trace_scopes.append(scope)
        self.trace_managers.append(manager)
        return manager

    def start_observation(
        self,
        attributes: ObservationAttributes,
    ) -> AbstractContextManager[object]:
        del attributes
        self.observation_start_calls += 1
        raise AssertionError("CLI must not start observations directly.")

    def record_event(self, event: EventObservation) -> None:
        del event
        raise AssertionError("CLI must not record events.")

    def record_trace_event(self, *, identity: object, event: EventObservation) -> None:
        del identity, event
        raise AssertionError("CLI must not record events.")

    def flush(self) -> None:
        return None

    def shutdown(self) -> None:
        self.shutdown_calls += 1
        if self.shutdown_error is not None:
            raise self.shutdown_error


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
        observability_client: (
            FakeObservabilityClient | RecordingObservabilityClient | None
        ) = None,
    ) -> None:
        self.index_profile = build_knowledge_index_profile(settings)
        self.observability_recorder: FakeObservabilityClient | RecordingObservabilityClient = (
            observability_client or FakeObservabilityClient()
        )
        self.observability_client = cast(
            ObservabilityClient,
            self.observability_recorder,
        )
        self.ensure_calls = 0
        self.index_calls: list[tuple[UUID, UUID, UUID]] = []
        self.index_trace_seeds: list[str | None] = []
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
        active_trace = current_trace_context()
        self.index_trace_seeds.append(
            None if active_trace is None else active_trace.trace_seed,
        )
        if self.index_error is not None:
            raise self.index_error
        return self.result

    async def close(self) -> None:
        self.closed += 1
        with suppress(Exception):
            self.observability_recorder.shutdown()
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
    ) -> KnowledgeIndexCommandRuntime:
        self.calls.append(settings)
        return self.runtime


def _index_argv() -> list[str]:
    return [
        "index-version",
        "--workspace-id",
        str(_WORKSPACE_ID),
        "--document-id",
        str(_DOCUMENT_ID),
        "--document-version-id",
        str(_VERSION_ID),
    ]


def _assert_content_free_metadata(metadata: object) -> None:
    payload = json.dumps(metadata, default=str).lower()
    for token in _FORBIDDEN_METADATA_TOKENS:
        assert token not in payload


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
        _index_argv(),
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
        _index_argv(),
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
    assert runtime.observability_recorder.shutdown_calls == 1
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
    assert runtime.observability_recorder.shutdown_calls == 1


async def test_observability_shutdown_on_indexing_failure() -> None:
    settings = create_settings()
    runtime = FakeRuntime(settings=settings)
    runtime.index_error = EmbeddingTimeoutError()
    factory = FakeRuntimeFactory(runtime)

    exit_code = await run_cli(
        _index_argv(),
        stdout=StringIO(),
        stderr=StringIO(),
        settings_factory=lambda: settings,
        runtime_factory=factory,
    )

    assert exit_code == 1
    assert runtime.closed == 1
    assert runtime.observability_recorder.shutdown_calls == 1


async def test_observability_shutdown_failure_preserves_successful_exit_code() -> None:
    settings = create_settings()
    runtime = FakeRuntime(settings=settings)
    runtime.observability_recorder.shutdown_error = RuntimeError(
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
    assert runtime.observability_recorder.shutdown_calls == 1
    assert stderr.getvalue() == ""


async def test_observability_shutdown_failure_preserves_indexing_failure() -> None:
    settings = create_settings()
    runtime = FakeRuntime(settings=settings)
    runtime.index_error = EmbeddingTimeoutError()
    runtime.observability_recorder.shutdown_error = RuntimeError(
        "observability shutdown failed",
    )
    factory = FakeRuntimeFactory(runtime)
    stderr = StringIO()

    exit_code = await run_cli(
        _index_argv(),
        stdout=StringIO(),
        stderr=stderr,
        settings_factory=lambda: settings,
        runtime_factory=factory,
    )

    assert exit_code == 1
    assert runtime.closed == 1
    assert runtime.observability_recorder.shutdown_calls == 1
    assert stderr.getvalue() == (
        "indexing_runtime_error: The embedding provider request exceeded its configured timeout.\n"
    )


async def test_index_version_creates_one_deterministic_trace_before_shutdown() -> None:
    settings = create_settings()
    observability = RecordingObservabilityClient()
    runtime = FakeRuntime(
        settings=settings,
        observability_client=observability,
    )
    factory = FakeRuntimeFactory(runtime)

    exit_code = await run_cli(
        _index_argv(),
        stdout=StringIO(),
        stderr=StringIO(),
        settings_factory=lambda: settings,
        runtime_factory=factory,
    )

    assert exit_code == 0
    assert len(observability.trace_attributes) == 1
    assert len(factory.calls) == 1

    attributes = observability.trace_attributes[0]
    execution_id = attributes.metadata["execution_id"]
    assert isinstance(execution_id, str)
    identity = knowledge_index_trace_identity(execution_id=execution_id)

    assert attributes.name == "knowledge-index"
    assert attributes.trace_seed == identity.trace_seed
    assert attributes.tags == identity.tags
    assert attributes.session_id is None
    assert attributes.metadata["workspace_id"] == str(_WORKSPACE_ID)
    assert attributes.metadata["document_id"] == str(_DOCUMENT_ID)
    assert attributes.metadata["document_version_id"] == str(_VERSION_ID)
    assert attributes.metadata["embedding_provider"] == "mock"
    assert attributes.metadata["embedding_model"] == "mock-hashing-embedding-v1"
    assert attributes.metadata["embedding_dimensions"] == 64
    assert attributes.metadata["batch_size"] == 64
    assert attributes.metadata["chunking_strategy"] == "markdown-token"
    assert attributes.metadata["chunking_version"] == "v1"
    assert "correlation_id" not in attributes.metadata
    _assert_content_free_metadata(attributes.metadata)

    manager = observability.trace_managers[0]
    scope = observability.trace_scopes[0]
    assert manager.entered is True
    assert manager.exited is True
    assert scope.updates[-1].status is ObservationStatus.OK
    assert len(observability.trace_attributes) == 1
    assert observability.observation_start_calls == 0
    assert runtime.index_trace_seeds == [identity.trace_seed]
    assert observability.lifecycle == [
        ("enter", "knowledge-index"),
        ("exit", "knowledge-index"),
    ]
    assert current_trace_context() is None
    assert observability.shutdown_calls == 1
    assert runtime.closed == 1
    assert observability.lifecycle[-1] == ("exit", "knowledge-index")


async def test_index_version_failure_closes_trace_before_shutdown() -> None:
    settings = create_settings()
    observability = RecordingObservabilityClient()
    runtime = FakeRuntime(
        settings=settings,
        observability_client=observability,
    )
    runtime.index_error = EmbeddingTimeoutError()
    factory = FakeRuntimeFactory(runtime)

    exit_code = await run_cli(
        _index_argv(),
        stdout=StringIO(),
        stderr=StringIO(),
        settings_factory=lambda: settings,
        runtime_factory=factory,
    )

    assert exit_code == 1
    assert len(observability.trace_attributes) == 1
    scope = observability.trace_scopes[0]
    assert scope.updates[-1].status is ObservationStatus.ERROR
    assert scope.updates[-1].error_code == "embedding_timeout"
    assert observability.trace_managers[0].exited is True
    assert observability.shutdown_calls == 1


async def test_trace_start_failure_preserves_success_exit_code() -> None:
    settings = create_settings()
    observability = RecordingObservabilityClient(fail_start=True)
    runtime = FakeRuntime(
        settings=settings,
        observability_client=observability,
    )
    factory = FakeRuntimeFactory(runtime)

    exit_code = await run_cli(
        _index_argv(),
        stdout=StringIO(),
        stderr=StringIO(),
        settings_factory=lambda: settings,
        runtime_factory=factory,
    )

    assert exit_code == 0
    assert runtime.index_calls == [
        (
            _WORKSPACE_ID,
            _DOCUMENT_ID,
            _VERSION_ID,
        )
    ]
    assert observability.shutdown_calls == 1


async def test_trace_update_failure_preserves_success_exit_code() -> None:
    settings = create_settings()
    observability = RecordingObservabilityClient(fail_update=True)
    runtime = FakeRuntime(
        settings=settings,
        observability_client=observability,
    )
    factory = FakeRuntimeFactory(runtime)

    exit_code = await run_cli(
        _index_argv(),
        stdout=StringIO(),
        stderr=StringIO(),
        settings_factory=lambda: settings,
        runtime_factory=factory,
    )

    assert exit_code == 0
    assert observability.trace_managers[0].exited is True
    assert observability.shutdown_calls == 1


async def test_trace_exit_failure_preserves_success_exit_code() -> None:
    settings = create_settings()
    observability = RecordingObservabilityClient(fail_exit=True)
    runtime = FakeRuntime(
        settings=settings,
        observability_client=observability,
    )
    factory = FakeRuntimeFactory(runtime)

    exit_code = await run_cli(
        _index_argv(),
        stdout=StringIO(),
        stderr=StringIO(),
        settings_factory=lambda: settings,
        runtime_factory=factory,
    )

    assert exit_code == 0
    assert observability.shutdown_calls == 1


async def test_shutdown_failure_preserves_success_after_traced_index() -> None:
    settings = create_settings()
    observability = RecordingObservabilityClient(
        shutdown_error=RuntimeError("observability shutdown failed"),
    )
    runtime = FakeRuntime(
        settings=settings,
        observability_client=observability,
    )
    factory = FakeRuntimeFactory(runtime)
    stderr = StringIO()

    exit_code = await run_cli(
        _index_argv(),
        stdout=StringIO(),
        stderr=stderr,
        settings_factory=lambda: settings,
        runtime_factory=factory,
    )

    assert exit_code == 0
    assert observability.trace_managers[0].exited is True
    assert observability.shutdown_calls == 1
    assert stderr.getvalue() == ""


async def test_original_failure_exit_code_remains_authoritative_with_trace() -> None:
    settings = create_settings()
    observability = RecordingObservabilityClient(
        fail_update=True,
        fail_exit=True,
        shutdown_error=RuntimeError("shutdown failed"),
    )
    runtime = FakeRuntime(
        settings=settings,
        observability_client=observability,
    )
    runtime.index_error = EmbeddingTimeoutError()
    factory = FakeRuntimeFactory(runtime)
    stderr = StringIO()

    exit_code = await run_cli(
        _index_argv(),
        stdout=StringIO(),
        stderr=stderr,
        settings_factory=lambda: settings,
        runtime_factory=factory,
    )

    assert exit_code == 1
    assert stderr.getvalue() == (
        "indexing_runtime_error: The embedding provider request exceeded its configured timeout.\n"
    )
    assert len(factory.calls) == 1
    assert observability.shutdown_calls == 1
