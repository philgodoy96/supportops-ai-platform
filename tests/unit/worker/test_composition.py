"""Unit tests for worker AI and executor composition."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any, cast
from unittest.mock import MagicMock, patch

import pytest
from langgraph.checkpoint.memory import MemorySaver
from pydantic import SecretStr
from sqlalchemy.ext.asyncio import AsyncSession

from supportops.agent_graph.application.human_approved_workflow import (
    HumanApprovedSupportWorkflowExecutor,
)
from supportops.agent_graph.application.resume_planning import (
    HumanApprovedGraphResumePlanner,
)
from supportops.agent_graph.application.workflow import (
    ControlledSupportWorkflowExecutor,
)
from supportops.agent_graph.domain.human_approved_state import (
    HUMAN_APPROVED_SUPPORT_WORKFLOW_VERSION,
)
from supportops.agent_graph.domain.state import (
    CONTROLLED_SUPPORT_WORKFLOW_VERSION,
)
from supportops.agent_tools.infrastructure.query_repository import (
    SqlAlchemyAgentToolCallQueryRepository,
)
from supportops.ai.embeddings.observability import ObservingEmbeddingProvider
from supportops.ai.gateway.tool_decisions import LLMToolDecisionGateway
from supportops.ai.providers.mock import (
    MOCK_TICKET_CLASSIFIER_MODEL,
    MockLLMProvider,
)
from supportops.ai.providers.openai import OpenAILLMProvider
from supportops.core.settings import Settings
from supportops.modules.agent_runs.application.deterministic_executor import (
    DeterministicTicketProcessingExecutor,
)
from supportops.modules.agent_runs.application.execution import (
    TerminalAgentRunExecutionError,
)
from supportops.modules.agent_runs.domain.models import (
    DETERMINISTIC_BASELINE_WORKFLOW_VERSION,
    INITIAL_TICKET_PROCESSING_WORKFLOW_NAME,
    TICKET_CLASSIFICATION_WORKFLOW_VERSION,
)
from supportops.modules.approvals.infrastructure.repository import (
    SqlAlchemyApprovalRequestRepository,
)
from supportops.modules.knowledge_documents.domain.models import (
    KnowledgeIndexProfile,
)
from supportops.modules.ticket_classifications.application.executor import (
    TicketClassificationExecutor,
)
from supportops.worker.composition import (
    WorkerControlledSupportRuntime,
    create_session_scoped_executor_registry,
    create_worker_controlled_support_runtime,
    create_worker_llm_runtime,
)


class NoOpTransactionManager:
    """Provide an unused transaction boundary for composition tests."""

    @asynccontextmanager
    async def transaction(
        self,
    ) -> AsyncIterator[None]:
        yield


def _create_settings(**overrides: object) -> Settings:
    values: dict[str, object] = {
        "postgresql_url": (
            "postgresql+asyncpg://supportops:supportops-local@localhost:5432/supportops"
        ),
        "qdrant_url": "http://localhost:6333",
    }
    values.update(overrides)
    settings_type = cast(Any, Settings)
    return cast(Settings, settings_type(_env_file=None, **values))


class FakeCheckpointRuntime:
    """Process-scoped checkpoint stand-in for composition tests."""

    def __init__(self) -> None:
        self.checkpointer = MemorySaver()
        self.setup_calls = 0
        self.close_calls = 0
        self.close_error: Exception | None = None

    async def setup(self) -> None:
        self.setup_calls += 1

    async def close(self) -> None:
        self.close_calls += 1
        if self.close_error is not None:
            raise self.close_error


class FakeEmbeddingProvider:
    """Process-scoped embedding stand-in for composition tests."""

    def __init__(self) -> None:
        self.provider_name = "mock"
        self.close_calls = 0
        self.close_error: Exception | None = None

    async def close(self) -> None:
        self.close_calls += 1
        if self.close_error is not None:
            raise self.close_error


class FakeQdrantClient:
    """Process-scoped Qdrant stand-in for composition tests."""

    def __init__(self) -> None:
        self.close_calls = 0
        self.close_error: Exception | None = None

    async def close(self) -> None:
        self.close_calls += 1
        if self.close_error is not None:
            raise self.close_error


class FakeObservabilityClient:
    """Process-scoped observability stand-in for composition tests."""

    def __init__(self) -> None:
        self.enabled = False
        self.shutdown_calls = 0
        self.shutdown_error: Exception | None = None

    def shutdown(self) -> None:
        self.shutdown_calls += 1
        if self.shutdown_error is not None:
            raise self.shutdown_error


def _mock_index_profile() -> KnowledgeIndexProfile:
    """Return one retrieval profile compatible with FakeEmbeddingProvider."""

    return KnowledgeIndexProfile(
        chunking_strategy="markdown-token",
        chunking_version="v1",
        tokenizer_encoding="cl100k_base",
        embedding_provider="mock",
        embedding_model="mock-hashing-embedding-v1",
        embedding_dimensions=64,
        knowledge_collection=("supportops-knowledge-mock-v1"),
        knowledge_vector_name="dense",
    )


def _embedding_provider_factory(
    provider: FakeEmbeddingProvider,
) -> object:
    """Return a factory matching the observability-aware signature."""

    def factory(
        configured_settings: Settings,
        *,
        observability_client: object | None = None,
    ) -> FakeEmbeddingProvider:
        del configured_settings, observability_client
        return provider

    return factory


def _capturing_embedding_provider_factory(
    provider: FakeEmbeddingProvider,
    captured_clients: list[object],
) -> object:
    """Return a factory that records the supplied observability client."""

    def factory(
        configured_settings: Settings,
        *,
        observability_client: object | None = None,
    ) -> FakeEmbeddingProvider:
        del configured_settings
        if observability_client is not None:
            captured_clients.append(observability_client)
        return provider

    return factory


async def test_creates_process_scoped_mock_runtime() -> None:
    runtime = create_worker_llm_runtime(
        provider_name="mock",
        openai_api_key=None,
        openai_model="gpt-5-nano",
        openai_base_url=None,
        request_timeout_seconds=12,
        transport_max_retries=2,
        max_repair_attempts=1,
    )

    try:
        assert isinstance(
            runtime.provider,
            MockLLMProvider,
        )
        assert runtime.provider.provider_name == "mock"
        assert runtime.model == (MOCK_TICKET_CLASSIFIER_MODEL)
    finally:
        await runtime.close()


async def test_worker_runtimes_share_one_observability_client() -> None:
    settings = _create_settings()
    checkpoint_runtime = FakeCheckpointRuntime()
    embedding_provider = FakeEmbeddingProvider()
    qdrant_client = FakeQdrantClient()
    observability_client = FakeObservabilityClient()
    embedding_clients: list[object] = []

    async def checkpoint_runtime_factory(
        *,
        database_url: SecretStr,
    ) -> FakeCheckpointRuntime:
        del database_url
        return checkpoint_runtime

    llm_runtime = create_worker_llm_runtime(
        provider_name="mock",
        openai_api_key=None,
        openai_model="gpt-5-nano",
        openai_base_url=None,
        request_timeout_seconds=12,
        transport_max_retries=2,
        max_repair_attempts=1,
        observability_client=cast(Any, observability_client),
    )
    controlled_runtime = await create_worker_controlled_support_runtime(
        settings=settings,
        checkpoint_runtime_factory=cast(
            Any,
            checkpoint_runtime_factory,
        ),
        embedding_provider_factory=cast(
            Any,
            _capturing_embedding_provider_factory(
                embedding_provider,
                embedding_clients,
            ),
        ),
        qdrant_client_factory=cast(
            Any,
            lambda configured_settings: qdrant_client,
        ),
        index_profile_factory=cast(
            Any,
            lambda configured_settings: object(),
        ),
        observability_client=cast(Any, observability_client),
    )

    try:
        assert llm_runtime.gateway._observability_client is cast(
            Any,
            observability_client,
        )
        assert controlled_runtime.observability_client is cast(
            Any,
            observability_client,
        )
        assert embedding_clients == [observability_client]
    finally:
        await controlled_runtime.close()
        await llm_runtime.close()

    assert observability_client.shutdown_calls == 1


async def test_session_factory_shares_observability_with_tool_decision_gateway() -> None:
    settings = _create_settings()
    checkpoint_runtime = FakeCheckpointRuntime()
    embedding_provider = FakeEmbeddingProvider()
    qdrant_client = FakeQdrantClient()
    observability_client = FakeObservabilityClient()
    created_clients: list[object] = []
    embedding_clients: list[object] = []

    class CapturingGateway(LLMToolDecisionGateway):
        def __init__(self, **kwargs: Any) -> None:
            created_clients.append(kwargs.get("observability_client"))
            super().__init__(**kwargs)

    async def checkpoint_runtime_factory(
        *,
        database_url: SecretStr,
    ) -> FakeCheckpointRuntime:
        del database_url
        return checkpoint_runtime

    llm_runtime = create_worker_llm_runtime(
        provider_name="mock",
        openai_api_key=None,
        openai_model="gpt-5-nano",
        openai_base_url=None,
        request_timeout_seconds=12,
        transport_max_retries=2,
        max_repair_attempts=1,
        observability_client=cast(Any, observability_client),
    )
    controlled_runtime = await create_worker_controlled_support_runtime(
        settings=settings,
        checkpoint_runtime_factory=cast(
            Any,
            checkpoint_runtime_factory,
        ),
        embedding_provider_factory=cast(
            Any,
            _capturing_embedding_provider_factory(
                embedding_provider,
                embedding_clients,
            ),
        ),
        qdrant_client_factory=cast(
            Any,
            lambda configured_settings: qdrant_client,
        ),
        index_profile_factory=cast(
            Any,
            lambda configured_settings: _mock_index_profile(),
        ),
        observability_client=cast(Any, observability_client),
    )
    session = cast(
        AsyncSession,
        MagicMock(spec=AsyncSession),
    )

    try:
        with patch(
            "supportops.worker.composition.LLMToolDecisionGateway",
            CapturingGateway,
        ):
            create_session_scoped_executor_registry(
                session=session,
                transaction_manager=NoOpTransactionManager(),
                gateway=llm_runtime.gateway,
                provider=llm_runtime.provider,
                model=llm_runtime.model,
                request_timeout_seconds=12,
                controlled_runtime=controlled_runtime,
                embedding_timeout_seconds=12,
            )

        assert len(created_clients) == 1
        assert created_clients[0] is cast(Any, observability_client)
        assert embedding_clients == [observability_client]
        assert llm_runtime.gateway._observability_client is cast(
            Any,
            observability_client,
        )
        assert controlled_runtime.observability_client is cast(
            Any,
            observability_client,
        )
    finally:
        await controlled_runtime.close()
        await llm_runtime.close()

    assert observability_client.shutdown_calls == 1


async def test_controlled_runtime_passes_observability_to_real_embedding_factory() -> None:
    settings = _create_settings()
    checkpoint_runtime = FakeCheckpointRuntime()
    qdrant_client = FakeQdrantClient()
    observability_client = FakeObservabilityClient()
    factory_calls = 0

    async def checkpoint_runtime_factory(
        *,
        database_url: SecretStr,
    ) -> FakeCheckpointRuntime:
        del database_url
        return checkpoint_runtime

    def embedding_provider_factory(
        configured_settings: Settings,
        *,
        observability_client: object | None = None,
    ) -> ObservingEmbeddingProvider:
        nonlocal factory_calls
        factory_calls += 1
        del configured_settings
        return ObservingEmbeddingProvider(
            provider=cast(Any, FakeEmbeddingProvider()),
            observability_client=cast(Any, observability_client),
        )

    runtime = await create_worker_controlled_support_runtime(
        settings=settings,
        checkpoint_runtime_factory=cast(
            Any,
            checkpoint_runtime_factory,
        ),
        embedding_provider_factory=cast(
            Any,
            embedding_provider_factory,
        ),
        qdrant_client_factory=cast(
            Any,
            lambda configured_settings: qdrant_client,
        ),
        index_profile_factory=cast(
            Any,
            lambda configured_settings: _mock_index_profile(),
        ),
        observability_client=cast(Any, observability_client),
    )

    try:
        assert factory_calls == 1
        assert isinstance(
            runtime.embedding_provider,
            ObservingEmbeddingProvider,
        )
        assert runtime.embedding_provider._observability_client is cast(
            Any,
            observability_client,
        )
        assert runtime.observability_client is cast(
            Any,
            observability_client,
        )
    finally:
        await runtime.close()

    assert observability_client.shutdown_calls == 1
    assert factory_calls == 1


async def test_creates_process_scoped_openai_runtime_without_network() -> None:
    runtime = create_worker_llm_runtime(
        provider_name="openai",
        openai_api_key="test-api-key",
        openai_model="gpt-5-nano",
        openai_base_url=None,
        request_timeout_seconds=12,
        transport_max_retries=2,
        max_repair_attempts=1,
    )

    try:
        assert isinstance(
            runtime.provider,
            OpenAILLMProvider,
        )
        assert runtime.provider.provider_name == "openai"
        assert runtime.provider.model == "gpt-5-nano"
        assert runtime.model == "gpt-5-nano"
    finally:
        await runtime.close()


async def test_openai_runtime_accepts_explicit_base_url() -> None:
    runtime = create_worker_llm_runtime(
        provider_name="openai",
        openai_api_key="test-api-key",
        openai_model="gpt-5-nano",
        openai_base_url="https://example.invalid/v1",
        request_timeout_seconds=12,
        transport_max_retries=0,
        max_repair_attempts=0,
    )

    try:
        assert isinstance(
            runtime.provider,
            OpenAILLMProvider,
        )
    finally:
        await runtime.close()


def test_openai_runtime_requires_api_key() -> None:
    with pytest.raises(
        ValueError,
        match=("openai_api_key is required when the OpenAI provider is selected"),
    ):
        create_worker_llm_runtime(
            provider_name="openai",
            openai_api_key=None,
            openai_model="gpt-5-nano",
            openai_base_url=None,
            request_timeout_seconds=12,
            transport_max_retries=2,
            max_repair_attempts=1,
        )


@pytest.mark.parametrize(
    "provider_name",
    [
        "",
        " mock",
        "unsupported",
    ],
)
def test_rejects_invalid_provider_name(
    provider_name: str,
) -> None:
    with pytest.raises(ValueError):
        create_worker_llm_runtime(
            provider_name=provider_name,
            openai_api_key=None,
            openai_model="gpt-5-nano",
            openai_base_url=None,
            request_timeout_seconds=12,
            transport_max_retries=2,
            max_repair_attempts=1,
        )


async def test_creates_process_scoped_controlled_support_runtime() -> None:
    settings = _create_settings()
    checkpoint_runtime = FakeCheckpointRuntime()
    embedding_provider = FakeEmbeddingProvider()
    qdrant_client = FakeQdrantClient()
    observability_client = FakeObservabilityClient()
    captured_database_urls: list[SecretStr] = []
    index_profile = object()

    async def checkpoint_runtime_factory(
        *,
        database_url: SecretStr,
    ) -> FakeCheckpointRuntime:
        captured_database_urls.append(database_url)
        return checkpoint_runtime

    runtime = await create_worker_controlled_support_runtime(
        settings=settings,
        checkpoint_runtime_factory=cast(
            Any,
            checkpoint_runtime_factory,
        ),
        embedding_provider_factory=cast(
            Any,
            _embedding_provider_factory(embedding_provider),
        ),
        qdrant_client_factory=cast(
            Any,
            lambda configured_settings: qdrant_client,
        ),
        index_profile_factory=cast(
            Any,
            lambda configured_settings: index_profile,
        ),
        observability_client_factory=cast(
            Any,
            lambda configured_settings: observability_client,
        ),
    )

    try:
        assert isinstance(runtime, WorkerControlledSupportRuntime)
        assert cast(Any, runtime.checkpoint_runtime) is checkpoint_runtime
        assert cast(Any, runtime.checkpoint_runtime).checkpointer is checkpoint_runtime.checkpointer
        assert cast(Any, runtime.embedding_provider) is embedding_provider
        assert cast(Any, runtime.qdrant_client) is qdrant_client
        assert cast(Any, runtime.index_profile) is index_profile
        assert runtime.observability_client is cast(Any, observability_client)
        assert runtime.vector_store is not None
        assert runtime.vector_searcher is not None
        assert checkpoint_runtime.setup_calls == 1
        assert len(captured_database_urls) == 1
        resolved_dsn = captured_database_urls[0].get_secret_value()
        assert resolved_dsn.startswith("postgresql://")
        assert "+asyncpg" not in resolved_dsn
        assert "supportops-local" in resolved_dsn
    finally:
        await runtime.close()

    assert checkpoint_runtime.close_calls == 1
    assert embedding_provider.close_calls == 1
    assert qdrant_client.close_calls == 1
    assert observability_client.shutdown_calls == 1


async def test_controlled_runtime_partial_construction_cleans_up() -> None:
    settings = _create_settings()
    checkpoint_runtime = FakeCheckpointRuntime()
    embedding_provider = FakeEmbeddingProvider()
    observability_client = FakeObservabilityClient()

    async def checkpoint_runtime_factory(
        *,
        database_url: SecretStr,
    ) -> FakeCheckpointRuntime:
        del database_url
        return checkpoint_runtime

    def failing_qdrant_factory(configured_settings: Settings) -> FakeQdrantClient:
        del configured_settings
        raise RuntimeError("qdrant unavailable")

    with pytest.raises(RuntimeError, match="qdrant unavailable"):
        await create_worker_controlled_support_runtime(
            settings=settings,
            checkpoint_runtime_factory=cast(
                Any,
                checkpoint_runtime_factory,
            ),
            embedding_provider_factory=cast(
                Any,
                _embedding_provider_factory(embedding_provider),
            ),
            qdrant_client_factory=cast(
                Any,
                failing_qdrant_factory,
            ),
            index_profile_factory=cast(
                Any,
                lambda configured_settings: object(),
            ),
            observability_client_factory=cast(
                Any,
                lambda configured_settings: observability_client,
            ),
        )

    assert checkpoint_runtime.setup_calls == 1
    assert checkpoint_runtime.close_calls == 1
    assert embedding_provider.close_calls == 1
    assert observability_client.shutdown_calls == 1


async def test_controlled_runtime_close_is_idempotent() -> None:
    settings = _create_settings()
    checkpoint_runtime = FakeCheckpointRuntime()
    embedding_provider = FakeEmbeddingProvider()
    qdrant_client = FakeQdrantClient()
    observability_client = FakeObservabilityClient()

    async def checkpoint_runtime_factory(
        *,
        database_url: SecretStr,
    ) -> FakeCheckpointRuntime:
        del database_url
        return checkpoint_runtime

    runtime = await create_worker_controlled_support_runtime(
        settings=settings,
        checkpoint_runtime_factory=cast(
            Any,
            checkpoint_runtime_factory,
        ),
        embedding_provider_factory=cast(
            Any,
            _embedding_provider_factory(embedding_provider),
        ),
        qdrant_client_factory=cast(
            Any,
            lambda configured_settings: qdrant_client,
        ),
        index_profile_factory=cast(
            Any,
            lambda configured_settings: object(),
        ),
        observability_client_factory=cast(
            Any,
            lambda configured_settings: observability_client,
        ),
    )

    await runtime.close()
    await runtime.close()

    assert checkpoint_runtime.close_calls == 1
    assert embedding_provider.close_calls == 1
    assert observability_client.shutdown_calls == 1


async def test_controlled_runtime_close_attempts_all_resources_when_one_fails() -> None:
    settings = _create_settings()
    checkpoint_runtime = FakeCheckpointRuntime()
    checkpoint_runtime.close_error = RuntimeError("checkpoint close failed")
    embedding_provider = FakeEmbeddingProvider()
    embedding_provider.close_error = RuntimeError("embedding close failed")
    qdrant_client = FakeQdrantClient()
    observability_client = FakeObservabilityClient()

    async def checkpoint_runtime_factory(
        *,
        database_url: SecretStr,
    ) -> FakeCheckpointRuntime:
        del database_url
        return checkpoint_runtime

    runtime = await create_worker_controlled_support_runtime(
        settings=settings,
        checkpoint_runtime_factory=cast(
            Any,
            checkpoint_runtime_factory,
        ),
        embedding_provider_factory=cast(
            Any,
            _embedding_provider_factory(embedding_provider),
        ),
        qdrant_client_factory=cast(
            Any,
            lambda configured_settings: qdrant_client,
        ),
        index_profile_factory=cast(
            Any,
            lambda configured_settings: object(),
        ),
        observability_client_factory=cast(
            Any,
            lambda configured_settings: observability_client,
        ),
    )

    with pytest.raises(RuntimeError, match="checkpoint close failed") as captured:
        await runtime.close()

    assert checkpoint_runtime.close_calls == 1
    assert embedding_provider.close_calls == 1
    assert qdrant_client.close_calls == 1
    assert observability_client.shutdown_calls == 1
    assert any(
        "additional controlled-support runtime resource failed to close" in note
        for note in getattr(captured.value, "__notes__", ())
    )


async def test_controlled_runtime_observability_shutdown_failure_is_isolated() -> None:
    settings = _create_settings()
    checkpoint_runtime = FakeCheckpointRuntime()
    embedding_provider = FakeEmbeddingProvider()
    qdrant_client = FakeQdrantClient()
    observability_client = FakeObservabilityClient()
    observability_client.shutdown_error = RuntimeError("shutdown failed")

    async def checkpoint_runtime_factory(
        *,
        database_url: SecretStr,
    ) -> FakeCheckpointRuntime:
        del database_url
        return checkpoint_runtime

    runtime = await create_worker_controlled_support_runtime(
        settings=settings,
        checkpoint_runtime_factory=cast(
            Any,
            checkpoint_runtime_factory,
        ),
        embedding_provider_factory=cast(
            Any,
            _embedding_provider_factory(embedding_provider),
        ),
        qdrant_client_factory=cast(
            Any,
            lambda configured_settings: qdrant_client,
        ),
        index_profile_factory=cast(
            Any,
            lambda configured_settings: object(),
        ),
        observability_client_factory=cast(
            Any,
            lambda configured_settings: observability_client,
        ),
    )

    await runtime.close()

    assert checkpoint_runtime.close_calls == 1
    assert embedding_provider.close_calls == 1
    assert qdrant_client.close_calls == 1
    assert observability_client.shutdown_calls == 1


async def _build_registry_with_stubs() -> tuple[Any, Any, AsyncSession]:
    llm_runtime = create_worker_llm_runtime(
        provider_name="mock",
        openai_api_key=None,
        openai_model="gpt-5-nano",
        openai_base_url=None,
        request_timeout_seconds=12,
        transport_max_retries=2,
        max_repair_attempts=1,
    )
    checkpoint_runtime = FakeCheckpointRuntime()
    embedding_provider = FakeEmbeddingProvider()
    qdrant_client = FakeQdrantClient()

    async def checkpoint_runtime_factory(
        *,
        database_url: SecretStr,
    ) -> FakeCheckpointRuntime:
        del database_url
        return checkpoint_runtime

    controlled_runtime = await create_worker_controlled_support_runtime(
        settings=_create_settings(),
        checkpoint_runtime_factory=cast(
            Any,
            checkpoint_runtime_factory,
        ),
        embedding_provider_factory=cast(
            Any,
            _embedding_provider_factory(embedding_provider),
        ),
        qdrant_client_factory=cast(
            Any,
            lambda configured_settings: qdrant_client,
        ),
        index_profile_factory=cast(
            Any,
            lambda configured_settings: _mock_index_profile(),
        ),
    )
    session = cast(
        AsyncSession,
        MagicMock(spec=AsyncSession),
    )
    registry = create_session_scoped_executor_registry(
        session=session,
        transaction_manager=NoOpTransactionManager(),
        gateway=llm_runtime.gateway,
        provider=llm_runtime.provider,
        model=llm_runtime.model,
        request_timeout_seconds=12,
        controlled_runtime=controlled_runtime,
        embedding_timeout_seconds=12,
    )

    return registry, (llm_runtime, controlled_runtime), session


async def test_session_factory_registers_four_workflows() -> None:
    registry, runtimes, session = await _build_registry_with_stubs()
    llm_runtime, controlled_runtime = runtimes

    try:
        baseline_executor = registry.resolve(
            workflow_name=(INITIAL_TICKET_PROCESSING_WORKFLOW_NAME),
            workflow_version=(DETERMINISTIC_BASELINE_WORKFLOW_VERSION),
        )
        classification_executor = registry.resolve(
            workflow_name=(INITIAL_TICKET_PROCESSING_WORKFLOW_NAME),
            workflow_version=(TICKET_CLASSIFICATION_WORKFLOW_VERSION),
        )
        controlled_executor = registry.resolve(
            workflow_name=(INITIAL_TICKET_PROCESSING_WORKFLOW_NAME),
            workflow_version=(CONTROLLED_SUPPORT_WORKFLOW_VERSION),
        )
        human_approved_executor = registry.resolve(
            workflow_name=(INITIAL_TICKET_PROCESSING_WORKFLOW_NAME),
            workflow_version=(HUMAN_APPROVED_SUPPORT_WORKFLOW_VERSION),
        )

        assert len(registry) == 4
        assert isinstance(
            baseline_executor,
            DeterministicTicketProcessingExecutor,
        )
        assert isinstance(
            classification_executor,
            TicketClassificationExecutor,
        )
        assert isinstance(
            controlled_executor,
            ControlledSupportWorkflowExecutor,
        )
        assert isinstance(
            human_approved_executor,
            HumanApprovedSupportWorkflowExecutor,
        )
        assert CONTROLLED_SUPPORT_WORKFLOW_VERSION == ("controlled-support-v1")
        assert not hasattr(controlled_executor, "_resume_planner")
        planner = human_approved_executor._resume_planner
        assert isinstance(planner, HumanApprovedGraphResumePlanner)
        approval_repo = planner._approval_request_repository
        tool_repo = planner._tool_call_query_repository
        assert type(approval_repo) is SqlAlchemyApprovalRequestRepository
        assert type(tool_repo) is SqlAlchemyAgentToolCallQueryRepository
        assert approval_repo._session is session
        assert tool_repo._session is session
        graph_nodes = cast(Any, human_approved_executor._graph).nodes
        assert "handle_approval_decision" in graph_nodes
        assert "execute_sensitive_tool" in graph_nodes
        assert "draft_grounded_recommendation" in graph_nodes
        assert "validate_recommendation" in graph_nodes
        assert "persist_recommendation" in graph_nodes
        assert "prepare_sensitive_action" in graph_nodes
        assert "await_human_approval" in graph_nodes
        assert "execute_read_only_tool" in graph_nodes
    finally:
        await controlled_runtime.close()
        await llm_runtime.close()


async def test_session_factory_rejects_unsupported_workflow_version() -> None:
    registry, runtimes, _session = await _build_registry_with_stubs()
    llm_runtime, controlled_runtime = runtimes

    try:
        with pytest.raises(TerminalAgentRunExecutionError) as captured:
            registry.resolve(
                workflow_name=(INITIAL_TICKET_PROCESSING_WORKFLOW_NAME),
                workflow_version="unsupported-version",
            )

        assert captured.value.error_code == "unsupported_workflow_version"
    finally:
        await controlled_runtime.close()
        await llm_runtime.close()


async def test_session_factory_rejects_invalid_request_timeout() -> None:
    llm_runtime = create_worker_llm_runtime(
        provider_name="mock",
        openai_api_key=None,
        openai_model="gpt-5-nano",
        openai_base_url=None,
        request_timeout_seconds=12,
        transport_max_retries=2,
        max_repair_attempts=1,
    )
    checkpoint_runtime = FakeCheckpointRuntime()
    embedding_provider = FakeEmbeddingProvider()
    qdrant_client = FakeQdrantClient()

    async def checkpoint_runtime_factory(
        *,
        database_url: SecretStr,
    ) -> FakeCheckpointRuntime:
        del database_url
        return checkpoint_runtime

    controlled_runtime = await create_worker_controlled_support_runtime(
        settings=_create_settings(),
        checkpoint_runtime_factory=cast(
            Any,
            checkpoint_runtime_factory,
        ),
        embedding_provider_factory=cast(
            Any,
            _embedding_provider_factory(embedding_provider),
        ),
        qdrant_client_factory=cast(
            Any,
            lambda configured_settings: qdrant_client,
        ),
        index_profile_factory=cast(
            Any,
            lambda configured_settings: object(),
        ),
    )
    session = cast(
        AsyncSession,
        MagicMock(spec=AsyncSession),
    )

    try:
        with pytest.raises(
            ValueError,
            match="request_timeout_seconds must be positive",
        ):
            create_session_scoped_executor_registry(
                session=session,
                transaction_manager=(NoOpTransactionManager()),
                gateway=llm_runtime.gateway,
                provider=llm_runtime.provider,
                model=llm_runtime.model,
                request_timeout_seconds=0,
                controlled_runtime=controlled_runtime,
                embedding_timeout_seconds=12,
            )
    finally:
        await controlled_runtime.close()
        await llm_runtime.close()
