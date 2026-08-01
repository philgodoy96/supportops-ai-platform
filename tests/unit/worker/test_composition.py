"""Unit tests for worker AI and executor composition."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import cast
from unittest.mock import MagicMock

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from supportops.ai.providers.mock import (
    MOCK_TICKET_CLASSIFIER_MODEL,
    MockLLMProvider,
)
from supportops.ai.providers.openai import OpenAILLMProvider
from supportops.modules.agent_runs.application.deterministic_executor import (
    DeterministicTicketProcessingExecutor,
)
from supportops.modules.agent_runs.domain.models import (
    DETERMINISTIC_BASELINE_WORKFLOW_VERSION,
    INITIAL_TICKET_PROCESSING_WORKFLOW_NAME,
)
from supportops.modules.ticket_classifications.application.executor import (
    TICKET_CLASSIFICATION_WORKFLOW_VERSION,
    TicketClassificationExecutor,
)
from supportops.worker.composition import (
    create_session_scoped_executor_registry,
    create_worker_llm_runtime,
)


class NoOpTransactionManager:
    """Provide an unused transaction boundary for composition tests."""

    @asynccontextmanager
    async def transaction(
        self,
    ) -> AsyncIterator[None]:
        yield


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


async def test_session_factory_registers_both_workflows() -> None:
    runtime = create_worker_llm_runtime(
        provider_name="mock",
        openai_api_key=None,
        openai_model="gpt-5-nano",
        openai_base_url=None,
        request_timeout_seconds=12,
        transport_max_retries=2,
        max_repair_attempts=1,
    )
    session = cast(
        AsyncSession,
        MagicMock(spec=AsyncSession),
    )

    try:
        registry = create_session_scoped_executor_registry(
            session=session,
            transaction_manager=NoOpTransactionManager(),
            gateway=runtime.gateway,
            model=runtime.model,
            request_timeout_seconds=12,
        )

        baseline_executor = registry.resolve(
            workflow_name=(INITIAL_TICKET_PROCESSING_WORKFLOW_NAME),
            workflow_version=(DETERMINISTIC_BASELINE_WORKFLOW_VERSION),
        )
        classification_executor = registry.resolve(
            workflow_name=(INITIAL_TICKET_PROCESSING_WORKFLOW_NAME),
            workflow_version=(TICKET_CLASSIFICATION_WORKFLOW_VERSION),
        )

        assert len(registry) == 2
        assert isinstance(
            baseline_executor,
            DeterministicTicketProcessingExecutor,
        )
        assert isinstance(
            classification_executor,
            TicketClassificationExecutor,
        )
    finally:
        await runtime.close()


async def test_session_factory_rejects_invalid_request_timeout() -> None:
    runtime = create_worker_llm_runtime(
        provider_name="mock",
        openai_api_key=None,
        openai_model="gpt-5-nano",
        openai_base_url=None,
        request_timeout_seconds=12,
        transport_max_retries=2,
        max_repair_attempts=1,
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
                gateway=runtime.gateway,
                model=runtime.model,
                request_timeout_seconds=0,
            )
    finally:
        await runtime.close()
