"""Unit tests for classification evaluation LLM composition."""

import pytest

from supportops.ai.providers.mock import (
    MOCK_TICKET_CLASSIFIER_MODEL,
    MockLLMProvider,
)
from supportops.ai.providers.openai import (
    OpenAILLMProvider,
)
from supportops.core.settings import LLMProviderName
from supportops.evaluation.ticket_classification.composition import (
    create_ticket_classification_evaluation_runtime,
)
from supportops.evaluation.ticket_classification.settings import (
    TicketClassificationEvaluationSettings,
)


def _settings(
    *,
    openai_api_key: str | None = None,
) -> TicketClassificationEvaluationSettings:
    return TicketClassificationEvaluationSettings(
        _env_file=None,
        openai_api_key=openai_api_key,
    )


async def test_composes_network_free_mock_runtime() -> None:
    runtime = create_ticket_classification_evaluation_runtime(
        provider_name=LLMProviderName.MOCK,
        settings=_settings(),
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


async def test_composes_openai_runtime_without_request() -> None:
    runtime = create_ticket_classification_evaluation_runtime(
        provider_name=LLMProviderName.OPENAI,
        settings=_settings(
            openai_api_key="test-evaluation-key",
        ),
    )

    try:
        assert isinstance(
            runtime.provider,
            OpenAILLMProvider,
        )
        assert runtime.provider.provider_name == "openai"
        assert runtime.model == "gpt-5-nano"
    finally:
        await runtime.close()


def test_openai_runtime_requires_api_key() -> None:
    with pytest.raises(
        ValueError,
        match=("openai_api_key is required when the OpenAI provider is selected"),
    ):
        create_ticket_classification_evaluation_runtime(
            provider_name=LLMProviderName.OPENAI,
            settings=_settings(),
        )


async def test_runtime_close_is_idempotent() -> None:
    runtime = create_ticket_classification_evaluation_runtime(
        provider_name=LLMProviderName.MOCK,
        settings=_settings(),
    )

    await runtime.close()
    await runtime.close()
