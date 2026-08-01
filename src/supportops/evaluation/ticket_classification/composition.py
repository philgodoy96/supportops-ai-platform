"""Process-scoped LLM composition for classification evaluation."""

from dataclasses import dataclass

from supportops.ai.gateway.contracts import LLMProvider
from supportops.ai.gateway.service import LLMGateway
from supportops.ai.providers.mock import (
    MOCK_LLM_PROVIDER_NAME,
    MOCK_TICKET_CLASSIFIER_MODEL,
    MockLLMProvider,
)
from supportops.ai.providers.openai import (
    OPENAI_LLM_PROVIDER_NAME,
    OpenAILLMProvider,
)
from supportops.core.settings import LLMProviderName
from supportops.evaluation.ticket_classification.settings import (
    TicketClassificationEvaluationSettings,
)


@dataclass(frozen=True, slots=True)
class TicketClassificationEvaluationLLMRuntime:
    """Own one provider and Gateway for an evaluation process."""

    provider: LLMProvider
    gateway: LLMGateway
    model: str

    async def close(self) -> None:
        """Release provider-owned process resources."""

        await self.provider.close()


def create_ticket_classification_evaluation_runtime(
    *,
    provider_name: LLMProviderName,
    settings: TicketClassificationEvaluationSettings,
) -> TicketClassificationEvaluationLLMRuntime:
    """Create one explicitly selected evaluation runtime."""

    provider: LLMProvider
    model: str

    if provider_name is LLMProviderName.MOCK:
        provider = MockLLMProvider(
            model=MOCK_TICKET_CLASSIFIER_MODEL,
        )
        model = MOCK_TICKET_CLASSIFIER_MODEL
    elif provider_name is LLMProviderName.OPENAI:
        api_key = (
            settings.openai_api_key.get_secret_value()
            if settings.openai_api_key is not None
            else None
        )

        if api_key is None:
            raise ValueError(
                "openai_api_key is required when the OpenAI provider is selected.",
            )

        provider = OpenAILLMProvider.create(
            api_key=api_key,
            model=settings.openai_model,
            timeout_seconds=(settings.llm_request_timeout_seconds),
            transport_max_retries=(settings.llm_transport_max_retries),
            base_url=settings.openai_base_url,
        )
        model = settings.openai_model
    else:
        raise ValueError(
            f"Unsupported LLM provider: {provider_name}.",
        )

    if provider.provider_name not in {
        MOCK_LLM_PROVIDER_NAME,
        OPENAI_LLM_PROVIDER_NAME,
    }:
        raise RuntimeError(
            "Evaluation runtime produced an unsupported provider identity.",
        )

    gateway = LLMGateway(
        provider=provider,
        max_repair_attempts=(settings.llm_max_repair_attempts),
    )

    return TicketClassificationEvaluationLLMRuntime(
        provider=provider,
        gateway=gateway,
        model=model,
    )
