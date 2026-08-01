"""Process-scoped LLM and session-scoped executor composition."""

from dataclasses import dataclass

from sqlalchemy.ext.asyncio import AsyncSession

from supportops.ai.gateway.contracts import LLMProvider
from supportops.ai.gateway.service import LLMGateway
from supportops.ai.providers.mock import (
    MOCK_TICKET_CLASSIFIER_MODEL,
    MockLLMProvider,
)
from supportops.ai.providers.openai import OpenAILLMProvider
from supportops.core.transactions import TransactionManager
from supportops.modules.agent_runs.application.deterministic_executor import (
    DeterministicTicketProcessingExecutor,
)
from supportops.modules.agent_runs.application.executor_registry import (
    AgentRunExecutorRegistration,
    AgentRunExecutorRegistry,
)
from supportops.modules.agent_runs.domain.models import (
    DETERMINISTIC_BASELINE_WORKFLOW_VERSION,
    INITIAL_TICKET_PROCESSING_WORKFLOW_NAME,
    TICKET_CLASSIFICATION_WORKFLOW_VERSION,
)
from supportops.modules.ticket_classifications.application.executor import (
    TicketClassificationExecutor,
)
from supportops.modules.ticket_classifications.infrastructure.repository import (
    SqlAlchemyClassificationPersistenceRepository,
)

MOCK_LLM_PROVIDER_NAME = "mock"
OPENAI_LLM_PROVIDER_NAME = "openai"


@dataclass(frozen=True, slots=True)
class WorkerLLMRuntime:
    """Own one process-scoped provider and application LLM Gateway."""

    provider: LLMProvider
    gateway: LLMGateway
    model: str

    async def close(self) -> None:
        """Release provider-owned process resources."""

        await self.provider.close()


def create_worker_llm_runtime(
    *,
    provider_name: str,
    openai_api_key: str | None,
    openai_model: str,
    openai_base_url: str | None,
    request_timeout_seconds: float,
    transport_max_retries: int,
    max_repair_attempts: int,
) -> WorkerLLMRuntime:
    """Create one explicitly configured process-scoped LLM runtime."""

    _validate_required_text(
        provider_name,
        field_name="provider_name",
    )

    provider: LLMProvider
    model: str

    if provider_name == MOCK_LLM_PROVIDER_NAME:
        provider = MockLLMProvider(
            model=MOCK_TICKET_CLASSIFIER_MODEL,
        )
        model = MOCK_TICKET_CLASSIFIER_MODEL
    elif provider_name == OPENAI_LLM_PROVIDER_NAME:
        if openai_api_key is None:
            raise ValueError(
                "openai_api_key is required when the OpenAI provider is selected.",
            )

        provider = OpenAILLMProvider.create(
            api_key=openai_api_key,
            model=openai_model,
            timeout_seconds=request_timeout_seconds,
            transport_max_retries=transport_max_retries,
            base_url=openai_base_url,
        )
        model = openai_model
    else:
        raise ValueError(
            f"Unsupported LLM provider: {provider_name}.",
        )

    gateway = LLMGateway(
        provider=provider,
        max_repair_attempts=max_repair_attempts,
    )

    return WorkerLLMRuntime(
        provider=provider,
        gateway=gateway,
        model=model,
    )


def create_session_scoped_executor_registry(
    *,
    session: AsyncSession,
    transaction_manager: TransactionManager,
    gateway: LLMGateway,
    model: str,
    request_timeout_seconds: float,
) -> AgentRunExecutorRegistry:
    """Create all workflow executors owned by one database session."""

    classification_repository = SqlAlchemyClassificationPersistenceRepository(
        session,
    )
    classification_executor = TicketClassificationExecutor(
        gateway=gateway,
        model=model,
        request_timeout_seconds=request_timeout_seconds,
        transaction_manager=transaction_manager,
        classification_repository=(classification_repository),
        execution_repository=classification_repository,
    )

    return AgentRunExecutorRegistry(
        (
            AgentRunExecutorRegistration(
                workflow_name=(INITIAL_TICKET_PROCESSING_WORKFLOW_NAME),
                workflow_version=(DETERMINISTIC_BASELINE_WORKFLOW_VERSION),
                executor=(DeterministicTicketProcessingExecutor()),
            ),
            AgentRunExecutorRegistration(
                workflow_name=(INITIAL_TICKET_PROCESSING_WORKFLOW_NAME),
                workflow_version=(TICKET_CLASSIFICATION_WORKFLOW_VERSION),
                executor=classification_executor,
            ),
        ),
    )


def _validate_required_text(
    value: str,
    *,
    field_name: str,
) -> None:
    if not value:
        raise ValueError(f"{field_name} is required.")

    if value != value.strip():
        raise ValueError(
            f"{field_name} must not contain surrounding whitespace.",
        )
