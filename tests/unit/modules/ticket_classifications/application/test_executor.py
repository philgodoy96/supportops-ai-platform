"""Unit tests for durable ticket-classification execution."""

from collections import deque
from collections.abc import AsyncIterator, Iterable
from contextlib import asynccontextmanager
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from uuid import UUID

import pytest
from pydantic import BaseModel

from supportops.agent_graph.domain.state import (
    CONTROLLED_SUPPORT_WORKFLOW_VERSION,
)
from supportops.ai.gateway.contracts import (
    LLMRequest,
    LLMTokenUsage,
)
from supportops.ai.gateway.errors import (
    LLMErrorCode,
    LLMRefusalError,
    LLMTimeoutError,
)
from supportops.ai.gateway.results import (
    LLMGatewayFailure,
    LLMGatewayResult,
    LLMInvocationStatus,
    LLMInvocationTrace,
)
from supportops.ai.prompts.ticket_classification_v1 import (
    TICKET_CLASSIFICATION_PROMPT_V1,
)
from supportops.ai.schemas.ticket_classification import (
    TICKET_CLASSIFICATION_SCHEMA_VERSION,
    TicketCategory,
    TicketClassificationResult,
    TicketIntent,
    TicketSentiment,
    TicketUrgency,
)
from supportops.modules.agent_runs.application.execution import (
    AgentRunExecutionContext,
    CompletedExecution,
    RetryableAgentRunExecutionError,
    TerminalAgentRunExecutionError,
)
from supportops.modules.agent_runs.domain.models import (
    INITIAL_TICKET_PROCESSING_TRIGGER_KEY,
    INITIAL_TICKET_PROCESSING_WORKFLOW_NAME,
    TICKET_CLASSIFICATION_WORKFLOW_VERSION,
    AgentRun,
    AgentRunAttempt,
    AgentRunStatus,
)
from supportops.modules.ticket_classifications.application.executor import (
    TicketClassificationExecutor,
)
from supportops.modules.ticket_classifications.application.persistence import (
    ClassificationPersistenceResult,
    PersistClassificationExecutionCommand,
)
from supportops.modules.ticket_classifications.domain.models import (
    TicketClassification,
)
from supportops.modules.tickets.domain.models import Ticket

_NOW = datetime(
    2026,
    8,
    1,
    20,
    0,
    tzinfo=UTC,
)
_WORKSPACE_ID = UUID(
    "08c88bd1-7175-4d63-832e-93cd021ca89c",
)
_TICKET_ID = UUID(
    "6ef81942-8a2c-4dbd-8cd2-95696cd8bac5",
)
_AGENT_RUN_ID = UUID(
    "3a275db7-aea0-43c4-a448-028b21856b9e",
)
_ATTEMPT_ID = UUID(
    "77dac0c6-787e-43c9-8c42-35f03effd274",
)
_LEASE_TOKEN = UUID(
    "0cba3b93-d23d-45a4-ab1d-4ff18180232d",
)
_EXECUTION_REQUEST_ID = UUID(
    "aa401887-2d56-4554-9c90-25bbf1f7cae1",
)
_INVOCATION_ID_1 = UUID(
    "0344ca31-57e6-409c-8076-f99cce99673e",
)
_INVOCATION_ID_2 = UUID(
    "cba06c01-b6a9-4579-8979-a962b8748ff6",
)
_CLASSIFICATION_ID = UUID(
    "47ef430d-f01c-4201-a260-ff2a140ab251",
)
_ZERO_COST = Decimal("0.000000000000")


class FakeGateway:
    """Return one configured gateway result or failure."""

    def __init__(
        self,
        outcome: LLMGatewayResult | LLMGatewayFailure,
    ) -> None:
        self.outcome = outcome
        self.requests: list[LLMRequest] = []

    async def generate(
        self,
        request: LLMRequest,
    ) -> LLMGatewayResult:
        self.requests.append(request)

        if isinstance(
            self.outcome,
            LLMGatewayFailure,
        ):
            raise self.outcome

        return self.outcome


class RecordingClassificationRepository:
    """Record classification queries and return one configured result."""

    def __init__(
        self,
        existing: TicketClassification | None = None,
    ) -> None:
        self.existing = existing
        self.queries: list[tuple[UUID, UUID]] = []

    async def add(
        self,
        classification: TicketClassification,
    ) -> None:
        raise AssertionError(
            "The executor must use fenced persistence.",
        )

    async def get_by_agent_run_id(
        self,
        *,
        workspace_id: UUID,
        agent_run_id: UUID,
    ) -> TicketClassification | None:
        self.queries.append(
            (
                workspace_id,
                agent_run_id,
            ),
        )
        return self.existing


class RecordingExecutionRepository:
    """Record fenced commands and return one configured outcome."""

    def __init__(
        self,
        result: ClassificationPersistenceResult = (ClassificationPersistenceResult.APPLIED),
    ) -> None:
        self.result = result
        self.commands: list[PersistClassificationExecutionCommand] = []

    async def persist_fenced(
        self,
        command: PersistClassificationExecutionCommand,
    ) -> ClassificationPersistenceResult:
        self.commands.append(command)
        return self.result


class RecordingTransactionManager:
    """Record application-owned transaction boundaries."""

    def __init__(self) -> None:
        self.enter_count = 0
        self.exit_count = 0

    @asynccontextmanager
    async def transaction(
        self,
    ) -> AsyncIterator[None]:
        self.enter_count += 1

        try:
            yield
        finally:
            self.exit_count += 1


class SequenceUuidFactory:
    """Return deterministic UUIDs in configured order."""

    def __init__(
        self,
        values: Iterable[UUID],
    ) -> None:
        self._values = deque(values)

    def __call__(self) -> UUID:
        if not self._values:
            raise AssertionError(
                "No configured UUID remains.",
            )

        return self._values.popleft()


class UnexpectedOutput(BaseModel):
    """Unexpected gateway output used to verify schema ownership."""

    value: str


def _context() -> AgentRunExecutionContext:
    ticket = Ticket.create(
        ticket_id=_TICKET_ID,
        workspace_id=_WORKSPACE_ID,
        subject="Invoice charge question",
        description=("A charge appears twice on the latest invoice."),
        ingestion_request_id=UUID(
            "af5c4264-a33d-422a-9537-94965d997d5c",
        ),
        correlation_id=UUID(
            "a22d37c9-43c0-4d96-bc86-6c314be99113",
        ),
        now=_NOW - timedelta(minutes=1),
    )
    initial_run = AgentRun.create_initial(
        agent_run_id=_AGENT_RUN_ID,
        workspace_id=_WORKSPACE_ID,
        ticket_id=_TICKET_ID,
        ingestion_request_id=(ticket.ingestion_request_id),
        correlation_id=ticket.correlation_id,
        workflow_version=(TICKET_CLASSIFICATION_WORKFLOW_VERSION),
        max_retryable_failures=3,
        now=_NOW - timedelta(minutes=1),
    )
    running_run = replace(
        initial_run,
        workflow_name=(INITIAL_TICKET_PROCESSING_WORKFLOW_NAME),
        workflow_version=(TICKET_CLASSIFICATION_WORKFLOW_VERSION),
        trigger_key=(INITIAL_TICKET_PROCESSING_TRIGGER_KEY),
        status=AgentRunStatus.RUNNING,
        attempt_count=1,
        lease_owner="worker-a",
        lease_token=_LEASE_TOKEN,
        lease_expires_at=(_NOW + timedelta(seconds=45)),
        first_started_at=_NOW,
        updated_at=_NOW,
    )
    attempt = AgentRunAttempt.start(
        attempt_id=_ATTEMPT_ID,
        agent_run_id=_AGENT_RUN_ID,
        attempt_number=1,
        worker_id="worker-a",
        lease_token=_LEASE_TOKEN,
        execution_request_id=_EXECUTION_REQUEST_ID,
        now=_NOW,
    )

    return AgentRunExecutionContext(
        agent_run=running_run,
        attempt=attempt,
        ticket=ticket,
    )


def _classification_output() -> TicketClassificationResult:
    return TicketClassificationResult(
        category=TicketCategory.BILLING,
        intent=TicketIntent.ASK_QUESTION,
        urgency=TicketUrgency.NORMAL,
        sentiment=TicketSentiment.NEUTRAL,
        requires_human_review=False,
        summary=("The customer is asking about a duplicated invoice charge."),
        schema_version=(TICKET_CLASSIFICATION_SCHEMA_VERSION),
    )


def _successful_trace(
    *,
    sequence: int = 1,
    provider: str = "mock",
    model: str = "mock-ticket-classifier-v1",
) -> LLMInvocationTrace:
    return LLMInvocationTrace(
        invocation_sequence=sequence,
        status=LLMInvocationStatus.SUCCEEDED,
        provider=provider,
        model=model,
        provider_request_id=f"request-{sequence}",
        usage=LLMTokenUsage(
            input_tokens=120,
            cached_input_tokens=0,
            output_tokens=24,
            reasoning_tokens=None,
            total_tokens=144,
        ),
        latency_ms=25,
        error_code=None,
    )


def _successful_result(
    *,
    provider: str = "mock",
    model: str = "mock-ticket-classifier-v1",
) -> LLMGatewayResult:
    return LLMGatewayResult(
        output=_classification_output(),
        invocations=(
            _successful_trace(
                provider=provider,
                model=model,
            ),
        ),
        accepted_invocation_sequence=1,
    )


def _timeout_failure() -> LLMGatewayFailure:
    error = LLMTimeoutError(
        provider_request_id="request-1",
    )

    return LLMGatewayFailure(
        error=error,
        invocations=(
            LLMInvocationTrace(
                invocation_sequence=1,
                status=LLMInvocationStatus.TIMED_OUT,
                provider="openai",
                model="gpt-5-nano",
                provider_request_id="request-1",
                usage=None,
                latency_ms=12_000,
                error_code=error.error_code,
            ),
        ),
    )


def _refusal_failure() -> LLMGatewayFailure:
    error = LLMRefusalError(
        provider_request_id="request-1",
    )

    return LLMGatewayFailure(
        error=error,
        invocations=(
            LLMInvocationTrace(
                invocation_sequence=1,
                status=LLMInvocationStatus.REFUSED,
                provider="openai",
                model="gpt-5-nano",
                provider_request_id="request-1",
                usage=None,
                latency_ms=100,
                error_code=error.error_code,
            ),
        ),
    )


def _executor(
    *,
    gateway: FakeGateway,
    query_repository: (RecordingClassificationRepository | None) = None,
    execution_repository: (RecordingExecutionRepository | None) = None,
    transaction_manager: (RecordingTransactionManager | None) = None,
    model: str = "mock-ticket-classifier-v1",
    uuid_values: Iterable[UUID] = (
        _INVOCATION_ID_1,
        _CLASSIFICATION_ID,
    ),
) -> tuple[
    TicketClassificationExecutor,
    RecordingClassificationRepository,
    RecordingExecutionRepository,
    RecordingTransactionManager,
]:
    query_repository = query_repository or RecordingClassificationRepository()
    execution_repository = execution_repository or RecordingExecutionRepository()
    transaction_manager = transaction_manager or RecordingTransactionManager()

    executor = TicketClassificationExecutor(
        gateway=gateway,  # type: ignore[arg-type]
        model=model,
        request_timeout_seconds=12,
        transaction_manager=transaction_manager,
        classification_repository=query_repository,
        execution_repository=execution_repository,
        utc_now=lambda: _NOW,
        uuid_factory=SequenceUuidFactory(uuid_values),
    )

    return (
        executor,
        query_repository,
        execution_repository,
        transaction_manager,
    )


async def test_executes_gateway_and_persists_classification() -> None:
    gateway = FakeGateway(_successful_result())
    (
        executor,
        query_repository,
        execution_repository,
        transaction_manager,
    ) = _executor(gateway=gateway)

    result = await executor.execute(_context())

    assert result == CompletedExecution()
    assert query_repository.queries == [
        (
            _WORKSPACE_ID,
            _AGENT_RUN_ID,
        ),
    ]
    assert transaction_manager.enter_count == 2
    assert transaction_manager.exit_count == 2
    assert len(gateway.requests) == 1

    request = gateway.requests[0]

    assert request.model == "mock-ticket-classifier-v1"
    assert request.timeout_seconds == 12
    assert request.output_schema is TicketClassificationResult
    assert "Invoice charge question" not in request.instructions
    assert "Invoice charge question" in request.input
    assert request.metadata["supportops_agent_run_attempt_id"] == str(_ATTEMPT_ID)
    assert request.metadata["supportops_prompt_id"] == "ticket-classification"
    assert request.metadata["supportops_prompt_version"] == "1"
    assert (
        request.metadata["supportops_prompt_content_hash"]
        == TICKET_CLASSIFICATION_PROMPT_V1.content_hash
    )

    assert len(execution_repository.commands) == 1
    command = execution_repository.commands[0]

    assert command.persisted_at == _NOW
    assert len(command.invocations) == 1

    invocation = command.invocations[0]

    assert invocation.id == _INVOCATION_ID_1
    assert invocation.status is LLMInvocationStatus.SUCCEEDED
    assert invocation.provider == "mock"
    assert invocation.model == "mock-ticket-classifier-v1"
    assert invocation.pricing_found is True
    assert invocation.estimated_total_cost_usd == _ZERO_COST

    classification = command.classification

    assert classification is not None
    assert classification.id == _CLASSIFICATION_ID
    assert classification.accepted_llm_invocation_id == invocation.id
    assert classification.category is TicketCategory.BILLING
    assert classification.provider == invocation.provider
    assert classification.model == invocation.model
    assert classification.prompt_content_hash == (TICKET_CLASSIFICATION_PROMPT_V1.content_hash)


async def test_persists_initial_and_repair_invocations() -> None:
    validation_error_code = LLMErrorCode.OUTPUT_VALIDATION_FAILED
    gateway_result = LLMGatewayResult(
        output=_classification_output(),
        invocations=(
            LLMInvocationTrace(
                invocation_sequence=1,
                status=(LLMInvocationStatus.VALIDATION_FAILED),
                provider="mock",
                model="mock-ticket-classifier-v1",
                provider_request_id="request-1",
                usage=LLMTokenUsage(
                    input_tokens=100,
                    cached_input_tokens=0,
                    output_tokens=10,
                    total_tokens=110,
                ),
                latency_ms=10,
                error_code=validation_error_code,
            ),
            _successful_trace(sequence=2),
        ),
        accepted_invocation_sequence=2,
    )
    execution_repository = RecordingExecutionRepository()
    executor, _, _, _ = _executor(
        gateway=FakeGateway(gateway_result),
        execution_repository=execution_repository,
        uuid_values=(
            _INVOCATION_ID_1,
            _INVOCATION_ID_2,
            _CLASSIFICATION_ID,
        ),
    )

    await executor.execute(_context())

    command = execution_repository.commands[0]

    assert [invocation.invocation_sequence for invocation in command.invocations] == [
        1,
        2,
    ]
    assert command.invocations[0].error_code == (validation_error_code)
    assert command.invocations[1].error_code is None
    assert command.classification is not None
    assert command.classification.accepted_llm_invocation_id == _INVOCATION_ID_2


async def test_existing_classification_skips_gateway() -> None:
    existing = TicketClassification.create(
        classification_id=_CLASSIFICATION_ID,
        workspace_id=_WORKSPACE_ID,
        ticket_id=_TICKET_ID,
        agent_run_id=_AGENT_RUN_ID,
        accepted_llm_invocation_id=_INVOCATION_ID_1,
        category=TicketCategory.BILLING,
        intent=TicketIntent.ASK_QUESTION,
        urgency=TicketUrgency.NORMAL,
        sentiment=TicketSentiment.NEUTRAL,
        requires_human_review=False,
        summary="An accepted classification already exists.",
        schema_version=(TICKET_CLASSIFICATION_SCHEMA_VERSION),
        prompt_id="ticket-classification",
        prompt_version=1,
        prompt_content_hash=(TICKET_CLASSIFICATION_PROMPT_V1.content_hash),
        provider="mock",
        model="mock-ticket-classifier-v1",
        now=_NOW,
    )
    gateway = FakeGateway(_successful_result())
    query_repository = RecordingClassificationRepository(
        existing=existing,
    )
    execution_repository = RecordingExecutionRepository()
    transaction_manager = RecordingTransactionManager()
    executor, _, _, _ = _executor(
        gateway=gateway,
        query_repository=query_repository,
        execution_repository=execution_repository,
        transaction_manager=transaction_manager,
        uuid_values=(),
    )

    await executor.execute(_context())

    assert gateway.requests == []
    assert execution_repository.commands == []
    assert transaction_manager.enter_count == 1
    assert transaction_manager.exit_count == 1


async def test_retryable_gateway_failure_is_persisted_then_raised() -> None:
    gateway = FakeGateway(_timeout_failure())
    execution_repository = RecordingExecutionRepository()
    executor, _, _, transaction_manager = _executor(
        gateway=gateway,
        execution_repository=execution_repository,
        model="gpt-5-nano",
        uuid_values=(_INVOCATION_ID_1,),
    )

    with pytest.raises(
        RetryableAgentRunExecutionError,
    ) as captured:
        await executor.execute(_context())

    assert captured.value.error_code == "llm_timeout"
    assert captured.value.error_summary == (
        "The LLM provider request exceeded its configured timeout."
    )
    assert transaction_manager.enter_count == 2
    assert len(execution_repository.commands) == 1

    command = execution_repository.commands[0]

    assert command.classification is None
    assert len(command.invocations) == 1
    assert command.invocations[0].error_code is (LLMErrorCode.TIMEOUT)


async def test_terminal_gateway_failure_is_persisted_then_raised() -> None:
    execution_repository = RecordingExecutionRepository()
    executor, _, _, _ = _executor(
        gateway=FakeGateway(_refusal_failure()),
        execution_repository=execution_repository,
        model="gpt-5-nano",
        uuid_values=(_INVOCATION_ID_1,),
    )

    with pytest.raises(
        TerminalAgentRunExecutionError,
    ) as captured:
        await executor.execute(_context())

    assert captured.value.error_code == "llm_refusal"
    assert captured.value.error_summary == (
        "The LLM provider refused to produce the requested output."
    )
    assert execution_repository.commands[0].classification is None


async def test_concurrent_accepted_classification_wins_failure_race() -> None:
    execution_repository = RecordingExecutionRepository(
        result=(ClassificationPersistenceResult.ALREADY_CLASSIFIED),
    )
    executor, _, _, _ = _executor(
        gateway=FakeGateway(_timeout_failure()),
        execution_repository=execution_repository,
        model="gpt-5-nano",
        uuid_values=(_INVOCATION_ID_1,),
    )

    await executor.execute(_context())

    assert len(execution_repository.commands) == 1


async def test_lost_lease_overrides_successful_gateway_result() -> None:
    execution_repository = RecordingExecutionRepository(
        result=ClassificationPersistenceResult.LEASE_LOST,
    )
    executor, _, _, _ = _executor(
        gateway=FakeGateway(_successful_result()),
        execution_repository=execution_repository,
    )

    with pytest.raises(
        RetryableAgentRunExecutionError,
    ) as captured:
        await executor.execute(_context())

    assert captured.value.error_code == ("classification_lease_lost")
    assert captured.value.error_summary == (
        "The AgentRun lease was lost before classification results could be persisted."
    )


async def test_unknown_pricing_remains_null() -> None:
    execution_repository = RecordingExecutionRepository()
    executor, _, _, _ = _executor(
        gateway=FakeGateway(
            _successful_result(
                provider="future-provider",
                model="future-model",
            ),
        ),
        execution_repository=execution_repository,
        model="future-model",
    )

    await executor.execute(_context())

    invocation = execution_repository.commands[0].invocations[0]

    assert invocation.pricing_found is False
    assert invocation.estimated_input_cost_usd is None
    assert invocation.estimated_cached_input_cost_usd is None
    assert invocation.estimated_output_cost_usd is None
    assert invocation.estimated_total_cost_usd is None


@pytest.mark.parametrize(
    (
        "field_name",
        "field_value",
        "expected_code",
    ),
    [
        (
            "workflow_name",
            "unknown-workflow",
            "unsupported_workflow",
        ),
        (
            "workflow_version",
            "unknown-version",
            "unsupported_workflow_version",
        ),
        (
            "trigger_key",
            "unknown-trigger",
            "unsupported_trigger",
        ),
    ],
)
async def test_rejects_unsupported_run_identity(
    field_name: str,
    field_value: str,
    expected_code: str,
) -> None:
    context = _context()
    if field_name == "workflow_name":
        agent_run = replace(
            context.agent_run,
            workflow_name=field_value,
        )
    elif field_name == "workflow_version":
        agent_run = replace(
            context.agent_run,
            workflow_version=field_value,
        )
    else:
        assert field_name == "trigger_key"
        agent_run = replace(
            context.agent_run,
            trigger_key=field_value,
        )
    modified_context = replace(
        context,
        agent_run=agent_run,
    )
    gateway = FakeGateway(_successful_result())
    executor, _, execution_repository, _ = _executor(
        gateway=gateway,
    )

    with pytest.raises(
        TerminalAgentRunExecutionError,
    ) as captured:
        await executor.execute(modified_context)

    assert captured.value.error_code == expected_code
    assert gateway.requests == []
    assert execution_repository.commands == []


async def test_accepts_controlled_support_workflow_version() -> None:
    context = _context()
    controlled_context = replace(
        context,
        agent_run=replace(
            context.agent_run,
            workflow_version=(CONTROLLED_SUPPORT_WORKFLOW_VERSION),
        ),
    )
    gateway = FakeGateway(_successful_result())
    executor, _, execution_repository, _ = _executor(
        gateway=gateway,
    )

    await executor.execute(controlled_context)

    assert len(gateway.requests) == 1
    assert len(execution_repository.commands) == 1
    assert (
        gateway.requests[0].metadata["supportops_workflow_version"]
        == CONTROLLED_SUPPORT_WORKFLOW_VERSION
    )


async def test_rejects_unexpected_gateway_output_schema() -> None:
    gateway_result = LLMGatewayResult(
        output=UnexpectedOutput(value="unexpected"),
        invocations=(_successful_trace(),),
        accepted_invocation_sequence=1,
    )
    execution_repository = RecordingExecutionRepository()
    executor, _, _, _ = _executor(
        gateway=FakeGateway(gateway_result),
        execution_repository=execution_repository,
    )

    with pytest.raises(
        RuntimeError,
        match="unexpected output schema",
    ):
        await executor.execute(_context())

    assert execution_repository.commands == []


@pytest.mark.parametrize(
    (
        "model",
        "request_timeout_seconds",
        "expected_message",
    ),
    [
        (
            "",
            12,
            "model is required",
        ),
        (
            " model",
            12,
            "must not contain surrounding whitespace",
        ),
        (
            "test-model",
            0,
            "request_timeout_seconds must be positive",
        ),
    ],
)
def test_rejects_invalid_configuration(
    model: str,
    request_timeout_seconds: float,
    expected_message: str,
) -> None:
    gateway = FakeGateway(_successful_result())

    with pytest.raises(
        ValueError,
        match=expected_message,
    ):
        TicketClassificationExecutor(
            gateway=gateway,  # type: ignore[arg-type]
            model=model,
            request_timeout_seconds=(request_timeout_seconds),
            transaction_manager=(RecordingTransactionManager()),
            classification_repository=(RecordingClassificationRepository()),
            execution_repository=(RecordingExecutionRepository()),
        )
