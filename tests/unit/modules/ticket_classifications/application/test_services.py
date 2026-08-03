"""Unit tests for classification inspection application services."""

from collections.abc import Sequence
from datetime import UTC, datetime
from decimal import Decimal
from uuid import UUID

import pytest

from supportops.ai.gateway.results import (
    LLMInvocationStatus,
)
from supportops.ai.schemas.ticket_classification import (
    TICKET_CLASSIFICATION_SCHEMA_VERSION,
    TicketCategory,
    TicketIntent,
    TicketSentiment,
    TicketUrgency,
)
from supportops.modules.agent_runs.application.errors import (
    AgentRunNotFoundError,
)
from supportops.modules.agent_runs.domain.models import (
    TICKET_CLASSIFICATION_WORKFLOW_VERSION,
    AgentRun,
    AgentRunAttempt,
)
from supportops.modules.ticket_classifications.application.errors import (
    TicketClassificationNotFoundError,
)
from supportops.modules.ticket_classifications.application.services import (
    GetAgentRunClassificationReference,
    GetTicketClassification,
    ListAgentRunLLMInvocations,
    ListTicketClassifications,
)
from supportops.modules.ticket_classifications.domain.inspection import (
    AgentRunClassificationReference,
    LLMInvocationInspection,
)
from supportops.modules.ticket_classifications.domain.models import (
    TicketClassification,
)
from supportops.modules.tickets.application.errors import (
    TicketNotFoundError,
)
from supportops.modules.tickets.domain.models import Ticket

_NOW = datetime(
    2026,
    8,
    1,
    20,
    30,
    tzinfo=UTC,
)
_WORKSPACE_ID = UUID(
    "11111111-2222-4333-8444-555555555555",
)
_OTHER_WORKSPACE_ID = UUID(
    "99999999-8888-4777-8666-555555555555",
)
_TICKET_ID = UUID(
    "22222222-3333-4444-8555-666666666666",
)
_AGENT_RUN_ID = UUID(
    "33333333-4444-4555-8666-777777777777",
)
_ATTEMPT_ID = UUID(
    "44444444-5555-4666-8777-888888888888",
)
_INVOCATION_ID = UUID(
    "55555555-6666-4777-8888-999999999999",
)
_CLASSIFICATION_ID = UUID(
    "66666666-7777-4888-8999-aaaaaaaaaaaa",
)
_INGESTION_REQUEST_ID = UUID(
    "77777777-8888-4999-8aaa-bbbbbbbbbbbb",
)
_CORRELATION_ID = UUID(
    "88888888-9999-4aaa-8bbb-cccccccccccc",
)
_PROMPT_HASH = "a" * 64
_ZERO_COST = Decimal("0.000000000000")


class FakeTicketRepository:
    """Store deterministic workspace-scoped ticket query results."""

    def __init__(
        self,
        *,
        tickets: Sequence[Ticket] = (),
    ) -> None:
        self._tickets = {
            (
                ticket.workspace_id,
                ticket.id,
            ): ticket
            for ticket in tickets
        }
        self.get_calls: list[tuple[UUID, UUID]] = []

    async def add(
        self,
        ticket: Ticket,
    ) -> None:
        raise AssertionError(
            "Inspection services must not write tickets.",
        )

    async def get(
        self,
        workspace_id: UUID,
        ticket_id: UUID,
    ) -> Ticket | None:
        self.get_calls.append(
            (
                workspace_id,
                ticket_id,
            ),
        )
        return self._tickets.get(
            (
                workspace_id,
                ticket_id,
            ),
        )

    async def list(
        self,
        workspace_id: UUID,
        *,
        limit: int,
        after_created_at: datetime | None = None,
        after_ticket_id: UUID | None = None,
    ) -> Sequence[Ticket]:
        raise AssertionError(
            "Classification services must not list tickets.",
        )


class FakeAgentRunQueryRepository:
    """Store deterministic workspace-scoped AgentRun query results."""

    def __init__(
        self,
        *,
        agent_runs: Sequence[AgentRun] = (),
    ) -> None:
        self._agent_runs = {
            (
                agent_run.workspace_id,
                agent_run.id,
            ): agent_run
            for agent_run in agent_runs
        }
        self.get_calls: list[tuple[UUID, UUID]] = []

    async def get(
        self,
        *,
        workspace_id: UUID,
        agent_run_id: UUID,
    ) -> AgentRun | None:
        self.get_calls.append(
            (
                workspace_id,
                agent_run_id,
            ),
        )
        return self._agent_runs.get(
            (
                workspace_id,
                agent_run_id,
            ),
        )

    async def list_attempts(
        self,
        *,
        agent_run_id: UUID,
    ) -> Sequence[AgentRunAttempt]:
        raise AssertionError(
            "Classification services must not list attempts.",
        )


class FakeClassificationQueryRepository:
    """Store deterministic classification inspection results."""

    def __init__(
        self,
        *,
        classifications: Sequence[TicketClassification] = (),
        references: Sequence[tuple[UUID, UUID, AgentRunClassificationReference]] = (),
        invocations: Sequence[LLMInvocationInspection] = (),
    ) -> None:
        self._classifications_by_id = {
            (
                classification.workspace_id,
                classification.id,
            ): classification
            for classification in classifications
        }
        self._classifications_by_run = {
            (
                classification.workspace_id,
                classification.agent_run_id,
            ): classification
            for classification in classifications
        }
        self._references = {
            (
                workspace_id,
                agent_run_id,
            ): reference
            for workspace_id, agent_run_id, reference in references
        }
        self._invocations = tuple(invocations)

        self.get_calls: list[tuple[UUID, UUID]] = []
        self.list_ticket_calls: list[
            tuple[
                UUID,
                UUID,
                int,
                datetime | None,
                UUID | None,
            ]
        ] = []
        self.list_invocation_calls: list[tuple[UUID, UUID]] = []
        self.reference_calls: list[tuple[UUID, UUID]] = []

    async def get(
        self,
        *,
        workspace_id: UUID,
        classification_id: UUID,
    ) -> TicketClassification | None:
        self.get_calls.append(
            (
                workspace_id,
                classification_id,
            ),
        )
        return self._classifications_by_id.get(
            (
                workspace_id,
                classification_id,
            ),
        )

    async def get_by_agent_run_id(
        self,
        *,
        workspace_id: UUID,
        agent_run_id: UUID,
    ) -> TicketClassification | None:
        return self._classifications_by_run.get(
            (
                workspace_id,
                agent_run_id,
            ),
        )

    async def get_reference_by_agent_run_id(
        self,
        *,
        workspace_id: UUID,
        agent_run_id: UUID,
    ) -> AgentRunClassificationReference | None:
        self.reference_calls.append(
            (
                workspace_id,
                agent_run_id,
            ),
        )
        return self._references.get(
            (
                workspace_id,
                agent_run_id,
            ),
        )

    async def list_by_ticket(
        self,
        *,
        workspace_id: UUID,
        ticket_id: UUID,
        limit: int,
        after_created_at: datetime | None = None,
        after_classification_id: UUID | None = None,
    ) -> Sequence[TicketClassification]:
        self.list_ticket_calls.append(
            (
                workspace_id,
                ticket_id,
                limit,
                after_created_at,
                after_classification_id,
            ),
        )

        matches = tuple(
            classification
            for classification in (self._classifications_by_id.values())
            if (
                classification.workspace_id == workspace_id
                and classification.ticket_id == ticket_id
            )
        )

        return tuple(
            sorted(
                matches,
                key=lambda classification: (
                    classification.created_at,
                    classification.id,
                ),
                reverse=True,
            )[:limit]
        )

    async def list_invocations_by_agent_run(
        self,
        *,
        workspace_id: UUID,
        agent_run_id: UUID,
    ) -> Sequence[LLMInvocationInspection]:
        self.list_invocation_calls.append(
            (
                workspace_id,
                agent_run_id,
            ),
        )

        return tuple(
            sorted(
                (
                    invocation
                    for invocation in self._invocations
                    if (
                        invocation.workspace_id == workspace_id
                        and invocation.agent_run_id == agent_run_id
                    )
                ),
                key=lambda invocation: (
                    invocation.attempt_number,
                    invocation.invocation_sequence,
                ),
            ),
        )


def _ticket() -> Ticket:
    return Ticket.create(
        ticket_id=_TICKET_ID,
        workspace_id=_WORKSPACE_ID,
        subject="Duplicated invoice charge",
        description=("The latest invoice contains the same charge twice."),
        ingestion_request_id=_INGESTION_REQUEST_ID,
        correlation_id=_CORRELATION_ID,
        now=_NOW,
    )


def _agent_run() -> AgentRun:
    return AgentRun.create_initial(
        agent_run_id=_AGENT_RUN_ID,
        workspace_id=_WORKSPACE_ID,
        ticket_id=_TICKET_ID,
        ingestion_request_id=_INGESTION_REQUEST_ID,
        correlation_id=_CORRELATION_ID,
        workflow_version=(TICKET_CLASSIFICATION_WORKFLOW_VERSION),
        max_retryable_failures=3,
        now=_NOW,
    )


def _classification() -> TicketClassification:
    return TicketClassification.create(
        classification_id=_CLASSIFICATION_ID,
        workspace_id=_WORKSPACE_ID,
        ticket_id=_TICKET_ID,
        agent_run_id=_AGENT_RUN_ID,
        accepted_llm_invocation_id=_INVOCATION_ID,
        category=TicketCategory.BILLING,
        intent=TicketIntent.ASK_QUESTION,
        urgency=TicketUrgency.NORMAL,
        sentiment=TicketSentiment.NEUTRAL,
        requires_human_review=False,
        summary="The customer is asking about a billing charge.",
        schema_version=(TICKET_CLASSIFICATION_SCHEMA_VERSION),
        prompt_id="ticket-classification",
        prompt_version=1,
        prompt_content_hash=_PROMPT_HASH,
        provider="mock",
        model="mock-ticket-classifier-v1",
        now=_NOW,
    )


def _invocation() -> LLMInvocationInspection:
    return LLMInvocationInspection(
        id=_INVOCATION_ID,
        workspace_id=_WORKSPACE_ID,
        ticket_id=_TICKET_ID,
        agent_run_id=_AGENT_RUN_ID,
        agent_run_attempt_id=_ATTEMPT_ID,
        attempt_number=1,
        invocation_sequence=1,
        status=LLMInvocationStatus.SUCCEEDED,
        provider="mock",
        model="mock-ticket-classifier-v1",
        prompt_id="ticket-classification",
        prompt_version=1,
        prompt_content_hash=_PROMPT_HASH,
        schema_version=(TICKET_CLASSIFICATION_SCHEMA_VERSION),
        input_tokens=100,
        cached_input_tokens=0,
        output_tokens=20,
        reasoning_tokens=None,
        total_tokens=120,
        pricing_catalog_version="pricing-v1",
        pricing_found=True,
        estimated_input_cost_usd=_ZERO_COST,
        estimated_cached_input_cost_usd=_ZERO_COST,
        estimated_output_cost_usd=_ZERO_COST,
        estimated_total_cost_usd=_ZERO_COST,
        latency_ms=25,
        error_code=None,
        created_at=_NOW,
    )


async def test_get_classification_returns_scoped_result() -> None:
    classification = _classification()
    repository = FakeClassificationQueryRepository(
        classifications=(classification,),
    )
    service = GetTicketClassification(
        repository=repository,
    )

    result = await service.execute(
        workspace_id=_WORKSPACE_ID,
        classification_id=_CLASSIFICATION_ID,
    )

    assert result == classification
    assert repository.get_calls == [
        (
            _WORKSPACE_ID,
            _CLASSIFICATION_ID,
        ),
    ]


@pytest.mark.parametrize(
    "workspace_id",
    [
        _WORKSPACE_ID,
        _OTHER_WORKSPACE_ID,
    ],
)
async def test_get_classification_uses_stable_not_found(
    workspace_id: UUID,
) -> None:
    repository = FakeClassificationQueryRepository()
    service = GetTicketClassification(
        repository=repository,
    )

    with pytest.raises(
        TicketClassificationNotFoundError,
        match="Ticket classification was not found",
    ):
        await service.execute(
            workspace_id=workspace_id,
            classification_id=_CLASSIFICATION_ID,
        )


async def test_list_ticket_classifications_validates_ticket() -> None:
    ticket = _ticket()
    classification = _classification()
    ticket_repository = FakeTicketRepository(
        tickets=(ticket,),
    )
    classification_repository = FakeClassificationQueryRepository(
        classifications=(classification,),
    )
    service = ListTicketClassifications(
        ticket_repository=ticket_repository,
        classification_repository=(classification_repository),
    )

    result = await service.execute(
        workspace_id=_WORKSPACE_ID,
        ticket_id=_TICKET_ID,
        limit=21,
        after_created_at=None,
        after_classification_id=None,
    )

    assert result == (classification,)
    assert ticket_repository.get_calls == [
        (
            _WORKSPACE_ID,
            _TICKET_ID,
        ),
    ]
    assert classification_repository.list_ticket_calls == [
        (
            _WORKSPACE_ID,
            _TICKET_ID,
            21,
            None,
            None,
        ),
    ]


async def test_list_ticket_classifications_hides_cross_workspace_ticket() -> None:
    classification_repository = FakeClassificationQueryRepository()
    service = ListTicketClassifications(
        ticket_repository=FakeTicketRepository(
            tickets=(_ticket(),),
        ),
        classification_repository=(classification_repository),
    )

    with pytest.raises(
        TicketNotFoundError,
        match="Ticket was not found",
    ):
        await service.execute(
            workspace_id=_OTHER_WORKSPACE_ID,
            ticket_id=_TICKET_ID,
            limit=20,
        )

    assert classification_repository.list_ticket_calls == []


async def test_list_invocations_validates_agent_run() -> None:
    invocation = _invocation()
    agent_run_repository = FakeAgentRunQueryRepository(
        agent_runs=(_agent_run(),),
    )
    classification_repository = FakeClassificationQueryRepository(
        invocations=(invocation,),
    )
    service = ListAgentRunLLMInvocations(
        agent_run_repository=agent_run_repository,
        classification_repository=(classification_repository),
    )

    result = await service.execute(
        workspace_id=_WORKSPACE_ID,
        agent_run_id=_AGENT_RUN_ID,
    )

    assert result == (invocation,)
    assert agent_run_repository.get_calls == [
        (
            _WORKSPACE_ID,
            _AGENT_RUN_ID,
        ),
    ]
    assert classification_repository.list_invocation_calls == [
        (
            _WORKSPACE_ID,
            _AGENT_RUN_ID,
        ),
    ]


async def test_list_invocations_hides_cross_workspace_run() -> None:
    classification_repository = FakeClassificationQueryRepository()
    service = ListAgentRunLLMInvocations(
        agent_run_repository=(
            FakeAgentRunQueryRepository(
                agent_runs=(_agent_run(),),
            )
        ),
        classification_repository=(classification_repository),
    )

    with pytest.raises(
        AgentRunNotFoundError,
        match="AgentRun was not found",
    ):
        await service.execute(
            workspace_id=_OTHER_WORKSPACE_ID,
            agent_run_id=_AGENT_RUN_ID,
        )

    assert classification_repository.list_invocation_calls == []


async def test_get_agent_run_classification_reference() -> None:
    reference = AgentRunClassificationReference.from_domain(
        _classification(),
    )
    classification_repository = FakeClassificationQueryRepository(
        references=(
            (
                _WORKSPACE_ID,
                _AGENT_RUN_ID,
                reference,
            ),
        ),
    )
    service = GetAgentRunClassificationReference(
        agent_run_repository=(
            FakeAgentRunQueryRepository(
                agent_runs=(_agent_run(),),
            )
        ),
        classification_repository=(classification_repository),
    )

    result = await service.execute(
        workspace_id=_WORKSPACE_ID,
        agent_run_id=_AGENT_RUN_ID,
    )

    assert result == reference
    assert classification_repository.reference_calls == [
        (
            _WORKSPACE_ID,
            _AGENT_RUN_ID,
        ),
    ]


async def test_agent_run_without_classification_returns_none() -> None:
    service = GetAgentRunClassificationReference(
        agent_run_repository=(
            FakeAgentRunQueryRepository(
                agent_runs=(_agent_run(),),
            )
        ),
        classification_repository=(FakeClassificationQueryRepository()),
    )

    result = await service.execute(
        workspace_id=_WORKSPACE_ID,
        agent_run_id=_AGENT_RUN_ID,
    )

    assert result is None
