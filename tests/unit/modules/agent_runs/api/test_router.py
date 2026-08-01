"""Unit tests for AgentRun inspection route orchestration."""

from datetime import UTC, datetime
from decimal import Decimal
from typing import cast
from uuid import UUID

from supportops.ai.gateway.results import (
    LLMInvocationStatus,
)
from supportops.ai.schemas.ticket_classification import (
    TICKET_CLASSIFICATION_SCHEMA_VERSION,
)
from supportops.application.agent_run_inspection import (
    AgentRunInspection,
    GetAgentRunInspection,
)
from supportops.modules.agent_runs.api.router import (
    get_agent_run,
    list_agent_run_llm_invocations,
)
from supportops.modules.agent_runs.domain.models import (
    TICKET_CLASSIFICATION_WORKFLOW_VERSION,
    AgentRun,
)
from supportops.modules.ticket_classifications.application.services import (
    ListAgentRunLLMInvocations,
)
from supportops.modules.ticket_classifications.domain.inspection import (
    AgentRunClassificationReference,
    LLMInvocationInspection,
)

_NOW = datetime(
    2026,
    8,
    1,
    21,
    45,
    tzinfo=UTC,
)
_WORKSPACE_ID = UUID(
    "11111111-aaaa-4111-8111-111111111111",
)
_TICKET_ID = UUID(
    "22222222-bbbb-4222-8222-222222222222",
)
_AGENT_RUN_ID = UUID(
    "33333333-cccc-4333-8333-333333333333",
)
_ATTEMPT_ID = UUID(
    "44444444-dddd-4444-8444-444444444444",
)
_INVOCATION_ID = UUID(
    "55555555-eeee-4555-8555-555555555555",
)
_CLASSIFICATION_ID = UUID(
    "66666666-ffff-4666-8666-666666666666",
)
_INGESTION_REQUEST_ID = UUID(
    "77777777-1111-4777-8777-777777777777",
)
_CORRELATION_ID = UUID(
    "88888888-2222-4888-8888-888888888888",
)
_PROMPT_HASH = "a" * 64
_ZERO_COST = Decimal("0.000000000000")


class RecordingInspectionService:
    """Return one configured AgentRun inspection."""

    def __init__(
        self,
        inspection: AgentRunInspection,
    ) -> None:
        self.inspection = inspection
        self.calls: list[tuple[UUID, UUID]] = []

    async def execute(
        self,
        *,
        workspace_id: UUID,
        agent_run_id: UUID,
    ) -> AgentRunInspection:
        self.calls.append(
            (
                workspace_id,
                agent_run_id,
            ),
        )
        return self.inspection


class RecordingInvocationService:
    """Return configured logical invocation history."""

    def __init__(
        self,
        invocations: tuple[
            LLMInvocationInspection,
            ...,
        ],
    ) -> None:
        self.invocations = invocations
        self.calls: list[tuple[UUID, UUID]] = []

    async def execute(
        self,
        *,
        workspace_id: UUID,
        agent_run_id: UUID,
    ) -> tuple[LLMInvocationInspection, ...]:
        self.calls.append(
            (
                workspace_id,
                agent_run_id,
            ),
        )
        return self.invocations


def _agent_run() -> AgentRun:
    return AgentRun.create_initial(
        agent_run_id=_AGENT_RUN_ID,
        workspace_id=_WORKSPACE_ID,
        ticket_id=_TICKET_ID,
        ingestion_request_id=_INGESTION_REQUEST_ID,
        correlation_id=_CORRELATION_ID,
        workflow_version=(TICKET_CLASSIFICATION_WORKFLOW_VERSION),
        max_attempts=3,
        now=_NOW,
    )


def _reference() -> AgentRunClassificationReference:
    return AgentRunClassificationReference(
        id=_CLASSIFICATION_ID,
        schema_version=(TICKET_CLASSIFICATION_SCHEMA_VERSION),
        created_at=_NOW,
    )


def _invocation(
    *,
    invocation_sequence: int,
) -> LLMInvocationInspection:
    return LLMInvocationInspection(
        id=UUID(
            f"55555555-eeee-4555-8555-{invocation_sequence:012d}",
        ),
        workspace_id=_WORKSPACE_ID,
        ticket_id=_TICKET_ID,
        agent_run_id=_AGENT_RUN_ID,
        agent_run_attempt_id=_ATTEMPT_ID,
        attempt_number=1,
        invocation_sequence=invocation_sequence,
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


async def test_agent_run_detail_projects_classification_reference() -> None:
    service = RecordingInspectionService(
        AgentRunInspection(
            agent_run=_agent_run(),
            classification=_reference(),
        ),
    )

    response = await get_agent_run(
        workspace_id=_WORKSPACE_ID,
        agent_run_id=_AGENT_RUN_ID,
        service=cast(
            GetAgentRunInspection,
            service,
        ),
    )

    assert response.id == _AGENT_RUN_ID
    assert response.classification is not None
    assert response.classification.id == (_CLASSIFICATION_ID)
    assert response.classification.schema_version == (TICKET_CLASSIFICATION_SCHEMA_VERSION)
    assert service.calls == [
        (
            _WORKSPACE_ID,
            _AGENT_RUN_ID,
        ),
    ]


async def test_agent_run_detail_supports_no_classification() -> None:
    service = RecordingInspectionService(
        AgentRunInspection(
            agent_run=_agent_run(),
            classification=None,
        ),
    )

    response = await get_agent_run(
        workspace_id=_WORKSPACE_ID,
        agent_run_id=_AGENT_RUN_ID,
        service=cast(
            GetAgentRunInspection,
            service,
        ),
    )

    assert response.classification is None


async def test_invocation_route_preserves_service_order() -> None:
    first = _invocation(
        invocation_sequence=1,
    )
    second = _invocation(
        invocation_sequence=2,
    )
    service = RecordingInvocationService(
        (
            first,
            second,
        ),
    )

    response = await list_agent_run_llm_invocations(
        workspace_id=_WORKSPACE_ID,
        agent_run_id=_AGENT_RUN_ID,
        service=cast(
            ListAgentRunLLMInvocations,
            service,
        ),
    )

    assert [item.invocation_sequence for item in response.items] == [
        1,
        2,
    ]
    assert response.items[0].provider == "mock"
    assert response.items[0].usage is not None
    assert response.items[0].usage.total_tokens == 120
    assert response.items[0].estimated_cost.total_cost_usd == _ZERO_COST
    assert service.calls == [
        (
            _WORKSPACE_ID,
            _AGENT_RUN_ID,
        ),
    ]


async def test_invocation_route_supports_empty_history() -> None:
    service = RecordingInvocationService(())

    response = await list_agent_run_llm_invocations(
        workspace_id=_WORKSPACE_ID,
        agent_run_id=_AGENT_RUN_ID,
        service=cast(
            ListAgentRunLLMInvocations,
            service,
        ),
    )

    assert response.items == []
