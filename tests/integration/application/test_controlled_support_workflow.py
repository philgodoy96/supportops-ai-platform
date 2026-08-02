"""PostgreSQL checkpoint integration for the controlled support graph."""

from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from uuid import UUID, uuid4

import pytest
from pydantic import SecretStr
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from supportops.agent_graph.application.recommendation_execution import (
    RecommendationExecutionOutcome,
)
from supportops.agent_graph.application.tool_observations import (
    ControlledToolObservationBundle,
)
from supportops.agent_graph.application.transitions import (
    attach_analysis_completion,
    attach_recommendation,
    attach_recommendation_invocation,
    reserve_decision_turn,
)
from supportops.agent_graph.application.workflow import (
    ControlledSupportWorkflowExecutor,
    ControlledSupportWorkflowNodes,
    compile_controlled_support_graph,
)
from supportops.agent_graph.domain.completion import (
    CompleteSupportAnalysisInput,
)
from supportops.agent_graph.domain.identity import (
    ControlledSupportGraphIdentity,
    derive_controlled_support_graph_identity,
)
from supportops.agent_graph.domain.state import (
    CONTROLLED_SUPPORT_WORKFLOW_VERSION,
    ControlledSupportGraphStateSnapshot,
    validate_controlled_support_state,
)
from supportops.agent_graph.infrastructure.checkpoints import (
    PostgresCheckpointRuntime,
    create_postgres_checkpoint_runtime,
)
from supportops.ai.gateway.tool_decisions import (
    LLMTerminalControlDecision,
)
from supportops.ai.schemas.ticket_classification import (
    TICKET_CLASSIFICATION_SCHEMA_VERSION,
    TicketCategory,
    TicketIntent,
    TicketSentiment,
    TicketUrgency,
)
from supportops.core.settings import Settings
from supportops.modules.agent_runs.application.execution import (
    AgentRunExecutionContext,
    RetryableAgentRunExecutionError,
)
from supportops.modules.agent_runs.domain.models import (
    AgentRun,
    AgentRunAttempt,
    AgentRunStatus,
)
from supportops.modules.support_recommendations.domain.models import (
    SupportRecommendation,
    SupportRecommendationAction,
)
from supportops.modules.ticket_classifications.domain.models import (
    TicketClassification,
)
from supportops.modules.tickets.domain.models import Ticket

pytestmark = pytest.mark.integration

_BASE_TIMESTAMP = datetime(
    2026,
    8,
    2,
    18,
    0,
    tzinfo=UTC,
)


@dataclass(frozen=True, slots=True)
class _DecisionOutcome:
    """Minimal successful result consumed by the graph node."""

    state: ControlledSupportGraphStateSnapshot
    decision: LLMTerminalControlDecision


class _NoopTransactionManager:
    """Expose the transaction shape required by graph nodes."""

    @asynccontextmanager
    async def transaction(
        self,
    ) -> AsyncIterator[None]:
        yield


class _ClassificationRepository:
    """Return one durable classification projection."""

    def __init__(
        self,
        classification: TicketClassification,
    ) -> None:
        self._classification = classification
        self.call_count = 0

    async def add(
        self,
        classification: TicketClassification,
    ) -> None:
        del classification

        raise AssertionError("Classification persistence must not run in this path.")

    async def get_by_agent_run_id(
        self,
        *,
        workspace_id: UUID,
        agent_run_id: UUID,
    ) -> TicketClassification | None:
        self.call_count += 1

        assert workspace_id == (self._classification.workspace_id)
        assert agent_run_id == (self._classification.agent_run_id)

        return self._classification


class _UnusedClassificationExecutor:
    """Fail if classification generation is unexpectedly repeated."""

    def __init__(self) -> None:
        self.call_count = 0

    async def execute(
        self,
        context: AgentRunExecutionContext,
    ) -> None:
        del context

        self.call_count += 1

        raise AssertionError("The persisted classification should have been reused.")


class _EmptyObservationAssembler:
    """Return deterministic empty tool history."""

    def __init__(self) -> None:
        self.call_count = 0

    async def assemble(
        self,
        *,
        state: ControlledSupportGraphStateSnapshot,
        context: object,
    ) -> ControlledToolObservationBundle:
        del state, context

        self.call_count += 1

        return ControlledToolObservationBundle(
            observations=(),
            citation_sources=(),
        )


class _TerminalDecisionExecutor:
    """Produce one terminal decision and count executions."""

    def __init__(
        self,
        *,
        accepted_invocation_id: UUID,
    ) -> None:
        self._accepted_invocation_id = accepted_invocation_id
        self.call_count = 0

    async def execute(
        self,
        *,
        state: ControlledSupportGraphStateSnapshot,
        context: AgentRunExecutionContext,
        tool_observations: tuple[
            Mapping[str, object],
            ...,
        ],
    ) -> _DecisionOutcome:
        del context

        self.call_count += 1

        assert tool_observations == ()

        reserved_state = reserve_decision_turn(state)
        completion = CompleteSupportAnalysisInput(
            recommended_action=(SupportRecommendationAction.RESPOND),
            evidence_sufficient=True,
            requires_human_review=False,
            decision_summary=(
                "The persisted classification provides sufficient evidence for a direct response."
            ),
        )
        completed_state = attach_analysis_completion(
            reserved_state,
            completion,
        )
        decision = LLMTerminalControlDecision(
            provider_tool_call_id=(f"integration-terminal-{self._accepted_invocation_id}"),
            control_name="complete_support_analysis",
            control_version=1,
            output=completion,
        )

        return _DecisionOutcome(
            state=completed_state,
            decision=decision,
        )


class _NoToolExecutor:
    """Confirm that the terminal-only path executes no tool."""

    def __init__(self) -> None:
        self.recovery_call_count = 0
        self.execution_call_count = 0

    async def recover_next_persisted_outcome(
        self,
        *,
        state: ControlledSupportGraphStateSnapshot,
        context: AgentRunExecutionContext,
    ) -> None:
        del state, context

        self.recovery_call_count += 1

        return None

    async def execute(
        self,
        **kwargs: object,
    ) -> None:
        del kwargs

        self.execution_call_count += 1

        raise AssertionError("The terminal-only decision must not execute a tool.")


class _FailOnceRecommendationExecutor:
    """Fail once, then produce the persisted recommendation state."""

    def __init__(
        self,
        *,
        recommendation_id: UUID,
        recommendation_invocation_id: UUID,
    ) -> None:
        self._recommendation_id = recommendation_id
        self._recommendation_invocation_id = recommendation_invocation_id
        self.call_count = 0

    async def execute(
        self,
        *,
        state: ControlledSupportGraphStateSnapshot,
        context: AgentRunExecutionContext,
    ) -> RecommendationExecutionOutcome:
        self.call_count += 1

        if self.call_count == 1:
            raise RetryableAgentRunExecutionError(
                error_code=("synthetic_recommendation_dependency"),
                error_summary=(
                    "The integration test interrupted "
                    "recommendation drafting after the terminal "
                    "decision checkpoint."
                ),
            )

        classification_id = state.classification_id

        if classification_id is None:
            raise AssertionError("Recommendation execution requires classification.")

        recommendation = SupportRecommendation.create(
            recommendation_id=self._recommendation_id,
            workspace_id=state.workspace_id,
            ticket_id=state.ticket_id,
            agent_run_id=state.agent_run_id,
            classification_id=classification_id,
            accepted_llm_invocation_id=(self._recommendation_invocation_id),
            recommended_action=(SupportRecommendationAction.RESPOND),
            response_text=("Follow the documented account recovery procedure."),
            requires_human_review=False,
            decision_summary=(
                "The persisted classification provides sufficient evidence for a direct response."
            ),
            prompt_id="support-recommendation-draft",
            prompt_version=1,
            prompt_content_hash="a" * 64,
            provider="integration",
            model="integration-model",
        )
        state_with_invocation = attach_recommendation_invocation(
            state,
            self._recommendation_invocation_id,
        )
        completed_state = attach_recommendation(
            state_with_invocation,
            recommendation,
        )

        assert context.agent_run.id == state.agent_run_id

        return RecommendationExecutionOutcome(
            state=completed_state,
            recommendation=recommendation,
            recovered=False,
        )


def _create_context(
    *,
    workspace_id: UUID,
    ticket_id: UUID,
    agent_run_id: UUID,
    attempt_id: UUID,
    lease_token: UUID,
) -> AgentRunExecutionContext:
    ticket = Ticket.create(
        ticket_id=ticket_id,
        workspace_id=workspace_id,
        subject="Unable to reset account access",
        description=("The customer cannot complete the documented account recovery procedure."),
        external_reference=None,
        ingestion_request_id=uuid4(),
        correlation_id=uuid4(),
        now=_BASE_TIMESTAMP,
    )
    queued_run = AgentRun.create_initial(
        agent_run_id=agent_run_id,
        workspace_id=workspace_id,
        ticket_id=ticket_id,
        ingestion_request_id=(ticket.ingestion_request_id),
        correlation_id=ticket.correlation_id,
        workflow_version=(CONTROLLED_SUPPORT_WORKFLOW_VERSION),
        max_attempts=3,
        now=_BASE_TIMESTAMP,
    )
    running_run = replace(
        queued_run,
        status=AgentRunStatus.RUNNING,
        attempt_count=1,
        lease_owner="integration-worker",
        lease_token=lease_token,
        lease_expires_at=(_BASE_TIMESTAMP + timedelta(minutes=5)),
        first_started_at=_BASE_TIMESTAMP,
        updated_at=_BASE_TIMESTAMP,
    )
    attempt = AgentRunAttempt.start(
        attempt_id=attempt_id,
        agent_run_id=agent_run_id,
        attempt_number=1,
        worker_id="integration-worker",
        lease_token=lease_token,
        execution_request_id=uuid4(),
        now=_BASE_TIMESTAMP,
    )

    return AgentRunExecutionContext(
        agent_run=running_run,
        attempt=attempt,
        ticket=ticket,
    )


def _create_classification(
    *,
    workspace_id: UUID,
    ticket_id: UUID,
    agent_run_id: UUID,
) -> TicketClassification:
    return TicketClassification.create(
        classification_id=uuid4(),
        workspace_id=workspace_id,
        ticket_id=ticket_id,
        agent_run_id=agent_run_id,
        accepted_llm_invocation_id=uuid4(),
        category=TicketCategory.ACCOUNT_ACCESS,
        intent=TicketIntent.REQUEST_ACCESS,
        urgency=TicketUrgency.NORMAL,
        sentiment=TicketSentiment.NEUTRAL,
        requires_human_review=False,
        summary=("The customer needs documented account recovery guidance."),
        schema_version=(TICKET_CLASSIFICATION_SCHEMA_VERSION),
        prompt_id="ticket-classification",
        prompt_version=1,
        prompt_content_hash="b" * 64,
        provider="integration",
        model="integration-model",
        now=_BASE_TIMESTAMP,
    )


def _checkpoint_database_url(
    settings: Settings,
) -> SecretStr:
    database_url = str(settings.postgresql_url)

    if database_url.startswith("postgresql+asyncpg://"):
        database_url = database_url.replace(
            "postgresql+asyncpg://",
            "postgresql://",
            1,
        )
    elif database_url.startswith("postgresql+psycopg://"):
        database_url = database_url.replace(
            "postgresql+psycopg://",
            "postgresql://",
            1,
        )

    if not database_url.startswith("postgresql://"):
        raise AssertionError(
            "Integration PostgreSQL URL cannot be converted to a Psycopg connection string."
        )

    return SecretStr(database_url)


async def _delete_checkpoint_thread(
    *,
    engine: AsyncEngine,
    identity: ControlledSupportGraphIdentity,
) -> None:
    parameters = {
        "thread_id": identity.thread_id,
        "checkpoint_ns": identity.checkpoint_namespace,
    }

    async with engine.begin() as connection:
        await connection.execute(
            text(
                "DELETE FROM checkpoint_writes "
                "WHERE thread_id = :thread_id "
                "AND checkpoint_ns = :checkpoint_ns"
            ),
            parameters,
        )
        await connection.execute(
            text(
                "DELETE FROM checkpoint_blobs "
                "WHERE thread_id = :thread_id "
                "AND checkpoint_ns = :checkpoint_ns"
            ),
            parameters,
        )
        await connection.execute(
            text(
                "DELETE FROM checkpoints "
                "WHERE thread_id = :thread_id "
                "AND checkpoint_ns = :checkpoint_ns"
            ),
            parameters,
        )


async def test_postgres_checkpoint_resume_skips_completed_nodes(
    integration_settings: Settings,
    postgresql_engine: AsyncEngine,
) -> None:
    workspace_id = uuid4()
    ticket_id = uuid4()
    agent_run_id = uuid4()
    attempt_id = uuid4()
    lease_token = uuid4()
    recommendation_id = uuid4()
    recommendation_invocation_id = uuid4()
    decision_invocation_id = uuid4()

    context = _create_context(
        workspace_id=workspace_id,
        ticket_id=ticket_id,
        agent_run_id=agent_run_id,
        attempt_id=attempt_id,
        lease_token=lease_token,
    )
    classification = _create_classification(
        workspace_id=workspace_id,
        ticket_id=ticket_id,
        agent_run_id=agent_run_id,
    )
    classification_repository = _ClassificationRepository(classification)
    classification_executor = _UnusedClassificationExecutor()
    observation_assembler = _EmptyObservationAssembler()
    decision_executor = _TerminalDecisionExecutor(
        accepted_invocation_id=decision_invocation_id,
    )
    tool_executor = _NoToolExecutor()
    recommendation_executor = _FailOnceRecommendationExecutor(
        recommendation_id=recommendation_id,
        recommendation_invocation_id=(recommendation_invocation_id),
    )
    checkpoint_runtime: PostgresCheckpointRuntime | None = None
    checkpoint_setup_complete = False
    identity = derive_controlled_support_graph_identity(agent_run_id)

    try:
        checkpoint_runtime = await create_postgres_checkpoint_runtime(
            database_url=_checkpoint_database_url(integration_settings)
        )
        await checkpoint_runtime.setup()
        checkpoint_setup_complete = True

        nodes = ControlledSupportWorkflowNodes(
            transaction_manager=(_NoopTransactionManager()),
            classification_repository=(classification_repository),
            classification_executor=(
                classification_executor  # type: ignore[arg-type]
            ),
            observation_assembler=(
                observation_assembler  # type: ignore[arg-type]
            ),
            decision_executor=(
                decision_executor  # type: ignore[arg-type]
            ),
            tool_executor=(tool_executor),  # type: ignore[arg-type]
            recommendation_executor=(
                recommendation_executor  # type: ignore[arg-type]
            ),
        )
        graph = compile_controlled_support_graph(
            nodes=nodes,
            checkpointer=(checkpoint_runtime.checkpointer),
        )
        executor = ControlledSupportWorkflowExecutor(graph=graph)

        with pytest.raises(
            RetryableAgentRunExecutionError,
        ) as captured:
            await executor.execute(context)

        assert captured.value.error_code == ("synthetic_recommendation_dependency")

        assert classification_repository.call_count == 1
        assert classification_executor.call_count == 0
        assert observation_assembler.call_count == 1
        assert decision_executor.call_count == 1
        assert tool_executor.recovery_call_count == 1
        assert tool_executor.execution_call_count == 0
        assert recommendation_executor.call_count == 1

        await executor.execute(context)

        assert classification_repository.call_count == 1
        assert classification_executor.call_count == 0
        assert observation_assembler.call_count == 1
        assert decision_executor.call_count == 1
        assert tool_executor.recovery_call_count == 1
        assert tool_executor.execution_call_count == 0
        assert recommendation_executor.call_count == 2

        checkpoint = await graph.aget_state(
            {
                "configurable": {
                    "thread_id": identity.thread_id,
                    "checkpoint_ns": (identity.checkpoint_namespace),
                }
            }
        )
        final_state = validate_controlled_support_state(checkpoint.values)

        assert final_state.workspace_id == workspace_id
        assert final_state.ticket_id == ticket_id
        assert final_state.agent_run_id == agent_run_id
        assert final_state.classification_id == (classification.id)
        assert final_state.decision_turn_count == 1
        assert final_state.tool_call_count == 0
        assert final_state.analysis_completion is not None
        assert final_state.recommendation_invocation_id == (recommendation_invocation_id)
        assert final_state.recommendation_id == (recommendation_id)
        assert final_state.current_error_code is None

        await executor.execute(context)

        assert classification_repository.call_count == 1
        assert observation_assembler.call_count == 1
        assert decision_executor.call_count == 1
        assert recommendation_executor.call_count == 2
    finally:
        if checkpoint_runtime is not None:
            await checkpoint_runtime.close()

        if checkpoint_setup_complete:
            await _delete_checkpoint_thread(
                engine=postgresql_engine,
                identity=identity,
            )
