"""Read contracts for controlled-support inspection persistence."""

from dataclasses import dataclass
from typing import Protocol
from uuid import UUID

from supportops.agent_tools.domain.audit import (
    AgentToolCall,
)
from supportops.modules.agent_runs.domain.models import (
    AgentRun,
    AgentRunAttempt,
)
from supportops.modules.support_recommendations.domain.models import (
    SupportRecommendation,
    SupportRecommendationCitation,
)
from supportops.modules.ticket_classifications.domain.models import (
    LLMInvocation,
    TicketClassification,
)


@dataclass(frozen=True, slots=True)
class ControlledSupportInspectionIdentity:
    """Identify one workspace-owned AgentRun inspection."""

    workspace_id: UUID
    ticket_id: UUID
    agent_run_id: UUID

    def __post_init__(self) -> None:
        identifiers = (
            self.workspace_id,
            self.ticket_id,
            self.agent_run_id,
        )

        if not all(isinstance(identifier, UUID) for identifier in identifiers):
            raise TypeError("Inspection identity values must be UUIDs.")


@dataclass(frozen=True, slots=True)
class ControlledSupportInspectionData:
    """Durable entities loaded for one inspection snapshot."""

    agent_run: AgentRun
    attempts: tuple[AgentRunAttempt, ...]
    classification: TicketClassification | None
    tool_calls: tuple[AgentToolCall, ...]
    llm_invocations: tuple[LLMInvocation, ...]
    recommendation: SupportRecommendation | None
    citations: tuple[
        SupportRecommendationCitation,
        ...,
    ]

    def __post_init__(self) -> None:
        attempt_numbers = tuple(attempt.attempt_number for attempt in self.attempts)
        expected_attempt_numbers = tuple(
            range(
                1,
                self.agent_run.attempt_count + 1,
            )
        )

        if attempt_numbers != expected_attempt_numbers:
            raise ValueError("AgentRun attempts must be complete, ordered, and contiguous.")

        attempt_ids = tuple(attempt.id for attempt in self.attempts)

        if len(set(attempt_ids)) != len(attempt_ids):
            raise ValueError("AgentRun attempt IDs must be unique.")

        for attempt in self.attempts:
            if attempt.agent_run_id != self.agent_run.id:
                raise ValueError("AgentRun attempt ownership does not match the inspection root.")

        _validate_classification_ownership(
            agent_run=self.agent_run,
            classification=self.classification,
        )

        attempt_number_by_id = {attempt.id: attempt.attempt_number for attempt in self.attempts}

        _validate_tool_calls(
            agent_run=self.agent_run,
            tool_calls=self.tool_calls,
            attempt_number_by_id=attempt_number_by_id,
        )
        _validate_llm_invocations(
            agent_run=self.agent_run,
            invocations=self.llm_invocations,
            attempt_number_by_id=attempt_number_by_id,
        )
        _validate_recommendation(
            agent_run=self.agent_run,
            classification=self.classification,
            recommendation=self.recommendation,
        )
        _validate_citations(
            recommendation=self.recommendation,
            citations=self.citations,
        )


class ControlledSupportInspectionRepository(Protocol):
    """Load one consistent controlled-support inspection snapshot."""

    async def get_inspection_data(
        self,
        identity: ControlledSupportInspectionIdentity,
    ) -> ControlledSupportInspectionData | None:
        """Return exact workspace-scoped data or no resource."""

        ...


def _validate_classification_ownership(
    *,
    agent_run: AgentRun,
    classification: TicketClassification | None,
) -> None:
    if classification is None:
        return

    ownership = (
        (
            classification.workspace_id,
            agent_run.workspace_id,
        ),
        (
            classification.ticket_id,
            agent_run.ticket_id,
        ),
        (
            classification.agent_run_id,
            agent_run.id,
        ),
    )

    if any(actual != expected for actual, expected in ownership):
        raise ValueError("Classification ownership does not match the inspection root.")


def _validate_tool_calls(
    *,
    agent_run: AgentRun,
    tool_calls: tuple[AgentToolCall, ...],
    attempt_number_by_id: dict[UUID, int],
) -> None:
    keys: list[tuple[int, int, UUID]] = []
    tool_call_ids: list[UUID] = []

    for tool_call in tool_calls:
        ownership = (
            (
                tool_call.workspace_id,
                agent_run.workspace_id,
            ),
            (
                tool_call.ticket_id,
                agent_run.ticket_id,
            ),
            (
                tool_call.agent_run_id,
                agent_run.id,
            ),
        )

        if any(actual != expected for actual, expected in ownership):
            raise ValueError("Tool-call ownership does not match the inspection root.")

        proposed_by_attempt_number = attempt_number_by_id.get(
            tool_call.proposed_by_agent_run_attempt_id,
        )

        if proposed_by_attempt_number is None:
            raise ValueError("Tool call references an unknown proposal AgentRun attempt.")

        if tool_call.executed_by_agent_run_attempt_id is not None:
            executed_by_attempt_number = attempt_number_by_id.get(
                tool_call.executed_by_agent_run_attempt_id,
            )

            if executed_by_attempt_number is None:
                raise ValueError("Tool call references an unknown execution AgentRun attempt.")

        keys.append(
            (
                proposed_by_attempt_number,
                tool_call.sequence,
                tool_call.id,
            )
        )
        tool_call_ids.append(tool_call.id)

    if tuple(keys) != tuple(sorted(keys)):
        raise ValueError("Tool calls must be ordered by proposal attempt and sequence.")

    if len(set(tool_call_ids)) != len(tool_call_ids):
        raise ValueError("Tool-call IDs must be unique.")


def _validate_llm_invocations(
    *,
    agent_run: AgentRun,
    invocations: tuple[LLMInvocation, ...],
    attempt_number_by_id: dict[UUID, int],
) -> None:
    keys: list[tuple[int, int]] = []
    invocation_ids: list[UUID] = []

    for invocation in invocations:
        ownership = (
            (
                invocation.workspace_id,
                agent_run.workspace_id,
            ),
            (
                invocation.ticket_id,
                agent_run.ticket_id,
            ),
            (
                invocation.agent_run_id,
                agent_run.id,
            ),
        )

        if any(actual != expected for actual, expected in ownership):
            raise ValueError("LLM invocation ownership does not match the inspection root.")

        attempt_number = attempt_number_by_id.get(invocation.agent_run_attempt_id)

        if attempt_number is None:
            raise ValueError("LLM invocation references an unknown AgentRun attempt.")

        keys.append(
            (
                attempt_number,
                invocation.invocation_sequence,
            )
        )
        invocation_ids.append(invocation.id)

    if tuple(keys) != tuple(sorted(keys)):
        raise ValueError("LLM invocations must be ordered by attempt and sequence.")

    if len(set(invocation_ids)) != len(invocation_ids):
        raise ValueError("LLM invocation IDs must be unique.")


def _validate_recommendation(
    *,
    agent_run: AgentRun,
    classification: TicketClassification | None,
    recommendation: SupportRecommendation | None,
) -> None:
    if recommendation is None:
        return

    ownership = (
        (
            recommendation.workspace_id,
            agent_run.workspace_id,
        ),
        (
            recommendation.ticket_id,
            agent_run.ticket_id,
        ),
        (
            recommendation.agent_run_id,
            agent_run.id,
        ),
    )

    if any(actual != expected for actual, expected in ownership):
        raise ValueError("Recommendation ownership does not match the inspection root.")

    if classification is None:
        raise ValueError("Recommendation requires a persisted classification.")

    if recommendation.classification_id != classification.id:
        raise ValueError(
            "Recommendation classification does not match the inspection classification."
        )


def _validate_citations(
    *,
    recommendation: SupportRecommendation | None,
    citations: tuple[
        SupportRecommendationCitation,
        ...,
    ],
) -> None:
    if recommendation is None:
        if citations:
            raise ValueError("Citations require a persisted recommendation.")

        return

    citation_ids: list[UUID] = []
    ordinals: list[int] = []

    for citation in citations:
        if citation.workspace_id != recommendation.workspace_id:
            raise ValueError("Citation workspace does not match the recommendation.")

        if citation.support_recommendation_id != recommendation.id:
            raise ValueError("Citation recommendation ownership does not match.")

        citation_ids.append(citation.id)
        ordinals.append(citation.ordinal)

    expected_ordinals = list(
        range(
            1,
            len(citations) + 1,
        )
    )

    if ordinals != expected_ordinals:
        raise ValueError("Citation ordinals must be contiguous and one-based.")

    if len(set(citation_ids)) != len(citation_ids):
        raise ValueError("Citation IDs must be unique.")
