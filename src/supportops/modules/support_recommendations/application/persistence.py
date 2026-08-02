"""Fenced persistence contracts for support recommendations."""

from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import StrEnum
from typing import Protocol
from uuid import UUID

from supportops.ai.gateway.results import LLMInvocationStatus
from supportops.modules.support_recommendations.domain.models import (
    SupportRecommendation,
    SupportRecommendationCitation,
)
from supportops.modules.ticket_classifications.domain.models import (
    LLMInvocation,
)


class SupportRecommendationPersistenceResult(StrEnum):
    """Outcome of one fenced recommendation persistence operation."""

    APPLIED = "applied"
    ALREADY_RECOMMENDED = "already_recommended"
    ALREADY_RECORDED = "already_recorded"
    LEASE_LOST = "lease_lost"


@dataclass(frozen=True, slots=True)
class PersistSupportRecommendationCommand:
    """Persist recommendation execution state under an active lease."""

    workspace_id: UUID
    ticket_id: UUID
    agent_run_id: UUID
    agent_run_attempt_id: UUID
    lease_token: UUID
    persisted_at: datetime
    invocations: tuple[LLMInvocation, ...]
    recommendation: SupportRecommendation | None
    citations: tuple[SupportRecommendationCitation, ...]

    def __post_init__(self) -> None:
        _validate_utc_timestamp(
            self.persisted_at,
            field_name="persisted_at",
        )
        _validate_invocations(self)

        if self.recommendation is None:
            if self.citations:
                raise ValueError("Citations require an accepted recommendation.")

            return

        _validate_recommendation(
            command=self,
            recommendation=self.recommendation,
        )
        _validate_citations(
            command=self,
            recommendation=self.recommendation,
            citations=self.citations,
        )


class SupportRecommendationExecutionRepository(Protocol):
    """Atomic persistence boundary for recommendation execution."""

    async def persist_fenced(
        self,
        command: PersistSupportRecommendationCommand,
    ) -> SupportRecommendationPersistenceResult:
        """Persist invocations and an optional recommendation."""

        ...


def _validate_invocations(
    command: PersistSupportRecommendationCommand,
) -> None:
    if not command.invocations:
        raise ValueError("At least one LLM invocation is required.")

    expected_sequences = tuple(
        range(
            1,
            len(command.invocations) + 1,
        )
    )
    actual_sequences = tuple(invocation.invocation_sequence for invocation in command.invocations)

    if actual_sequences != expected_sequences:
        raise ValueError("Invocation sequences must be contiguous, ordered, and start at one.")

    invocation_ids = {invocation.id for invocation in command.invocations}

    if len(invocation_ids) != len(command.invocations):
        raise ValueError("Invocation identifiers must be unique.")

    for invocation in command.invocations:
        _validate_invocation_ownership(
            command=command,
            invocation=invocation,
        )

        if invocation.created_at > command.persisted_at:
            raise ValueError("persisted_at must not precede an invocation timestamp.")


def _validate_invocation_ownership(
    *,
    command: PersistSupportRecommendationCommand,
    invocation: LLMInvocation,
) -> None:
    ownership_values = (
        (
            invocation.workspace_id,
            command.workspace_id,
            "workspace",
        ),
        (
            invocation.ticket_id,
            command.ticket_id,
            "ticket",
        ),
        (
            invocation.agent_run_id,
            command.agent_run_id,
            "AgentRun",
        ),
        (
            invocation.agent_run_attempt_id,
            command.agent_run_attempt_id,
            "AgentRunAttempt",
        ),
    )

    for actual, expected, resource_name in ownership_values:
        if actual != expected:
            raise ValueError(
                f"Invocation {resource_name} ownership does not match the persistence command."
            )


def _validate_recommendation(
    *,
    command: PersistSupportRecommendationCommand,
    recommendation: SupportRecommendation,
) -> None:
    ownership_values = (
        (
            recommendation.workspace_id,
            command.workspace_id,
            "workspace",
        ),
        (
            recommendation.ticket_id,
            command.ticket_id,
            "ticket",
        ),
        (
            recommendation.agent_run_id,
            command.agent_run_id,
            "AgentRun",
        ),
    )

    for actual, expected, resource_name in ownership_values:
        if actual != expected:
            raise ValueError(
                f"Recommendation {resource_name} ownership does not match the persistence command."
            )

    if recommendation.created_at > command.persisted_at:
        raise ValueError("persisted_at must not precede the recommendation timestamp.")

    accepted_invocation = next(
        (
            invocation
            for invocation in command.invocations
            if invocation.id == recommendation.accepted_llm_invocation_id
        ),
        None,
    )

    if accepted_invocation is None:
        raise ValueError(
            "The recommendation must reference an invocation included in the persistence command."
        )

    final_invocation = command.invocations[-1]

    if accepted_invocation.id != final_invocation.id:
        raise ValueError("The accepted recommendation invocation must be the final invocation.")

    if accepted_invocation.status is not LLMInvocationStatus.SUCCEEDED:
        raise ValueError("The accepted recommendation invocation must have succeeded.")

    _validate_accepted_provenance(
        recommendation=recommendation,
        invocation=accepted_invocation,
    )


def _validate_accepted_provenance(
    *,
    recommendation: SupportRecommendation,
    invocation: LLMInvocation,
) -> None:
    provenance_values = (
        (
            recommendation.prompt_id,
            invocation.prompt_id,
            "prompt_id",
        ),
        (
            recommendation.prompt_version,
            invocation.prompt_version,
            "prompt_version",
        ),
        (
            recommendation.prompt_content_hash,
            invocation.prompt_content_hash,
            "prompt_content_hash",
        ),
        (
            recommendation.schema_version,
            invocation.schema_version,
            "schema_version",
        ),
        (
            recommendation.provider,
            invocation.provider,
            "provider",
        ),
        (
            recommendation.model,
            invocation.model,
            "model",
        ),
    )

    for actual, expected, field_name in provenance_values:
        if actual != expected:
            raise ValueError(
                f"Recommendation and accepted invocation must share {field_name} provenance."
            )


def _validate_citations(
    *,
    command: PersistSupportRecommendationCommand,
    recommendation: SupportRecommendation,
    citations: tuple[SupportRecommendationCitation, ...],
) -> None:
    expected_ordinals = tuple(
        range(
            1,
            len(citations) + 1,
        )
    )
    actual_ordinals = tuple(citation.ordinal for citation in citations)

    if actual_ordinals != expected_ordinals:
        raise ValueError("Citation ordinals must be contiguous, ordered, and start at one.")

    citation_ids = {citation.id for citation in citations}

    if len(citation_ids) != len(citations):
        raise ValueError("Citation identifiers must be unique.")

    chunk_ids = {citation.chunk_id for citation in citations}

    if len(chunk_ids) != len(citations):
        raise ValueError("A recommendation cannot cite the same chunk more than once.")

    for citation in citations:
        if citation.workspace_id != command.workspace_id:
            raise ValueError("Citation workspace ownership does not match the persistence command.")

        if citation.support_recommendation_id != recommendation.id:
            raise ValueError(
                "Citation recommendation ownership does not match the accepted recommendation."
            )

        if citation.created_at > command.persisted_at:
            raise ValueError("persisted_at must not precede a citation timestamp.")


def _validate_utc_timestamp(
    value: datetime,
    *,
    field_name: str,
) -> None:
    if value.tzinfo is None or value.utcoffset() != timedelta(0):
        raise ValueError(f"{field_name} must be a UTC-aware timestamp.")
