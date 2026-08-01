"""Fenced persistence contracts for ticket-classification execution."""

from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import StrEnum
from typing import Protocol
from uuid import UUID

from supportops.ai.gateway.results import LLMInvocationStatus
from supportops.modules.ticket_classifications.domain.models import (
    LLMInvocation,
    TicketClassification,
)


class ClassificationPersistenceResult(StrEnum):
    """Outcome of one fenced classification persistence operation."""

    APPLIED = "applied"
    ALREADY_CLASSIFIED = "already_classified"
    ALREADY_RECORDED = "already_recorded"
    LEASE_LOST = "lease_lost"


@dataclass(frozen=True, slots=True)
class PersistClassificationExecutionCommand:
    """Persist one execution result while the AgentRun lease is valid."""

    workspace_id: UUID
    ticket_id: UUID
    agent_run_id: UUID
    agent_run_attempt_id: UUID
    lease_token: UUID
    persisted_at: datetime
    invocations: tuple[LLMInvocation, ...]
    classification: TicketClassification | None

    def __post_init__(self) -> None:
        _validate_utc_timestamp(
            self.persisted_at,
            field_name="persisted_at",
        )

        if not self.invocations:
            raise ValueError(
                "At least one LLM invocation is required.",
            )

        expected_sequences = tuple(
            range(
                1,
                len(self.invocations) + 1,
            ),
        )
        actual_sequences = tuple(invocation.invocation_sequence for invocation in self.invocations)

        if actual_sequences != expected_sequences:
            raise ValueError(
                "Invocation sequences must be contiguous, ordered, and start at one.",
            )

        invocation_ids = {invocation.id for invocation in self.invocations}
        if len(invocation_ids) != len(self.invocations):
            raise ValueError(
                "Invocation identifiers must be unique.",
            )

        for invocation in self.invocations:
            _validate_invocation_ownership(
                command=self,
                invocation=invocation,
            )

        if self.classification is None:
            if any(
                invocation.status is LLMInvocationStatus.SUCCEEDED
                for invocation in self.invocations
            ):
                raise ValueError(
                    "A successful invocation requires an accepted classification.",
                )
            return

        _validate_classification_ownership(
            command=self,
            classification=self.classification,
        )

        successful_invocations = tuple(
            invocation
            for invocation in self.invocations
            if invocation.status is LLMInvocationStatus.SUCCEEDED
        )

        if len(successful_invocations) != 1:
            raise ValueError(
                "An accepted classification requires exactly one successful invocation.",
            )

        accepted_invocation = successful_invocations[0]

        if accepted_invocation.id != self.invocations[-1].id:
            raise ValueError(
                "The successful invocation must be the final invocation.",
            )

        if self.classification.accepted_llm_invocation_id != accepted_invocation.id:
            raise ValueError(
                "The classification must reference the successful invocation.",
            )

        _validate_accepted_provenance(
            classification=self.classification,
            invocation=accepted_invocation,
        )


class ClassificationExecutionRepository(Protocol):
    """Atomic persistence boundary for one classified AgentRun attempt."""

    async def persist_fenced(
        self,
        command: PersistClassificationExecutionCommand,
    ) -> ClassificationPersistenceResult:
        """Persist invocations and optional classification under lease fencing."""

        ...


def _validate_invocation_ownership(
    *,
    command: PersistClassificationExecutionCommand,
    invocation: LLMInvocation,
) -> None:
    expected_values = (
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

    for actual, expected, resource_name in expected_values:
        if actual != expected:
            raise ValueError(
                f"Invocation {resource_name} ownership does not match the persistence command.",
            )


def _validate_classification_ownership(
    *,
    command: PersistClassificationExecutionCommand,
    classification: TicketClassification,
) -> None:
    expected_values = (
        (
            classification.workspace_id,
            command.workspace_id,
            "workspace",
        ),
        (
            classification.ticket_id,
            command.ticket_id,
            "ticket",
        ),
        (
            classification.agent_run_id,
            command.agent_run_id,
            "AgentRun",
        ),
    )

    for actual, expected, resource_name in expected_values:
        if actual != expected:
            raise ValueError(
                f"Classification {resource_name} ownership does not match the persistence command.",
            )


def _validate_accepted_provenance(
    *,
    classification: TicketClassification,
    invocation: LLMInvocation,
) -> None:
    provenance_values = (
        (
            classification.prompt_id,
            invocation.prompt_id,
            "prompt_id",
        ),
        (
            classification.prompt_version,
            invocation.prompt_version,
            "prompt_version",
        ),
        (
            classification.prompt_content_hash,
            invocation.prompt_content_hash,
            "prompt_content_hash",
        ),
        (
            classification.schema_version,
            invocation.schema_version,
            "schema_version",
        ),
        (
            classification.provider,
            invocation.provider,
            "provider",
        ),
        (
            classification.model,
            invocation.model,
            "model",
        ),
    )

    for actual, expected, field_name in provenance_values:
        if actual != expected:
            raise ValueError(
                f"Classification and accepted invocation must share {field_name} provenance.",
            )


def _validate_utc_timestamp(
    value: datetime,
    *,
    field_name: str,
) -> None:
    if value.tzinfo is None or value.utcoffset() != timedelta(0):
        raise ValueError(
            f"{field_name} must be a UTC-aware timestamp.",
        )
