"""Application contracts for executing a claimed AgentRun."""

from dataclasses import dataclass
from typing import Protocol

from supportops.modules.agent_runs.domain.models import (
    AGENT_RUN_ERROR_CODE_MAX_LENGTH,
    AGENT_RUN_ERROR_SUMMARY_MAX_LENGTH,
    AgentRun,
    AgentRunAttempt,
    AgentRunStatus,
)
from supportops.modules.tickets.domain.models import Ticket


@dataclass(frozen=True, slots=True)
class AgentRunExecutionContext:
    """Claimed AgentRun, active attempt, and ticket for one executor invocation."""

    agent_run: AgentRun
    attempt: AgentRunAttempt
    ticket: Ticket

    def __post_init__(self) -> None:
        if self.agent_run.status is not AgentRunStatus.RUNNING:
            raise ValueError(
                "AgentRun execution requires a running AgentRun.",
            )

        if self.agent_run.lease_token is None:
            raise ValueError(
                "AgentRun execution requires an active lease token.",
            )

        if self.agent_run.ticket_id != self.ticket.id:
            raise ValueError(
                "AgentRun and ticket must reference the same ticket.",
            )

        if self.agent_run.workspace_id != self.ticket.workspace_id:
            raise ValueError(
                "AgentRun and ticket must belong to the same workspace.",
            )

        if self.attempt.agent_run_id != self.agent_run.id:
            raise ValueError(
                "AgentRun execution attempt must reference the same AgentRun.",
            )

        if self.attempt.attempt_number != self.agent_run.attempt_count:
            raise ValueError(
                "AgentRun execution attempt number must match the AgentRun.",
            )

        if self.attempt.lease_token != self.agent_run.lease_token:
            raise ValueError(
                "AgentRun execution attempt must share the AgentRun lease token.",
            )

        if self.attempt.worker_id != self.agent_run.lease_owner:
            raise ValueError(
                "AgentRun execution attempt must share the AgentRun worker.",
            )

        if self.attempt.is_finished:
            raise ValueError(
                "AgentRun execution requires an active attempt.",
            )


class AgentRunExecutor(Protocol):
    """Execute one claimed AgentRun outside its claim transaction."""

    async def execute(
        self,
        context: AgentRunExecutionContext,
    ) -> None:
        """Execute the claimed workflow or raise a typed execution error."""

        ...


class AgentRunExecutionError(Exception):
    """Base class for safe, classified executor failures."""

    def __init__(
        self,
        *,
        error_code: str,
        error_summary: str,
    ) -> None:
        _validate_error_text(
            error_code,
            field_name="error_code",
            maximum_length=AGENT_RUN_ERROR_CODE_MAX_LENGTH,
        )
        _validate_error_text(
            error_summary,
            field_name="error_summary",
            maximum_length=AGENT_RUN_ERROR_SUMMARY_MAX_LENGTH,
        )

        self.error_code = error_code
        self.error_summary = error_summary

        super().__init__(error_summary)


class RetryableAgentRunExecutionError(AgentRunExecutionError):
    """Failure that may be retried while attempt budget remains."""


class TerminalAgentRunExecutionError(AgentRunExecutionError):
    """Failure that must terminate the AgentRun immediately."""


def _validate_error_text(
    value: str,
    *,
    field_name: str,
    maximum_length: int,
) -> None:
    if not value:
        raise ValueError(f"{field_name} is required.")

    if value != value.strip():
        raise ValueError(
            f"{field_name} must not contain surrounding whitespace.",
        )

    if len(value) > maximum_length:
        raise ValueError(
            f"{field_name} exceeds the maximum length.",
        )
