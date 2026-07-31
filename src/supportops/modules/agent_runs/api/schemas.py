"""HTTP response schemas for AgentRun inspection."""

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel

from supportops.modules.agent_runs.domain.models import (
    AgentRun,
    AgentRunAttempt,
    AgentRunAttemptOutcome,
    AgentRunStatus,
)


class AgentRunWorkflowResponse(BaseModel):
    """Public workflow identity for an AgentRun."""

    name: str
    version: str
    trigger_key: str


class AgentRunErrorResponse(BaseModel):
    """Safe persisted processing error metadata."""

    code: str
    summary: str


class AgentRunResponse(BaseModel):
    """Public workspace-scoped AgentRun representation."""

    id: UUID
    workspace_id: UUID
    ticket_id: UUID
    status: AgentRunStatus
    workflow: AgentRunWorkflowResponse
    attempt_count: int
    max_attempts: int
    available_at: datetime
    first_started_at: datetime | None
    completed_at: datetime | None
    last_error: AgentRunErrorResponse | None
    correlation_id: UUID
    created_at: datetime
    updated_at: datetime

    @classmethod
    def from_domain(
        cls,
        agent_run: AgentRun,
    ) -> "AgentRunResponse":
        """Project an AgentRun without exposing lease ownership secrets."""

        last_error = None

        if agent_run.last_error_code is not None and agent_run.last_error_summary is not None:
            last_error = AgentRunErrorResponse(
                code=agent_run.last_error_code,
                summary=agent_run.last_error_summary,
            )

        return cls(
            id=agent_run.id,
            workspace_id=agent_run.workspace_id,
            ticket_id=agent_run.ticket_id,
            status=agent_run.status,
            workflow=AgentRunWorkflowResponse(
                name=agent_run.workflow_name,
                version=agent_run.workflow_version,
                trigger_key=agent_run.trigger_key,
            ),
            attempt_count=agent_run.attempt_count,
            max_attempts=agent_run.max_attempts,
            available_at=agent_run.available_at,
            first_started_at=agent_run.first_started_at,
            completed_at=agent_run.completed_at,
            last_error=last_error,
            correlation_id=agent_run.correlation_id,
            created_at=agent_run.created_at,
            updated_at=agent_run.updated_at,
        )


class AgentRunAttemptResponse(BaseModel):
    """Public representation of one AgentRun execution attempt."""

    id: UUID
    attempt_number: int
    worker_id: str
    started_at: datetime
    finished_at: datetime | None
    outcome: AgentRunAttemptOutcome | None
    error: AgentRunErrorResponse | None

    @classmethod
    def from_domain(
        cls,
        attempt: AgentRunAttempt,
    ) -> "AgentRunAttemptResponse":
        """Project an attempt without exposing fencing identifiers."""

        error = None

        if attempt.error_code is not None and attempt.error_summary is not None:
            error = AgentRunErrorResponse(
                code=attempt.error_code,
                summary=attempt.error_summary,
            )

        return cls(
            id=attempt.id,
            attempt_number=attempt.attempt_number,
            worker_id=attempt.worker_id,
            started_at=attempt.started_at,
            finished_at=attempt.finished_at,
            outcome=attempt.outcome,
            error=error,
        )


class AgentRunAttemptListResponse(BaseModel):
    """Ordered execution-attempt history."""

    items: list[AgentRunAttemptResponse]
