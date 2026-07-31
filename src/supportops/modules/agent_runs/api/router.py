"""Workspace-scoped AgentRun inspection HTTP routes."""

from uuid import UUID

from fastapi import APIRouter

from supportops.modules.agent_runs.api.dependencies import (
    GetAgentRunDependency,
    ListAgentRunAttemptsDependency,
)
from supportops.modules.agent_runs.api.schemas import (
    AgentRunAttemptListResponse,
    AgentRunAttemptResponse,
    AgentRunResponse,
)

router = APIRouter(
    prefix="/workspaces/{workspace_id}/agent-runs",
    tags=["agent-runs"],
)


@router.get(
    "/{agent_run_id}",
    response_model=AgentRunResponse,
)
async def get_agent_run(
    workspace_id: UUID,
    agent_run_id: UUID,
    service: GetAgentRunDependency,
) -> AgentRunResponse:
    """Retrieve an AgentRun only through its workspace boundary."""

    agent_run = await service.execute(
        workspace_id=workspace_id,
        agent_run_id=agent_run_id,
    )

    return AgentRunResponse.from_domain(agent_run)


@router.get(
    "/{agent_run_id}/attempts",
    response_model=AgentRunAttemptListResponse,
)
async def list_agent_run_attempts(
    workspace_id: UUID,
    agent_run_id: UUID,
    service: ListAgentRunAttemptsDependency,
) -> AgentRunAttemptListResponse:
    """List the ordered attempt history for one scoped AgentRun."""

    attempts = await service.execute(
        workspace_id=workspace_id,
        agent_run_id=agent_run_id,
    )

    return AgentRunAttemptListResponse(
        items=[AgentRunAttemptResponse.from_domain(attempt) for attempt in attempts],
    )
