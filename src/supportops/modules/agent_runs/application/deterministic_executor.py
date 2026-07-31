"""Deterministic baseline executor for initial ticket processing."""

from supportops.modules.agent_runs.application.execution import (
    AgentRunExecutionContext,
    TerminalAgentRunExecutionError,
)
from supportops.modules.agent_runs.domain.models import (
    DETERMINISTIC_BASELINE_WORKFLOW_VERSION,
    INITIAL_TICKET_PROCESSING_TRIGGER_KEY,
    INITIAL_TICKET_PROCESSING_WORKFLOW_NAME,
)


class DeterministicTicketProcessingExecutor:
    """Validate and process the baseline ticket workflow deterministically."""

    async def execute(
        self,
        context: AgentRunExecutionContext,
    ) -> None:
        """Complete the supported baseline workflow without external I/O."""

        run = context.agent_run

        if run.workflow_name != INITIAL_TICKET_PROCESSING_WORKFLOW_NAME:
            raise TerminalAgentRunExecutionError(
                error_code="unsupported_workflow",
                error_summary=(
                    "The AgentRun workflow is not supported by the configured executor."
                ),
            )

        if run.workflow_version != DETERMINISTIC_BASELINE_WORKFLOW_VERSION:
            raise TerminalAgentRunExecutionError(
                error_code="unsupported_workflow_version",
                error_summary=(
                    "The AgentRun workflow version is not supported by the configured executor."
                ),
            )

        if run.trigger_key != INITIAL_TICKET_PROCESSING_TRIGGER_KEY:
            raise TerminalAgentRunExecutionError(
                error_code="unsupported_trigger",
                error_summary=("The AgentRun trigger is not supported by the configured executor."),
            )
