"""Graph adapter for approved sensitive tool execution."""

from dataclasses import dataclass

from supportops.agent_graph.domain.human_approved_state import (
    HumanApprovalCheckpointStatus,
    HumanApprovedSensitiveExecutionOutput,
    HumanApprovedSupportGraphState,
    validate_human_approved_support_state,
)
from supportops.agent_tools.application.sensitive_execution import (
    ExecuteApprovedTicketEscalation,
)
from supportops.modules.agent_runs.application.execution import (
    AgentRunExecutionContext,
)


@dataclass(frozen=True, slots=True)
class SensitiveToolExecutionNode:
    """Execute one approved escalation and project safe output."""

    executor: ExecuteApprovedTicketEscalation

    async def execute(
        self,
        state: HumanApprovedSupportGraphState,
        context: AgentRunExecutionContext,
    ) -> HumanApprovedSupportGraphState:
        """Run the grant-backed internal mutation exactly once."""

        snapshot = validate_human_approved_support_state(state)
        if snapshot.approval_request_id is None or snapshot.agent_tool_call_id is None:
            raise ValueError(
                "Sensitive execution requires approval identifiers.",
            )

        result = await self.executor.execute(
            context=context,
            approval_request_id=snapshot.approval_request_id,
            agent_tool_call_id=snapshot.agent_tool_call_id,
        )

        return snapshot.model_copy(
            update={
                "approval_status": (HumanApprovalCheckpointStatus.APPROVED),
                "current_error_code": None,
                "sensitive_execution_output": (
                    HumanApprovedSensitiveExecutionOutput.model_validate(
                        result.output.model_dump(mode="json"),
                    )
                ),
            },
        ).to_graph_state()
