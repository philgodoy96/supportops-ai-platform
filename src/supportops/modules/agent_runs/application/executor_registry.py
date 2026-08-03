"""Versioned dispatch for AgentRun workflow executors."""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from types import MappingProxyType

from supportops.modules.agent_runs.application.execution import (
    AgentRunExecutionContext,
    AgentRunExecutionResult,
    AgentRunExecutor,
    TerminalAgentRunExecutionError,
)
from supportops.modules.agent_runs.domain.models import (
    AGENT_RUN_WORKFLOW_NAME_MAX_LENGTH,
    AGENT_RUN_WORKFLOW_VERSION_MAX_LENGTH,
)


@dataclass(frozen=True, slots=True)
class AgentRunExecutorRegistration:
    """Bind one exact workflow name and version to an executor."""

    workflow_name: str
    workflow_version: str
    executor: AgentRunExecutor

    def __post_init__(self) -> None:
        _validate_workflow_identifier(
            self.workflow_name,
            field_name="workflow_name",
            maximum_length=AGENT_RUN_WORKFLOW_NAME_MAX_LENGTH,
        )
        _validate_workflow_identifier(
            self.workflow_version,
            field_name="workflow_version",
            maximum_length=AGENT_RUN_WORKFLOW_VERSION_MAX_LENGTH,
        )

        if not callable(
            getattr(
                self.executor,
                "execute",
                None,
            ),
        ):
            raise TypeError(
                "executor must implement the AgentRunExecutor contract.",
            )


class DuplicateAgentRunExecutorRegistrationError(
    ValueError,
):
    """Raised when an exact workflow and version are registered twice."""


class AgentRunExecutorRegistry:
    """Resolve and execute one exact versioned AgentRun workflow."""

    __slots__ = (
        "_executors",
        "_workflow_names",
    )

    _executors: Mapping[
        tuple[str, str],
        AgentRunExecutor,
    ]
    _workflow_names: frozenset[str]

    def __init__(
        self,
        registrations: Iterable[AgentRunExecutorRegistration],
    ) -> None:
        executors: dict[
            tuple[str, str],
            AgentRunExecutor,
        ] = {}

        for registration in registrations:
            key = (
                registration.workflow_name,
                registration.workflow_version,
            )

            if key in executors:
                raise (
                    DuplicateAgentRunExecutorRegistrationError(
                        "Duplicate AgentRun executor registration: "
                        f"{registration.workflow_name}/"
                        f"{registration.workflow_version}.",
                    )
                )

            executors[key] = registration.executor

        self._executors = MappingProxyType(executors)
        self._workflow_names = frozenset(workflow_name for workflow_name, _ in executors)

    def resolve(
        self,
        *,
        workflow_name: str,
        workflow_version: str,
    ) -> AgentRunExecutor:
        """Resolve one executor by exact workflow name and version."""

        executor = self._executors.get(
            (
                workflow_name,
                workflow_version,
            ),
        )

        if executor is not None:
            return executor

        if workflow_name not in self._workflow_names:
            raise TerminalAgentRunExecutionError(
                error_code="unsupported_workflow",
                error_summary=(
                    "The AgentRun workflow is not supported by the configured executor registry."
                ),
            )

        raise TerminalAgentRunExecutionError(
            error_code="unsupported_workflow_version",
            error_summary=(
                "The AgentRun workflow version is not supported "
                "by the configured executor registry."
            ),
        )

    async def execute(
        self,
        context: AgentRunExecutionContext,
    ) -> AgentRunExecutionResult:
        """Resolve and execute the exact workflow registered for the run."""

        executor = self.resolve(
            workflow_name=context.agent_run.workflow_name,
            workflow_version=(context.agent_run.workflow_version),
        )

        return await executor.execute(context)

    def __len__(self) -> int:
        """Return the number of exact workflow registrations."""

        return len(self._executors)


def _validate_workflow_identifier(
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
