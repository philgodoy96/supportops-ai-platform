"""Immutable runtime bindings for controlled read-only tools."""

from collections.abc import (
    Callable,
    Iterable,
    Mapping,
)
from dataclasses import dataclass
from types import MappingProxyType
from typing import Protocol

from pydantic import JsonValue

from supportops.agent_tools.domain.contracts import (
    StrictToolSchema,
    ToolAuditPolicy,
    ToolDefinition,
    ToolSafetyLevel,
)
from supportops.agent_tools.domain.errors import (
    ToolDuplicateDefinitionError,
    ToolNotFoundError,
    ToolSafetyViolationError,
    ToolVersionNotFoundError,
)

type ToolAuditProjector = Callable[
    [StrictToolSchema],
    Mapping[str, JsonValue],
]


class ToolHandler(Protocol):
    """Application-owned handler for one controlled tool."""

    async def execute(
        self,
        context: object,
        arguments: StrictToolSchema,
    ) -> object:
        """Execute one read-only operation."""

        ...


@dataclass(frozen=True, slots=True)
class ExecutableToolBinding:
    """Bind immutable policy metadata to one runtime handler."""

    definition: ToolDefinition
    handler: ToolHandler
    safe_input_projector: ToolAuditProjector
    safe_output_projector: ToolAuditProjector

    def __post_init__(self) -> None:
        if self.definition.safety_level is not ToolSafetyLevel.READ_ONLY:
            raise ToolSafetyViolationError()

        if self.definition.audit_policy is not ToolAuditPolicy.SAFE_PROJECTION:
            raise ValueError("Executable tools require safe audit projections.")

        handler_execute = getattr(
            self.handler,
            "execute",
            None,
        )

        if not callable(handler_execute):
            raise TypeError("handler must expose an asynchronous execute method.")

        if not callable(self.safe_input_projector):
            raise TypeError("safe_input_projector must be callable.")

        if not callable(self.safe_output_projector):
            raise TypeError("safe_output_projector must be callable.")


class ExecutableToolRegistry:
    """Immutable exact-version registry for executable bindings."""

    def __init__(
        self,
        bindings: Iterable[ExecutableToolBinding],
    ) -> None:
        ordered_bindings = tuple(
            sorted(
                bindings,
                key=lambda binding: (
                    binding.definition.name,
                    binding.definition.version,
                ),
            )
        )
        bindings_by_key: dict[
            tuple[str, int],
            ExecutableToolBinding,
        ] = {}
        versions_by_name: dict[str, set[int]] = {}

        for binding in ordered_bindings:
            key = (
                binding.definition.name,
                binding.definition.version,
            )

            if key in bindings_by_key:
                raise ToolDuplicateDefinitionError()

            bindings_by_key[key] = binding
            versions_by_name.setdefault(
                binding.definition.name,
                set(),
            ).add(binding.definition.version)

        self._bindings = ordered_bindings
        self._bindings_by_key: Mapping[
            tuple[str, int],
            ExecutableToolBinding,
        ] = MappingProxyType(bindings_by_key)
        self._versions_by_name: Mapping[
            str,
            frozenset[int],
        ] = MappingProxyType(
            {name: frozenset(versions) for name, versions in versions_by_name.items()}
        )

    @property
    def bindings(
        self,
    ) -> tuple[ExecutableToolBinding, ...]:
        """Return bindings in deterministic identity order."""

        return self._bindings

    @property
    def definitions(
        self,
    ) -> tuple[ToolDefinition, ...]:
        """Return immutable definitions in deterministic order."""

        return tuple(binding.definition for binding in self._bindings)

    def lookup(
        self,
        *,
        name: str,
        version: int,
    ) -> ExecutableToolBinding:
        """Resolve one exact executable tool identity."""

        binding = self._bindings_by_key.get(
            (
                name,
                version,
            )
        )

        if binding is not None:
            return binding

        if name not in self._versions_by_name:
            raise ToolNotFoundError()

        raise ToolVersionNotFoundError()
