"""Immutable proposal bindings for sensitive-write tools."""

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from types import MappingProxyType

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

type SensitiveInputProjector = Callable[
    [StrictToolSchema],
    Mapping[str, JsonValue],
]
type ApprovalReasonProjector = Callable[[StrictToolSchema], str]


@dataclass(frozen=True, slots=True)
class SensitiveToolBinding:
    """Bind proposal policy to one non-executable sensitive tool."""

    definition: ToolDefinition
    safe_input_projector: SensitiveInputProjector
    approval_reason_projector: ApprovalReasonProjector

    def __post_init__(self) -> None:
        if self.definition.safety_level is not ToolSafetyLevel.SENSITIVE_WRITE:
            raise ToolSafetyViolationError()

        if self.definition.audit_policy is not ToolAuditPolicy.SAFE_PROJECTION:
            raise ValueError(
                "Sensitive tools require safe audit projections.",
            )

        if not callable(self.safe_input_projector):
            raise TypeError(
                "safe_input_projector must be callable.",
            )
        if not callable(self.approval_reason_projector):
            raise TypeError(
                "approval_reason_projector must be callable.",
            )


class SensitiveToolRegistry:
    """Immutable exact-version registry for sensitive proposals."""

    def __init__(
        self,
        bindings: Iterable[SensitiveToolBinding],
    ) -> None:
        ordered_bindings = tuple(
            sorted(
                bindings,
                key=lambda binding: (
                    binding.definition.name,
                    binding.definition.version,
                ),
            ),
        )
        bindings_by_key: dict[
            tuple[str, int],
            SensitiveToolBinding,
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
            SensitiveToolBinding,
        ] = MappingProxyType(bindings_by_key)
        self._versions_by_name: Mapping[
            str,
            frozenset[int],
        ] = MappingProxyType(
            {name: frozenset(versions) for name, versions in versions_by_name.items()},
        )

    @property
    def bindings(self) -> tuple[SensitiveToolBinding, ...]:
        """Return bindings in deterministic identity order."""

        return self._bindings

    @property
    def definitions(self) -> tuple[ToolDefinition, ...]:
        """Return model-visible definitions in deterministic order."""

        return tuple(binding.definition for binding in self._bindings)

    def lookup(
        self,
        *,
        name: str,
        version: int,
    ) -> SensitiveToolBinding:
        """Resolve one exact sensitive proposal binding."""

        binding = self._bindings_by_key.get((name, version))
        if binding is not None:
            return binding

        if name not in self._versions_by_name:
            raise ToolNotFoundError()

        raise ToolVersionNotFoundError()
