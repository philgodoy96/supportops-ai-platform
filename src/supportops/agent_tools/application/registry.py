"""Immutable application-owned controlled tool registry."""

from collections.abc import Iterable
from types import MappingProxyType

from supportops.agent_tools.domain.contracts import (
    ProviderToolDefinition,
    ToolDefinition,
    ToolReference,
    ToolSafetyLevel,
)
from supportops.agent_tools.domain.errors import (
    ToolDuplicateDefinitionError,
    ToolNotFoundError,
    ToolProviderSelectionError,
    ToolSafetyViolationError,
    ToolVersionNotFoundError,
)

_READ_ONLY_SAFETY_LEVELS = frozenset(
    {
        ToolSafetyLevel.READ_ONLY,
    }
)


class ToolRegistry:
    """Provide immutable exact-version lookup and safe projection."""

    __slots__ = (
        "_definitions",
        "_definitions_by_key",
        "_versions_by_name",
    )

    def __init__(
        self,
        definitions: Iterable[ToolDefinition],
    ) -> None:
        ordered_definitions = tuple(
            sorted(
                definitions,
                key=lambda definition: (
                    definition.name,
                    definition.version,
                ),
            )
        )
        definitions_by_key: dict[
            tuple[str, int],
            ToolDefinition,
        ] = {}
        versions_by_name: dict[str, set[int]] = {}

        for definition in ordered_definitions:
            key = (
                definition.name,
                definition.version,
            )

            if key in definitions_by_key:
                raise ToolDuplicateDefinitionError()

            definitions_by_key[key] = definition
            versions_by_name.setdefault(
                definition.name,
                set(),
            ).add(definition.version)

        self._definitions = ordered_definitions
        self._definitions_by_key = MappingProxyType(definitions_by_key)
        self._versions_by_name = MappingProxyType(
            {name: frozenset(versions) for name, versions in versions_by_name.items()}
        )

    @property
    def definitions(self) -> tuple[ToolDefinition, ...]:
        """Return immutable definitions in deterministic order."""

        return self._definitions

    def lookup(
        self,
        *,
        name: str,
        version: int,
    ) -> ToolDefinition:
        """Resolve an exact registered tool identity."""

        definition = self._definitions_by_key.get(
            (
                name,
                version,
            )
        )

        if definition is not None:
            return definition

        if name not in self._versions_by_name:
            raise ToolNotFoundError()

        raise ToolVersionNotFoundError()

    def project_provider_definitions(
        self,
        references: Iterable[ToolReference],
        *,
        allowed_safety_levels: frozenset[ToolSafetyLevel] = _READ_ONLY_SAFETY_LEVELS,
    ) -> tuple[ProviderToolDefinition, ...]:
        """Expose only explicitly selected and permitted tools."""

        selected_references = tuple(
            sorted(
                references,
                key=lambda reference: (
                    reference.name,
                    reference.version,
                ),
            )
        )
        selected_keys: set[tuple[str, int]] = set()
        selected_names: set[str] = set()
        provider_definitions: list[ProviderToolDefinition] = []

        for reference in selected_references:
            key = (
                reference.name,
                reference.version,
            )

            if key in selected_keys:
                raise ToolProviderSelectionError()

            if reference.name in selected_names:
                raise ToolProviderSelectionError()

            definition = self.lookup(
                name=reference.name,
                version=reference.version,
            )

            if definition.safety_level not in allowed_safety_levels:
                raise ToolSafetyViolationError()

            selected_keys.add(key)
            selected_names.add(reference.name)
            provider_definitions.append(definition.to_provider_definition())

        return tuple(provider_definitions)
