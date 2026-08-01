"""Explicit immutable lookup registry for versioned prompts."""

from collections.abc import Iterable, Mapping
from types import MappingProxyType

from supportops.ai.prompts.definitions import PromptDefinition

PromptKey = tuple[str, int]


class DuplicatePromptDefinitionError(ValueError):
    """Raised when a prompt ID and version are registered more than once."""


class PromptDefinitionNotFoundError(LookupError):
    """Raised when an explicitly requested prompt version is unsupported."""


class PromptRegistry:
    """Immutable registry keyed by explicit prompt ID and version."""

    __slots__ = ("_definitions",)

    _definitions: Mapping[PromptKey, PromptDefinition]

    def __init__(
        self,
        definitions: Iterable[PromptDefinition],
    ) -> None:
        definitions_by_key: dict[PromptKey, PromptDefinition] = {}

        for definition in definitions:
            key = (
                definition.prompt_id,
                definition.version,
            )

            if key in definitions_by_key:
                raise DuplicatePromptDefinitionError(
                    "Duplicate prompt definition: "
                    f"{definition.prompt_id} version {definition.version}.",
                )

            definitions_by_key[key] = definition

        self._definitions = MappingProxyType(definitions_by_key)

    def get(
        self,
        *,
        prompt_id: str,
        version: int,
    ) -> PromptDefinition:
        """Return one explicitly requested prompt definition."""

        key = (
            prompt_id,
            version,
        )

        try:
            return self._definitions[key]
        except KeyError as error:
            raise PromptDefinitionNotFoundError(
                f"Unsupported prompt: {prompt_id} version {version}.",
            ) from error

    def __len__(self) -> int:
        """Return the number of registered prompt versions."""

        return len(self._definitions)
