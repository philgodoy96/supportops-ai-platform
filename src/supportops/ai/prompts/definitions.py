"""Immutable definitions for versioned application-owned prompts."""

import hashlib
import json
from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class PromptDefinition:
    """Immutable versioned prompt definition stored in the repository."""

    prompt_id: str
    version: int
    description: str
    instructions: str
    input_template: str
    output_schema_id: str
    content_hash: str = field(init=False)

    def __post_init__(self) -> None:
        _validate_required_text(self.prompt_id, field_name="prompt_id")
        _validate_required_text(self.description, field_name="description")
        _validate_required_text(self.instructions, field_name="instructions")
        _validate_required_text(
            self.input_template,
            field_name="input_template",
        )
        _validate_required_text(
            self.output_schema_id,
            field_name="output_schema_id",
        )

        if self.version <= 0:
            raise ValueError("version must be positive.")

        object.__setattr__(
            self,
            "content_hash",
            _compute_content_hash(self),
        )


@dataclass(frozen=True, slots=True)
class RenderedPrompt:
    """Prompt definition paired with one rendered untrusted input."""

    definition: PromptDefinition
    input: str

    def __post_init__(self) -> None:
        _validate_required_text(self.input, field_name="input")

    @property
    def instructions(self) -> str:
        """Return the immutable instructions associated with the prompt."""

        return self.definition.instructions


def _compute_content_hash(definition: PromptDefinition) -> str:
    canonical_content = json.dumps(
        {
            "description": definition.description,
            "input_template": definition.input_template,
            "instructions": definition.instructions,
            "output_schema_id": definition.output_schema_id,
            "prompt_id": definition.prompt_id,
            "version": definition.version,
        },
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )

    return hashlib.sha256(
        canonical_content.encode("utf-8"),
    ).hexdigest()


def _validate_required_text(
    value: str,
    *,
    field_name: str,
) -> None:
    if not value:
        raise ValueError(f"{field_name} is required.")

    if value != value.strip():
        raise ValueError(
            f"{field_name} must not contain surrounding whitespace.",
        )
