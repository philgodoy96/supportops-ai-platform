"""Unit tests for immutable prompt definitions."""

from collections.abc import Callable
from dataclasses import FrozenInstanceError, replace

import pytest

from supportops.ai.prompts.definitions import (
    PromptDefinition,
    RenderedPrompt,
)


def _definition() -> PromptDefinition:
    return PromptDefinition(
        prompt_id="example-prompt",
        version=1,
        description="Example prompt definition.",
        instructions="Follow the application-owned contract.",
        input_template="Input: {{payload}}",
        output_schema_id="example-schema-v1",
    )


def test_equal_prompt_content_produces_equal_hashes() -> None:
    first = _definition()
    second = _definition()

    assert first.content_hash == second.content_hash
    assert len(first.content_hash) == 64


@pytest.mark.parametrize(
    "changed_definition",
    [
        lambda: replace(
            _definition(),
            description="Changed description.",
        ),
        lambda: replace(
            _definition(),
            instructions="Changed instructions.",
        ),
        lambda: replace(
            _definition(),
            input_template="Changed: {{payload}}",
        ),
        lambda: replace(
            _definition(),
            output_schema_id="example-schema-v2",
        ),
        lambda: replace(
            _definition(),
            version=2,
        ),
    ],
)
def test_behavioral_prompt_changes_produce_new_hashes(
    changed_definition: Callable[[], PromptDefinition],
) -> None:
    changed = changed_definition()

    assert changed.content_hash != _definition().content_hash


@pytest.mark.parametrize(
    "invalid_definition",
    [
        lambda: replace(_definition(), prompt_id=""),
        lambda: replace(_definition(), description=" "),
        lambda: replace(_definition(), instructions=""),
        lambda: replace(_definition(), input_template=""),
        lambda: replace(_definition(), output_schema_id=""),
        lambda: replace(_definition(), version=0),
    ],
)
def test_rejects_invalid_prompt_definitions(
    invalid_definition: Callable[[], PromptDefinition],
) -> None:
    with pytest.raises(ValueError):
        invalid_definition()


def test_prompt_definition_is_immutable() -> None:
    definition = _definition()

    with pytest.raises(FrozenInstanceError):
        definition.version = 2  # type: ignore[misc]


def test_rendered_prompt_exposes_definition_instructions() -> None:
    definition = _definition()
    rendered = RenderedPrompt(
        definition=definition,
        input="Input: example",
    )

    assert rendered.instructions == definition.instructions
    assert rendered.definition.content_hash == definition.content_hash
