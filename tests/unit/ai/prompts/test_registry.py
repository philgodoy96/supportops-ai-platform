"""Unit tests for explicit versioned prompt lookup."""

import pytest

from supportops.ai.prompts.definitions import PromptDefinition
from supportops.ai.prompts.registry import (
    DuplicatePromptDefinitionError,
    PromptDefinitionNotFoundError,
    PromptRegistry,
)


def _definition(
    *,
    prompt_id: str = "example-prompt",
    version: int = 1,
) -> PromptDefinition:
    return PromptDefinition(
        prompt_id=prompt_id,
        version=version,
        description="Example prompt definition.",
        instructions="Follow the application-owned contract.",
        input_template="Input: {{payload}}",
        output_schema_id="example-schema-v1",
    )


def test_returns_explicit_prompt_version() -> None:
    version_one = _definition(version=1)
    version_two = _definition(version=2)
    registry = PromptRegistry(
        (
            version_one,
            version_two,
        ),
    )

    result = registry.get(
        prompt_id="example-prompt",
        version=1,
    )

    assert result is version_one


def test_rejects_duplicate_prompt_id_and_version() -> None:
    with pytest.raises(
        DuplicatePromptDefinitionError,
        match="example-prompt version 1",
    ):
        PromptRegistry(
            (
                _definition(),
                _definition(),
            ),
        )


def test_allows_same_prompt_id_with_different_versions() -> None:
    registry = PromptRegistry(
        (
            _definition(version=1),
            _definition(version=2),
        ),
    )

    assert len(registry) == 2


def test_allows_same_version_for_different_prompt_ids() -> None:
    registry = PromptRegistry(
        (
            _definition(prompt_id="first-prompt"),
            _definition(prompt_id="second-prompt"),
        ),
    )

    assert len(registry) == 2


def test_missing_prompt_version_fails_explicitly() -> None:
    registry = PromptRegistry(
        (_definition(version=1),),
    )

    with pytest.raises(
        PromptDefinitionNotFoundError,
        match="example-prompt version 2",
    ):
        registry.get(
            prompt_id="example-prompt",
            version=2,
        )
