"""Unit tests for deterministic tool-call fingerprints."""

import json
from typing import Annotated
from uuid import UUID

import pytest
from pydantic import Field

from supportops.agent_tools.domain.contracts import (
    StrictToolSchema,
    ToolAuditPolicy,
    ToolDefinition,
    ToolFailurePolicy,
    ToolSafetyLevel,
)
from supportops.agent_tools.domain.errors import (
    ToolInputValidationError,
)
from supportops.agent_tools.domain.fingerprints import (
    canonicalize_validated_tool_arguments,
    create_tool_call_fingerprint,
)


class FingerprintInput(StrictToolSchema):
    """Validated input used by fingerprint tests."""

    top_k: Annotated[
        int,
        Field(
            strict=True,
            ge=1,
            le=10,
        ),
    ]
    document_ids: tuple[UUID, ...] | None


class OtherInput(StrictToolSchema):
    """Different schema used to verify type enforcement."""

    service_name: str


class FingerprintOutput(StrictToolSchema):
    """Synthetic tool output contract."""

    result_count: Annotated[
        int,
        Field(
            strict=True,
            ge=0,
        ),
    ]


def create_definition(
    *,
    name: str = "search_knowledge",
    version: int = 1,
) -> ToolDefinition:
    """Create a valid fingerprint test definition."""

    return ToolDefinition(
        name=name,
        version=version,
        description="Search active workspace knowledge.",
        input_schema=FingerprintInput,
        output_schema=FingerprintOutput,
        safety_level=ToolSafetyLevel.READ_ONLY,
        timeout_seconds=15,
        failure_policy=(ToolFailurePolicy.RETRY_AGENT_RUN),
        audit_policy=ToolAuditPolicy.SAFE_PROJECTION,
    )


def test_canonical_arguments_use_stable_json() -> None:
    first_document_id = UUID("11111111-1111-4111-8111-111111111111")
    second_document_id = UUID("22222222-2222-4222-8222-222222222222")
    arguments = FingerprintInput(
        top_k=5,
        document_ids=(
            first_document_id,
            second_document_id,
        ),
    )

    canonical_arguments = canonicalize_validated_tool_arguments(arguments)

    assert canonical_arguments == json.dumps(
        {
            "document_ids": [
                str(first_document_id),
                str(second_document_id),
            ],
            "top_k": 5,
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def test_semantically_identical_arguments_share_fingerprint() -> None:
    document_id = UUID("11111111-1111-4111-8111-111111111111")
    definition = create_definition()

    first_arguments = FingerprintInput.model_validate(
        {
            "top_k": 5,
            "document_ids": [str(document_id)],
        }
    )
    second_arguments = FingerprintInput.model_validate(
        {
            "document_ids": [str(document_id)],
            "top_k": 5,
        }
    )

    first_fingerprint = create_tool_call_fingerprint(
        definition=definition,
        arguments=first_arguments,
    )
    second_fingerprint = create_tool_call_fingerprint(
        definition=definition,
        arguments=second_arguments,
    )

    assert first_fingerprint == second_fingerprint
    assert len(first_fingerprint) == 64
    assert first_fingerprint.isascii()
    assert first_fingerprint.islower()


def test_argument_change_changes_fingerprint() -> None:
    definition = create_definition()

    first_fingerprint = create_tool_call_fingerprint(
        definition=definition,
        arguments=FingerprintInput(
            top_k=3,
            document_ids=None,
        ),
    )
    second_fingerprint = create_tool_call_fingerprint(
        definition=definition,
        arguments=FingerprintInput(
            top_k=5,
            document_ids=None,
        ),
    )

    assert first_fingerprint != second_fingerprint


def test_tool_version_changes_fingerprint() -> None:
    arguments = FingerprintInput(
        top_k=5,
        document_ids=None,
    )

    version_one = create_tool_call_fingerprint(
        definition=create_definition(version=1),
        arguments=arguments,
    )
    version_two = create_tool_call_fingerprint(
        definition=create_definition(version=2),
        arguments=arguments,
    )

    assert version_one != version_two


def test_tool_name_changes_fingerprint() -> None:
    arguments = FingerprintInput(
        top_k=5,
        document_ids=None,
    )

    knowledge_fingerprint = create_tool_call_fingerprint(
        definition=create_definition(name="search_knowledge"),
        arguments=arguments,
    )
    alternate_fingerprint = create_tool_call_fingerprint(
        definition=create_definition(name="search_alternate"),
        arguments=arguments,
    )

    assert knowledge_fingerprint != alternate_fingerprint


def test_fingerprint_rejects_wrong_validated_schema() -> None:
    definition = create_definition()
    invalid_arguments = OtherInput(
        service_name="payments",
    )

    with pytest.raises(
        ToolInputValidationError,
        match="arguments are invalid",
    ):
        create_tool_call_fingerprint(
            definition=definition,
            arguments=invalid_arguments,
        )
