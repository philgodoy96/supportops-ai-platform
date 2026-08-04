"""Unit tests for privacy-aware observability export policies."""

from enum import StrEnum
from uuid import UUID

import pytest

from supportops.observability.errors import (
    ObservabilityPrivacyPolicyError,
    ObservabilitySerializationError,
)
from supportops.observability.privacy import (
    ExportFieldPolicy,
    MetadataOnlyExportPolicy,
    PrivacySanitizer,
    RedactedContentExportPolicy,
    SanitizationLimits,
)


class ExampleStatus(StrEnum):
    SUCCEEDED = "succeeded"


def test_metadata_only_omits_input_and_output_content() -> None:
    policy = MetadataOnlyExportPolicy()
    fields = ExportFieldPolicy(
        metadata_paths=frozenset(
            {
                ("workspace_id",),
                ("provider",),
            }
        ),
        input_paths=frozenset({("prompt",)}),
        output_paths=frozenset({("response",)}),
    )

    sanitized = policy.sanitize(
        metadata={
            "workspace_id": "workspace-1",
            "provider": "mock",
        },
        field_policy=fields,
        input_data={"prompt": "sensitive prompt"},
        output_data={"response": "sensitive output"},
    )

    assert sanitized.metadata == {
        "workspace_id": "workspace-1",
        "provider": "mock",
    }
    assert sanitized.input_data is None
    assert sanitized.output_data is None


def test_redacted_content_requires_explicit_allowlisted_paths() -> None:
    policy = RedactedContentExportPolicy()
    fields = ExportFieldPolicy(
        metadata_paths=frozenset({("operation",)}),
        input_paths=frozenset({("trusted", "summary")}),
        output_paths=frozenset({("result", "status")}),
    )

    sanitized = policy.sanitize(
        metadata={"operation": "classification"},
        field_policy=fields,
        input_data={
            "trusted": {
                "summary": "Synthetic request",
                "internal_note": "must be removed",
            },
            "untrusted": "must be removed",
        },
        output_data={
            "result": {
                "status": "completed",
                "raw_output": "must be removed",
            }
        },
    )

    assert sanitized.input_data == {
        "trusted": {
            "summary": "Synthetic request",
        }
    }
    assert sanitized.output_data == {
        "result": {
            "status": "completed",
        }
    }


def test_redacted_content_rejects_unstructured_root_strings() -> None:
    policy = RedactedContentExportPolicy()
    fields = ExportFieldPolicy(
        metadata_paths=frozenset(),
        input_paths=frozenset({("prompt",)}),
    )

    with pytest.raises(
        ObservabilityPrivacyPolicyError,
        match="structured mapping",
    ):
        policy.sanitize(
            metadata={},
            field_policy=fields,
            input_data="raw prompt",
        )


def test_unknown_fields_are_removed_recursively() -> None:
    sanitizer = PrivacySanitizer()

    sanitized = sanitizer.sanitize_metadata(
        {
            "workflow": {
                "name": "ticket-processing",
                "version": "human-approved-support-v1",
                "raw_state": "must be removed",
            },
            "unknown": "must be removed",
        },
        allowed_paths=frozenset(
            {
                ("workflow", "name"),
                ("workflow", "version"),
            }
        ),
    )

    assert sanitized == {
        "workflow": {
            "name": "ticket-processing",
            "version": "human-approved-support-v1",
        }
    }


def test_email_addresses_are_masked() -> None:
    sanitizer = PrivacySanitizer()

    sanitized = sanitizer.sanitize_metadata(
        {"message": "Contact engineer@example.com for details."},
        allowed_paths=frozenset({("message",)}),
    )

    assert sanitized["message"] == ("Contact <redacted-email> for details.")


def test_phone_numbers_are_masked() -> None:
    sanitizer = PrivacySanitizer()

    sanitized = sanitizer.sanitize_metadata(
        {"message": "Call +1 (415) 555-0100 immediately."},
        allowed_paths=frozenset({("message",)}),
    )

    assert sanitized["message"] == ("Call <redacted-phone> immediately.")


def test_prefixed_api_keys_are_masked() -> None:
    sanitizer = PrivacySanitizer()

    sanitized = sanitizer.sanitize_metadata(
        {"message": ("The synthetic key is sk-lf-test-secret-1234567890.")},
        allowed_paths=frozenset({("message",)}),
    )

    assert sanitized["message"] == ("The synthetic key is <redacted-credential>.")


def test_authorization_headers_are_removed_even_when_allowlisted() -> None:
    sanitizer = PrivacySanitizer()

    sanitized = sanitizer.sanitize_metadata(
        {
            "authorization": "Bearer synthetic-token-value",
            "operation": "classification",
        },
        allowed_paths=frozenset(
            {
                ("authorization",),
                ("operation",),
            }
        ),
    )

    assert sanitized == {"operation": "classification"}


def test_database_url_fields_are_removed() -> None:
    sanitizer = PrivacySanitizer()

    sanitized = sanitizer.sanitize_metadata(
        {
            "database_url": ("postgresql://user:password@localhost/supportops"),
            "operation": "indexing",
        },
        allowed_paths=frozenset(
            {
                ("database_url",),
                ("operation",),
            }
        ),
    )

    assert sanitized == {"operation": "indexing"}


def test_database_urls_inside_allowlisted_strings_are_masked() -> None:
    sanitizer = PrivacySanitizer()

    sanitized = sanitizer.sanitize_metadata(
        {"message": ("Connection failed for postgresql://user:password@localhost/supportops")},
        allowed_paths=frozenset({("message",)}),
    )

    assert sanitized["message"] == ("Connection failed for <redacted-database-url>")


def test_strings_are_truncated_to_configured_limit() -> None:
    sanitizer = PrivacySanitizer(
        limits=SanitizationLimits(
            max_string_length=20,
            max_collection_length=32,
            max_depth=8,
        )
    )

    sanitized = sanitizer.sanitize_metadata(
        {"message": "abcdefghijklmnopqrstuvwxyz"},
        allowed_paths=frozenset({("message",)}),
    )

    assert sanitized["message"] == "abcdefghijklmnopqrs…"
    assert len(sanitized["message"]) == 20


def test_collections_are_bounded() -> None:
    sanitizer = PrivacySanitizer(
        limits=SanitizationLimits(
            max_string_length=512,
            max_collection_length=2,
            max_depth=8,
        )
    )

    sanitized = sanitizer.sanitize_metadata(
        {"chunk_ids": ["chunk-1", "chunk-2", "chunk-3"]},
        allowed_paths=frozenset({("chunk_ids",)}),
    )

    assert sanitized["chunk_ids"] == ["chunk-1", "chunk-2"]


@pytest.mark.parametrize(
    "forbidden_key",
    [
        "vector",
        "embedding_vectors",
        "checkpoint_payload",
        "raw_checkpoint",
        "sensitive_execution_grant",
        "grant_token",
        "lease_token",
        "traceback",
    ],
)
def test_forbidden_structures_are_removed_even_when_allowlisted(
    forbidden_key: str,
) -> None:
    sanitizer = PrivacySanitizer()

    sanitized = sanitizer.sanitize_metadata(
        {
            forbidden_key: "sensitive-value",
            "operation": "workflow",
        },
        allowed_paths=frozenset(
            {
                (forbidden_key,),
                ("operation",),
            }
        ),
    )

    assert sanitized == {"operation": "workflow"}


def test_exception_objects_are_rejected_without_serializing_message() -> None:
    sanitizer = PrivacySanitizer()

    with pytest.raises(
        ObservabilityPrivacyPolicyError,
        match="exception objects must not be exported",
    ) as exception_info:
        sanitizer.sanitize_metadata(
            {"error": RuntimeError("secret provider response")},
            allowed_paths=frozenset({("error",)}),
        )

    assert "secret provider response" not in str(exception_info.value)


def test_binary_values_are_rejected() -> None:
    sanitizer = PrivacySanitizer()

    with pytest.raises(
        ObservabilitySerializationError,
        match="binary values must not be exported",
    ):
        sanitizer.sanitize_metadata(
            {"payload": b"raw bytes"},
            allowed_paths=frozenset({("payload",)}),
        )


def test_safe_uuid_values_are_preserved() -> None:
    sanitizer = PrivacySanitizer()
    workspace_id = UUID("11111111-1111-4111-8111-111111111111")

    sanitized = sanitizer.sanitize_metadata(
        {
            "workspace_id": workspace_id,
            "agent_run_id": str(workspace_id),
        },
        allowed_paths=frozenset(
            {
                ("workspace_id",),
                ("agent_run_id",),
            }
        ),
    )

    assert sanitized == {
        "workspace_id": str(workspace_id),
        "agent_run_id": str(workspace_id),
    }


def test_safe_enum_values_are_preserved() -> None:
    sanitizer = PrivacySanitizer()

    sanitized = sanitizer.sanitize_metadata(
        {"status": ExampleStatus.SUCCEEDED},
        allowed_paths=frozenset({("status",)}),
    )

    assert sanitized == {"status": "succeeded"}


def test_excessive_depth_is_omitted() -> None:
    sanitizer = PrivacySanitizer(
        limits=SanitizationLimits(
            max_string_length=512,
            max_collection_length=32,
            max_depth=2,
        )
    )

    sanitized = sanitizer.sanitize_metadata(
        {
            "level_one": {
                "level_two": {
                    "level_three": "too deep",
                }
            }
        },
        allowed_paths=frozenset(
            {
                (
                    "level_one",
                    "level_two",
                    "level_three",
                )
            }
        ),
    )

    assert sanitized == {
        "level_one": {
            "level_two": {},
        }
    }


def test_non_finite_floats_are_rejected() -> None:
    sanitizer = PrivacySanitizer()

    with pytest.raises(
        ObservabilitySerializationError,
        match="non-finite floats",
    ):
        sanitizer.sanitize_metadata(
            {"score": float("nan")},
            allowed_paths=frozenset({("score",)}),
        )


def test_export_field_policy_rejects_empty_paths() -> None:
    with pytest.raises(
        ValueError,
        match="must not contain an empty path",
    ):
        ExportFieldPolicy(
            metadata_paths=frozenset({()}),
        )


def test_metadata_only_policy_satisfies_export_protocol() -> None:
    from supportops.observability.privacy import (
        ObservabilityExportPolicy,
    )

    assert isinstance(
        MetadataOnlyExportPolicy(),
        ObservabilityExportPolicy,
    )


def test_redacted_policy_satisfies_export_protocol() -> None:
    from supportops.observability.privacy import (
        ObservabilityExportPolicy,
    )

    assert isinstance(
        RedactedContentExportPolicy(),
        ObservabilityExportPolicy,
    )
