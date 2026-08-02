"""Deterministic fingerprints for validated controlled tool calls."""

import hashlib
import json

from supportops.agent_tools.domain.contracts import (
    StrictToolSchema,
    ToolDefinition,
)
from supportops.agent_tools.domain.errors import (
    ToolInputValidationError,
)


def canonicalize_validated_tool_arguments(
    arguments: StrictToolSchema,
) -> str:
    """Serialize validated arguments using stable canonical JSON."""

    try:
        payload = arguments.model_dump(
            mode="json",
            by_alias=True,
            exclude_none=False,
        )

        return json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ToolInputValidationError() from exc


def create_tool_call_fingerprint(
    *,
    definition: ToolDefinition,
    arguments: StrictToolSchema,
) -> str:
    """Hash exact tool identity and canonical validated arguments."""

    if not isinstance(arguments, definition.input_schema):
        raise ToolInputValidationError()

    canonical_arguments = canonicalize_validated_tool_arguments(arguments)
    fingerprint_payload = json.dumps(
        {
            "arguments": json.loads(canonical_arguments),
            "tool_name": definition.name,
            "tool_version": definition.version,
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )

    return hashlib.sha256(fingerprint_payload.encode("utf-8")).hexdigest()
