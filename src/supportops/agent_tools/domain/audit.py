"""Immutable audit records for controlled tool-call outcomes."""

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from types import MappingProxyType
from typing import cast
from uuid import UUID, uuid4

from pydantic import JsonValue, TypeAdapter, ValidationError

from supportops.agent_tools.domain.contracts import (
    ToolSafetyLevel,
)

AGENT_TOOL_CALL_NAME_MAX_LENGTH = 64
AGENT_TOOL_CALL_PROVIDER_CALL_ID_MAX_LENGTH = 255
AGENT_TOOL_CALL_ERROR_CODE_MAX_LENGTH = 128
AGENT_TOOL_CALL_SAFE_INPUT_MAX_BYTES = 8_192
AGENT_TOOL_CALL_SAFE_OUTPUT_MAX_BYTES = 32_768

_SHA256_HEXADECIMAL_LENGTH = 64
_STABLE_ERROR_CODE_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")
_JSON_OBJECT_ADAPTER = TypeAdapter(dict[str, JsonValue])


class AgentToolCallStatus(StrEnum):
    """Terminal outcomes persisted for controlled tool calls."""

    SUCCEEDED = "succeeded"
    FAILED = "failed"
    TIMED_OUT = "timed_out"
    REJECTED = "rejected"


@dataclass(frozen=True, slots=True)
class AgentToolCall:
    """One terminal application-owned audit record for a tool call."""

    id: UUID
    workspace_id: UUID
    ticket_id: UUID
    agent_run_id: UUID
    agent_run_attempt_id: UUID
    sequence: int
    provider_tool_call_id: str | None
    tool_name: str
    tool_version: int
    safety_level: ToolSafetyLevel
    status: AgentToolCallStatus
    input_fingerprint: str
    safe_input: Mapping[str, JsonValue]
    safe_output: Mapping[str, JsonValue] | None
    latency_ms: int
    error_code: str | None
    started_at: datetime
    finished_at: datetime

    def __post_init__(self) -> None:
        if self.sequence <= 0:
            raise ValueError("sequence must be positive.")

        _validate_optional_bounded_identifier(
            self.provider_tool_call_id,
            field_name="provider_tool_call_id",
            maximum_length=(AGENT_TOOL_CALL_PROVIDER_CALL_ID_MAX_LENGTH),
        )
        _validate_bounded_identifier(
            self.tool_name,
            field_name="tool_name",
            maximum_length=AGENT_TOOL_CALL_NAME_MAX_LENGTH,
        )

        if self.tool_version <= 0:
            raise ValueError("tool_version must be positive.")

        if not isinstance(
            self.safety_level,
            ToolSafetyLevel,
        ):
            raise ValueError("safety_level must use the supported taxonomy.")

        if not isinstance(
            self.status,
            AgentToolCallStatus,
        ):
            raise ValueError("status must be a supported AgentToolCallStatus.")

        _validate_sha256_hash(
            self.input_fingerprint,
            field_name="input_fingerprint",
        )

        safe_input = _validate_safe_json_object(
            self.safe_input,
            field_name="safe_input",
            maximum_bytes=(AGENT_TOOL_CALL_SAFE_INPUT_MAX_BYTES),
        )
        object.__setattr__(
            self,
            "safe_input",
            safe_input,
        )

        if self.safe_output is not None:
            safe_output = _validate_safe_json_object(
                self.safe_output,
                field_name="safe_output",
                maximum_bytes=(AGENT_TOOL_CALL_SAFE_OUTPUT_MAX_BYTES),
            )
            object.__setattr__(
                self,
                "safe_output",
                safe_output,
            )

        if self.latency_ms < 0:
            raise ValueError("latency_ms must be non-negative.")

        _validate_utc_timestamp(
            self.started_at,
            field_name="started_at",
        )
        _validate_utc_timestamp(
            self.finished_at,
            field_name="finished_at",
        )

        if self.finished_at < self.started_at:
            raise ValueError("finished_at must not precede started_at.")

        _validate_terminal_outcome(self)

    @classmethod
    def create(
        cls,
        *,
        workspace_id: UUID,
        ticket_id: UUID,
        agent_run_id: UUID,
        agent_run_attempt_id: UUID,
        sequence: int,
        provider_tool_call_id: str | None,
        tool_name: str,
        tool_version: int,
        safety_level: ToolSafetyLevel,
        status: AgentToolCallStatus,
        input_fingerprint: str,
        safe_input: Mapping[str, JsonValue],
        safe_output: Mapping[str, JsonValue] | None,
        latency_ms: int,
        error_code: str | None,
        started_at: datetime,
        finished_at: datetime,
        tool_call_id: UUID | None = None,
    ) -> "AgentToolCall":
        """Create one terminal controlled-tool audit record."""

        return cls(
            id=tool_call_id or uuid4(),
            workspace_id=workspace_id,
            ticket_id=ticket_id,
            agent_run_id=agent_run_id,
            agent_run_attempt_id=agent_run_attempt_id,
            sequence=sequence,
            provider_tool_call_id=provider_tool_call_id,
            tool_name=tool_name,
            tool_version=tool_version,
            safety_level=safety_level,
            status=status,
            input_fingerprint=input_fingerprint,
            safe_input=safe_input,
            safe_output=safe_output,
            latency_ms=latency_ms,
            error_code=error_code,
            started_at=started_at,
            finished_at=finished_at,
        )


def _validate_terminal_outcome(
    tool_call: AgentToolCall,
) -> None:
    if tool_call.status is AgentToolCallStatus.SUCCEEDED:
        if tool_call.error_code is not None:
            raise ValueError("Successful tool calls cannot define an error_code.")

        if tool_call.safe_output is None:
            raise ValueError("Successful tool calls require safe_output.")

        return

    _validate_bounded_identifier(
        tool_call.error_code,
        field_name="error_code",
        maximum_length=(AGENT_TOOL_CALL_ERROR_CODE_MAX_LENGTH),
    )

    if _STABLE_ERROR_CODE_PATTERN.fullmatch(cast(str, tool_call.error_code)) is None:
        raise ValueError("error_code must use stable lowercase snake case.")

    if tool_call.safe_output is not None:
        raise ValueError("Unsuccessful tool calls cannot define safe_output.")


def _validate_safe_json_object(
    value: Mapping[str, JsonValue],
    *,
    field_name: str,
    maximum_bytes: int,
) -> Mapping[str, JsonValue]:
    try:
        validated = _JSON_OBJECT_ADAPTER.validate_python(
            dict(value),
            strict=True,
        )
        canonical_json = json.dumps(
            validated,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (
        TypeError,
        ValueError,
        ValidationError,
    ) as exc:
        raise ValueError(f"{field_name} must be a JSON-compatible object.") from exc

    if len(canonical_json.encode("utf-8")) > maximum_bytes:
        raise ValueError(f"{field_name} exceeds the supported size.")

    defensive_copy = cast(
        dict[str, JsonValue],
        json.loads(canonical_json),
    )

    return MappingProxyType(defensive_copy)


def _validate_sha256_hash(
    value: str,
    *,
    field_name: str,
) -> None:
    if len(value) != _SHA256_HEXADECIMAL_LENGTH:
        raise ValueError(f"{field_name} must be a lowercase SHA-256 hash.")

    if any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{field_name} must be a lowercase SHA-256 hash.")


def _validate_bounded_identifier(
    value: str | None,
    *,
    field_name: str,
    maximum_length: int,
) -> None:
    if value is None or not value:
        raise ValueError(f"{field_name} is required.")

    if value != value.strip():
        raise ValueError(f"{field_name} must not contain surrounding whitespace.")

    if len(value) > maximum_length:
        raise ValueError(f"{field_name} exceeds the maximum length.")


def _validate_optional_bounded_identifier(
    value: str | None,
    *,
    field_name: str,
    maximum_length: int,
) -> None:
    if value is None:
        return

    _validate_bounded_identifier(
        value,
        field_name=field_name,
        maximum_length=maximum_length,
    )


def _validate_utc_timestamp(
    value: datetime,
    *,
    field_name: str,
) -> None:
    if value.tzinfo is None or value.utcoffset() != timedelta(0):
        raise ValueError(f"{field_name} must be a UTC-aware timestamp.")


def utc_now() -> datetime:
    """Return a UTC timestamp for tool-call factories."""

    return datetime.now(UTC)
