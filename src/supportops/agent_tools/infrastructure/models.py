"""SQLAlchemy persistence model for controlled tool-call audits."""

from datetime import datetime
from uuid import UUID

from pydantic import JsonValue
from sqlalchemy import (
    BigInteger,
    CheckConstraint,
    DateTime,
    ForeignKeyConstraint,
    Index,
    Integer,
    String,
    UniqueConstraint,
    Uuid,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from supportops.agent_tools.domain.audit import (
    AGENT_TOOL_CALL_ERROR_CODE_MAX_LENGTH,
    AGENT_TOOL_CALL_NAME_MAX_LENGTH,
    AGENT_TOOL_CALL_PROVIDER_CALL_ID_MAX_LENGTH,
    AGENT_TOOL_CALL_SAFE_INPUT_MAX_BYTES,
    AGENT_TOOL_CALL_SAFE_OUTPUT_MAX_BYTES,
    AgentToolCall,
    AgentToolCallStatus,
)
from supportops.agent_tools.domain.contracts import ToolSafetyLevel
from supportops.infrastructure.postgresql.base import Base

_INPUT_FINGERPRINT_LENGTH = 64
_STATUS_MAX_LENGTH = 32
_SAFETY_LEVEL_MAX_LENGTH = 32

_TOOL_CALL_STATUS_SQL_VALUES = ", ".join(f"'{member.value}'" for member in AgentToolCallStatus)
_TOOL_SAFETY_SQL_VALUES = ", ".join(f"'{member.value}'" for member in ToolSafetyLevel)


class AgentToolCallRecord(Base):
    """Persisted terminal audit outcome for one controlled tool call."""

    __tablename__ = "agent_tool_calls"

    id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True),
        primary_key=True,
    )
    workspace_id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True),
        nullable=False,
    )
    ticket_id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True),
        nullable=False,
    )
    agent_run_id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True),
        nullable=False,
    )
    agent_run_attempt_id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True),
        nullable=False,
    )
    sequence: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )
    provider_tool_call_id: Mapped[str | None] = mapped_column(
        String(AGENT_TOOL_CALL_PROVIDER_CALL_ID_MAX_LENGTH),
        nullable=True,
    )
    tool_name: Mapped[str] = mapped_column(
        String(AGENT_TOOL_CALL_NAME_MAX_LENGTH),
        nullable=False,
    )
    tool_version: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )
    safety_level: Mapped[str] = mapped_column(
        String(_SAFETY_LEVEL_MAX_LENGTH),
        nullable=False,
    )
    status: Mapped[str] = mapped_column(
        String(_STATUS_MAX_LENGTH),
        nullable=False,
    )
    input_fingerprint: Mapped[str] = mapped_column(
        String(_INPUT_FINGERPRINT_LENGTH),
        nullable=False,
    )
    safe_input: Mapped[dict[str, JsonValue]] = mapped_column(
        JSONB,
        nullable=False,
    )
    safe_output: Mapped[dict[str, JsonValue] | None] = mapped_column(
        JSONB,
        nullable=True,
    )
    latency_ms: Mapped[int] = mapped_column(
        BigInteger,
        nullable=False,
    )
    error_code: Mapped[str | None] = mapped_column(
        String(AGENT_TOOL_CALL_ERROR_CODE_MAX_LENGTH),
        nullable=True,
    )
    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
    )
    finished_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
    )

    __table_args__ = (
        ForeignKeyConstraint(
            [
                "workspace_id",
                "ticket_id",
                "agent_run_id",
            ],
            [
                "agent_runs.workspace_id",
                "agent_runs.ticket_id",
                "agent_runs.id",
            ],
            name=("fk_agent_tool_calls_workspace_ticket_agent_run"),
            ondelete="CASCADE",
        ),
        ForeignKeyConstraint(
            [
                "agent_run_id",
                "agent_run_attempt_id",
            ],
            [
                "agent_run_attempts.agent_run_id",
                "agent_run_attempts.id",
            ],
            name="fk_agent_tool_calls_agent_run_attempt",
            ondelete="CASCADE",
        ),
        UniqueConstraint(
            "agent_run_attempt_id",
            "sequence",
            name="uq_agent_tool_calls_attempt_sequence",
        ),
        UniqueConstraint(
            "agent_run_attempt_id",
            "provider_tool_call_id",
            name=("uq_agent_tool_calls_attempt_provider_call"),
        ),
        CheckConstraint(
            "sequence >= 1",
            name="agent_tool_call_sequence_positive",
        ),
        CheckConstraint(
            (
                "provider_tool_call_id IS NULL OR ("
                "provider_tool_call_id = "
                "btrim(provider_tool_call_id) "
                "AND char_length(provider_tool_call_id) "
                "BETWEEN 1 AND 255"
                ")"
            ),
            name="agent_tool_call_provider_call_id_format",
        ),
        CheckConstraint(
            (
                "tool_name = btrim(tool_name) "
                "AND char_length(tool_name) BETWEEN 1 AND 64 "
                "AND tool_name ~ '^[a-z][a-z0-9_]*$'"
            ),
            name="agent_tool_call_tool_name_format",
        ),
        CheckConstraint(
            "tool_version >= 1",
            name="agent_tool_call_tool_version_positive",
        ),
        CheckConstraint(
            f"safety_level IN ({_TOOL_SAFETY_SQL_VALUES})",
            name="agent_tool_call_safety_level",
        ),
        CheckConstraint(
            f"status IN ({_TOOL_CALL_STATUS_SQL_VALUES})",
            name="agent_tool_call_status",
        ),
        CheckConstraint(
            "input_fingerprint ~ '^[0-9a-f]{64}$'",
            name="agent_tool_call_input_fingerprint",
        ),
        CheckConstraint(
            "jsonb_typeof(safe_input) = 'object'",
            name="agent_tool_call_safe_input_object",
        ),
        CheckConstraint(
            (f"octet_length(safe_input::text) <= {AGENT_TOOL_CALL_SAFE_INPUT_MAX_BYTES}"),
            name="agent_tool_call_safe_input_size",
        ),
        CheckConstraint(
            ("safe_output IS NULL OR jsonb_typeof(safe_output) = 'object'"),
            name="agent_tool_call_safe_output_object",
        ),
        CheckConstraint(
            (
                "safe_output IS NULL "
                "OR octet_length(safe_output::text) "
                f"<= {AGENT_TOOL_CALL_SAFE_OUTPUT_MAX_BYTES}"
            ),
            name="agent_tool_call_safe_output_size",
        ),
        CheckConstraint(
            "latency_ms >= 0",
            name="agent_tool_call_latency_non_negative",
        ),
        CheckConstraint(
            (
                "error_code IS NULL OR ("
                "error_code = btrim(error_code) "
                "AND char_length(error_code) BETWEEN 1 AND 128 "
                "AND error_code ~ '^[a-z][a-z0-9_]*$'"
                ")"
            ),
            name="agent_tool_call_error_code_format",
        ),
        CheckConstraint(
            (
                "("
                "status = 'succeeded' "
                "AND safe_output IS NOT NULL "
                "AND error_code IS NULL"
                ") OR ("
                "status IN ('failed', 'timed_out', 'rejected') "
                "AND safe_output IS NULL "
                "AND error_code IS NOT NULL"
                ")"
            ),
            name="agent_tool_call_terminal_outcome",
        ),
        CheckConstraint(
            "finished_at >= started_at",
            name="agent_tool_call_timestamp_order",
        ),
        Index(
            "ix_agent_tool_calls_workspace_run_sequence",
            "workspace_id",
            "agent_run_id",
            "sequence",
        ),
    )

    @classmethod
    def from_domain(
        cls,
        tool_call: AgentToolCall,
    ) -> "AgentToolCallRecord":
        """Create a persistence record from a terminal audit entity."""

        return cls(
            id=tool_call.id,
            workspace_id=tool_call.workspace_id,
            ticket_id=tool_call.ticket_id,
            agent_run_id=tool_call.agent_run_id,
            agent_run_attempt_id=(tool_call.agent_run_attempt_id),
            sequence=tool_call.sequence,
            provider_tool_call_id=(tool_call.provider_tool_call_id),
            tool_name=tool_call.tool_name,
            tool_version=tool_call.tool_version,
            safety_level=tool_call.safety_level.value,
            status=tool_call.status.value,
            input_fingerprint=tool_call.input_fingerprint,
            safe_input=dict(tool_call.safe_input),
            safe_output=(
                dict(tool_call.safe_output) if tool_call.safe_output is not None else None
            ),
            latency_ms=tool_call.latency_ms,
            error_code=tool_call.error_code,
            started_at=tool_call.started_at,
            finished_at=tool_call.finished_at,
        )

    def to_domain(self) -> AgentToolCall:
        """Map the persistence record to a terminal audit entity."""

        return AgentToolCall(
            id=self.id,
            workspace_id=self.workspace_id,
            ticket_id=self.ticket_id,
            agent_run_id=self.agent_run_id,
            agent_run_attempt_id=self.agent_run_attempt_id,
            sequence=self.sequence,
            provider_tool_call_id=self.provider_tool_call_id,
            tool_name=self.tool_name,
            tool_version=self.tool_version,
            safety_level=ToolSafetyLevel(self.safety_level),
            status=AgentToolCallStatus(self.status),
            input_fingerprint=self.input_fingerprint,
            safe_input=self.safe_input,
            safe_output=self.safe_output,
            latency_ms=self.latency_ms,
            error_code=self.error_code,
            started_at=self.started_at,
            finished_at=self.finished_at,
        )
