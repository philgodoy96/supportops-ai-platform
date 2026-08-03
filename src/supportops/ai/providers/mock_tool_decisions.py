"""Explicit scripted outcomes for mock LLM tool decisions."""

import json
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from uuid import UUID

from supportops.ai.gateway.contracts import LLMTokenUsage
from supportops.ai.gateway.tool_decisions import (
    COMPLETE_SUPPORT_ANALYSIS_CONTROL_NAME,
)

_MAX_ARGUMENTS_JSON_LENGTH = 20_000
_MAX_REPEATED_OUTCOMES = 10


class MockToolDecisionOutcomeKind(StrEnum):
    """Explicit outcomes supported by mock tool decisions."""

    FUNCTION_CALL = "function_call"
    REFUSAL = "refusal"
    TIMEOUT = "timeout"
    RETRYABLE_PROVIDER_ERROR = "retryable_provider_error"
    TERMINAL_PROVIDER_ERROR = "terminal_provider_error"
    INCOMPLETE_RESPONSE = "incomplete_response"


@dataclass(frozen=True, slots=True)
class MockToolDecisionOutcome:
    """One explicitly scripted mock tool-decision result."""

    kind: MockToolDecisionOutcomeKind
    function_name: str | None = None
    arguments_json: str | None = None
    usage: LLMTokenUsage | None = None
    finish_reason: str | None = None

    def __post_init__(self) -> None:
        if self.kind is MockToolDecisionOutcomeKind.FUNCTION_CALL:
            if self.function_name is None:
                raise ValueError("A function-call outcome requires function_name.")

            if self.arguments_json is None:
                raise ValueError("A function-call outcome requires arguments_json.")

            if self.finish_reason is None:
                raise ValueError("A function-call outcome requires finish_reason.")

            _validate_required_text(
                self.function_name,
                field_name="function_name",
            )
            _validate_required_text(
                self.arguments_json,
                field_name="arguments_json",
            )
            _validate_required_text(
                self.finish_reason,
                field_name="finish_reason",
            )

            if len(self.arguments_json) > _MAX_ARGUMENTS_JSON_LENGTH:
                raise ValueError("arguments_json exceeds the supported size.")

            return

        if self.function_name is not None:
            raise ValueError("Failure outcomes must not define function_name.")

        if self.arguments_json is not None:
            raise ValueError("Failure outcomes must not define arguments_json.")

        if self.usage is not None:
            raise ValueError("Failure outcomes must not define usage.")

        if self.finish_reason is not None:
            raise ValueError("Failure outcomes must not define finish_reason.")

    @classmethod
    def function_call(
        cls,
        *,
        function_name: str,
        arguments: Mapping[str, object],
        usage: LLMTokenUsage | None = None,
        finish_reason: str = "completed",
    ) -> "MockToolDecisionOutcome":
        """Create one valid JSON function-call outcome."""

        return cls.raw_function_call(
            function_name=function_name,
            arguments_json=json.dumps(
                dict(arguments),
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            ),
            usage=usage,
            finish_reason=finish_reason,
        )

    @classmethod
    def raw_function_call(
        cls,
        *,
        function_name: str,
        arguments_json: str,
        usage: LLMTokenUsage | None = None,
        finish_reason: str = "completed",
    ) -> "MockToolDecisionOutcome":
        """Create one function call without parsing its arguments."""

        return cls(
            kind=MockToolDecisionOutcomeKind.FUNCTION_CALL,
            function_name=function_name,
            arguments_json=arguments_json,
            usage=usage,
            finish_reason=finish_reason,
        )

    @classmethod
    def search_knowledge(
        cls,
        *,
        query: str = "support guidance",
        top_k: int = 5,
        document_ids: tuple[UUID, ...] | None = None,
        usage: LLMTokenUsage | None = None,
    ) -> "MockToolDecisionOutcome":
        """Script one search_knowledge function call."""

        serialized_document_ids = (
            None if document_ids is None else [str(document_id) for document_id in document_ids]
        )

        return cls.function_call(
            function_name="search_knowledge",
            arguments={
                "query": query,
                "top_k": top_k,
                "document_ids": serialized_document_ids,
            },
            usage=usage,
        )

    @classmethod
    def lookup_service_status(
        cls,
        *,
        service_name: str,
        usage: LLMTokenUsage | None = None,
    ) -> "MockToolDecisionOutcome":
        """Script one lookup_service_status function call."""

        return cls.function_call(
            function_name="lookup_service_status",
            arguments={
                "service_name": service_name,
            },
            usage=usage,
        )

    @classmethod
    def complete_support_analysis(
        cls,
        *,
        recommended_action: str,
        evidence_sufficient: bool,
        requires_human_review: bool,
        decision_summary: str,
        usage: LLMTokenUsage | None = None,
    ) -> "MockToolDecisionOutcome":
        """Script the non-executable terminal control action."""

        return cls.function_call(
            function_name=(COMPLETE_SUPPORT_ANALYSIS_CONTROL_NAME),
            arguments={
                "recommended_action": recommended_action,
                "evidence_sufficient": evidence_sufficient,
                "requires_human_review": (requires_human_review),
                "decision_summary": decision_summary,
            },
            usage=usage,
        )

    @classmethod
    def escalate_ticket(
        cls,
        *,
        target_queue: str = "security_operations",
        reason: str = ("The ticket evidence supports internal escalation."),
        usage: LLMTokenUsage | None = None,
    ) -> "MockToolDecisionOutcome":
        """Script one bounded sensitive escalation proposal."""

        return cls.function_call(
            function_name="escalate_ticket",
            arguments={
                "target_queue": target_queue,
                "reason": reason,
            },
            usage=usage,
        )

    @classmethod
    def complete_human_approved_support_analysis(
        cls,
        *,
        recommended_action: str,
        evidence_sufficient: bool,
        requires_human_review: bool,
        decision_summary: str,
        usage: LLMTokenUsage | None = None,
    ) -> "MockToolDecisionOutcome":
        """Script the human-approved terminal control action."""

        return cls.function_call(
            function_name=("complete_human_approved_support_analysis"),
            arguments={
                "schema_version": ("human-approved-support-decision-v1"),
                "recommended_action": recommended_action,
                "evidence_sufficient": evidence_sufficient,
                "requires_human_review": (requires_human_review),
                "decision_summary": decision_summary,
            },
            usage=usage,
        )

    @classmethod
    def unknown_tool(
        cls,
        *,
        tool_name: str = "unknown_tool",
        arguments: Mapping[str, object] | None = None,
    ) -> "MockToolDecisionOutcome":
        """Script a provider-selected function unknown to the app."""

        return cls.function_call(
            function_name=tool_name,
            arguments=arguments or {},
        )

    @classmethod
    def malformed_arguments(
        cls,
        *,
        function_name: str = "search_knowledge",
        arguments_json: str = "{malformed-json",
    ) -> "MockToolDecisionOutcome":
        """Script a function call containing malformed JSON."""

        return cls.raw_function_call(
            function_name=function_name,
            arguments_json=arguments_json,
        )

    @classmethod
    def repeated_tool_call(
        cls,
        *,
        function_name: str,
        arguments: Mapping[str, object],
        repetitions: int = 2,
    ) -> tuple["MockToolDecisionOutcome", ...]:
        """Create an explicit repeated-call script."""

        if repetitions < 2:
            raise ValueError("repetitions must be at least two.")

        if repetitions > _MAX_REPEATED_OUTCOMES:
            raise ValueError("repetitions exceeds the supported maximum.")

        outcome = cls.function_call(
            function_name=function_name,
            arguments=arguments,
        )

        return tuple(outcome for _ in range(repetitions))

    @classmethod
    def refusal(cls) -> "MockToolDecisionOutcome":
        """Script one provider refusal."""

        return cls(kind=MockToolDecisionOutcomeKind.REFUSAL)

    @classmethod
    def timeout(cls) -> "MockToolDecisionOutcome":
        """Script one provider timeout."""

        return cls(kind=MockToolDecisionOutcomeKind.TIMEOUT)

    @classmethod
    def retryable_provider_error(
        cls,
    ) -> "MockToolDecisionOutcome":
        """Script one retryable provider failure."""

        return cls(kind=(MockToolDecisionOutcomeKind.RETRYABLE_PROVIDER_ERROR))

    @classmethod
    def terminal_provider_error(
        cls,
    ) -> "MockToolDecisionOutcome":
        """Script one terminal provider failure."""

        return cls(kind=(MockToolDecisionOutcomeKind.TERMINAL_PROVIDER_ERROR))

    @classmethod
    def incomplete_response(
        cls,
    ) -> "MockToolDecisionOutcome":
        """Script one incomplete provider response."""

        return cls(kind=(MockToolDecisionOutcomeKind.INCOMPLETE_RESPONSE))


class MockToolDecisionOutcomeQueueExhaustedError(RuntimeError):
    """Raised when no scripted tool decision remains."""


def _validate_required_text(
    value: str,
    *,
    field_name: str,
) -> None:
    if not value:
        raise ValueError(f"{field_name} is required.")

    if value != value.strip():
        raise ValueError(f"{field_name} must not contain surrounding whitespace.")
