"""JSON-compatible state contracts for the human-approved support graph."""

from collections.abc import Mapping
from enum import StrEnum
from typing import Annotated, Literal, TypedDict, cast
from uuid import UUID

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    StrictBool,
    StringConstraints,
    ValidationError,
    model_validator,
)

from supportops.agent_graph.domain.resume_planning import (
    ApprovalResumeDecisionStatus,
)
from supportops.ai.schemas.ticket_classification import (
    TicketCategory,
    TicketIntent,
    TicketSentiment,
    TicketUrgency,
)

HumanApprovedSupportStateSchemaVersion = Literal["human-approved-support-state-v1"]
HumanApprovedSupportWorkflowName = Literal["ticket-processing"]
HumanApprovedSupportWorkflowVersion = Literal["human-approved-support-v1"]
HumanApprovedSupportGraphVersion = Literal["graph-v1"]

HUMAN_APPROVED_SUPPORT_STATE_SCHEMA_VERSION: HumanApprovedSupportStateSchemaVersion = (
    "human-approved-support-state-v1"
)
HUMAN_APPROVED_SUPPORT_WORKFLOW_NAME: HumanApprovedSupportWorkflowName = "ticket-processing"
HUMAN_APPROVED_SUPPORT_WORKFLOW_VERSION: HumanApprovedSupportWorkflowVersion = (
    "human-approved-support-v1"
)
HUMAN_APPROVED_SUPPORT_GRAPH_VERSION: HumanApprovedSupportGraphVersion = "graph-v1"

HUMAN_APPROVED_GRAPH_STATE_MAX_STEPS = 64
HUMAN_APPROVED_GRAPH_STATE_MAX_TOOL_CALLS = 10
HUMAN_APPROVED_GRAPH_STATE_MAX_DECISION_TURNS = 11
HUMAN_APPROVED_GRAPH_STATE_MAX_DECISION_SUMMARY_LENGTH = 500
HUMAN_APPROVED_GRAPH_STATE_MAX_REQUEST_REASON_LENGTH = 1000

ToolCallFingerprint = Annotated[
    str,
    StringConstraints(pattern=r"^[0-9a-f]{64}$"),
]
ClassificationSummary = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        min_length=1,
        max_length=500,
    ),
]
DecisionSummary = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        min_length=1,
        max_length=HUMAN_APPROVED_GRAPH_STATE_MAX_DECISION_SUMMARY_LENGTH,
    ),
]
ApprovalRequestReason = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        min_length=1,
        max_length=HUMAN_APPROVED_GRAPH_STATE_MAX_REQUEST_REASON_LENGTH,
    ),
]
GraphErrorCode = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        min_length=1,
        max_length=128,
        pattern=r"^[a-z][a-z0-9_]*$",
    ),
]


class HumanApprovedSensitiveExecutionOutput(BaseModel):
    """Safe projection of one granted escalate_ticket execution."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    escalation_id: UUID
    ticket_id: UUID
    target_queue: Literal[
        "billing_operations",
        "engineering_support",
        "security_operations",
        "support_operations",
    ]
    status: Literal["escalated"]


class HumanApprovedDecisionKind(StrEnum):
    """Decision kinds understood by the human-approved graph."""

    TERMINAL = "terminal"
    READ_ONLY_TOOL = "read_only_tool"
    SENSITIVE_TOOL = "sensitive_tool"


class HumanApprovalCheckpointStatus(StrEnum):
    """Approval status projected into checkpoint state."""

    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXPIRED = "expired"


class HumanApprovedRecommendationStage(StrEnum):
    """Recommendation progression after approval-aware drafting."""

    DRAFTED = "drafted"
    VALIDATED = "validated"
    PERSISTED = "persisted"


class HumanApprovedApprovalResumePayload(BaseModel):
    """Bounded resume value projected into checkpoint state."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    approval_request_id: UUID
    agent_tool_call_id: UUID
    decision_status: ApprovalResumeDecisionStatus


class HumanApprovedSupportGraphState(TypedDict):
    """Bounded JSON-compatible state checkpointed by LangGraph."""

    state_schema_version: str
    workflow_name: str
    workflow_version: str
    graph_version: str
    workspace_id: str
    ticket_id: str
    agent_run_id: str
    run_context_loaded: bool

    classification_id: str | None
    classification_category: str | None
    classification_intent: str | None
    classification_urgency: str | None
    classification_sentiment: str | None
    classification_requires_human_review: bool | None
    classification_summary: str | None

    graph_step_count: int
    decision_turn_count: int
    tool_call_count: int

    decision_kind: str | None
    decision_invocation_id: str | None
    decision_summary: str | None
    analysis_recommended_action: str | None
    analysis_evidence_sufficient: bool | None
    analysis_requires_human_review: bool | None

    proposed_tool_provider_call_id: str | None
    proposed_tool_name: str | None
    proposed_tool_version: int | None
    proposed_tool_input: dict[str, JsonValue] | None
    proposed_tool_fingerprint: str | None
    approval_request_reason: str | None

    agent_tool_call_id: str | None
    approval_request_id: str | None
    approval_status: str | None
    approval_expires_at: str | None
    approval_resume_payload: dict[str, JsonValue] | None

    sensitive_execution_output: dict[str, JsonValue] | None

    recommendation_invocation_id: str | None
    recommendation_id: str | None
    recommendation_stage: str | None
    current_error_code: str | None


class HumanApprovedGraphStateIncompatibleError(ValueError):
    """Raised when checkpointed state cannot be safely resumed."""

    error_code = "human_approved_graph_state_incompatible"
    retryable = False

    def __init__(self) -> None:
        super().__init__(
            "Human-approved support graph state is incompatible with the current schema.",
        )


class HumanApprovedSupportGraphStateSnapshot(BaseModel):
    """Strict validation for initial and recovered graph state."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    state_schema_version: HumanApprovedSupportStateSchemaVersion
    workflow_name: HumanApprovedSupportWorkflowName
    workflow_version: HumanApprovedSupportWorkflowVersion
    graph_version: HumanApprovedSupportGraphVersion

    workspace_id: UUID
    ticket_id: UUID
    agent_run_id: UUID
    run_context_loaded: StrictBool = False

    classification_id: UUID | None = None
    classification_category: TicketCategory | None = None
    classification_intent: TicketIntent | None = None
    classification_urgency: TicketUrgency | None = None
    classification_sentiment: TicketSentiment | None = None
    classification_requires_human_review: StrictBool | None = None
    classification_summary: ClassificationSummary | None = None

    graph_step_count: Annotated[
        int,
        Field(
            strict=True,
            ge=0,
            le=HUMAN_APPROVED_GRAPH_STATE_MAX_STEPS,
        ),
    ] = 0
    decision_turn_count: Annotated[
        int,
        Field(
            strict=True,
            ge=0,
            le=HUMAN_APPROVED_GRAPH_STATE_MAX_DECISION_TURNS,
        ),
    ] = 0
    tool_call_count: Annotated[
        int,
        Field(
            strict=True,
            ge=0,
            le=HUMAN_APPROVED_GRAPH_STATE_MAX_TOOL_CALLS,
        ),
    ] = 0

    decision_kind: HumanApprovedDecisionKind | None = None
    decision_invocation_id: UUID | None = None
    decision_summary: DecisionSummary | None = None
    analysis_recommended_action: str | None = None
    analysis_evidence_sufficient: StrictBool | None = None
    analysis_requires_human_review: StrictBool | None = None

    proposed_tool_provider_call_id: str | None = None
    proposed_tool_name: str | None = None
    proposed_tool_version: (
        Annotated[
            int,
            Field(strict=True, ge=1, le=1000),
        ]
        | None
    ) = None
    proposed_tool_input: dict[str, JsonValue] | None = None
    proposed_tool_fingerprint: ToolCallFingerprint | None = None
    approval_request_reason: ApprovalRequestReason | None = None

    agent_tool_call_id: UUID | None = None
    approval_request_id: UUID | None = None
    approval_status: HumanApprovalCheckpointStatus | None = None
    approval_expires_at: str | None = None
    approval_resume_payload: HumanApprovedApprovalResumePayload | None = None

    sensitive_execution_output: HumanApprovedSensitiveExecutionOutput | None = None

    recommendation_invocation_id: UUID | None = None
    recommendation_id: UUID | None = None
    recommendation_stage: HumanApprovedRecommendationStage | None = None
    current_error_code: GraphErrorCode | None = None

    @model_validator(mode="after")
    def validate_state_relationships(
        self,
    ) -> "HumanApprovedSupportGraphStateSnapshot":
        """Reject internally inconsistent checkpoint state."""

        classification_values = (
            self.classification_category,
            self.classification_intent,
            self.classification_urgency,
            self.classification_sentiment,
            self.classification_requires_human_review,
            self.classification_summary,
        )
        if self.classification_id is None and any(
            value is not None for value in classification_values
        ):
            raise ValueError(
                "classification details require classification_id.",
            )
        if self.classification_id is not None and any(
            value is None for value in classification_values
        ):
            raise ValueError(
                "classification_id requires complete classification details.",
            )

        if self.tool_call_count > self.decision_turn_count:
            raise ValueError(
                "tool_call_count must not exceed decision_turn_count.",
            )

        decision_values = (
            self.decision_invocation_id,
            self.decision_summary,
        )
        if self.decision_kind is None and any(value is not None for value in decision_values):
            raise ValueError(
                "decision details require decision_kind.",
            )
        if self.decision_kind is not None and any(value is None for value in decision_values):
            raise ValueError(
                "decision_kind requires invocation and summary.",
            )

        terminal_values = (
            self.analysis_recommended_action,
            self.analysis_evidence_sufficient,
            self.analysis_requires_human_review,
        )
        if self.decision_kind is HumanApprovedDecisionKind.TERMINAL:
            if any(value is None for value in terminal_values):
                raise ValueError(
                    "Terminal decisions require complete analysis output.",
                )
        elif any(value is not None for value in terminal_values):
            raise ValueError(
                "Analysis output is valid only for terminal decisions.",
            )

        proposal_values = (
            self.proposed_tool_provider_call_id,
            self.proposed_tool_name,
            self.proposed_tool_version,
            self.proposed_tool_input,
            self.proposed_tool_fingerprint,
            self.approval_request_reason,
        )
        has_any_proposal_value = any(value is not None for value in proposal_values)
        has_all_proposal_values = all(value is not None for value in proposal_values)
        if has_any_proposal_value and not has_all_proposal_values:
            raise ValueError(
                "Sensitive proposals require complete tool identity, "
                "input, fingerprint, and reason.",
            )
        if has_all_proposal_values and (
            self.decision_kind is not HumanApprovedDecisionKind.SENSITIVE_TOOL
        ):
            raise ValueError(
                "Sensitive proposal fields require a sensitive decision.",
            )
        if (
            self.decision_kind is HumanApprovedDecisionKind.SENSITIVE_TOOL
            and not has_all_proposal_values
        ):
            raise ValueError(
                "Sensitive decisions require complete proposal fields.",
            )

        approval_values = (
            self.agent_tool_call_id,
            self.approval_request_id,
            self.approval_status,
            self.approval_expires_at,
        )
        has_any_approval_value = any(value is not None for value in approval_values)
        has_all_approval_values = all(value is not None for value in approval_values)
        if has_any_approval_value and not has_all_approval_values:
            raise ValueError(
                "Approval checkpoint fields must be populated together.",
            )
        if has_all_approval_values and not has_all_proposal_values:
            raise ValueError(
                "Approval checkpoint fields require a sensitive proposal.",
            )

        if self.approval_resume_payload is not None:
            if self.approval_status is not HumanApprovalCheckpointStatus.PENDING:
                raise ValueError(
                    "approval_resume_payload requires pending approval status.",
                )
            if (
                self.approval_request_id != self.approval_resume_payload.approval_request_id
                or self.agent_tool_call_id != self.approval_resume_payload.agent_tool_call_id
            ):
                raise ValueError(
                    "approval_resume_payload must match checkpoint approval identifiers.",
                )

        if self.sensitive_execution_output is not None:
            if self.approval_status is not HumanApprovalCheckpointStatus.APPROVED:
                raise ValueError(
                    "sensitive_execution_output requires approved status.",
                )
            if self.proposed_tool_name != "escalate_ticket" or self.proposed_tool_version != 1:
                raise ValueError(
                    "sensitive_execution_output requires escalate_ticket v1.",
                )

        if self.recommendation_invocation_id is not None and self.decision_kind is None:
            raise ValueError(
                "recommendation_invocation_id requires a decision.",
            )
        if self.recommendation_id is not None and self.recommendation_invocation_id is None:
            raise ValueError(
                "recommendation_id requires recommendation_invocation_id.",
            )
        if self.recommendation_stage is not None and self.recommendation_id is None:
            raise ValueError(
                "recommendation_stage requires recommendation_id.",
            )

        return self

    def to_graph_state(self) -> HumanApprovedSupportGraphState:
        """Return a JSON-compatible checkpoint representation."""

        return cast(
            HumanApprovedSupportGraphState,
            self.model_dump(mode="json"),
        )


def create_initial_human_approved_support_state(
    *,
    workspace_id: UUID,
    ticket_id: UUID,
    agent_run_id: UUID,
) -> HumanApprovedSupportGraphState:
    """Create the minimal valid state for a new workflow."""

    return HumanApprovedSupportGraphStateSnapshot(
        state_schema_version=(HUMAN_APPROVED_SUPPORT_STATE_SCHEMA_VERSION),
        workflow_name=HUMAN_APPROVED_SUPPORT_WORKFLOW_NAME,
        workflow_version=HUMAN_APPROVED_SUPPORT_WORKFLOW_VERSION,
        graph_version=HUMAN_APPROVED_SUPPORT_GRAPH_VERSION,
        workspace_id=workspace_id,
        ticket_id=ticket_id,
        agent_run_id=agent_run_id,
    ).to_graph_state()


def validate_human_approved_support_state(
    state: Mapping[str, object],
) -> HumanApprovedSupportGraphStateSnapshot:
    """Validate initial or recovered graph state."""

    try:
        return HumanApprovedSupportGraphStateSnapshot.model_validate(
            dict(state),
        )
    except ValidationError as exc:
        raise HumanApprovedGraphStateIncompatibleError() from exc
