"""JSON-compatible state contracts for the controlled support graph."""

from collections.abc import Mapping
from typing import Annotated, Literal, TypedDict, cast
from uuid import UUID

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StringConstraints,
    ValidationError,
    model_validator,
)

from supportops.ai.schemas.ticket_classification import (
    TicketCategory,
    TicketIntent,
    TicketSentiment,
    TicketUrgency,
)

ControlledSupportStateSchemaVersion = Literal["controlled-support-state-v1"]
ControlledSupportWorkflowName = Literal["ticket-processing"]
ControlledSupportWorkflowVersion = Literal["controlled-support-v1"]
ControlledSupportGraphVersion = Literal["graph-v1"]
SupportRecommendedAction = Literal[
    "respond",
    "request_more_information",
    "recommend_escalation",
]

CONTROLLED_SUPPORT_STATE_SCHEMA_VERSION: ControlledSupportStateSchemaVersion = (
    "controlled-support-state-v1"
)
CONTROLLED_SUPPORT_WORKFLOW_NAME: ControlledSupportWorkflowName = "ticket-processing"
CONTROLLED_SUPPORT_WORKFLOW_VERSION: ControlledSupportWorkflowVersion = "controlled-support-v1"
CONTROLLED_SUPPORT_GRAPH_VERSION: ControlledSupportGraphVersion = "graph-v1"

CONTROLLED_SUPPORT_GRAPH_STATE_MAX_STEPS = 64
CONTROLLED_SUPPORT_GRAPH_STATE_MAX_TOOL_CALLS = 10
CONTROLLED_SUPPORT_GRAPH_STATE_MAX_DECISION_TURNS = 11
CONTROLLED_SUPPORT_GRAPH_STATE_MAX_RETRIEVAL_QUERIES = 10
CONTROLLED_SUPPORT_GRAPH_STATE_MAX_RETRIEVED_CHUNKS = 30
CONTROLLED_SUPPORT_GRAPH_STATE_MAX_DECISION_SUMMARY_LENGTH = 500

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
        max_length=(CONTROLLED_SUPPORT_GRAPH_STATE_MAX_DECISION_SUMMARY_LENGTH),
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
GraphStepCount = Annotated[
    int,
    Field(
        strict=True,
        ge=0,
        le=CONTROLLED_SUPPORT_GRAPH_STATE_MAX_STEPS,
    ),
]
ToolCallCount = Annotated[
    int,
    Field(
        strict=True,
        ge=0,
        le=CONTROLLED_SUPPORT_GRAPH_STATE_MAX_TOOL_CALLS,
    ),
]
DecisionTurnCount = Annotated[
    int,
    Field(
        strict=True,
        ge=0,
        le=CONTROLLED_SUPPORT_GRAPH_STATE_MAX_DECISION_TURNS,
    ),
]


class SupportAnalysisCompletionState(TypedDict):
    """JSON-compatible terminal analysis decision stored in graph state."""

    recommended_action: str
    evidence_sufficient: bool
    requires_human_review: bool
    decision_summary: str


class ControlledSupportGraphState(TypedDict):
    """Bounded JSON-compatible state checkpointed by LangGraph."""

    state_schema_version: str
    workflow_name: str
    workflow_version: str
    graph_version: str
    workspace_id: str
    ticket_id: str
    agent_run_id: str
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
    seen_tool_call_fingerprints: list[str]
    tool_call_ids: list[str]
    retrieval_query_ids: list[str]
    retrieved_chunk_ids: list[str]
    service_status_tool_call_ids: list[str]
    analysis_completion: SupportAnalysisCompletionState | None
    recommendation_invocation_id: str | None
    recommendation_id: str | None
    current_error_code: str | None


class GraphStateIncompatibleError(ValueError):
    """Raised when checkpointed graph state cannot be safely resumed."""

    error_code = "graph_state_incompatible"
    retryable = False

    def __init__(self) -> None:
        super().__init__("Controlled support graph state is incompatible with the current schema.")


class SupportAnalysisCompletionSnapshot(BaseModel):
    """Strict validation for a checkpointed terminal analysis decision."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    recommended_action: SupportRecommendedAction
    evidence_sufficient: StrictBool
    requires_human_review: StrictBool
    decision_summary: DecisionSummary


class ControlledSupportGraphStateSnapshot(BaseModel):
    """Strict typed validation for initial and recovered graph state."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    state_schema_version: ControlledSupportStateSchemaVersion
    workflow_name: ControlledSupportWorkflowName
    workflow_version: ControlledSupportWorkflowVersion
    graph_version: ControlledSupportGraphVersion
    workspace_id: UUID
    ticket_id: UUID
    agent_run_id: UUID
    classification_id: UUID | None = None
    classification_category: TicketCategory | None = None
    classification_intent: TicketIntent | None = None
    classification_urgency: TicketUrgency | None = None
    classification_sentiment: TicketSentiment | None = None
    classification_requires_human_review: StrictBool | None = None
    classification_summary: ClassificationSummary | None = None
    graph_step_count: GraphStepCount = 0
    decision_turn_count: DecisionTurnCount = 0
    tool_call_count: ToolCallCount = 0
    seen_tool_call_fingerprints: tuple[
        ToolCallFingerprint,
        ...,
    ] = Field(
        default=(),
        max_length=CONTROLLED_SUPPORT_GRAPH_STATE_MAX_TOOL_CALLS,
    )
    tool_call_ids: tuple[UUID, ...] = Field(
        default=(),
        max_length=CONTROLLED_SUPPORT_GRAPH_STATE_MAX_TOOL_CALLS,
    )
    retrieval_query_ids: tuple[UUID, ...] = Field(
        default=(),
        max_length=CONTROLLED_SUPPORT_GRAPH_STATE_MAX_RETRIEVAL_QUERIES,
    )
    retrieved_chunk_ids: tuple[UUID, ...] = Field(
        default=(),
        max_length=CONTROLLED_SUPPORT_GRAPH_STATE_MAX_RETRIEVED_CHUNKS,
    )
    service_status_tool_call_ids: tuple[UUID, ...] = Field(
        default=(),
        max_length=CONTROLLED_SUPPORT_GRAPH_STATE_MAX_TOOL_CALLS,
    )
    analysis_completion: SupportAnalysisCompletionSnapshot | None = None
    recommendation_invocation_id: UUID | None = None
    recommendation_id: UUID | None = None
    current_error_code: GraphErrorCode | None = None

    @model_validator(mode="after")
    def validate_state_relationships(
        self,
    ) -> "ControlledSupportGraphStateSnapshot":
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
            raise ValueError("classification details require classification_id.")

        if self.classification_id is not None and any(
            value is None for value in classification_values
        ):
            raise ValueError("classification_id requires complete classification details.")

        if self.tool_call_count > self.decision_turn_count:
            raise ValueError("tool_call_count must not exceed decision_turn_count.")

        if len(self.tool_call_ids) != self.tool_call_count:
            raise ValueError("tool_call_ids must match tool_call_count.")

        if len(self.seen_tool_call_fingerprints) != self.tool_call_count:
            raise ValueError("seen_tool_call_fingerprints must match tool_call_count.")

        _require_unique_values(
            self.seen_tool_call_fingerprints,
            field_name="seen_tool_call_fingerprints",
        )
        _require_unique_values(
            self.tool_call_ids,
            field_name="tool_call_ids",
        )
        _require_unique_values(
            self.retrieval_query_ids,
            field_name="retrieval_query_ids",
        )
        _require_unique_values(
            self.retrieved_chunk_ids,
            field_name="retrieved_chunk_ids",
        )
        _require_unique_values(
            self.service_status_tool_call_ids,
            field_name="service_status_tool_call_ids",
        )

        if self.retrieved_chunk_ids and not self.retrieval_query_ids:
            raise ValueError("retrieved_chunk_ids require retrieval_query_ids.")

        if not set(self.service_status_tool_call_ids).issubset(self.tool_call_ids):
            raise ValueError("service_status_tool_call_ids must reference persisted tool calls.")

        if self.analysis_completion is not None and self.decision_turn_count == 0:
            raise ValueError("analysis_completion requires at least one decision turn.")

        if self.recommendation_invocation_id is not None and self.analysis_completion is None:
            raise ValueError("recommendation_invocation_id requires analysis_completion.")

        if self.recommendation_id is not None and self.recommendation_invocation_id is None:
            raise ValueError("recommendation_id requires recommendation_invocation_id.")

        return self

    def to_graph_state(self) -> ControlledSupportGraphState:
        """Return a JSON-compatible state representation."""

        return cast(
            ControlledSupportGraphState,
            self.model_dump(mode="json"),
        )


def create_initial_controlled_support_state(
    *,
    workspace_id: UUID,
    ticket_id: UUID,
    agent_run_id: UUID,
) -> ControlledSupportGraphState:
    """Create the minimal valid state for a new controlled workflow."""

    snapshot = ControlledSupportGraphStateSnapshot(
        state_schema_version=CONTROLLED_SUPPORT_STATE_SCHEMA_VERSION,
        workflow_name=CONTROLLED_SUPPORT_WORKFLOW_NAME,
        workflow_version=CONTROLLED_SUPPORT_WORKFLOW_VERSION,
        graph_version=CONTROLLED_SUPPORT_GRAPH_VERSION,
        workspace_id=workspace_id,
        ticket_id=ticket_id,
        agent_run_id=agent_run_id,
    )

    return snapshot.to_graph_state()


def validate_controlled_support_state(
    state: Mapping[str, object],
) -> ControlledSupportGraphStateSnapshot:
    """Validate initial or recovered state before workflow execution."""

    try:
        return ControlledSupportGraphStateSnapshot.model_validate(dict(state))
    except ValidationError as exc:
        raise GraphStateIncompatibleError() from exc


def _require_unique_values(
    values: tuple[object, ...],
    *,
    field_name: str,
) -> None:
    if len(set(values)) != len(values):
        raise ValueError(f"{field_name} must contain unique values.")
