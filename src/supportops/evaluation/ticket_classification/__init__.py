"""Versioned evaluation foundation for ticket classification."""

from supportops.evaluation.ticket_classification.comparison import (
    TicketClassificationComparisonEvidenceKind,
    TicketClassificationMetricDelta,
    TicketClassificationOptionalMetricDelta,
    TicketClassificationPairedComparison,
    TicketClassificationPairedComparisonContent,
    TicketClassificationPairedComparisonError,
    TicketClassificationPairedGateEvaluation,
    TicketClassificationPairedGateStatus,
    compare_ticket_classification_prediction_sets,
    load_ticket_classification_paired_comparison,
    write_ticket_classification_paired_comparison,
)
from supportops.evaluation.ticket_classification.failure_analysis import (
    TicketClassificationFailureAnalysis,
    TicketClassificationFailureAnalysisContent,
    TicketClassificationFailureAnalysisError,
    TicketClassificationFailureEvidenceKind,
    TicketClassificationFailureEvidenceSummary,
    TicketClassificationFailureObservation,
    TicketClassificationFailureSafetyImpact,
    TicketClassificationFailureType,
    load_ticket_classification_failure_analysis,
)

__all__ = [
    "TicketClassificationComparisonEvidenceKind",
    "TicketClassificationFailureAnalysis",
    "TicketClassificationFailureAnalysisContent",
    "TicketClassificationFailureAnalysisError",
    "TicketClassificationFailureEvidenceKind",
    "TicketClassificationFailureEvidenceSummary",
    "TicketClassificationFailureObservation",
    "TicketClassificationFailureSafetyImpact",
    "TicketClassificationFailureType",
    "TicketClassificationMetricDelta",
    "TicketClassificationOptionalMetricDelta",
    "TicketClassificationPairedComparison",
    "TicketClassificationPairedComparisonContent",
    "TicketClassificationPairedComparisonError",
    "TicketClassificationPairedGateEvaluation",
    "TicketClassificationPairedGateStatus",
    "compare_ticket_classification_prediction_sets",
    "load_ticket_classification_failure_analysis",
    "load_ticket_classification_paired_comparison",
    "write_ticket_classification_paired_comparison",
]
