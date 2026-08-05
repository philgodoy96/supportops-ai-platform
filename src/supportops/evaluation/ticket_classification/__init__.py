"""Versioned evaluation foundation for ticket classification."""

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
    "TicketClassificationFailureAnalysis",
    "TicketClassificationFailureAnalysisContent",
    "TicketClassificationFailureAnalysisError",
    "TicketClassificationFailureEvidenceKind",
    "TicketClassificationFailureEvidenceSummary",
    "TicketClassificationFailureObservation",
    "TicketClassificationFailureSafetyImpact",
    "TicketClassificationFailureType",
    "load_ticket_classification_failure_analysis",
]
