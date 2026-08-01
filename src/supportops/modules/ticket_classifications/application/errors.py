"""Expected ticket-classification inspection errors."""


class TicketClassificationNotFoundError(Exception):
    """Raised when a scoped classification lookup does not resolve."""
