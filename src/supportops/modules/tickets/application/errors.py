"""Expected support ticket application errors."""


class TicketNotFoundError(Exception):
    """Raised when a scoped ticket lookup does not resolve."""


class TicketExternalReferenceConflictApplicationError(Exception):
    """Raised when an external reference conflicts in one workspace."""
