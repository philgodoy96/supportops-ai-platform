"""Expected workspace application errors."""


class WorkspaceNotFoundError(Exception):
    """Raised when a requested workspace does not exist."""


class WorkspaceSlugConflictApplicationError(Exception):
    """Raised when a workspace slug is already in use."""
