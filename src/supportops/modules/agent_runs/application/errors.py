"""Expected AgentRun application errors."""


class AgentRunNotFoundError(Exception):
    """Raised when a workspace-scoped AgentRun lookup does not resolve."""
