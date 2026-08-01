"""Expected versioned knowledge-document application errors."""


class DocumentNotFoundError(Exception):
    """Raised when a scoped document lookup does not resolve."""


class DocumentVersionNotFoundError(Exception):
    """Raised when a scoped document-version lookup does not resolve."""


class DocumentExternalReferenceConflictApplicationError(Exception):
    """Raised when an external reference conflicts in one workspace."""


class DocumentVersionContentConflictApplicationError(Exception):
    """Raised when normalized content already exists for one document."""


class DocumentVersionNumberConflictApplicationError(Exception):
    """Raised when concurrent version allocation conflicts."""


class DocumentVersionNotReadyError(Exception):
    """Raised when a non-ready version is selected for activation."""
