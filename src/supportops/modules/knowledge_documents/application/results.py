"""Application result types for knowledge-document operations."""

from dataclasses import dataclass

from supportops.modules.knowledge_documents.domain.models import (
    Document,
    DocumentVersion,
)


@dataclass(frozen=True, slots=True)
class CreateDocumentResult:
    """Document identity and its atomically created first version."""

    document: Document
    version: DocumentVersion
