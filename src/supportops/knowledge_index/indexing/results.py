"""Result types for explicit document-version indexing."""

from dataclasses import dataclass
from decimal import Decimal

from supportops.modules.knowledge_documents.domain.models import (
    DocumentVersion,
    DocumentVersionStatus,
)


@dataclass(frozen=True, slots=True)
class IndexDocumentVersionResult:
    """Successful ready-version indexing outcome."""

    version: DocumentVersion
    already_ready: bool

    def __post_init__(self) -> None:
        if self.version.status is not DocumentVersionStatus.READY:
            raise ValueError("Indexing results require a ready document version.")

    @property
    def chunk_count(self) -> int:
        """Return the verified authoritative chunk count."""

        assert self.version.chunk_count is not None
        return self.version.chunk_count

    @property
    def embedding_input_tokens(self) -> int:
        """Return provider-reported input-token usage."""

        assert self.version.embedding_input_tokens is not None
        return self.version.embedding_input_tokens

    @property
    def estimated_cost_usd(self) -> Decimal | None:
        """Return the estimated embedding cost when pricing is known."""

        return self.version.embedding_estimated_cost_usd

    @property
    def pricing_catalog_version(self) -> str:
        """Return the pricing catalog used for the estimate."""

        assert self.version.embedding_pricing_catalog_version is not None
        return self.version.embedding_pricing_catalog_version
