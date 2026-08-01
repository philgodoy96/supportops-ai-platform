"""Tests for document versions and indexing lifecycle state."""

from dataclasses import FrozenInstanceError, replace
from datetime import UTC, datetime
from decimal import Decimal
from uuid import UUID

import pytest

from supportops.modules.knowledge_documents.domain.content import (
    compute_content_sha256,
)
from supportops.modules.knowledge_documents.domain.models import (
    DocumentMediaType,
    DocumentVersion,
    DocumentVersionStatus,
    KnowledgeIndexProfile,
)

WORKSPACE_ID = UUID("032c8c87-57cc-4d14-bfbd-04968b4e8cd4")
DOCUMENT_ID = UUID("276046a2-28ec-4cb1-8bb6-a2ff70f9064b")
NOW = datetime(2026, 8, 1, 18, 0, tzinfo=UTC)
LATER = datetime(2026, 8, 1, 18, 5, tzinfo=UTC)
LATEST = datetime(2026, 8, 1, 18, 10, tzinfo=UTC)


def create_pending_version() -> DocumentVersion:
    return DocumentVersion.create_pending(
        workspace_id=WORKSPACE_ID,
        document_id=DOCUMENT_ID,
        version_number=1,
        media_type=DocumentMediaType.TEXT_MARKDOWN,
        content=("\ufeff# Database incidents\r\nRestart the connection pool.\r"),
        now=NOW,
    )


def create_profile() -> KnowledgeIndexProfile:
    return KnowledgeIndexProfile(
        chunking_strategy="markdown-token",
        chunking_version="v1",
        tokenizer_encoding="cl100k_base",
        embedding_provider="mock",
        embedding_model="mock-hashing-embedding-v1",
        embedding_dimensions=64,
        knowledge_collection="supportops-knowledge-mock-v1",
        knowledge_vector_name="dense",
    )


def test_version_normalizes_and_hashes_stored_content() -> None:
    version = create_pending_version()

    assert version.content == ("# Database incidents\nRestart the connection pool.\n")
    assert version.content_sha256 == compute_content_sha256(version.content)
    assert version.status is DocumentVersionStatus.PENDING
    assert version.index_profile is None
    assert version.version_number == 1


@pytest.mark.parametrize(
    "media_type",
    [
        DocumentMediaType.TEXT_PLAIN,
        DocumentMediaType.TEXT_MARKDOWN,
    ],
)
def test_version_accepts_supported_media_types(
    media_type: DocumentMediaType,
) -> None:
    version = DocumentVersion.create_pending(
        workspace_id=WORKSPACE_ID,
        document_id=DOCUMENT_ID,
        version_number=1,
        media_type=media_type,
        content="Restart the connection pool.",
    )

    assert version.media_type is media_type


def test_version_rejects_unsupported_media_type() -> None:
    version = create_pending_version()

    with pytest.raises(
        ValueError,
        match="supported DocumentMediaType",
    ):
        replace(
            version,
            media_type="application/pdf",  # type: ignore[arg-type]
        )


def test_version_rejects_content_hash_mismatch() -> None:
    version = create_pending_version()

    with pytest.raises(
        ValueError,
        match="content_sha256 must match",
    ):
        replace(
            version,
            content_sha256="0" * 64,
        )


def test_version_content_is_immutable() -> None:
    version = create_pending_version()

    with pytest.raises(FrozenInstanceError):
        version.content = "Changed"  # type: ignore[misc]


def test_version_binds_profile_once() -> None:
    version = create_pending_version()
    profile = create_profile()

    bound = version.bind_index_profile(
        profile,
        now=LATER,
    )

    assert bound.index_profile == profile
    assert bound.updated_at == LATER
    assert (
        bound.bind_index_profile(
            profile,
            now=LATEST,
        )
        is bound
    )


def test_version_rejects_mismatched_retry_profile() -> None:
    version = create_pending_version().bind_index_profile(
        create_profile(),
        now=LATER,
    )
    mismatched = replace(
        create_profile(),
        embedding_dimensions=1536,
    )

    with pytest.raises(
        ValueError,
        match="does not match the persisted profile",
    ):
        version.bind_index_profile(
            mismatched,
            now=LATEST,
        )


def test_version_records_failure_and_prepares_retry() -> None:
    bound = create_pending_version().bind_index_profile(
        create_profile(),
        now=LATER,
    )

    failed = bound.mark_failed(
        error_code="embedding_timeout",
        chunk_count=3,
        now=LATEST,
    )
    retry = failed.prepare_retry(now=LATEST)

    assert failed.status is DocumentVersionStatus.FAILED
    assert failed.last_error_code == "embedding_timeout"
    assert failed.chunk_count == 3
    assert retry.status is DocumentVersionStatus.PENDING
    assert retry.index_profile == create_profile()
    assert retry.chunk_count == 3
    assert retry.last_error_code is None


def test_version_marks_ready_with_cost_provenance() -> None:
    bound = create_pending_version().bind_index_profile(
        create_profile(),
        now=LATER,
    )

    ready = bound.mark_ready(
        chunk_count=3,
        embedding_input_tokens=240,
        embedding_estimated_cost_usd=Decimal("0"),
        embedding_pricing_catalog_version=("supportops-embedding-pricing-2026-08-01"),
        indexed_at=LATEST,
    )

    assert ready.status is DocumentVersionStatus.READY
    assert ready.chunk_count == 3
    assert ready.embedding_input_tokens == 240
    assert ready.embedding_estimated_cost_usd == Decimal("0")
    assert ready.indexed_at == LATEST
    assert ready.last_error_code is None


def test_version_preserves_unknown_cost_as_none() -> None:
    bound = create_pending_version().bind_index_profile(
        create_profile(),
        now=LATER,
    )

    ready = bound.mark_ready(
        chunk_count=1,
        embedding_input_tokens=25,
        embedding_estimated_cost_usd=None,
        embedding_pricing_catalog_version=("supportops-embedding-pricing-2026-08-01"),
        indexed_at=LATEST,
    )

    assert ready.embedding_estimated_cost_usd is None


def test_version_rejects_partial_index_profile() -> None:
    version = create_pending_version()

    with pytest.raises(
        ValueError,
        match="populated or cleared together",
    ):
        replace(
            version,
            chunking_strategy="markdown-token",
        )


def test_version_rejects_ready_rewrite() -> None:
    ready = (
        create_pending_version()
        .bind_index_profile(
            create_profile(),
            now=LATER,
        )
        .mark_ready(
            chunk_count=1,
            embedding_input_tokens=25,
            embedding_estimated_cost_usd=Decimal("0"),
            embedding_pricing_catalog_version=("supportops-embedding-pricing-2026-08-01"),
            indexed_at=LATEST,
        )
    )

    with pytest.raises(
        ValueError,
        match="cannot be rewritten",
    ):
        ready.mark_ready(
            chunk_count=1,
            embedding_input_tokens=25,
            embedding_estimated_cost_usd=Decimal("0"),
            embedding_pricing_catalog_version=("supportops-embedding-pricing-2026-08-01"),
            indexed_at=LATEST,
        )
