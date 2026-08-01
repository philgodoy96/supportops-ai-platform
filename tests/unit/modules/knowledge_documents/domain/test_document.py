"""Tests for knowledge-document identity and activation."""

from dataclasses import FrozenInstanceError
from datetime import UTC, datetime
from decimal import Decimal
from uuid import UUID

import pytest

from supportops.modules.knowledge_documents.domain.models import (
    Document,
    DocumentMediaType,
    DocumentVersion,
    KnowledgeIndexProfile,
)

WORKSPACE_ID = UUID("032c8c87-57cc-4d14-bfbd-04968b4e8cd4")
OTHER_WORKSPACE_ID = UUID("4aefba3b-b57e-47d1-889e-bb28762fa1ed")
DOCUMENT_ID = UUID("276046a2-28ec-4cb1-8bb6-a2ff70f9064b")
VERSION_ID = UUID("09036916-84cf-4a58-bdf4-09bc52716ec5")
NOW = datetime(2026, 8, 1, 18, 0, tzinfo=UTC)
LATER = datetime(2026, 8, 1, 18, 5, tzinfo=UTC)


def create_ready_version() -> DocumentVersion:
    version = DocumentVersion.create_pending(
        workspace_id=WORKSPACE_ID,
        document_id=DOCUMENT_ID,
        version_number=1,
        media_type=DocumentMediaType.TEXT_MARKDOWN,
        content=("# Database incidents\nRestart the connection pool.\n"),
        document_version_id=VERSION_ID,
        now=NOW,
    ).bind_index_profile(
        KnowledgeIndexProfile(
            chunking_strategy="markdown-token",
            chunking_version="v1",
            tokenizer_encoding="cl100k_base",
            embedding_provider="mock",
            embedding_model="mock-hashing-embedding-v1",
            embedding_dimensions=64,
            knowledge_collection=("supportops-knowledge-mock-v1"),
            knowledge_vector_name="dense",
        ),
        now=NOW,
    )

    return version.mark_ready(
        chunk_count=1,
        embedding_input_tokens=12,
        embedding_estimated_cost_usd=Decimal("0"),
        embedding_pricing_catalog_version=("supportops-embedding-pricing-2026-08-01"),
        indexed_at=LATER,
    )


def test_document_create_normalizes_metadata() -> None:
    document = Document.create(
        workspace_id=WORKSPACE_ID,
        title="  Database Incident Runbook  ",
        external_reference=("  runbook-database-incidents  "),
        document_id=DOCUMENT_ID,
        now=NOW,
    )

    assert document.title == "Database Incident Runbook"
    assert document.external_reference == "runbook-database-incidents"
    assert document.active_version_id is None
    assert document.created_at == NOW
    assert document.updated_at == NOW


@pytest.mark.parametrize(
    "title",
    ["", "   ", "a" * 201],
)
def test_document_create_rejects_invalid_title(
    title: str,
) -> None:
    with pytest.raises(ValueError):
        Document.create(
            workspace_id=WORKSPACE_ID,
            title=title,
        )


@pytest.mark.parametrize(
    "external_reference",
    ["", "   ", "a" * 129],
)
def test_document_create_rejects_invalid_external_reference(
    external_reference: str,
) -> None:
    with pytest.raises(ValueError):
        Document.create(
            workspace_id=WORKSPACE_ID,
            title="Database Incident Runbook",
            external_reference=external_reference,
        )


def test_document_workspace_ownership_is_immutable() -> None:
    document = Document.create(
        workspace_id=WORKSPACE_ID,
        title="Database Incident Runbook",
    )

    with pytest.raises(FrozenInstanceError):
        document.workspace_id = OTHER_WORKSPACE_ID  # type: ignore[misc]


def test_document_activates_ready_owned_version() -> None:
    document = Document.create(
        workspace_id=WORKSPACE_ID,
        title="Database Incident Runbook",
        document_id=DOCUMENT_ID,
        now=NOW,
    )

    activated = document.activate_version(
        create_ready_version(),
        now=LATER,
    )

    assert activated.active_version_id == VERSION_ID
    assert activated.updated_at == LATER
    assert document.active_version_id is None


def test_document_activation_is_idempotent() -> None:
    document = Document.create(
        workspace_id=WORKSPACE_ID,
        title="Database Incident Runbook",
        document_id=DOCUMENT_ID,
        now=NOW,
    )
    version = create_ready_version()
    activated = document.activate_version(
        version,
        now=LATER,
    )

    assert activated.activate_version(version, now=LATER) is activated


def test_document_rejects_pending_version_activation() -> None:
    document = Document.create(
        workspace_id=WORKSPACE_ID,
        title="Database Incident Runbook",
        document_id=DOCUMENT_ID,
        now=NOW,
    )
    pending = DocumentVersion.create_pending(
        workspace_id=WORKSPACE_ID,
        document_id=DOCUMENT_ID,
        version_number=1,
        media_type=DocumentMediaType.TEXT_MARKDOWN,
        content="# Database incidents\n",
        now=NOW,
    )

    with pytest.raises(
        ValueError,
        match="Only ready document versions may become active",
    ):
        document.activate_version(
            pending,
            now=LATER,
        )


def test_document_rejects_cross_workspace_activation() -> None:
    document = Document.create(
        workspace_id=OTHER_WORKSPACE_ID,
        title="Database Incident Runbook",
        document_id=DOCUMENT_ID,
        now=NOW,
    )

    with pytest.raises(
        ValueError,
        match="same document and workspace",
    ):
        document.activate_version(
            create_ready_version(),
            now=LATER,
        )
