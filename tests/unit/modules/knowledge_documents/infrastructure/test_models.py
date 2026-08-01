"""Unit tests for knowledge-document persistence models."""

from datetime import UTC, datetime
from decimal import Decimal
from typing import cast
from uuid import UUID

from sqlalchemy import (
    CheckConstraint,
    ForeignKeyConstraint,
    Index,
    Table,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB

from supportops.modules.knowledge_documents.domain.models import (
    Document,
    DocumentChunk,
    DocumentMediaType,
    DocumentVersion,
    KnowledgeIndexProfile,
)
from supportops.modules.knowledge_documents.infrastructure.models import (
    DocumentChunkRecord,
    DocumentRecord,
    DocumentVersionRecord,
)

WORKSPACE_ID = UUID("032c8c87-57cc-4d14-bfbd-04968b4e8cd4")
DOCUMENT_ID = UUID("276046a2-28ec-4cb1-8bb6-a2ff70f9064b")
VERSION_ID = UUID("09036916-84cf-4a58-bdf4-09bc52716ec5")
CHUNK_ID = UUID("d841f8a8-bae3-5d93-bf93-211b9d96eaa6")
CREATED_AT = datetime(
    2026,
    8,
    1,
    18,
    0,
    tzinfo=UTC,
)
INDEXED_AT = datetime(
    2026,
    8,
    1,
    18,
    5,
    tzinfo=UTC,
)


def create_document() -> Document:
    """Create one deterministic document for mapping tests."""

    return Document(
        id=DOCUMENT_ID,
        workspace_id=WORKSPACE_ID,
        title="Database Incident Runbook",
        external_reference="runbook-database-incidents",
        active_version_id=VERSION_ID,
        created_at=CREATED_AT,
        updated_at=INDEXED_AT,
    )


def create_ready_version() -> DocumentVersion:
    """Create one ready version with complete index provenance."""

    pending = DocumentVersion.create_pending(
        workspace_id=WORKSPACE_ID,
        document_id=DOCUMENT_ID,
        version_number=1,
        media_type=DocumentMediaType.TEXT_MARKDOWN,
        content=(
            "# Connection exhaustion\n\nRestart the connection pool before increasing limits.\n"
        ),
        document_version_id=VERSION_ID,
        now=CREATED_AT,
    )
    profiled = pending.bind_index_profile(
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
        now=CREATED_AT,
    )
    return profiled.mark_ready(
        chunk_count=1,
        embedding_input_tokens=20,
        embedding_estimated_cost_usd=Decimal("0"),
        embedding_pricing_catalog_version=("supportops-embedding-pricing-2026-08-01"),
        indexed_at=INDEXED_AT,
    )


def create_chunk() -> DocumentChunk:
    """Create one deterministic chunk for mapping tests."""

    chunk = DocumentChunk.create(
        document_version=create_ready_version(),
        ordinal=0,
        section_path=("Connection exhaustion",),
        content=("Restart the connection pool before increasing limits."),
        token_count=9,
        now=CREATED_AT,
    )

    assert chunk.id != CHUNK_ID
    return chunk


def constraint_names(
    table: Table,
) -> set[str]:
    """Return named checks and unique constraints from one table."""

    return {
        constraint.name
        for constraint in table.constraints
        if isinstance(
            constraint,
            (CheckConstraint, UniqueConstraint),
        )
        and isinstance(constraint.name, str)
    }


def foreign_key_constraint(
    table: Table,
    *,
    name: str,
) -> ForeignKeyConstraint:
    """Return one named composite foreign-key constraint."""

    return next(
        constraint for constraint in table.foreign_key_constraints if constraint.name == name
    )


def index_by_name(
    table: Table,
    *,
    name: str,
) -> Index:
    """Return one named index."""

    return next(index for index in table.indexes if index.name == name)


def test_document_record_round_trip_preserves_domain_values() -> None:
    document = create_document()

    record = DocumentRecord.from_domain(document)

    assert record.to_domain() == document


def test_document_version_record_round_trip_preserves_domain_values() -> None:
    version = create_ready_version()

    record = DocumentVersionRecord.from_domain(version)

    assert record.to_domain() == version


def test_document_chunk_record_round_trip_preserves_domain_values() -> None:
    chunk = create_chunk()

    record = DocumentChunkRecord.from_domain(chunk)

    assert record.section_path == ["Connection exhaustion"]
    assert record.to_domain() == chunk


def test_document_table_declares_expected_constraints() -> None:
    table = cast(Table, DocumentRecord.__table__)

    assert {
        "uq_knowledge_documents_workspace_id",
        "uq_knowledge_documents_workspace_external_reference",
        "ck_knowledge_documents_document_title_trimmed",
        "ck_knowledge_documents_document_title_length",
        ("ck_knowledge_documents_document_external_reference_format"),
        "ck_knowledge_documents_document_timestamp_order",
    }.issubset(constraint_names(table))


def test_document_table_declares_workspace_foreign_key() -> None:
    table = cast(Table, DocumentRecord.__table__)
    foreign_keys = [
        foreign_key
        for foreign_key in table.c.workspace_id.foreign_keys
        if foreign_key.target_fullname == "workspaces.id"
    ]

    assert len(foreign_keys) == 1
    assert foreign_keys[0].ondelete == "RESTRICT"


def test_document_table_declares_owned_active_version_foreign_key() -> None:
    table = cast(Table, DocumentRecord.__table__)
    constraint = foreign_key_constraint(
        table,
        name="fk_knowledge_documents_active_version",
    )

    assert constraint.use_alter
    assert constraint.ondelete == "RESTRICT"
    assert [element.target_fullname for element in constraint.elements] == [
        "knowledge_document_versions.workspace_id",
        "knowledge_document_versions.document_id",
        "knowledge_document_versions.id",
    ]


def test_document_table_declares_listing_and_active_indexes() -> None:
    table = cast(Table, DocumentRecord.__table__)
    listing_index = index_by_name(
        table,
        name="ix_knowledge_documents_workspace_created_id",
    )
    active_index = index_by_name(
        table,
        name=("ix_knowledge_documents_workspace_active_version"),
    )

    assert [str(expression) for expression in listing_index.expressions] == [
        "knowledge_documents.workspace_id",
        "knowledge_documents.created_at DESC",
        "knowledge_documents.id DESC",
    ]
    assert [str(expression) for expression in active_index.expressions] == [
        "knowledge_documents.workspace_id",
        "knowledge_documents.active_version_id",
    ]
    assert active_index.dialect_options["postgresql"]["where"] is not None


def test_document_version_table_declares_expected_constraints() -> None:
    table = cast(Table, DocumentVersionRecord.__table__)

    assert {
        ("uq_knowledge_document_versions_workspace_document_id"),
        ("uq_knowledge_document_versions_document_version_number"),
        ("uq_knowledge_document_versions_document_content_sha256"),
        ("ck_knowledge_document_versions_document_version_number_positive"),
        ("ck_knowledge_document_versions_document_version_media_type"),
        ("ck_knowledge_document_versions_document_version_content_non_whitespace"),
        ("ck_knowledge_document_versions_document_version_content_sha256"),
        ("ck_knowledge_document_versions_document_version_status"),
        ("ck_knowledge_document_versions_document_version_index_profile_completeness"),
        ("ck_knowledge_document_versions_document_version_embedding_dimensions_positive"),
        ("ck_knowledge_document_versions_document_version_embedding_tokens_non_negative"),
        ("ck_knowledge_document_versions_document_version_embedding_cost_non_negative"),
        ("ck_knowledge_document_versions_document_version_chunk_count_non_negative"),
        ("ck_knowledge_document_versions_document_version_status_state"),
        ("ck_knowledge_document_versions_document_version_timestamp_order"),
        ("ck_knowledge_document_versions_document_version_indexed_timestamp_order"),
    }.issubset(constraint_names(table))


def test_document_version_table_declares_document_ownership_foreign_key() -> None:
    table = cast(Table, DocumentVersionRecord.__table__)
    constraint = foreign_key_constraint(
        table,
        name="fk_knowledge_document_versions_document",
    )

    assert constraint.ondelete == "RESTRICT"
    assert [element.target_fullname for element in constraint.elements] == [
        "knowledge_documents.workspace_id",
        "knowledge_documents.id",
    ]


def test_document_version_table_declares_listing_index() -> None:
    table = cast(Table, DocumentVersionRecord.__table__)
    index = index_by_name(
        table,
        name=("ix_knowledge_document_versions_workspace_document_number"),
    )

    assert [str(expression) for expression in index.expressions] == [
        "knowledge_document_versions.workspace_id",
        "knowledge_document_versions.document_id",
        "knowledge_document_versions.version_number DESC",
        "knowledge_document_versions.id DESC",
    ]


def test_document_chunk_table_declares_expected_constraints() -> None:
    table = cast(Table, DocumentChunkRecord.__table__)

    assert {
        ("uq_knowledge_document_chunks_version_ordinal"),
        ("ck_knowledge_document_chunks_document_chunk_ordinal_non_negative"),
        ("ck_knowledge_document_chunks_document_chunk_section_path"),
        ("ck_knowledge_document_chunks_document_chunk_content_non_whitespace"),
        ("ck_knowledge_document_chunks_document_chunk_content_sha256"),
        ("ck_knowledge_document_chunks_document_chunk_token_count_positive"),
        ("ck_knowledge_document_chunks_document_chunk_chunking_strategy_format"),
        ("ck_knowledge_document_chunks_document_chunk_chunking_version_format"),
    }.issubset(constraint_names(table))


def test_document_chunk_table_declares_version_ownership_foreign_key() -> None:
    table = cast(Table, DocumentChunkRecord.__table__)
    constraint = foreign_key_constraint(
        table,
        name="fk_knowledge_document_chunks_version",
    )

    assert constraint.ondelete == "RESTRICT"
    assert [element.target_fullname for element in constraint.elements] == [
        "knowledge_document_versions.workspace_id",
        "knowledge_document_versions.document_id",
        "knowledge_document_versions.id",
    ]


def test_document_chunk_table_uses_jsonb_section_path() -> None:
    table = cast(Table, DocumentChunkRecord.__table__)

    assert isinstance(table.c.section_path.type, JSONB)
    assert not table.c.section_path.nullable


def test_document_chunk_table_declares_hydration_index() -> None:
    table = cast(Table, DocumentChunkRecord.__table__)
    index = index_by_name(
        table,
        name=("ix_knowledge_document_chunks_workspace_version_id"),
    )

    assert [str(expression) for expression in index.expressions] == [
        "knowledge_document_chunks.workspace_id",
        "knowledge_document_chunks.document_version_id",
        "knowledge_document_chunks.id",
    ]
