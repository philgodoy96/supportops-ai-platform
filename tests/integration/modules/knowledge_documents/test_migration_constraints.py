"""Integration tests for versioned knowledge migration constraints."""

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from hashlib import sha256
from uuid import UUID

import pytest
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncEngine

pytestmark = pytest.mark.integration

WORKSPACE_ID = UUID("7c639906-170a-4a6e-8c43-42b45aa2604a")
DOCUMENT_ID = UUID("5f63db19-89d2-48f9-806d-4e05264657af")
DOCUMENT_VERSION_ID = UUID("050c7c44-7f9f-4f5b-8082-b03b5d5e1711")
CHUNK_ID = UUID("c219597f-5455-5b4d-996f-607421f69ee5")
CREATED_AT = datetime(2026, 8, 1, 22, 38, tzinfo=UTC)
READY_AT = CREATED_AT + timedelta(minutes=5)
DOCUMENT_CONTENT = "# Database incidents\nRestart the connection pool.\n"
DOCUMENT_CONTENT_SHA256 = sha256(DOCUMENT_CONTENT.encode("utf-8")).hexdigest()
CHUNK_CONTENT = "Restart the connection pool."
CHUNK_CONTENT_SHA256 = sha256(CHUNK_CONTENT.encode("utf-8")).hexdigest()


async def insert_profiled_pending_version(
    engine: AsyncEngine,
) -> None:
    """Insert one workspace, document, and profiled pending version."""

    async with engine.begin() as connection:
        await connection.execute(
            text(
                """
                INSERT INTO workspaces (
                    id,
                    name,
                    slug,
                    created_at,
                    updated_at
                )
                VALUES (
                    :id,
                    :name,
                    :slug,
                    :created_at,
                    :updated_at
                )
                """
            ),
            {
                "id": WORKSPACE_ID,
                "name": "Knowledge Migration Workspace",
                "slug": "knowledge-migration-workspace",
                "created_at": CREATED_AT,
                "updated_at": CREATED_AT,
            },
        )
        await connection.execute(
            text(
                """
                INSERT INTO knowledge_documents (
                    id,
                    workspace_id,
                    title,
                    external_reference,
                    active_version_id,
                    created_at,
                    updated_at
                )
                VALUES (
                    :id,
                    :workspace_id,
                    :title,
                    :external_reference,
                    NULL,
                    :created_at,
                    :updated_at
                )
                """
            ),
            {
                "id": DOCUMENT_ID,
                "workspace_id": WORKSPACE_ID,
                "title": "Database Incident Runbook",
                "external_reference": "runbook-database-incidents",
                "created_at": CREATED_AT,
                "updated_at": CREATED_AT,
            },
        )
        await connection.execute(
            text(
                """
                INSERT INTO knowledge_document_versions (
                    id,
                    workspace_id,
                    document_id,
                    version_number,
                    media_type,
                    content,
                    content_sha256,
                    status,
                    chunking_strategy,
                    chunking_version,
                    tokenizer_encoding,
                    embedding_provider,
                    embedding_model,
                    embedding_dimensions,
                    knowledge_collection,
                    knowledge_vector_name,
                    embedding_input_tokens,
                    embedding_estimated_cost_usd,
                    embedding_pricing_catalog_version,
                    chunk_count,
                    indexed_at,
                    last_error_code,
                    created_at,
                    updated_at
                )
                VALUES (
                    :id,
                    :workspace_id,
                    :document_id,
                    1,
                    'text/markdown',
                    :content,
                    :content_sha256,
                    'pending',
                    'markdown-token',
                    'v1',
                    'cl100k_base',
                    'mock',
                    'mock-hashing-embedding-v1',
                    64,
                    'supportops-knowledge-mock-v1',
                    'dense',
                    NULL,
                    NULL,
                    NULL,
                    NULL,
                    NULL,
                    NULL,
                    :created_at,
                    :updated_at
                )
                """
            ),
            {
                "id": DOCUMENT_VERSION_ID,
                "workspace_id": WORKSPACE_ID,
                "document_id": DOCUMENT_ID,
                "content": DOCUMENT_CONTENT,
                "content_sha256": DOCUMENT_CONTENT_SHA256,
                "created_at": CREATED_AT,
                "updated_at": CREATED_AT,
            },
        )


async def insert_matching_chunk(
    engine: AsyncEngine,
) -> None:
    """Insert one chunk matching the persisted version profile."""

    async with engine.begin() as connection:
        await connection.execute(
            text(
                """
                INSERT INTO knowledge_document_chunks (
                    id,
                    workspace_id,
                    document_id,
                    document_version_id,
                    ordinal,
                    section_path,
                    content,
                    content_sha256,
                    token_count,
                    chunking_strategy,
                    chunking_version,
                    created_at
                )
                VALUES (
                    :id,
                    :workspace_id,
                    :document_id,
                    :document_version_id,
                    0,
                    '["Database incidents"]'::jsonb,
                    :content,
                    :content_sha256,
                    6,
                    'markdown-token',
                    'v1',
                    :created_at
                )
                """
            ),
            {
                "id": CHUNK_ID,
                "workspace_id": WORKSPACE_ID,
                "document_id": DOCUMENT_ID,
                "document_version_id": DOCUMENT_VERSION_ID,
                "content": CHUNK_CONTENT,
                "content_sha256": CHUNK_CONTENT_SHA256,
                "created_at": CREATED_AT,
            },
        )


async def mark_version_ready(
    engine: AsyncEngine,
) -> None:
    """Transition the profiled version to ready with complete provenance."""

    async with engine.begin() as connection:
        await connection.execute(
            text(
                """
                UPDATE knowledge_document_versions
                SET
                    status = 'ready',
                    embedding_input_tokens = :embedding_input_tokens,
                    embedding_estimated_cost_usd = :estimated_cost,
                    embedding_pricing_catalog_version = :catalog_version,
                    chunk_count = 1,
                    indexed_at = :indexed_at,
                    updated_at = :updated_at
                WHERE id = :document_version_id
                """
            ),
            {
                "embedding_input_tokens": 18,
                "estimated_cost": Decimal("0"),
                "catalog_version": ("supportops-embedding-pricing-2026-08-01"),
                "indexed_at": READY_AT,
                "updated_at": READY_AT,
                "document_version_id": DOCUMENT_VERSION_ID,
            },
        )


async def test_pending_version_cannot_be_activated_but_ready_version_can(
    postgresql_engine: AsyncEngine,
    clean_business_tables: None,
) -> None:
    """Require explicit readiness before changing the active pointer."""

    await insert_profiled_pending_version(postgresql_engine)
    await insert_matching_chunk(postgresql_engine)

    with pytest.raises(IntegrityError):
        async with postgresql_engine.begin() as connection:
            await connection.execute(
                text(
                    """
                    UPDATE knowledge_documents
                    SET
                        active_version_id = :document_version_id,
                        updated_at = :updated_at
                    WHERE id = :document_id
                    """
                ),
                {
                    "document_version_id": DOCUMENT_VERSION_ID,
                    "updated_at": READY_AT,
                    "document_id": DOCUMENT_ID,
                },
            )

    await mark_version_ready(postgresql_engine)

    async with postgresql_engine.begin() as connection:
        await connection.execute(
            text(
                """
                UPDATE knowledge_documents
                SET
                    active_version_id = :document_version_id,
                    updated_at = :updated_at
                WHERE id = :document_id
                """
            ),
            {
                "document_version_id": DOCUMENT_VERSION_ID,
                "updated_at": READY_AT,
                "document_id": DOCUMENT_ID,
            },
        )
        active_version_id = await connection.scalar(
            text(
                """
                SELECT active_version_id
                FROM knowledge_documents
                WHERE id = :document_id
                """
            ),
            {"document_id": DOCUMENT_ID},
        )

    assert active_version_id == DOCUMENT_VERSION_ID


async def test_ready_version_cannot_be_rewritten(
    postgresql_engine: AsyncEngine,
    clean_business_tables: None,
) -> None:
    """Protect source content and indexing provenance after readiness."""

    await insert_profiled_pending_version(postgresql_engine)
    await insert_matching_chunk(postgresql_engine)
    await mark_version_ready(postgresql_engine)

    with pytest.raises(IntegrityError):
        async with postgresql_engine.begin() as connection:
            await connection.execute(
                text(
                    """
                    UPDATE knowledge_document_versions
                    SET content = :content
                    WHERE id = :document_version_id
                    """
                ),
                {
                    "content": "Rewritten source content.",
                    "document_version_id": DOCUMENT_VERSION_ID,
                },
            )


async def test_chunk_profile_must_match_document_version_profile(
    postgresql_engine: AsyncEngine,
    clean_business_tables: None,
) -> None:
    """Reject chunks whose persisted chunking profile differs."""

    await insert_profiled_pending_version(postgresql_engine)

    with pytest.raises(IntegrityError):
        async with postgresql_engine.begin() as connection:
            await connection.execute(
                text(
                    """
                    INSERT INTO knowledge_document_chunks (
                        id,
                        workspace_id,
                        document_id,
                        document_version_id,
                        ordinal,
                        section_path,
                        content,
                        content_sha256,
                        token_count,
                        chunking_strategy,
                        chunking_version,
                        created_at
                    )
                    VALUES (
                        :id,
                        :workspace_id,
                        :document_id,
                        :document_version_id,
                        0,
                        '[]'::jsonb,
                        :content,
                        :content_sha256,
                        6,
                        'markdown-token',
                        'v2',
                        :created_at
                    )
                    """
                ),
                {
                    "id": CHUNK_ID,
                    "workspace_id": WORKSPACE_ID,
                    "document_id": DOCUMENT_ID,
                    "document_version_id": DOCUMENT_VERSION_ID,
                    "content": CHUNK_CONTENT,
                    "content_sha256": CHUNK_CONTENT_SHA256,
                    "created_at": CREATED_AT,
                },
            )

    await insert_matching_chunk(postgresql_engine)

    async with postgresql_engine.connect() as connection:
        chunk_count = await connection.scalar(
            text(
                """
                SELECT count(*)
                FROM knowledge_document_chunks
                WHERE document_version_id = :document_version_id
                """
            ),
            {"document_version_id": DOCUMENT_VERSION_ID},
        )

    assert chunk_count == 1
