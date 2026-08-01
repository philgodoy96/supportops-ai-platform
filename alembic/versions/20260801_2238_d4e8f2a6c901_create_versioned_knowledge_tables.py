"""create versioned knowledge tables

Revision ID: d4e8f2a6c901
Revises: b7d3f1e5a9c2
Create Date: 2026-08-01 22:38:00+00:00
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "d4e8f2a6c901"
down_revision: str | Sequence[str] | None = "b7d3f1e5a9c2"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Apply the migration."""
    op.create_table(
        "knowledge_documents",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("workspace_id", sa.Uuid(), nullable=False),
        sa.Column("title", sa.String(length=200), nullable=False),
        sa.Column(
            "external_reference",
            sa.String(length=128),
            nullable=True,
        ),
        sa.Column("active_version_id", sa.Uuid(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
        ),
        sa.CheckConstraint(
            "title = btrim(title)",
            name=op.f(
                "ck_knowledge_documents_document_title_trimmed",
            ),
        ),
        sa.CheckConstraint(
            "char_length(title) BETWEEN 1 AND 200",
            name=op.f(
                "ck_knowledge_documents_document_title_length",
            ),
        ),
        sa.CheckConstraint(
            (
                "external_reference IS NULL OR ("
                "external_reference = btrim(external_reference) "
                "AND char_length(external_reference) BETWEEN 1 AND 128"
                ")"
            ),
            name=op.f(
                "ck_knowledge_documents_document_external_reference_format",
            ),
        ),
        sa.CheckConstraint(
            "updated_at >= created_at",
            name=op.f(
                "ck_knowledge_documents_document_timestamp_order",
            ),
        ),
        sa.ForeignKeyConstraint(
            ["workspace_id"],
            ["workspaces.id"],
            name=op.f(
                "fk_knowledge_documents_workspace_id_workspaces",
            ),
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint(
            "id",
            name=op.f("pk_knowledge_documents"),
        ),
        sa.UniqueConstraint(
            "workspace_id",
            "id",
            name="uq_knowledge_documents_workspace_id",
        ),
        sa.UniqueConstraint(
            "workspace_id",
            "external_reference",
            name=("uq_knowledge_documents_workspace_external_reference"),
        ),
    )
    op.create_index(
        "ix_knowledge_documents_workspace_created_id",
        "knowledge_documents",
        [
            "workspace_id",
            sa.literal_column("created_at DESC"),
            sa.literal_column("id DESC"),
        ],
        unique=False,
    )
    op.create_index(
        "ix_knowledge_documents_workspace_active_version",
        "knowledge_documents",
        ["workspace_id", "active_version_id"],
        unique=False,
        postgresql_where=sa.text("active_version_id IS NOT NULL"),
    )

    op.create_table(
        "knowledge_document_versions",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("workspace_id", sa.Uuid(), nullable=False),
        sa.Column("document_id", sa.Uuid(), nullable=False),
        sa.Column("version_number", sa.Integer(), nullable=False),
        sa.Column(
            "media_type",
            sa.String(length=32),
            nullable=False,
        ),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column(
            "content_sha256",
            sa.String(length=64),
            nullable=False,
        ),
        sa.Column(
            "status",
            sa.String(length=32),
            nullable=False,
        ),
        sa.Column(
            "chunking_strategy",
            sa.String(length=128),
            nullable=True,
        ),
        sa.Column(
            "chunking_version",
            sa.String(length=128),
            nullable=True,
        ),
        sa.Column(
            "tokenizer_encoding",
            sa.String(length=128),
            nullable=True,
        ),
        sa.Column(
            "embedding_provider",
            sa.String(length=128),
            nullable=True,
        ),
        sa.Column(
            "embedding_model",
            sa.String(length=128),
            nullable=True,
        ),
        sa.Column(
            "embedding_dimensions",
            sa.Integer(),
            nullable=True,
        ),
        sa.Column(
            "knowledge_collection",
            sa.String(length=128),
            nullable=True,
        ),
        sa.Column(
            "knowledge_vector_name",
            sa.String(length=128),
            nullable=True,
        ),
        sa.Column(
            "embedding_input_tokens",
            sa.BigInteger(),
            nullable=True,
        ),
        sa.Column(
            "embedding_estimated_cost_usd",
            sa.Numeric(
                precision=20,
                scale=12,
                asdecimal=True,
            ),
            nullable=True,
        ),
        sa.Column(
            "embedding_pricing_catalog_version",
            sa.String(length=128),
            nullable=True,
        ),
        sa.Column("chunk_count", sa.Integer(), nullable=True),
        sa.Column(
            "indexed_at",
            sa.DateTime(timezone=True),
            nullable=True,
        ),
        sa.Column(
            "last_error_code",
            sa.String(length=64),
            nullable=True,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
        ),
        sa.CheckConstraint(
            "version_number >= 1",
            name=op.f(
                "ck_knowledge_document_versions_document_version_number_positive",
            ),
        ),
        sa.CheckConstraint(
            "media_type IN ('text/plain', 'text/markdown')",
            name=op.f(
                "ck_knowledge_document_versions_document_version_media_type",
            ),
        ),
        sa.CheckConstraint(
            "content ~ '[^[:space:]]'",
            name=op.f(
                "ck_knowledge_document_versions_document_version_content_non_whitespace",
            ),
        ),
        sa.CheckConstraint(
            "content_sha256 ~ '^[0-9a-f]{64}$'",
            name=op.f(
                "ck_knowledge_document_versions_document_version_content_sha256",
            ),
        ),
        sa.CheckConstraint(
            "status IN ('pending', 'ready', 'failed')",
            name=op.f(
                "ck_knowledge_document_versions_document_version_status",
            ),
        ),
        sa.CheckConstraint(
            (
                "("
                "chunking_strategy IS NULL "
                "AND chunking_version IS NULL "
                "AND tokenizer_encoding IS NULL "
                "AND embedding_provider IS NULL "
                "AND embedding_model IS NULL "
                "AND embedding_dimensions IS NULL "
                "AND knowledge_collection IS NULL "
                "AND knowledge_vector_name IS NULL"
                ") OR ("
                "chunking_strategy IS NOT NULL "
                "AND chunking_version IS NOT NULL "
                "AND tokenizer_encoding IS NOT NULL "
                "AND embedding_provider IS NOT NULL "
                "AND embedding_model IS NOT NULL "
                "AND embedding_dimensions IS NOT NULL "
                "AND knowledge_collection IS NOT NULL "
                "AND knowledge_vector_name IS NOT NULL"
                ")"
            ),
            name=op.f(
                "ck_knowledge_document_versions_document_version_index_profile_completeness",
            ),
        ),
        sa.CheckConstraint(
            (
                "chunking_strategy IS NULL OR ("
                "chunking_strategy = btrim(chunking_strategy) "
                "AND char_length(chunking_strategy) BETWEEN 1 AND 128"
                ")"
            ),
            name=op.f(
                "ck_knowledge_document_versions_document_version_chunking_strategy_format",
            ),
        ),
        sa.CheckConstraint(
            (
                "chunking_version IS NULL OR ("
                "chunking_version = btrim(chunking_version) "
                "AND char_length(chunking_version) BETWEEN 1 AND 128"
                ")"
            ),
            name=op.f(
                "ck_knowledge_document_versions_document_version_chunking_version_format",
            ),
        ),
        sa.CheckConstraint(
            (
                "tokenizer_encoding IS NULL OR ("
                "tokenizer_encoding = btrim(tokenizer_encoding) "
                "AND char_length(tokenizer_encoding) BETWEEN 1 AND 128"
                ")"
            ),
            name=op.f(
                "ck_knowledge_document_versions_document_version_tokenizer_encoding_format",
            ),
        ),
        sa.CheckConstraint(
            (
                "embedding_provider IS NULL OR ("
                "embedding_provider = btrim(embedding_provider) "
                "AND char_length(embedding_provider) BETWEEN 1 AND 128"
                ")"
            ),
            name=op.f(
                "ck_knowledge_document_versions_document_version_embedding_provider_format",
            ),
        ),
        sa.CheckConstraint(
            (
                "embedding_model IS NULL OR ("
                "embedding_model = btrim(embedding_model) "
                "AND char_length(embedding_model) BETWEEN 1 AND 128"
                ")"
            ),
            name=op.f(
                "ck_knowledge_document_versions_document_version_embedding_model_format",
            ),
        ),
        sa.CheckConstraint(
            (
                "knowledge_collection IS NULL OR ("
                "knowledge_collection = btrim(knowledge_collection) "
                "AND char_length(knowledge_collection) BETWEEN 1 AND 128"
                ")"
            ),
            name=op.f(
                "ck_knowledge_document_versions_document_version_knowledge_collection_format",
            ),
        ),
        sa.CheckConstraint(
            (
                "knowledge_vector_name IS NULL OR ("
                "knowledge_vector_name = btrim(knowledge_vector_name) "
                "AND char_length(knowledge_vector_name) BETWEEN 1 AND 128"
                ")"
            ),
            name=op.f(
                "ck_knowledge_document_versions_document_version_knowledge_vector_name_format",
            ),
        ),
        sa.CheckConstraint(
            ("embedding_dimensions IS NULL OR embedding_dimensions > 0"),
            name=op.f(
                "ck_knowledge_document_versions_document_version_embedding_dimensions_positive",
            ),
        ),
        sa.CheckConstraint(
            ("embedding_input_tokens IS NULL OR embedding_input_tokens >= 0"),
            name=op.f(
                "ck_knowledge_document_versions_document_version_embedding_tokens_non_negative",
            ),
        ),
        sa.CheckConstraint(
            ("embedding_estimated_cost_usd IS NULL OR embedding_estimated_cost_usd >= 0"),
            name=op.f(
                "ck_knowledge_document_versions_document_version_embedding_cost_non_negative",
            ),
        ),
        sa.CheckConstraint(
            (
                "embedding_pricing_catalog_version IS NULL OR ("
                "embedding_pricing_catalog_version = "
                "btrim(embedding_pricing_catalog_version) "
                "AND char_length(embedding_pricing_catalog_version) "
                "BETWEEN 1 AND 128"
                ")"
            ),
            name=op.f(
                "ck_knowledge_document_versions_document_version_pricing_catalog_format",
            ),
        ),
        sa.CheckConstraint(
            "chunk_count IS NULL OR chunk_count >= 0",
            name=op.f(
                "ck_knowledge_document_versions_document_version_chunk_count_non_negative",
            ),
        ),
        sa.CheckConstraint(
            (
                "last_error_code IS NULL OR ("
                "last_error_code = btrim(last_error_code) "
                "AND char_length(last_error_code) BETWEEN 1 AND 64"
                ")"
            ),
            name=op.f(
                "ck_knowledge_document_versions_document_version_error_code_format",
            ),
        ),
        sa.CheckConstraint(
            (
                "("
                "status = 'pending' "
                "AND indexed_at IS NULL "
                "AND last_error_code IS NULL "
                "AND embedding_input_tokens IS NULL "
                "AND embedding_estimated_cost_usd IS NULL "
                "AND embedding_pricing_catalog_version IS NULL "
                "AND ("
                "chunking_strategy IS NOT NULL "
                "OR chunk_count IS NULL"
                ")"
                ") OR ("
                "status = 'failed' "
                "AND chunking_strategy IS NOT NULL "
                "AND chunk_count IS NOT NULL "
                "AND indexed_at IS NULL "
                "AND last_error_code IS NOT NULL "
                "AND embedding_input_tokens IS NULL "
                "AND embedding_estimated_cost_usd IS NULL "
                "AND embedding_pricing_catalog_version IS NULL"
                ") OR ("
                "status = 'ready' "
                "AND chunking_strategy IS NOT NULL "
                "AND chunk_count > 0 "
                "AND embedding_input_tokens IS NOT NULL "
                "AND embedding_pricing_catalog_version IS NOT NULL "
                "AND indexed_at IS NOT NULL "
                "AND last_error_code IS NULL"
                ")"
            ),
            name=op.f(
                "ck_knowledge_document_versions_document_version_status_state",
            ),
        ),
        sa.CheckConstraint(
            "updated_at >= created_at",
            name=op.f(
                "ck_knowledge_document_versions_document_version_timestamp_order",
            ),
        ),
        sa.CheckConstraint(
            ("indexed_at IS NULL OR (indexed_at >= created_at AND indexed_at <= updated_at)"),
            name=op.f(
                "ck_knowledge_document_versions_document_version_indexed_timestamp_order",
            ),
        ),
        sa.ForeignKeyConstraint(
            ["workspace_id", "document_id"],
            [
                "knowledge_documents.workspace_id",
                "knowledge_documents.id",
            ],
            name="fk_knowledge_document_versions_document",
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint(
            "id",
            name=op.f("pk_knowledge_document_versions"),
        ),
        sa.UniqueConstraint(
            "workspace_id",
            "document_id",
            "id",
            name=("uq_knowledge_document_versions_workspace_document_id"),
        ),
        sa.UniqueConstraint(
            "workspace_id",
            "document_id",
            "id",
            "chunking_strategy",
            "chunking_version",
            name="uq_knowledge_document_versions_chunk_profile",
        ),
        sa.UniqueConstraint(
            "document_id",
            "version_number",
            name=("uq_knowledge_document_versions_document_version_number"),
        ),
        sa.UniqueConstraint(
            "document_id",
            "content_sha256",
            name=("uq_knowledge_document_versions_document_content_sha256"),
        ),
    )
    op.create_index(
        "ix_knowledge_document_versions_workspace_document_number",
        "knowledge_document_versions",
        [
            "workspace_id",
            "document_id",
            sa.literal_column("version_number DESC"),
            sa.literal_column("id DESC"),
        ],
        unique=False,
    )

    op.create_foreign_key(
        "fk_knowledge_documents_active_version",
        "knowledge_documents",
        "knowledge_document_versions",
        ["workspace_id", "id", "active_version_id"],
        ["workspace_id", "document_id", "id"],
        ondelete="RESTRICT",
    )

    op.create_table(
        "knowledge_document_chunks",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("workspace_id", sa.Uuid(), nullable=False),
        sa.Column("document_id", sa.Uuid(), nullable=False),
        sa.Column(
            "document_version_id",
            sa.Uuid(),
            nullable=False,
        ),
        sa.Column("ordinal", sa.Integer(), nullable=False),
        sa.Column(
            "section_path",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
        ),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column(
            "content_sha256",
            sa.String(length=64),
            nullable=False,
        ),
        sa.Column("token_count", sa.Integer(), nullable=False),
        sa.Column(
            "chunking_strategy",
            sa.String(length=128),
            nullable=False,
        ),
        sa.Column(
            "chunking_version",
            sa.String(length=128),
            nullable=False,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
        ),
        sa.CheckConstraint(
            "ordinal >= 0",
            name=op.f(
                "ck_knowledge_document_chunks_document_chunk_ordinal_non_negative",
            ),
        ),
        sa.CheckConstraint(
            ("jsonb_typeof(section_path) = 'array' AND jsonb_array_length(section_path) <= 6"),
            name=op.f(
                "ck_knowledge_document_chunks_document_chunk_section_path",
            ),
        ),
        sa.CheckConstraint(
            "content ~ '[^[:space:]]'",
            name=op.f(
                "ck_knowledge_document_chunks_document_chunk_content_non_whitespace",
            ),
        ),
        sa.CheckConstraint(
            "content_sha256 ~ '^[0-9a-f]{64}$'",
            name=op.f(
                "ck_knowledge_document_chunks_document_chunk_content_sha256",
            ),
        ),
        sa.CheckConstraint(
            "token_count > 0",
            name=op.f(
                "ck_knowledge_document_chunks_document_chunk_token_count_positive",
            ),
        ),
        sa.CheckConstraint(
            (
                "chunking_strategy = btrim(chunking_strategy) "
                "AND char_length(chunking_strategy) BETWEEN 1 AND 128"
            ),
            name=op.f(
                "ck_knowledge_document_chunks_document_chunk_chunking_strategy_format",
            ),
        ),
        sa.CheckConstraint(
            (
                "chunking_version = btrim(chunking_version) "
                "AND char_length(chunking_version) BETWEEN 1 AND 128"
            ),
            name=op.f(
                "ck_knowledge_document_chunks_document_chunk_chunking_version_format",
            ),
        ),
        sa.ForeignKeyConstraint(
            [
                "workspace_id",
                "document_id",
                "document_version_id",
                "chunking_strategy",
                "chunking_version",
            ],
            [
                "knowledge_document_versions.workspace_id",
                "knowledge_document_versions.document_id",
                "knowledge_document_versions.id",
                "knowledge_document_versions.chunking_strategy",
                "knowledge_document_versions.chunking_version",
            ],
            name=("fk_knowledge_document_chunks_version_profile"),
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint(
            "id",
            name=op.f("pk_knowledge_document_chunks"),
        ),
        sa.UniqueConstraint(
            "document_version_id",
            "ordinal",
            name=("uq_knowledge_document_chunks_version_ordinal"),
        ),
    )
    op.create_index(
        "ix_knowledge_document_chunks_workspace_version_id",
        "knowledge_document_chunks",
        ["workspace_id", "document_version_id", "id"],
        unique=False,
    )

    op.execute(
        """
        CREATE FUNCTION require_ready_knowledge_document_active_version()
        RETURNS trigger
        LANGUAGE plpgsql
        AS $$
        BEGIN
            IF NEW.active_version_id IS NULL THEN
                RETURN NEW;
            END IF;

            IF NOT EXISTS (
                SELECT 1
                FROM knowledge_document_versions
                WHERE id = NEW.active_version_id
                  AND workspace_id = NEW.workspace_id
                  AND document_id = NEW.id
                  AND status = 'ready'
            ) THEN
                RAISE EXCEPTION
                    'active knowledge document version must be ready'
                    USING
                        ERRCODE = '23514',
                        CONSTRAINT = 'knowledge_documents_active_version_ready';
            END IF;

            RETURN NEW;
        END;
        $$;
        """
    )
    op.execute(
        """
        CREATE TRIGGER trg_knowledge_documents_require_ready_active_version
        BEFORE UPDATE OF active_version_id, workspace_id, id
        ON knowledge_documents
        FOR EACH ROW
        EXECUTE FUNCTION require_ready_knowledge_document_active_version();
        """
    )
    op.execute(
        """
        CREATE FUNCTION prevent_ready_knowledge_document_version_rewrite()
        RETURNS trigger
        LANGUAGE plpgsql
        AS $$
        BEGIN
            IF OLD.status = 'ready' AND NEW IS DISTINCT FROM OLD THEN
                RAISE EXCEPTION
                    'ready knowledge document versions are immutable'
                    USING
                        ERRCODE = '23514',
                        CONSTRAINT = 'knowledge_document_versions_ready_immutable';
            END IF;

            RETURN NEW;
        END;
        $$;
        """
    )
    op.execute(
        """
        CREATE TRIGGER trg_knowledge_document_versions_preserve_ready
        BEFORE UPDATE
        ON knowledge_document_versions
        FOR EACH ROW
        EXECUTE FUNCTION prevent_ready_knowledge_document_version_rewrite();
        """
    )


def downgrade() -> None:
    """Revert the migration."""
    op.execute(
        """
        DROP TRIGGER IF EXISTS
        trg_knowledge_document_versions_preserve_ready
        ON knowledge_document_versions;
        """
    )
    op.execute(
        """
        DROP FUNCTION IF EXISTS
        prevent_ready_knowledge_document_version_rewrite();
        """
    )
    op.execute(
        """
        DROP TRIGGER IF EXISTS
        trg_knowledge_documents_require_ready_active_version
        ON knowledge_documents;
        """
    )
    op.execute(
        """
        DROP FUNCTION IF EXISTS
        require_ready_knowledge_document_active_version();
        """
    )

    op.drop_index(
        "ix_knowledge_document_chunks_workspace_version_id",
        table_name="knowledge_document_chunks",
    )
    op.drop_table("knowledge_document_chunks")

    op.drop_constraint(
        "fk_knowledge_documents_active_version",
        "knowledge_documents",
        type_="foreignkey",
    )
    op.drop_index(
        "ix_knowledge_document_versions_workspace_document_number",
        table_name="knowledge_document_versions",
    )
    op.drop_table("knowledge_document_versions")

    op.drop_index(
        "ix_knowledge_documents_workspace_active_version",
        table_name="knowledge_documents",
        postgresql_where=sa.text("active_version_id IS NOT NULL"),
    )
    op.drop_index(
        "ix_knowledge_documents_workspace_created_id",
        table_name="knowledge_documents",
    )
    op.drop_table("knowledge_documents")
