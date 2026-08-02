"""PostgreSQL query adapters for authoritative knowledge retrieval."""

from collections.abc import Sequence
from uuid import UUID

from sqlalchemy import and_, select
from sqlalchemy.ext.asyncio import AsyncSession

from supportops.knowledge_retrieval.contracts import (
    ActiveKnowledgeVersion,
)
from supportops.modules.knowledge_documents.domain.models import (
    DocumentChunk,
    DocumentVersionStatus,
)
from supportops.modules.knowledge_documents.infrastructure.models import (
    DocumentChunkRecord,
    DocumentRecord,
    DocumentVersionRecord,
)


class SqlAlchemyActiveKnowledgeVersionResolver:
    """Resolve workspace-owned active ready versions from PostgreSQL."""

    def __init__(
        self,
        session: AsyncSession,
    ) -> None:
        self._session = session

    async def resolve(
        self,
        *,
        workspace_id: UUID,
        document_ids: tuple[UUID, ...],
    ) -> Sequence[ActiveKnowledgeVersion]:
        """Return active ready versions within one workspace boundary."""

        statement = (
            select(
                DocumentRecord,
                DocumentVersionRecord,
            )
            .join(
                DocumentVersionRecord,
                and_(
                    DocumentVersionRecord.workspace_id == DocumentRecord.workspace_id,
                    DocumentVersionRecord.document_id == DocumentRecord.id,
                    DocumentVersionRecord.id == DocumentRecord.active_version_id,
                ),
            )
            .where(
                DocumentRecord.workspace_id == workspace_id,
                DocumentVersionRecord.status == DocumentVersionStatus.READY.value,
            )
            .order_by(
                DocumentRecord.id.asc(),
            )
        )

        if document_ids:
            statement = statement.where(DocumentRecord.id.in_(document_ids))

        result = await self._session.execute(statement)

        active_versions: list[ActiveKnowledgeVersion] = []

        for document_record, version_record in result.tuples().all():
            version = version_record.to_domain()
            index_profile = version.index_profile

            if version.status is not DocumentVersionStatus.READY or index_profile is None:
                raise RuntimeError("Active knowledge version violates the ready-profile invariant.")

            active_versions.append(
                ActiveKnowledgeVersion(
                    workspace_id=(document_record.workspace_id),
                    document_id=document_record.id,
                    document_title=document_record.title,
                    document_external_reference=(document_record.external_reference),
                    document_version_id=version.id,
                    version_number=(version.version_number),
                    media_type=version.media_type,
                    index_profile=index_profile,
                )
            )

        return tuple(active_versions)


class SqlAlchemyKnowledgeChunkHydrator:
    """Bulk-load authoritative chunk content through workspace ownership."""

    def __init__(
        self,
        session: AsyncSession,
    ) -> None:
        self._session = session

    async def hydrate(
        self,
        *,
        workspace_id: UUID,
        chunk_ids: tuple[UUID, ...],
    ) -> Sequence[DocumentChunk]:
        """Return existing workspace-owned chunks in requested-ID order."""

        if not chunk_ids:
            return ()

        if len(set(chunk_ids)) != len(chunk_ids):
            raise ValueError("chunk_ids must not contain duplicates.")

        statement = select(DocumentChunkRecord).where(
            DocumentChunkRecord.workspace_id == workspace_id,
            DocumentChunkRecord.id.in_(chunk_ids),
        )

        result = await self._session.execute(statement)
        chunks_by_id = {record.id: record.to_domain() for record in result.scalars().all()}

        return tuple(chunks_by_id[chunk_id] for chunk_id in chunk_ids if chunk_id in chunks_by_id)
