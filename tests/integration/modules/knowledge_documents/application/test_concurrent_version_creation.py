"""Concurrent PostgreSQL tests for document version creation."""

import asyncio
from collections.abc import Sequence
from datetime import UTC, datetime
from uuid import UUID

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from supportops.infrastructure.postgresql.transaction import (
    SqlAlchemyTransactionManager,
)
from supportops.modules.knowledge_documents.application.errors import (
    DocumentVersionContentConflictApplicationError,
)
from supportops.modules.knowledge_documents.application.results import (
    CreateDocumentResult,
)
from supportops.modules.knowledge_documents.application.services import (
    CreateDocument,
    CreateDocumentVersion,
)
from supportops.modules.knowledge_documents.domain.models import (
    DocumentMediaType,
    DocumentVersion,
)
from supportops.modules.knowledge_documents.infrastructure.repository import (
    SqlAlchemyDocumentRepository,
    SqlAlchemyDocumentVersionRepository,
)
from supportops.modules.workspaces.domain.models import Workspace
from supportops.modules.workspaces.infrastructure.repository import (
    SqlAlchemyWorkspaceRepository,
)

pytestmark = pytest.mark.integration

_WORKSPACE_ID = UUID("032c8c87-57cc-4d14-bfbd-04968b4e8cd4")
_CREATED_AT = datetime(
    2026,
    8,
    1,
    23,
    30,
    tzinfo=UTC,
)


async def seed_document(
    session_factory: async_sessionmaker[AsyncSession],
) -> CreateDocumentResult:
    """Persist one workspace, document, and initial document version."""

    async with session_factory() as session:
        transaction_manager = SqlAlchemyTransactionManager(session)
        workspace_repository = SqlAlchemyWorkspaceRepository(session)

        async with transaction_manager.transaction():
            await workspace_repository.add(
                Workspace(
                    id=_WORKSPACE_ID,
                    name="Platform Support",
                    slug="platform-support",
                    created_at=_CREATED_AT,
                    updated_at=_CREATED_AT,
                )
            )

        service = CreateDocument(
            workspace_repository=workspace_repository,
            document_repository=SqlAlchemyDocumentRepository(session),
            version_repository=SqlAlchemyDocumentVersionRepository(session),
            transaction_manager=transaction_manager,
        )
        return await service.execute(
            workspace_id=_WORKSPACE_ID,
            title="Database Incident Runbook",
            external_reference="runbook-database-incidents",
            media_type=DocumentMediaType.TEXT_MARKDOWN,
            content=("# Database incidents\n\nRestart the connection pool before escalating.\n"),
        )


async def create_version_after_barrier(
    *,
    session_factory: async_sessionmaker[AsyncSession],
    barrier: asyncio.Barrier,
    document_id: UUID,
    content: str,
) -> DocumentVersion:
    """Start one version-creation request with an independent session."""

    async with session_factory() as session:
        service = CreateDocumentVersion(
            document_repository=SqlAlchemyDocumentRepository(session),
            version_repository=SqlAlchemyDocumentVersionRepository(session),
            transaction_manager=SqlAlchemyTransactionManager(session),
        )

        await barrier.wait()

        return await service.execute(
            workspace_id=_WORKSPACE_ID,
            document_id=document_id,
            media_type=DocumentMediaType.TEXT_MARKDOWN,
            content=content,
        )


async def load_versions(
    *,
    session_factory: async_sessionmaker[AsyncSession],
    document_id: UUID,
) -> Sequence[DocumentVersion]:
    """Load every persisted version in descending version order."""

    async with session_factory() as session:
        repository = SqlAlchemyDocumentVersionRepository(session)
        return await repository.list(
            workspace_id=_WORKSPACE_ID,
            document_id=document_id,
            limit=20,
        )


async def test_concurrent_distinct_versions_receive_unique_numbers(
    postgresql_session_factory: async_sessionmaker[AsyncSession],
    clean_business_tables: None,
) -> None:
    """Serialize version allocation through the owning document row."""

    created = await seed_document(postgresql_session_factory)
    barrier = asyncio.Barrier(2)

    second, third = await asyncio.gather(
        create_version_after_barrier(
            session_factory=postgresql_session_factory,
            barrier=barrier,
            document_id=created.document.id,
            content=(
                "# Database incidents\n\nEscalate after two failed connection-pool restarts.\n"
            ),
        ),
        create_version_after_barrier(
            session_factory=postgresql_session_factory,
            barrier=barrier,
            document_id=created.document.id,
            content=(
                "# Database incidents\n\nFail over when the primary database is unavailable.\n"
            ),
        ),
    )

    assert {second.version_number, third.version_number} == {2, 3}
    assert second.id != third.id
    assert second.content_sha256 != third.content_sha256

    persisted = await load_versions(
        session_factory=postgresql_session_factory,
        document_id=created.document.id,
    )

    assert [version.version_number for version in persisted] == [3, 2, 1]
    assert len({version.version_number for version in persisted}) == 3
    assert len({version.id for version in persisted}) == 3


async def test_concurrent_equivalent_content_creates_one_version(
    postgresql_session_factory: async_sessionmaker[AsyncSession],
    clean_business_tables: None,
) -> None:
    """Reject the request that loses the normalized-content race."""

    created = await seed_document(postgresql_session_factory)
    barrier = asyncio.Barrier(2)

    outcomes = await asyncio.gather(
        create_version_after_barrier(
            session_factory=postgresql_session_factory,
            barrier=barrier,
            document_id=created.document.id,
            content=("\ufeff# Connection exhaustion\r\n\r\nEscalate after two failed restarts.\r"),
        ),
        create_version_after_barrier(
            session_factory=postgresql_session_factory,
            barrier=barrier,
            document_id=created.document.id,
            content=("# Connection exhaustion\n\nEscalate after two failed restarts.\n"),
        ),
        return_exceptions=True,
    )

    successful_versions = [outcome for outcome in outcomes if isinstance(outcome, DocumentVersion)]
    failures = [outcome for outcome in outcomes if isinstance(outcome, BaseException)]

    assert len(successful_versions) == 1
    assert successful_versions[0].version_number == 2

    assert len(failures) == 1
    assert isinstance(
        failures[0],
        DocumentVersionContentConflictApplicationError,
    )
    assert str(failures[0]) == ("Document content already exists for this document.")

    persisted = await load_versions(
        session_factory=postgresql_session_factory,
        document_id=created.document.id,
    )

    assert [version.version_number for version in persisted] == [2, 1]
    assert (
        persisted[0].content == "# Connection exhaustion\n\nEscalate after two failed restarts.\n"
    )
