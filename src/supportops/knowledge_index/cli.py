"""Operator-controlled command-line interface for knowledge indexing."""

import argparse
import asyncio
import json
import sys
from collections.abc import Awaitable, Callable, Sequence
from typing import Protocol, TextIO
from uuid import UUID

from pydantic import ValidationError

from supportops.ai.embeddings.errors import (
    EmbeddingError,
)
from supportops.core.settings import (
    EmbeddingProviderName,
    Settings,
)
from supportops.knowledge_index.composition import (
    KnowledgeIndexCompositionError,
    KnowledgeIndexRuntime,
    create_knowledge_index_runtime,
)
from supportops.knowledge_index.indexing.errors import (
    KnowledgeIndexingError,
)
from supportops.knowledge_index.indexing.results import (
    IndexDocumentVersionResult,
)
from supportops.knowledge_index.vector_store.contracts import (
    KnowledgeVectorStoreError,
)
from supportops.modules.knowledge_documents.domain.models import (
    KnowledgeIndexProfile,
)

_EXIT_SUCCESS = 0
_EXIT_RUNTIME_FAILURE = 1
_EXIT_USAGE_OR_CONFIGURATION_FAILURE = 2


class KnowledgeIndexCLIError(ValueError):
    """Raised when CLI execution is not explicitly safe."""


class ExternalEmbeddingProviderPermissionRequiredError(KnowledgeIndexCLIError):
    """Raised when an external provider was not acknowledged."""


class KnowledgeIndexCommandRuntime(Protocol):
    """Runtime behavior required by the indexing CLI."""

    @property
    def index_profile(self) -> KnowledgeIndexProfile:
        """Return the configured immutable index profile."""
        ...

    async def ensure_collection(self) -> None:
        """Create or validate the configured collection."""
        ...

    async def index_version(
        self,
        *,
        workspace_id: UUID,
        document_id: UUID,
        document_version_id: UUID,
    ) -> IndexDocumentVersionResult:
        """Index one workspace-owned document version."""
        ...

    async def close(self) -> None:
        """Release process-scoped resources."""
        ...


class KnowledgeIndexRuntimeFactory(Protocol):
    """Construct one process-scoped indexing runtime."""

    def __call__(
        self,
        *,
        settings: Settings,
    ) -> Awaitable[KnowledgeIndexCommandRuntime]:
        """Return one runtime for a command execution."""
        ...


type SettingsFactory = Callable[[], Settings]


def main() -> None:
    """Run the knowledge indexing CLI."""

    raise SystemExit(asyncio.run(run_cli()))


async def run_cli(
    argv: Sequence[str] | None = None,
    *,
    stdout: TextIO = sys.stdout,
    stderr: TextIO = sys.stderr,
    settings_factory: SettingsFactory = Settings,
    runtime_factory: KnowledgeIndexRuntimeFactory = (create_knowledge_index_runtime),
) -> int:
    """Execute one indexing command with testable process boundaries."""

    parser = build_parser()
    arguments = parser.parse_args(argv)
    runtime: KnowledgeIndexCommandRuntime | None = None
    exit_code = _EXIT_SUCCESS

    try:
        settings = settings_factory()
        _validate_external_provider_permission(
            provider_name=settings.embedding_provider,
            allow_external_provider=(arguments.allow_external_provider),
        )

        runtime = await runtime_factory(settings=settings)

        if arguments.command == "ensure-collection":
            await runtime.ensure_collection()
            _write_collection_summary(
                stdout=stdout,
                profile=runtime.index_profile,
            )
        elif arguments.command == "index-version":
            result = await runtime.index_version(
                workspace_id=arguments.workspace_id,
                document_id=arguments.document_id,
                document_version_id=(arguments.document_version_id),
            )
            _write_indexing_summary(
                stdout=stdout,
                result=result,
            )
        else:
            raise RuntimeError("Knowledge indexing parser produced an unsupported command.")
    except (
        KnowledgeIndexCLIError,
        KnowledgeIndexCompositionError,
        ValidationError,
        ValueError,
    ) as error:
        _write_expected_error(
            stderr=stderr,
            error=error,
        )
        exit_code = _EXIT_USAGE_OR_CONFIGURATION_FAILURE
    except (
        EmbeddingError,
        KnowledgeIndexingError,
        KnowledgeVectorStoreError,
    ) as error:
        _write_operational_error(
            stderr=stderr,
            error=error,
        )
        exit_code = _EXIT_RUNTIME_FAILURE
    except Exception:
        _write_runtime_error(
            stderr=stderr,
            message=("Knowledge indexing failed unexpectedly."),
        )
        exit_code = _EXIT_RUNTIME_FAILURE

    if runtime is not None:
        try:
            await runtime.close()
        except Exception:
            if exit_code == _EXIT_SUCCESS:
                _write_runtime_error(
                    stderr=stderr,
                    message=("Knowledge indexing resources could not be closed safely."),
                )
                exit_code = _EXIT_RUNTIME_FAILURE

    return exit_code


def build_parser() -> argparse.ArgumentParser:
    """Build the stable knowledge indexing command surface."""

    parser = argparse.ArgumentParser(
        prog="supportops-index-knowledge",
        description=("Create and operate the rebuildable knowledge vector projection."),
    )
    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
    )

    ensure_parser = subparsers.add_parser(
        "ensure-collection",
        help=("Create or validate the configured Qdrant knowledge collection."),
    )
    _add_external_provider_permission_argument(ensure_parser)

    index_parser = subparsers.add_parser(
        "index-version",
        help=("Index one workspace-owned knowledge document version."),
    )
    index_parser.add_argument(
        "--workspace-id",
        type=UUID,
        required=True,
        help="Owning workspace UUID.",
    )
    index_parser.add_argument(
        "--document-id",
        type=UUID,
        required=True,
        help="Owning knowledge document UUID.",
    )
    index_parser.add_argument(
        "--document-version-id",
        type=UUID,
        required=True,
        help="Knowledge document version UUID.",
    )
    _add_external_provider_permission_argument(index_parser)

    return parser


def _add_external_provider_permission_argument(
    parser: argparse.ArgumentParser,
) -> None:
    parser.add_argument(
        "--allow-external-provider",
        action="store_true",
        help=("Required explicit acknowledgement before OpenAI may be initialized."),
    )


def _validate_external_provider_permission(
    *,
    provider_name: EmbeddingProviderName,
    allow_external_provider: bool,
) -> None:
    if provider_name is EmbeddingProviderName.OPENAI and not allow_external_provider:
        raise (
            ExternalEmbeddingProviderPermissionRequiredError(
                "OpenAI knowledge indexing requires --allow-external-provider."
            )
        )

    if provider_name is EmbeddingProviderName.MOCK and allow_external_provider:
        raise KnowledgeIndexCLIError(
            "--allow-external-provider is valid only "
            "when the OpenAI embedding provider is configured."
        )


def _write_collection_summary(
    *,
    stdout: TextIO,
    profile: KnowledgeIndexProfile,
) -> None:
    payload = {
        "command": "ensure-collection",
        "collection": (profile.knowledge_collection),
        "dimensions": (profile.embedding_dimensions),
        "embedding_model": (profile.embedding_model),
        "embedding_provider": (profile.embedding_provider),
        "status": "compatible",
        "vector_name": (profile.knowledge_vector_name),
    }
    _write_json(
        stream=stdout,
        payload=payload,
    )


def _write_indexing_summary(
    *,
    stdout: TextIO,
    result: IndexDocumentVersionResult,
) -> None:
    version = result.version

    payload = {
        "already_ready": result.already_ready,
        "chunk_count": result.chunk_count,
        "command": "index-version",
        "document_id": str(version.document_id),
        "document_version_id": str(version.id),
        "embedding_estimated_cost_usd": (
            str(result.estimated_cost_usd) if result.estimated_cost_usd is not None else None
        ),
        "embedding_input_tokens": (result.embedding_input_tokens),
        "embedding_model": (version.embedding_model),
        "embedding_provider": (version.embedding_provider),
        "indexed_at": (version.indexed_at.isoformat() if version.indexed_at is not None else None),
        "knowledge_collection": (version.knowledge_collection),
        "pricing_catalog_version": (result.pricing_catalog_version),
        "status": version.status.value,
        "workspace_id": str(version.workspace_id),
    }
    _write_json(
        stream=stdout,
        payload=payload,
    )


def _write_json(
    *,
    stream: TextIO,
    payload: dict[str, object],
) -> None:
    stream.write(
        json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    )


def _write_expected_error(
    *,
    stderr: TextIO,
    error: Exception,
) -> None:
    message = (
        "Knowledge indexing configuration is invalid."
        if isinstance(error, ValidationError)
        else str(error)
    )
    stderr.write(f"indexing_cli_error: {message}\n")


def _write_operational_error(
    *,
    stderr: TextIO,
    error: Exception,
) -> None:
    stderr.write(f"indexing_runtime_error: {error}\n")


def _write_runtime_error(
    *,
    stderr: TextIO,
    message: str,
) -> None:
    stderr.write(f"indexing_runtime_error: {message}\n")


def default_runtime_factory(
    *,
    settings: Settings,
) -> Awaitable[KnowledgeIndexRuntime]:
    """Expose the concrete factory for typed executable wiring."""

    return create_knowledge_index_runtime(settings=settings)


if __name__ == "__main__":
    main()
