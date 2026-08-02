"""Unit tests for controlled support tool composition."""

from supportops.agent_tools.domain.contracts import (
    ToolSafetyLevel,
)
from supportops.agent_tools.tools.registry import (
    create_controlled_support_tool_registry,
)
from supportops.agent_tools.tools.search_knowledge import (
    SearchKnowledgeInput,
    SearchKnowledgeOutput,
)
from supportops.agent_tools.tools.service_status import (
    DeterministicServiceStatusCatalog,
    LookupServiceStatusInput,
    LookupServiceStatusOutput,
)
from supportops.knowledge_retrieval.contracts import (
    KnowledgeSearchRequest,
    KnowledgeSearchResult,
)


class EmptyKnowledgeSearch:
    """Return deterministic empty authoritative retrieval."""

    async def execute(
        self,
        request: KnowledgeSearchRequest,
    ) -> KnowledgeSearchResult:
        return KnowledgeSearchResult(
            request=request,
            searched_version_count=0,
            evidence=(),
        )


def test_composes_exact_controlled_support_registry() -> None:
    registry = create_controlled_support_tool_registry(
        knowledge_search=EmptyKnowledgeSearch(),
        service_status_catalog=(DeterministicServiceStatusCatalog(())),
    )

    assert [definition.name for definition in registry.definitions] == [
        "lookup_service_status",
        "search_knowledge",
    ]

    assert all(
        definition.safety_level is ToolSafetyLevel.READ_ONLY for definition in registry.definitions
    )


def test_registry_exposes_exact_schemas() -> None:
    registry = create_controlled_support_tool_registry(
        knowledge_search=EmptyKnowledgeSearch(),
        service_status_catalog=(DeterministicServiceStatusCatalog(())),
    )

    status_binding = registry.lookup(
        name="lookup_service_status",
        version=1,
    )
    search_binding = registry.lookup(
        name="search_knowledge",
        version=1,
    )

    assert status_binding.definition.input_schema is (LookupServiceStatusInput)
    assert status_binding.definition.output_schema is (LookupServiceStatusOutput)
    assert search_binding.definition.input_schema is (SearchKnowledgeInput)
    assert search_binding.definition.output_schema is (SearchKnowledgeOutput)


def test_registry_accepts_explicit_timeouts() -> None:
    registry = create_controlled_support_tool_registry(
        knowledge_search=EmptyKnowledgeSearch(),
        service_status_catalog=(DeterministicServiceStatusCatalog(())),
        search_timeout_seconds=12,
        service_status_timeout_seconds=3,
    )

    assert (
        registry.lookup(
            name="search_knowledge",
            version=1,
        ).definition.timeout_seconds
        == 12
    )

    assert (
        registry.lookup(
            name="lookup_service_status",
            version=1,
        ).definition.timeout_seconds
        == 3
    )
