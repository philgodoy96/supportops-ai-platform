"""Composition helpers for controlled support tools."""

from supportops.agent_tools.application.bindings import (
    ExecutableToolRegistry,
)
from supportops.agent_tools.tools.search_knowledge import (
    KnowledgeSearchService,
    create_search_knowledge_binding,
)
from supportops.agent_tools.tools.service_status import (
    DeterministicServiceStatusCatalog,
    create_lookup_service_status_binding,
)


def create_controlled_support_tool_registry(
    *,
    knowledge_search: KnowledgeSearchService,
    service_status_catalog: (DeterministicServiceStatusCatalog),
    search_timeout_seconds: float = 15,
    service_status_timeout_seconds: float = 5,
) -> ExecutableToolRegistry:
    """Compose the immutable controlled support tool registry."""

    return ExecutableToolRegistry(
        (
            create_search_knowledge_binding(
                service=knowledge_search,
                timeout_seconds=search_timeout_seconds,
            ),
            create_lookup_service_status_binding(
                catalog=service_status_catalog,
                timeout_seconds=(service_status_timeout_seconds),
            ),
        )
    )
