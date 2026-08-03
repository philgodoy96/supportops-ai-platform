"""Unit tests for the sensitive proposal registry."""

import pytest

from supportops.agent_tools.application.sensitive_bindings import (
    SensitiveToolBinding,
    SensitiveToolRegistry,
)
from supportops.agent_tools.domain.contracts import (
    ToolSafetyLevel,
)
from supportops.agent_tools.domain.errors import (
    ToolNotFoundError,
    ToolSafetyViolationError,
    ToolVersionNotFoundError,
)
from supportops.agent_tools.tools.escalate_ticket import (
    create_escalate_ticket_binding,
)
from supportops.agent_tools.tools.service_status import (
    DeterministicServiceStatusCatalog,
    create_lookup_service_status_binding,
)


def test_registry_resolves_exact_sensitive_version() -> None:
    binding = create_escalate_ticket_binding()
    registry = SensitiveToolRegistry((binding,))

    assert (
        registry.lookup(
            name=binding.definition.name,
            version=binding.definition.version,
        )
        is binding
    )
    assert registry.definitions == (binding.definition,)


def test_binding_rejects_read_only_definition() -> None:
    definition = create_lookup_service_status_binding(
        catalog=DeterministicServiceStatusCatalog(()),
    ).definition

    with pytest.raises(ToolSafetyViolationError):
        SensitiveToolBinding(
            definition=definition,
            safe_input_projector=lambda arguments: {},
            approval_reason_projector=lambda arguments: "Approval reason.",
        )

    assert definition.safety_level is ToolSafetyLevel.READ_ONLY


def test_registry_distinguishes_missing_name_and_version() -> None:
    registry = SensitiveToolRegistry(
        (create_escalate_ticket_binding(),),
    )

    with pytest.raises(ToolNotFoundError):
        registry.lookup(name="missing_tool", version=1)

    with pytest.raises(ToolVersionNotFoundError):
        registry.lookup(name="escalate_ticket", version=2)
