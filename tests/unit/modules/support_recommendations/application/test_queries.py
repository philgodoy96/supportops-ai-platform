"""Unit tests for recommendation query contracts."""

from typing import get_type_hints
from uuid import UUID

from supportops.modules.support_recommendations.application.queries import (
    SupportRecommendationQueryRepository,
)


def test_query_repository_exposes_workspace_scoped_lookup() -> None:
    annotations = get_type_hints(SupportRecommendationQueryRepository.get_by_agent_run_id)

    assert annotations["workspace_id"] is UUID
    assert annotations["agent_run_id"] is UUID
    assert "return" in annotations
