"""Unit tests for shared persisted tool-audit schemas."""

from uuid import UUID

import pytest
from pydantic import JsonValue

from supportops.agent_graph.application.tool_audit_schemas import (
    PersistedToolAuditOutputError,
    parse_persisted_search_knowledge_output,
    parse_persisted_service_status_output,
)

_RETRIEVAL_QUERY_ID = UUID("10000000-0000-4000-8000-000000000001")
_DOCUMENT_ID = UUID("20000000-0000-4000-8000-000000000002")
_DOCUMENT_VERSION_ID = UUID("30000000-0000-4000-8000-000000000003")
_FIRST_CHUNK_ID = UUID("40000000-0000-4000-8000-000000000004")
_SECOND_CHUNK_ID = UUID("50000000-0000-4000-8000-000000000005")


def _evidence(
    *,
    rank: int,
    score: float,
    chunk_id: UUID,
    chunk_ordinal: int,
) -> dict[str, JsonValue]:
    return {
        "rank": rank,
        "score": score,
        "document_id": str(_DOCUMENT_ID),
        "document_version_id": str(_DOCUMENT_VERSION_ID),
        "chunk_id": str(chunk_id),
        "chunk_ordinal": chunk_ordinal,
        "content_sha256": f"{rank:x}" * 64,
    }


def _search_output() -> dict[str, JsonValue]:
    return {
        "retrieval_query_id": str(_RETRIEVAL_QUERY_ID),
        "searched_version_count": 1,
        "result_count": 2,
        "evidence": [
            _evidence(
                rank=1,
                score=0.91,
                chunk_id=_FIRST_CHUNK_ID,
                chunk_ordinal=0,
            ),
            _evidence(
                rank=2,
                score=0.81,
                chunk_id=_SECOND_CHUNK_ID,
                chunk_ordinal=1,
            ),
        ],
    }


def test_parses_deterministic_search_projection() -> None:
    output = parse_persisted_search_knowledge_output(_search_output())

    assert output.retrieval_query_id == (_RETRIEVAL_QUERY_ID)
    assert output.result_count == 2
    assert tuple(evidence.rank for evidence in output.evidence) == (
        1,
        2,
    )


def test_rejects_non_contiguous_evidence_ranks() -> None:
    payload = _search_output()
    payload["result_count"] = 1
    payload["evidence"] = [
        _evidence(
            rank=2,
            score=0.81,
            chunk_id=_SECOND_CHUNK_ID,
            chunk_ordinal=1,
        )
    ]

    with pytest.raises(
        PersistedToolAuditOutputError,
        match="knowledge-search",
    ):
        parse_persisted_search_knowledge_output(payload)


def test_rejects_duplicate_evidence_chunks() -> None:
    payload = _search_output()
    second = _evidence(
        rank=2,
        score=0.81,
        chunk_id=_FIRST_CHUNK_ID,
        chunk_ordinal=1,
    )
    payload["evidence"] = [
        _evidence(
            rank=1,
            score=0.91,
            chunk_id=_FIRST_CHUNK_ID,
            chunk_ordinal=0,
        ),
        second,
    ]

    with pytest.raises(
        PersistedToolAuditOutputError,
        match="knowledge-search",
    ):
        parse_persisted_search_knowledge_output(payload)


def test_rejects_evidence_out_of_score_order() -> None:
    payload = _search_output()
    payload["evidence"] = [
        _evidence(
            rank=1,
            score=0.71,
            chunk_id=_FIRST_CHUNK_ID,
            chunk_ordinal=0,
        ),
        _evidence(
            rank=2,
            score=0.81,
            chunk_id=_SECOND_CHUNK_ID,
            chunk_ordinal=1,
        ),
    ]

    with pytest.raises(
        PersistedToolAuditOutputError,
        match="knowledge-search",
    ):
        parse_persisted_search_knowledge_output(payload)


def test_parses_consistent_service_status_projection() -> None:
    output = parse_persisted_service_status_output(
        {
            "service_name": "payments-api",
            "status": "degraded",
            "incident_reference": "incident-local-001",
            "has_incident": True,
            "source": "deterministic_catalog",
        }
    )

    assert output.service_name == "payments-api"
    assert output.status.value == "degraded"
    assert output.has_incident is True


def test_rejects_inconsistent_service_incident_fields() -> None:
    with pytest.raises(
        PersistedToolAuditOutputError,
        match="service-status",
    ):
        parse_persisted_service_status_output(
            {
                "service_name": "payments-api",
                "status": "operational",
                "incident_reference": None,
                "has_incident": True,
                "source": "deterministic_catalog",
            }
        )
