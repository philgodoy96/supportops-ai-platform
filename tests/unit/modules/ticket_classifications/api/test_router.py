"""Unit tests for classification inspection route orchestration."""

from datetime import UTC, datetime, timedelta
from typing import cast
from uuid import UUID

from supportops.ai.schemas.ticket_classification import (
    TICKET_CLASSIFICATION_SCHEMA_VERSION,
    TicketCategory,
    TicketIntent,
    TicketSentiment,
    TicketUrgency,
)
from supportops.modules.ticket_classifications.api.router import (
    get_ticket_classification,
    list_ticket_classifications,
)
from supportops.modules.ticket_classifications.application.pagination import (
    decode_ticket_classification_cursor,
    encode_ticket_classification_cursor,
)
from supportops.modules.ticket_classifications.application.services import (
    GetTicketClassification,
    ListTicketClassifications,
)
from supportops.modules.ticket_classifications.domain.models import (
    TicketClassification,
)

_NOW = datetime(
    2026,
    8,
    1,
    21,
    15,
    tzinfo=UTC,
)
_WORKSPACE_ID = UUID(
    "11111111-aaaa-4111-8111-111111111111",
)
_TICKET_ID = UUID(
    "22222222-bbbb-4222-8222-222222222222",
)
_AGENT_RUN_ID = UUID(
    "33333333-cccc-4333-8333-333333333333",
)
_INVOCATION_ID = UUID(
    "44444444-dddd-4444-8444-444444444444",
)
_FIRST_CLASSIFICATION_ID = UUID(
    "ffffffff-ffff-4fff-8fff-ffffffffffff",
)
_SECOND_CLASSIFICATION_ID = UUID(
    "eeeeeeee-eeee-4eee-8eee-eeeeeeeeeeee",
)
_THIRD_CLASSIFICATION_ID = UUID(
    "dddddddd-dddd-4ddd-8ddd-dddddddddddd",
)
_PROMPT_HASH = "a" * 64


class RecordingGetService:
    """Return one configured classification."""

    def __init__(
        self,
        classification: TicketClassification,
    ) -> None:
        self.classification = classification
        self.calls: list[tuple[UUID, UUID]] = []

    async def execute(
        self,
        *,
        workspace_id: UUID,
        classification_id: UUID,
    ) -> TicketClassification:
        self.calls.append(
            (
                workspace_id,
                classification_id,
            ),
        )
        return self.classification


class RecordingListService:
    """Return one configured classification page."""

    def __init__(
        self,
        classifications: tuple[
            TicketClassification,
            ...,
        ],
    ) -> None:
        self.classifications = classifications
        self.calls: list[
            tuple[
                UUID,
                UUID,
                int,
                datetime | None,
                UUID | None,
            ]
        ] = []

    async def execute(
        self,
        *,
        workspace_id: UUID,
        ticket_id: UUID,
        limit: int,
        after_created_at: datetime | None = None,
        after_classification_id: UUID | None = None,
    ) -> tuple[TicketClassification, ...]:
        self.calls.append(
            (
                workspace_id,
                ticket_id,
                limit,
                after_created_at,
                after_classification_id,
            ),
        )
        return self.classifications


def _classification(
    *,
    classification_id: UUID,
    created_at: datetime,
) -> TicketClassification:
    return TicketClassification.create(
        classification_id=classification_id,
        workspace_id=_WORKSPACE_ID,
        ticket_id=_TICKET_ID,
        agent_run_id=_AGENT_RUN_ID,
        accepted_llm_invocation_id=_INVOCATION_ID,
        category=TicketCategory.BILLING,
        intent=TicketIntent.ASK_QUESTION,
        urgency=TicketUrgency.NORMAL,
        sentiment=TicketSentiment.NEUTRAL,
        requires_human_review=False,
        summary=("The customer is asking about a duplicated invoice charge."),
        schema_version=(TICKET_CLASSIFICATION_SCHEMA_VERSION),
        prompt_id="ticket-classification",
        prompt_version=1,
        prompt_content_hash=_PROMPT_HASH,
        provider="mock",
        model="mock-ticket-classifier-v1",
        now=created_at,
    )


async def test_detail_route_projects_classification() -> None:
    classification = _classification(
        classification_id=_FIRST_CLASSIFICATION_ID,
        created_at=_NOW,
    )
    service = RecordingGetService(
        classification,
    )

    response = await get_ticket_classification(
        workspace_id=_WORKSPACE_ID,
        classification_id=_FIRST_CLASSIFICATION_ID,
        service=cast(
            GetTicketClassification,
            service,
        ),
    )

    assert response.id == _FIRST_CLASSIFICATION_ID
    assert response.accepted_invocation_id == (_INVOCATION_ID)
    assert service.calls == [
        (
            _WORKSPACE_ID,
            _FIRST_CLASSIFICATION_ID,
        ),
    ]


async def test_list_route_uses_lookahead_pagination() -> None:
    first = _classification(
        classification_id=_FIRST_CLASSIFICATION_ID,
        created_at=_NOW,
    )
    second = _classification(
        classification_id=_SECOND_CLASSIFICATION_ID,
        created_at=_NOW - timedelta(minutes=1),
    )
    third = _classification(
        classification_id=_THIRD_CLASSIFICATION_ID,
        created_at=_NOW - timedelta(minutes=2),
    )
    service = RecordingListService(
        (
            first,
            second,
            third,
        ),
    )

    response = await list_ticket_classifications(
        workspace_id=_WORKSPACE_ID,
        ticket_id=_TICKET_ID,
        service=cast(
            ListTicketClassifications,
            service,
        ),
        page_size=2,
        cursor=None,
    )

    assert [item.id for item in response.items] == [
        _FIRST_CLASSIFICATION_ID,
        _SECOND_CLASSIFICATION_ID,
    ]
    assert response.next_cursor is not None
    assert service.calls == [
        (
            _WORKSPACE_ID,
            _TICKET_ID,
            3,
            None,
            None,
        ),
    ]

    position = decode_ticket_classification_cursor(
        response.next_cursor,
    )

    assert position.created_at == second.created_at
    assert position.classification_id == second.id


async def test_list_route_decodes_keyset_cursor() -> None:
    position_created_at = _NOW - timedelta(minutes=5)
    cursor = encode_ticket_classification_cursor(
        created_at=position_created_at,
        classification_id=_SECOND_CLASSIFICATION_ID,
    )
    service = RecordingListService(())

    response = await list_ticket_classifications(
        workspace_id=_WORKSPACE_ID,
        ticket_id=_TICKET_ID,
        service=cast(
            ListTicketClassifications,
            service,
        ),
        page_size=20,
        cursor=cursor,
    )

    assert response.items == []
    assert response.next_cursor is None
    assert service.calls == [
        (
            _WORKSPACE_ID,
            _TICKET_ID,
            21,
            position_created_at,
            _SECOND_CLASSIFICATION_ID,
        ),
    ]
