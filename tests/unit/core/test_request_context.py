"""Unit tests for HTTP request context primitives."""

import asyncio
from uuid import UUID

import pytest

from supportops.core.request_context import (
    RequestContext,
    create_request_context,
    get_request_context,
    request_context_scope,
)


def test_create_request_context_generates_request_id_and_defaults_correlation_id() -> None:
    context = create_request_context()

    assert context.request_id.version == 4
    assert context.correlation_id == context.request_id


def test_create_request_context_accepts_valid_incoming_correlation_id() -> None:
    incoming_correlation_id = UUID("c11375a8-80e6-4aa2-838f-f342cfcb99ae")

    context = create_request_context(str(incoming_correlation_id).upper())

    assert context.request_id.version == 4
    assert context.request_id != incoming_correlation_id
    assert context.correlation_id == incoming_correlation_id


@pytest.mark.parametrize(
    "incoming_correlation_id",
    [
        "",
        "not-a-uuid",
        " c11375a8-80e6-4aa2-838f-f342cfcb99ae ",
    ],
)
def test_create_request_context_rejects_invalid_correlation_id(
    incoming_correlation_id: str,
) -> None:
    context = create_request_context(incoming_correlation_id)

    assert context.correlation_id == context.request_id


def test_request_context_scope_restores_previous_context() -> None:
    outer_context = RequestContext(
        request_id=UUID("93c0a0e3-7fcc-4ef2-bfeb-ad7c1105c612"),
        correlation_id=UUID("fc38bdaf-757e-4230-ae2d-935a139d3d4f"),
    )
    inner_context = RequestContext(
        request_id=UUID("a2af2b25-d58a-4823-b27f-7802491989e6"),
        correlation_id=UUID("474f8f95-e1a7-4925-a999-77ed077e408f"),
    )

    assert get_request_context() is None

    with request_context_scope(outer_context):
        assert get_request_context() == outer_context

        with request_context_scope(inner_context):
            assert get_request_context() == inner_context

        assert get_request_context() == outer_context

    assert get_request_context() is None


def test_request_context_scope_cleans_up_after_exception() -> None:
    context = RequestContext(
        request_id=UUID("47982bd2-b64b-4082-8eee-757efbf7e494"),
        correlation_id=UUID("89258483-2524-4271-badb-4751cc8a7a29"),
    )

    with (
        pytest.raises(RuntimeError, match="controlled failure"),
        request_context_scope(context),
    ):
        assert get_request_context() == context
        raise RuntimeError("controlled failure")

    assert get_request_context() is None


async def test_request_context_is_isolated_between_async_tasks() -> None:
    first_context = RequestContext(
        request_id=UUID("cabda2ee-625e-4ff7-8897-cad4d44aab31"),
        correlation_id=UUID("fabefc91-7032-4484-87ae-e4430773a85d"),
    )
    second_context = RequestContext(
        request_id=UUID("d99eae85-7516-4578-be81-25aecc8508a5"),
        correlation_id=UUID("1a3de719-f933-4634-86d1-d85ed7ea7267"),
    )
    barrier = asyncio.Barrier(2)

    async def observe_context(context: RequestContext) -> RequestContext:
        with request_context_scope(context):
            await barrier.wait()
            observed = get_request_context()
            assert observed is not None
            return observed

    observed_contexts = list(
        await asyncio.gather(
            observe_context(first_context),
            observe_context(second_context),
        )
    )

    assert observed_contexts == [first_context, second_context]
    assert get_request_context() is None
