"""PostgreSQL exception inspection helpers."""

from collections import deque


def get_constraint_name(
    error: BaseException,
) -> str | None:
    """Return a database constraint name from a wrapped exception."""

    pending: deque[BaseException] = deque([error])
    visited: set[int] = set()

    while pending:
        current = pending.popleft()
        current_id = id(current)

        if current_id in visited:
            continue

        visited.add(current_id)

        constraint_name = getattr(
            current,
            "constraint_name",
            None,
        )

        if isinstance(constraint_name, str):
            return constraint_name

        for attribute_name in (
            "orig",
            "__cause__",
            "__context__",
        ):
            nested = getattr(
                current,
                attribute_name,
                None,
            )

            if isinstance(nested, BaseException):
                pending.append(nested)

    return None
