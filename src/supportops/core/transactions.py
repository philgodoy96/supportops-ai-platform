"""Application transaction boundary contracts."""

from contextlib import AbstractAsyncContextManager
from typing import Protocol


class TransactionManager(Protocol):
    """Own the transaction boundary for one application use case."""

    def transaction(
        self,
    ) -> AbstractAsyncContextManager[None]:
        """Return an atomic transaction context."""

        ...
