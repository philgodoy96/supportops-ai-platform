"""Async Qdrant client construction and disposal."""

from qdrant_client import AsyncQdrantClient

from supportops.core.settings import Settings


def create_qdrant_client(settings: Settings) -> AsyncQdrantClient:
    """Create an async Qdrant client from validated settings."""

    return AsyncQdrantClient(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key,
        timeout=int(settings.dependency_health_timeout_seconds),
        prefer_grpc=False,
    )


async def close_qdrant_client(client: AsyncQdrantClient) -> None:
    """Release resources owned by an async Qdrant client."""

    await client.close()
