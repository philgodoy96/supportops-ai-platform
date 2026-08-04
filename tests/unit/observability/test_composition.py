"""Unit tests for observability client composition."""

from typing import Any

from pydantic import SecretStr

from supportops.core.settings import Settings
from supportops.observability.composition import (
    create_observability_client,
)
from supportops.observability.langfuse import (
    LangfuseObservabilityClient,
)
from supportops.observability.noop import (
    NoOpObservabilityClient,
)


class FakeSdk:
    def create_trace_id(self, *, seed: str) -> str:
        del seed
        return "a" * 32

    def start_as_current_observation(
        self,
        **kwargs: object,
    ) -> Any:
        del kwargs
        raise AssertionError("not used")

    def create_event(self, **kwargs: object) -> Any:
        del kwargs
        raise AssertionError("not used")

    def flush(self) -> None:
        return None

    def shutdown(self) -> None:
        return None


def test_composition_defaults_to_noop() -> None:
    client = create_observability_client(Settings())

    assert isinstance(client, NoOpObservabilityClient)


def test_composition_builds_langfuse_with_secret_values() -> None:
    captured: dict[str, object] = {}

    def sdk_factory(**kwargs: object) -> FakeSdk:
        captured.update(kwargs)
        return FakeSdk()

    settings = Settings(
        ai_observability_provider="langfuse",
        langfuse_public_key=SecretStr("pk-lf-test-public"),
        langfuse_secret_key=SecretStr("sk-lf-test-secret"),
    )

    client = create_observability_client(
        settings,
        sdk_factory=sdk_factory,
    )

    assert isinstance(
        client,
        LangfuseObservabilityClient,
    )
    assert captured["public_key"] == "pk-lf-test-public"
    assert captured["secret_key"] == "sk-lf-test-secret"
    assert "mask_otel_spans" in captured
