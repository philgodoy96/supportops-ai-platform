"""Tests for the tiktoken tokenizer adapter."""

from collections.abc import Sequence
from typing import cast

import pytest
from tiktoken import Encoding

import supportops.knowledge_index.chunking.tokenizer as tokenizer_module
from supportops.knowledge_index.chunking.tokenizer import (
    TiktokenTokenizer,
)


class FakeEncoding:
    """Minimal deterministic encoding used without external assets."""

    def __init__(
        self,
        *,
        name: str,
    ) -> None:
        self.name = name

    def encode(
        self,
        text: str,
        *,
        disallowed_special: set[str],
    ) -> list[int]:
        """Encode each Unicode code point."""

        assert not disallowed_special
        return [ord(character) for character in text]

    def decode(
        self,
        tokens: Sequence[int],
    ) -> str:
        """Decode Unicode code points."""

        return "".join(chr(token) for token in tokens)


def test_tokenizer_uses_explicit_encoding_and_round_trips_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    encoding = FakeEncoding(name="cl100k_base")

    monkeypatch.setattr(
        tokenizer_module,
        "get_encoding",
        lambda name: cast(Encoding, encoding),
    )

    tokenizer = TiktokenTokenizer(encoding_name="cl100k_base")
    tokens = tokenizer.encode("Restart the connection pool.")

    assert tokenizer.encoding_name == "cl100k_base"
    assert tokenizer.decode(tokens) == ("Restart the connection pool.")


def test_tokenizer_rejects_loaded_encoding_identity_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    encoding = FakeEncoding(name="o200k_base")

    monkeypatch.setattr(
        tokenizer_module,
        "get_encoding",
        lambda name: cast(Encoding, encoding),
    )

    with pytest.raises(
        ValueError,
        match=(
            r"Loaded tokenizer encoding does not match "
            r"the requested name\."
        ),
    ):
        TiktokenTokenizer(encoding_name="cl100k_base")


@pytest.mark.parametrize(
    "encoding_name",
    [
        "",
        " cl100k_base",
        "cl100k_base ",
    ],
)
def test_tokenizer_rejects_invalid_encoding_name(
    encoding_name: str,
) -> None:
    with pytest.raises(ValueError):
        TiktokenTokenizer(encoding_name=encoding_name)
