"""Tiktoken adapter used by knowledge-document chunking."""

from collections.abc import Sequence

from tiktoken import Encoding, get_encoding

from supportops.knowledge_index.chunking.contracts import TextTokenizer


class TiktokenTokenizer(TextTokenizer):
    """Expose one explicit tiktoken encoding behind an owned contract."""

    def __init__(
        self,
        *,
        encoding_name: str,
    ) -> None:
        if not encoding_name:
            raise ValueError("encoding_name is required.")
        if encoding_name != encoding_name.strip():
            raise ValueError("encoding_name must not contain surrounding whitespace.")

        encoding = get_encoding(encoding_name)
        if encoding.name != encoding_name:
            raise ValueError("Loaded tokenizer encoding does not match the requested name.")

        self._encoding_name = encoding_name
        self._encoding: Encoding = encoding

    @property
    def encoding_name(self) -> str:
        """Return the explicit encoding identity."""

        return self._encoding_name

    def encode(self, text: str) -> tuple[int, ...]:
        """Encode ordinary source text without interpreting special tokens."""

        return tuple(
            self._encoding.encode(
                text,
                disallowed_special=set(),
            )
        )

    def decode(self, tokens: Sequence[int]) -> str:
        """Decode one deterministic token sequence."""

        return self._encoding.decode(list(tokens))
