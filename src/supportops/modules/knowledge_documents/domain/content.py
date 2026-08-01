"""Deterministic source-content normalization and hashing."""

from hashlib import sha256

_UTF8_BYTE_ORDER_MARK = "\ufeff"
_SHA256_HEX_LENGTH = 64
_LOWERCASE_HEXADECIMAL_CHARACTERS = frozenset("0123456789abcdef")


def normalize_document_content(content: str) -> str:
    """Normalize accepted textual source content without rewriting its meaning."""

    normalized = content
    if normalized.startswith(_UTF8_BYTE_ORDER_MARK):
        normalized = normalized.removeprefix(_UTF8_BYTE_ORDER_MARK)

    normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")
    if not normalized.strip():
        raise ValueError("Document content must contain non-whitespace text.")

    return normalized


def compute_content_sha256(content: str) -> str:
    """Return a lowercase SHA-256 hash for the exact UTF-8 content bytes."""

    return sha256(content.encode("utf-8")).hexdigest()


def validate_content_sha256(value: str, *, field_name: str) -> None:
    """Validate one lowercase hexadecimal SHA-256 value."""

    if len(value) != _SHA256_HEX_LENGTH:
        raise ValueError(f"{field_name} must be a lowercase SHA-256 hash.")

    if any(character not in _LOWERCASE_HEXADECIMAL_CHARACTERS for character in value):
        raise ValueError(f"{field_name} must be a lowercase SHA-256 hash.")
