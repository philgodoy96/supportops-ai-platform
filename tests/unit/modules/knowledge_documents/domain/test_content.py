"""Tests for deterministic source normalization and hashing."""

from hashlib import sha256

import pytest

from supportops.modules.knowledge_documents.domain.content import (
    compute_content_sha256,
    normalize_document_content,
)


def test_normalization_removes_bom_and_normalizes_line_endings() -> None:
    content = "\ufeff# Runbook\r\n\rConnection recovery\rFinal line\n"

    normalized = normalize_document_content(content)

    assert normalized == ("# Runbook\n\nConnection recovery\nFinal line\n")


def test_normalization_preserves_meaningful_whitespace() -> None:
    content = "```text\nvalue  \n```\n\nParagraph with trailing spaces  \n"

    assert normalize_document_content(content) == content


@pytest.mark.parametrize(
    "content",
    ["", "   ", "\r\n\t\r"],
)
def test_normalization_rejects_whitespace_only_content(
    content: str,
) -> None:
    with pytest.raises(
        ValueError,
        match=r"Document content must contain non-whitespace text\.",
    ):
        normalize_document_content(content)


def test_content_hash_uses_exact_utf8_bytes() -> None:
    content = "Database recovery — primary region\n"

    assert compute_content_sha256(content) == sha256(content.encode("utf-8")).hexdigest()
