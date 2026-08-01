"""Deterministic Markdown-aware token chunking."""

import re
from dataclasses import dataclass
from uuid import UUID

from supportops.knowledge_index.chunking.contracts import (
    ChunkingPolicy,
    TextTokenizer,
)
from supportops.modules.knowledge_documents.domain.models import (
    DocumentChunk,
    DocumentMediaType,
    DocumentVersion,
    DocumentVersionStatus,
)

_HEADING_PATTERN = re.compile(
    r"^[ \t]{0,3}(?P<marks>#{1,6})[ \t]+"
    r"(?P<title>.+?)(?:[ \t]+#+[ \t]*)?$"
)
_FENCE_START_PATTERN = re.compile(r"^[ \t]{0,3}(?P<marker>`{3,}|~{3,}).*$")


@dataclass(frozen=True, slots=True)
class _SemanticBlock:
    section_path: tuple[str, ...]
    content: str


@dataclass(frozen=True, slots=True)
class _SemanticSection:
    section_path: tuple[str, ...]
    blocks: tuple[str, ...]


class MarkdownTokenChunker:
    """Create bounded overlapping chunks with semantic end preferences."""

    def __init__(
        self,
        *,
        policy: ChunkingPolicy,
        tokenizer: TextTokenizer,
    ) -> None:
        if tokenizer.encoding_name != policy.tokenizer_encoding:
            raise ValueError("Tokenizer encoding does not match the chunking policy.")

        self._policy = policy
        self._tokenizer = tokenizer

    @property
    def policy(self) -> ChunkingPolicy:
        """Return the immutable chunking policy."""

        return self._policy

    def chunk(
        self,
        document_version: DocumentVersion,
    ) -> tuple[DocumentChunk, ...]:
        """Create deterministic authoritative chunks for one pending version."""

        self._validate_document_version(document_version)

        blocks = _parse_semantic_blocks(
            document_version.content,
            markdown=(document_version.media_type is DocumentMediaType.TEXT_MARKDOWN),
        )
        sections = _group_semantic_sections(blocks)

        chunks: list[DocumentChunk] = []
        for section in sections:
            for content, token_count in self._chunk_section(section):
                chunks.append(
                    DocumentChunk.create(
                        document_version=document_version,
                        ordinal=len(chunks),
                        section_path=section.section_path,
                        content=content,
                        token_count=token_count,
                        now=document_version.created_at,
                    )
                )

        if not chunks:
            raise RuntimeError("Knowledge chunking produced no authoritative chunks.")

        return tuple(chunks)

    def _validate_document_version(
        self,
        document_version: DocumentVersion,
    ) -> None:
        if document_version.status is not DocumentVersionStatus.PENDING:
            raise ValueError("Only pending document versions may be chunked.")

        profile = document_version.index_profile
        if profile is None:
            raise ValueError("Document version requires an index profile before chunking.")

        if (
            profile.chunking_strategy != self._policy.strategy
            or profile.chunking_version != self._policy.version
            or profile.tokenizer_encoding != self._policy.tokenizer_encoding
        ):
            raise ValueError(
                "Document version index profile does not match the configured chunking policy."
            )

    def _chunk_section(
        self,
        section: _SemanticSection,
    ) -> tuple[tuple[str, int], ...]:
        section_content = "".join(section.blocks)
        section_tokens = self._tokenizer.encode(section_content)
        if not section_tokens:
            return ()

        preferred_boundaries = self._preferred_token_boundaries(section.blocks)
        windows: list[tuple[str, int]] = []
        start = 0

        while start < len(section_tokens):
            maximum_end = min(
                start + self._policy.max_tokens,
                len(section_tokens),
            )
            proposed_end = self._select_window_end(
                start=start,
                maximum_end=maximum_end,
                total_tokens=len(section_tokens),
                preferred_boundaries=preferred_boundaries,
            )
            end, content, token_count = self._fit_window(
                tokens=section_tokens,
                start=start,
                proposed_end=proposed_end,
            )
            windows.append((content, token_count))

            if end == len(section_tokens):
                break

            next_start = end - self._policy.overlap_tokens
            if next_start <= start:
                raise RuntimeError("Chunking policy did not make forward progress.")
            start = next_start

        return tuple(windows)

    def _preferred_token_boundaries(
        self,
        blocks: tuple[str, ...],
    ) -> tuple[int, ...]:
        if len(blocks) <= 1:
            return ()

        boundaries: list[int] = []
        prefix = ""

        for block in blocks[:-1]:
            prefix += block
            boundary = len(self._tokenizer.encode(prefix))
            if boundary > 0:
                boundaries.append(boundary)

        return tuple(boundaries)

    def _select_window_end(
        self,
        *,
        start: int,
        maximum_end: int,
        total_tokens: int,
        preferred_boundaries: tuple[int, ...],
    ) -> int:
        if maximum_end == total_tokens:
            return maximum_end

        minimum_semantic_end = start + self._policy.overlap_tokens + 1
        candidates = (
            boundary
            for boundary in preferred_boundaries
            if minimum_semantic_end <= boundary <= maximum_end
        )

        return max(candidates, default=maximum_end)

    def _fit_window(
        self,
        *,
        tokens: tuple[int, ...],
        start: int,
        proposed_end: int,
    ) -> tuple[int, str, int]:
        end = proposed_end

        while end > start:
            content = self._tokenizer.decode(tokens[start:end])
            token_count = len(self._tokenizer.encode(content))

            if content.strip() and token_count <= self._policy.max_tokens:
                return end, content, token_count

            end -= 1

        raise RuntimeError("Unable to create a non-empty chunk within the token limit.")


def _parse_semantic_blocks(
    content: str,
    *,
    markdown: bool,
) -> tuple[_SemanticBlock, ...]:
    lines = content.splitlines(keepends=True)
    blocks: list[_SemanticBlock] = []
    section_path: tuple[str, ...] = ()
    pending_prefix = ""
    index = 0

    while index < len(lines):
        current_line = _line_body(lines[index])

        if not current_line.strip():
            pending_prefix += lines[index]
            index += 1
            continue

        fence = _parse_fence_start(current_line) if markdown else None
        heading = _parse_heading(current_line) if markdown else None

        if fence is not None:
            marker_character, marker_length = fence
            start = index
            index += 1

            while index < len(lines):
                candidate = _line_body(lines[index])
                index += 1
                if _is_closing_fence(
                    candidate,
                    marker_character=marker_character,
                    marker_length=marker_length,
                ):
                    break

            index = _consume_blank_lines(lines, index)
            blocks.append(
                _SemanticBlock(
                    section_path=section_path,
                    content=(pending_prefix + "".join(lines[start:index])),
                )
            )
            pending_prefix = ""
            continue

        if heading is not None:
            level, title = heading
            retained_depth = min(
                level - 1,
                len(section_path),
            )
            section_path = (
                *section_path[:retained_depth],
                title,
            )

            start = index
            index += 1
            index = _consume_blank_lines(lines, index)
            blocks.append(
                _SemanticBlock(
                    section_path=section_path,
                    content=(pending_prefix + "".join(lines[start:index])),
                )
            )
            pending_prefix = ""
            continue

        start = index
        index += 1

        while index < len(lines):
            candidate = _line_body(lines[index])
            if not candidate.strip():
                break
            if markdown and (
                _parse_heading(candidate) is not None or _parse_fence_start(candidate) is not None
            ):
                break
            index += 1

        index = _consume_blank_lines(lines, index)
        blocks.append(
            _SemanticBlock(
                section_path=section_path,
                content=(pending_prefix + "".join(lines[start:index])),
            )
        )
        pending_prefix = ""

    if pending_prefix:
        raise RuntimeError("Knowledge source ended without a semantic content block.")

    return tuple(blocks)


def _group_semantic_sections(
    blocks: tuple[_SemanticBlock, ...],
) -> tuple[_SemanticSection, ...]:
    sections: list[_SemanticSection] = []

    for block in blocks:
        if sections and sections[-1].section_path == block.section_path:
            current = sections[-1]
            sections[-1] = _SemanticSection(
                section_path=current.section_path,
                blocks=(*current.blocks, block.content),
            )
            continue

        sections.append(
            _SemanticSection(
                section_path=block.section_path,
                blocks=(block.content,),
            )
        )

    return tuple(sections)


def _parse_heading(
    line: str,
) -> tuple[int, str] | None:
    match = _HEADING_PATTERN.match(line)
    if match is None:
        return None

    title = match.group("title").strip()
    if not title:
        return None

    return len(match.group("marks")), title


def _parse_fence_start(
    line: str,
) -> tuple[str, int] | None:
    match = _FENCE_START_PATTERN.match(line)
    if match is None:
        return None

    marker = match.group("marker")
    return marker[0], len(marker)


def _is_closing_fence(
    line: str,
    *,
    marker_character: str,
    marker_length: int,
) -> bool:
    stripped = line.lstrip(" \t")
    indentation = len(line) - len(stripped)
    if indentation > 3:
        return False

    marker_run_length = len(stripped) - len(stripped.lstrip(marker_character))
    if marker_run_length < marker_length:
        return False

    return not stripped[marker_run_length:].strip()


def _consume_blank_lines(
    lines: list[str],
    index: int,
) -> int:
    while index < len(lines) and not _line_body(lines[index]).strip():
        index += 1

    return index


def _line_body(line: str) -> str:
    return line[:-1] if line.endswith("\n") else line


def chunk_ids(
    chunks: tuple[DocumentChunk, ...],
) -> tuple[UUID, ...]:
    """Return deterministic chunk identifiers for diagnostics and tests."""

    return tuple(chunk.id for chunk in chunks)
