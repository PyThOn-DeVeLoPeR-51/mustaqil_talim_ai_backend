from __future__ import annotations

from dataclasses import dataclass
import hashlib
import re

from app.rag.extractors import ExtractedBlock


@dataclass(frozen=True)
class TextChunk:
    chunk_index: int
    content: str
    content_hash: str
    page_number_start: int | None
    page_number_end: int | None
    section_title: str | None
    char_start: int
    char_end: int
    char_count: int


@dataclass(frozen=True)
class _MergedBlock:
    text: str
    page_number: int | None
    section_title: str | None
    global_start: int


def _normalize_block_text(text: str) -> str:
    """Normalize spacing while preserving paragraph boundaries."""
    text = text.replace("\u00a0", " ").replace("\r", "\n")
    paragraphs: list[str] = []
    for raw in re.split(r"\n\s*\n", text):
        clean = re.sub(r"[ \t]+", " ", raw).strip()
        if clean:
            paragraphs.append(clean)
    return "\n\n".join(paragraphs)


def _merge_adjacent_blocks(blocks: list[ExtractedBlock]) -> list[_MergedBlock]:
    """
    Merge adjacent extractor blocks that belong to the same logical region.

    PDF extraction already yields one block per page, so page boundaries remain
    strict. DOCX extraction yields one block per paragraph; adjacent paragraphs
    in the same section are merged before character-based chunking. This avoids
    the old 1-paragraph ~= 1-chunk behaviour.
    """
    merged: list[_MergedBlock] = []
    buffer: list[str] = []
    current_page: int | None = None
    current_section: str | None = None
    group_start = 0
    global_offset = 0

    def flush() -> None:
        nonlocal buffer
        if not buffer:
            return
        text = "\n\n".join(buffer).strip()
        if text:
            merged.append(
                _MergedBlock(
                    text=text,
                    page_number=current_page,
                    section_title=current_section,
                    global_start=group_start,
                )
            )
        buffer = []

    for block in blocks:
        text = _normalize_block_text(block.text)
        if not text:
            continue

        # PDF: never merge across page boundaries.
        # DOCX: merge consecutive paragraphs while section_title stays the same.
        logical_key = (block.page_number, block.section_title)
        current_key = (current_page, current_section)

        if buffer and logical_key != current_key:
            flush()
            group_start = global_offset

        if not buffer:
            current_page = block.page_number
            current_section = block.section_title
            group_start = global_offset

        buffer.append(text)
        global_offset += len(text) + 2  # virtual paragraph separator

    flush()
    return merged


_BOUNDARY_RE = re.compile(r"(?:\n\s*\n+|[.!?…][\"”’»)]*\s+)")


def _find_split(text: str, start: int, target_end: int, min_end: int) -> int:
    if target_end >= len(text):
        return len(text)

    window = text[min_end:target_end]
    # Prefer paragraph/sentence boundaries, then ordinary whitespace.
    boundaries = [match.end() for match in _BOUNDARY_RE.finditer(window)]
    if boundaries:
        return min_end + boundaries[-1]

    whitespace = window.rfind(" ")
    if whitespace >= 0:
        return min_end + whitespace

    return target_end


def _find_overlap_start(text: str, chunk_start: int, chunk_end: int, overlap: int) -> int:
    """Return a sentence/paragraph-aware start for the next chunk.

    The previous implementation used ``chunk_end - overlap`` and only moved to
    the next whitespace. That preserved words but could start the next chunk in
    the middle of a sentence. Here we choose the nearest logical boundary so
    retrieved content is readable while still retaining useful context.
    """
    if overlap <= 0:
        return chunk_end

    desired = max(chunk_start, chunk_end - overlap)
    boundaries: list[int] = []
    for match in _BOUNDARY_RE.finditer(text[chunk_start:chunk_end]):
        boundary = chunk_start + match.end()
        if chunk_start < boundary < chunk_end:
            boundaries.append(boundary)

    # Prefer a boundary at/after the target so overlap does not grow too much.
    for boundary in boundaries:
        if boundary >= desired:
            return boundary

    # If the trailing sentence is longer than the requested overlap, keep the
    # whole sentence rather than cutting it in half.
    before = [boundary for boundary in boundaries if boundary < desired]
    if before:
        return before[-1]

    # Extremely long single sentence fallback: at least avoid cutting a word.
    whitespace = re.search(r"\s+", text[desired:chunk_end])
    if whitespace:
        return desired + whitespace.end()
    return max(chunk_start + 1, desired)


def chunk_blocks(
    blocks: list[ExtractedBlock],
    *,
    chunk_size: int = 1800,
    overlap: int = 250,
) -> list[TextChunk]:
    if chunk_size < 200:
        raise ValueError("chunk_size kamida 200 belgi bo‘lishi kerak.")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap 0 dan katta/teng va chunk_size dan kichik bo‘lishi kerak.")

    chunks: list[TextChunk] = []

    for block in _merge_adjacent_blocks(blocks):
        text = block.text
        start = 0

        while start < len(text):
            target_end = min(start + chunk_size, len(text))
            min_end = min(start + max(120, chunk_size // 2), target_end)
            end = _find_split(text, start, target_end, min_end)
            if end <= start:
                end = target_end

            content = text[start:end].strip()
            if content:
                leading_trim = len(text[start:end]) - len(text[start:end].lstrip())
                actual_start = start + leading_trim
                char_start = block.global_start + actual_start
                char_end = char_start + len(content)
                chunks.append(
                    TextChunk(
                        chunk_index=len(chunks),
                        content=content,
                        content_hash=hashlib.sha256(content.encode("utf-8")).hexdigest(),
                        page_number_start=block.page_number,
                        page_number_end=block.page_number,
                        section_title=block.section_title,
                        char_start=char_start,
                        char_end=char_end,
                        char_count=len(content),
                    )
                )

            if end >= len(text):
                break

            next_start = _find_overlap_start(text, start, end, overlap)
            if next_start <= start:
                next_start = min(end, start + 1)
            start = next_start

    return chunks
