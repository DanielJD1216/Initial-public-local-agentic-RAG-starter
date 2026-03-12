from __future__ import annotations

import re
from typing import Iterable

from .models import ChunkRecord, DocumentMetadata, ParsedSection
from .utils import estimate_tokens


def build_chunks(
    metadata: DocumentMetadata,
    sections: list[ParsedSection],
    *,
    max_chunk_tokens: int,
    overlap_tokens: int,
) -> list[ChunkRecord]:
    chunks: list[ChunkRecord] = []
    for section in sections:
        segment_texts = _split_section(section.text, max_chunk_tokens=max_chunk_tokens, overlap_tokens=overlap_tokens)
        for part_index, segment in enumerate(segment_texts, start=1):
            chunk_index = len(chunks)
            location_label = _build_location_label(section, part_index, len(segment_texts))
            chunk = ChunkRecord(
                chunk_id=f"{metadata.doc_id}-chunk-{chunk_index:04d}",
                doc_id=metadata.doc_id,
                source_path=metadata.source_path,
                content_type=metadata.content_type,
                checksum=metadata.checksum,
                parser_version=metadata.parser_version,
                title=metadata.title,
                ingested_at=metadata.ingested_at,
                access_scope=metadata.access_scope,
                access_principals=list(metadata.access_principals),
                chunk_index=chunk_index,
                section_path=section.section_path,
                text=segment,
                location_label=location_label,
                page_number=section.page_number,
                line_start=section.line_start,
                line_end=section.line_end,
                token_count=estimate_tokens(segment),
            )
            chunk.validate()
            chunks.append(chunk)
    return chunks


def _split_section(text: str, *, max_chunk_tokens: int, overlap_tokens: int) -> list[str]:
    paragraphs = [item.strip() for item in re.split(r"\n\s*\n", text) if item.strip()]
    if not paragraphs:
        return []

    chunk_texts = _group_by_token_budget(paragraphs, max_chunk_tokens=max_chunk_tokens)
    final_chunks: list[str] = []
    for chunk_text in chunk_texts:
        if estimate_tokens(chunk_text) <= max_chunk_tokens:
            final_chunks.append(chunk_text)
            continue
        sentences = re.split(r"(?<=[.!?])\s+", chunk_text)
        sentence_groups = _group_by_token_budget(sentences, max_chunk_tokens=max_chunk_tokens)
        for group in sentence_groups:
            if estimate_tokens(group) <= max_chunk_tokens:
                final_chunks.append(group)
                continue
            final_chunks.extend(_split_by_words(group, max_chunk_tokens=max_chunk_tokens, overlap_tokens=overlap_tokens))
    return final_chunks


def _group_by_token_budget(items: Iterable[str], *, max_chunk_tokens: int) -> list[str]:
    groups: list[str] = []
    buffer: list[str] = []
    current_tokens = 0
    for item in items:
        item = item.strip()
        if not item:
            continue
        item_tokens = estimate_tokens(item)
        if buffer and current_tokens + item_tokens > max_chunk_tokens:
            groups.append("\n\n".join(buffer).strip())
            buffer = [item]
            current_tokens = item_tokens
            continue
        buffer.append(item)
        current_tokens += item_tokens
    if buffer:
        groups.append("\n\n".join(buffer).strip())
    return groups


def _split_by_words(text: str, *, max_chunk_tokens: int, overlap_tokens: int) -> list[str]:
    words = text.split()
    if not words:
        return []
    max_words = max(1, int(max_chunk_tokens / 0.75))
    overlap_words = max(0, int(overlap_tokens / 0.75))
    chunks: list[str] = []
    start = 0
    while start < len(words):
        end = min(len(words), start + max_words)
        chunks.append(" ".join(words[start:end]))
        if end >= len(words):
            break
        start = max(end - overlap_words, start + 1)
    return chunks


def _build_location_label(section: ParsedSection, part_index: int, total_parts: int) -> str:
    parts: list[str] = []
    if section.page_number is not None:
        parts.append(f"Page {section.page_number}")
    if section.section_path:
        parts.append(section.section_path)
    if section.line_start is not None and section.line_end is not None:
        parts.append(f"lines {section.line_start}-{section.line_end}")
    if total_parts > 1:
        parts.append(f"part {part_index}/{total_parts}")
    return " | ".join(parts)
