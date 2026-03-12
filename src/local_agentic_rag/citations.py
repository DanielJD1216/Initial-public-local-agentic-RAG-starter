from __future__ import annotations

from .models import AnswerCitation, ChunkRecord


def format_citation(chunk: ChunkRecord) -> str:
    return f"{chunk.title} — {chunk.location_label}"


def build_citations(chunk_ids: list[str], chunk_lookup: dict[str, ChunkRecord], reasons: dict[str, str] | None = None) -> list[AnswerCitation]:
    reasons = reasons or {}
    citations: list[AnswerCitation] = []
    for chunk_id in chunk_ids:
        chunk = chunk_lookup.get(chunk_id)
        if chunk is None:
            continue
        citations.append(
            AnswerCitation(
                chunk_id=chunk_id,
                citation=format_citation(chunk),
                reason=reasons.get(chunk_id, ""),
            )
        )
    return citations
