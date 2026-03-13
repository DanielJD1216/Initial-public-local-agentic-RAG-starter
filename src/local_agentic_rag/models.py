from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(slots=True)
class ParsedSection:
    text: str
    section_path: str
    page_number: int | None = None
    line_start: int | None = None
    line_end: int | None = None


@dataclass(slots=True)
class ParsedDocument:
    source_path: str
    content_type: str
    detected_title: str
    sections: list[ParsedSection]


@dataclass(slots=True)
class DocumentMetadata:
    doc_id: str
    source_path: str
    content_type: str
    checksum: str
    parser_version: str
    title: str
    ingested_at: str
    access_scope: str
    access_principals: list[str]
    file_size_bytes: int
    modified_at: str
    ingest_mode: str
    ingest_model: str
    ingest_fingerprint: str
    chunking_strategy: str

    def validate(self) -> None:
        required = {
            "doc_id": self.doc_id,
            "source_path": self.source_path,
            "content_type": self.content_type,
            "checksum": self.checksum,
            "parser_version": self.parser_version,
            "title": self.title,
            "ingested_at": self.ingested_at,
            "access_scope": self.access_scope,
            "ingest_mode": self.ingest_mode,
            "ingest_model": self.ingest_model,
            "ingest_fingerprint": self.ingest_fingerprint,
            "chunking_strategy": self.chunking_strategy,
        }
        missing = [name for name, value in required.items() if not str(value).strip()]
        if missing:
            raise ValueError(f"Document metadata missing required fields: {', '.join(missing)}")
        if not self.access_principals:
            raise ValueError("Document metadata must include at least one access principal.")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class CorpusSummary:
    document_count: int
    chunk_count: int
    public_document_count: int
    restricted_document_count: int
    principals: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class PromptSuggestion:
    label: str
    prompt: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class CorpusIngestSummary:
    document_count: int
    mode: str | None
    ingest_model: str | None
    ingest_fingerprint: str | None
    chunking_strategy: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ChunkRecord:
    chunk_id: str
    doc_id: str
    source_path: str
    content_type: str
    checksum: str
    parser_version: str
    title: str
    ingested_at: str
    access_scope: str
    access_principals: list[str]
    chunk_index: int
    section_path: str
    text: str
    location_label: str
    page_number: int | None = None
    line_start: int | None = None
    line_end: int | None = None
    token_count: int = 0

    def validate(self) -> None:
        required = {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "source_path": self.source_path,
            "content_type": self.content_type,
            "checksum": self.checksum,
            "parser_version": self.parser_version,
            "title": self.title,
            "ingested_at": self.ingested_at,
            "access_scope": self.access_scope,
            "section_path": self.section_path,
            "location_label": self.location_label,
            "text": self.text,
        }
        missing = [name for name, value in required.items() if not str(value).strip()]
        if missing:
            raise ValueError(f"Chunk metadata missing required fields: {', '.join(missing)}")
        if self.chunk_index < 0:
            raise ValueError("chunk_index must be non-negative.")
        if not self.access_principals:
            raise ValueError("Chunk metadata must include at least one access principal.")

    def citation_label(self) -> str:
        filename = Path(self.source_path).name
        return f"{self.title} ({filename}) - {self.location_label}"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class EmbeddingRecord:
    chunk_id: str
    embedding_model: str
    vector: list[float]


@dataclass(slots=True)
class RetrievalHit:
    chunk: ChunkRecord
    score: float
    source: str
    rank: int

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["chunk"] = self.chunk.to_dict()
        return payload


@dataclass(slots=True)
class RetrievalAttempt:
    query: str
    query_type: str
    keyword_hits: list[RetrievalHit]
    vector_hits: list[RetrievalHit]
    fused_hits: list[RetrievalHit]
    evidence_score: float
    rewritten: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "query_type": self.query_type,
            "keyword_hits": [item.to_dict() for item in self.keyword_hits],
            "vector_hits": [item.to_dict() for item in self.vector_hits],
            "fused_hits": [item.to_dict() for item in self.fused_hits],
            "evidence_score": self.evidence_score,
            "rewritten": self.rewritten,
        }


@dataclass(slots=True)
class AnswerCitation:
    chunk_id: str
    citation: str
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class AgentTrace:
    initial_query: str
    query_type: str
    rewritten_query: str | None
    attempts: list[RetrievalAttempt] = field(default_factory=list)
    verification_notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "initial_query": self.initial_query,
            "query_type": self.query_type,
            "rewritten_query": self.rewritten_query,
            "attempts": [attempt.to_dict() for attempt in self.attempts],
            "verification_notes": self.verification_notes,
        }


@dataclass(slots=True)
class AnswerResult:
    question: str
    answer: str
    grounded: bool
    citations: list[AnswerCitation]
    trace: AgentTrace
    retrieved_chunks: list[ChunkRecord]
    status: str = "ok"

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "answer": self.answer,
            "grounded": self.grounded,
            "citations": [citation.to_dict() for citation in self.citations],
            "trace": self.trace.to_dict(),
            "retrieved_chunks": [chunk.to_dict() for chunk in self.retrieved_chunks],
            "status": self.status,
        }
