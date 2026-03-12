from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from .chunking import build_chunks
from .clients import EmbeddingClient
from .config import AppConfig
from .metadata import load_sidecar_metadata, normalize_access_principals
from .models import DocumentMetadata, utc_now_iso
from .parsers import parse_document
from .retrieval import HybridRetriever
from .storage import SQLiteStore
from .utils import PARSER_VERSION, compute_sha256, discover_documents, slugify


@dataclass(slots=True)
class IngestionReport:
    processed: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    deleted: list[str] = field(default_factory=list)
    errors: dict[str, str] = field(default_factory=dict)


@dataclass(slots=True)
class IngestionService:
    config: AppConfig
    store: SQLiteStore
    embedding_client: EmbeddingClient
    retriever: HybridRetriever

    def ingest(self, *, prune_missing: bool, force_embeddings: bool = False) -> IngestionReport:
        report = IngestionReport()
        current_files = discover_documents(self.config.paths.documents)
        current_paths = {str(path.resolve()) for path in current_files}

        if prune_missing:
            for source_path in self.store.list_document_sources():
                if source_path not in current_paths:
                    self.store.delete_document_by_source_path(source_path)
                    report.deleted.append(source_path)

        for path in current_files:
            source_path = str(path.resolve())
            try:
                checksum = compute_sha256(path)
                existing = self.store.get_document_by_source_path(source_path)
                if (
                    existing is not None
                    and existing.checksum == checksum
                    and existing.parser_version == PARSER_VERSION
                    and not force_embeddings
                ):
                    report.skipped.append(source_path)
                    continue
                parsed = parse_document(path)
                sidecar = load_sidecar_metadata(path)
                metadata = self._build_document_metadata(path, parsed.detected_title, parsed.content_type, checksum, sidecar)
                chunks = build_chunks(
                    metadata,
                    parsed.sections,
                    max_chunk_tokens=self.config.retrieval.max_chunk_tokens,
                    overlap_tokens=self.config.retrieval.overlap_tokens,
                )
                if not chunks:
                    raise ValueError("No text content could be extracted from the document.")
                vectors = self.embedding_client.embed_texts([chunk.text for chunk in chunks])
                embedding_map = {chunk.chunk_id: vector for chunk, vector in zip(chunks, vectors, strict=True)}
                self.store.replace_chunks(
                    metadata,
                    chunks,
                    embedding_map,
                    embedding_model=self.config.models.embedding_model,
                )
                report.processed.append(source_path)
            except Exception as exc:  # pragma: no cover - integration tests cover happy path
                report.errors[source_path] = str(exc)

        self.retriever.rebuild_from_store()
        return report

    def _build_document_metadata(
        self,
        path: Path,
        detected_title: str,
        content_type: str,
        checksum: str,
        sidecar: dict[str, object],
    ) -> DocumentMetadata:
        access_scope = str(sidecar.get("access_scope") or self.config.permissions.default_access_scope)
        access_principals = normalize_access_principals(sidecar.get("access_principals")) or list(
            self.config.permissions.default_access_principals
        )
        stat = path.stat()
        modified_at = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat()
        relative_path = path.resolve().relative_to(self.config.paths.documents.resolve())
        doc_slug = slugify(str(relative_path))
        doc_id = f"doc-{doc_slug}"
        metadata = DocumentMetadata(
            doc_id=doc_id,
            source_path=str(path.resolve()),
            content_type=content_type,
            checksum=checksum,
            parser_version=PARSER_VERSION,
            title=str(sidecar.get("title") or detected_title or path.stem.replace("_", " ").title()),
            ingested_at=utc_now_iso(),
            access_scope=access_scope,
            access_principals=access_principals,
            file_size_bytes=stat.st_size,
            modified_at=modified_at,
        )
        metadata.validate()
        return metadata
