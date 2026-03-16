from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from .chunking import build_chunks
from .ingest_bridge import (
    LOCAL_INGEST_MODEL,
    EnrichedDocument,
    IngestEnrichmentClient,
    compute_ingest_fingerprint,
)
from .clients import EmbeddingClient
from .config import AppConfig
from .metadata import load_sidecar_metadata, normalize_access_principals
from .models import DocumentMetadata, ParsedSection, utc_now_iso
from .parsers import parse_document
from .planning_artifacts import build_planning_artifacts
from .retrieval import HybridRetriever
from .sensitivity import should_auto_restrict_document
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
    ingest_enrichment_client: IngestEnrichmentClient | None = None

    def ingest(self, *, prune_missing: bool, force_embeddings: bool = False) -> IngestionReport:
        report = IngestionReport()
        current_files = discover_documents(self.config.paths.documents)
        current_paths = {str(path.resolve()) for path in current_files}
        ingest_fingerprint = self._current_ingest_fingerprint()

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
                    and existing.ingest_fingerprint == ingest_fingerprint
                    and not force_embeddings
                ):
                    report.skipped.append(source_path)
                    continue
                parsed = parse_document(path)
                sidecar = load_sidecar_metadata(path)
                enriched = self._maybe_enrich_document(
                    source_path=source_path,
                    content_type=parsed.content_type,
                    detected_title=parsed.detected_title,
                    sections=parsed.sections,
                )
                effective_sections = self._effective_sections(parsed.sections, enriched)
                semantic_chunks = (
                    enriched.semantic_chunks
                    if enriched and self.config.ingest.semantic_chunking and enriched.semantic_chunks
                    else None
                )
                metadata = self._build_document_metadata(
                    path,
                    enriched.title if enriched is not None else parsed.detected_title,
                    parsed.content_type,
                    checksum,
                    sidecar,
                    sections=effective_sections,
                    bridge_metadata=enriched.metadata if enriched is not None and self.config.ingest.metadata_enrichment else None,
                    ingest_fingerprint=ingest_fingerprint,
                    chunking_strategy="semantic" if semantic_chunks else "heuristic",
                )
                chunks = build_chunks(
                    metadata,
                    effective_sections,
                    max_chunk_tokens=self.config.retrieval.max_chunk_tokens,
                    overlap_tokens=self.config.retrieval.overlap_tokens,
                    semantic_chunks=semantic_chunks,
                )
                if not chunks:
                    raise ValueError("No text content could be extracted from the document.")
                planning_artifact, entities = build_planning_artifacts(
                    metadata=metadata,
                    sections=effective_sections,
                    chunks=chunks,
                )
                vectors = self.embedding_client.embed_texts([chunk.text for chunk in chunks])
                embedding_map = {chunk.chunk_id: vector for chunk, vector in zip(chunks, vectors, strict=True)}
                self.store.replace_chunks(
                    metadata,
                    chunks,
                    embedding_map,
                    embedding_model=self.config.models.embedding_model,
                    planning_artifact=planning_artifact,
                    entities=entities,
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
        sections: list[ParsedSection],
        *,
        bridge_metadata: dict[str, object] | None,
        ingest_fingerprint: str,
        chunking_strategy: str,
    ) -> DocumentMetadata:
        explicit_scope = sidecar.get("access_scope") or (bridge_metadata or {}).get("access_scope")
        explicit_principals = sidecar.get("access_principals") or (bridge_metadata or {}).get("access_principals")
        if explicit_scope or explicit_principals:
            access_scope = str(explicit_scope or self.config.permissions.default_access_scope)
            access_principals = normalize_access_principals(explicit_principals) or list(
                self.config.permissions.default_access_principals
            )
        elif self._should_auto_restrict(detected_title, sections):
            access_scope = "restricted"
            access_principals = list(self.config.permissions.auto_restrict_principals)
        else:
            access_scope = self.config.permissions.default_access_scope
            access_principals = list(self.config.permissions.default_access_principals)
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
            title=str(
                sidecar.get("title")
                or (bridge_metadata or {}).get("title")
                or detected_title
                or path.stem.replace("_", " ").title()
            ),
            ingested_at=utc_now_iso(),
            access_scope=access_scope,
            access_principals=access_principals,
            file_size_bytes=stat.st_size,
            modified_at=modified_at,
            ingest_mode=self.config.ingest.mode,
            ingest_model=self._current_ingest_model(),
            ingest_fingerprint=ingest_fingerprint,
            chunking_strategy=chunking_strategy,
        )
        metadata.validate()
        return metadata

    def _should_auto_restrict(self, detected_title: str, sections: list[ParsedSection]) -> bool:
        if not self.config.permissions.auto_restrict_enabled:
            return False
        return should_auto_restrict_document(
            title=detected_title,
            sections=sections,
            markers=self.config.permissions.auto_restrict_markers,
        )

    def _current_ingest_fingerprint(self) -> str:
        bridge_enabled = self.config.ingest.mode == "bridge"
        return compute_ingest_fingerprint(
            parser_version=PARSER_VERSION,
            mode=self.config.ingest.mode,
            ingest_model=self._current_ingest_model(),
            cleanup=self.config.ingest.cleanup if bridge_enabled else False,
            semantic_chunking=self.config.ingest.semantic_chunking if bridge_enabled else False,
            metadata_enrichment=self.config.ingest.metadata_enrichment if bridge_enabled else False,
            max_chunk_tokens=self.config.retrieval.max_chunk_tokens,
            overlap_tokens=self.config.retrieval.overlap_tokens,
        )

    def _current_ingest_model(self) -> str:
        if self.config.ingest.mode == "bridge":
            return self.config.ingest.bridge_model
        return LOCAL_INGEST_MODEL

    def _maybe_enrich_document(
        self,
        *,
        source_path: str,
        content_type: str,
        detected_title: str,
        sections: list[ParsedSection],
    ) -> EnrichedDocument | None:
        if self.config.ingest.mode != "bridge":
            return None
        if self.ingest_enrichment_client is None:
            raise RuntimeError("Bridge ingest mode is enabled, but no ingest bridge client is configured.")
        return self.ingest_enrichment_client.enrich_document(
            source_path=source_path,
            content_type=content_type,
            detected_title=detected_title,
            sections=[
                {
                    "text": section.text,
                    "section_path": section.section_path,
                    "page_number": section.page_number,
                    "line_start": section.line_start,
                    "line_end": section.line_end,
                }
                for section in sections
            ],
            stage_flags={
                "cleanup": self.config.ingest.cleanup,
                "semantic_chunking": self.config.ingest.semantic_chunking,
                "metadata_enrichment": self.config.ingest.metadata_enrichment,
            },
        )

    def _effective_sections(
        self,
        parsed_sections: list[ParsedSection],
        enriched: EnrichedDocument | None,
    ) -> list[ParsedSection]:
        if enriched is None:
            return parsed_sections
        return [
            ParsedSection(
                text=section.text,
                section_path=section.section_path,
                page_number=section.page_number,
                line_start=section.line_start,
                line_end=section.line_end,
            )
            for section in enriched.sections
        ]
