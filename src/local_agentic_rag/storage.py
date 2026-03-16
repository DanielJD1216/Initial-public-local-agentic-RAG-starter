from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path

import numpy as np

from .models import (
    ChunkRecord,
    CorpusIngestSummary,
    CorpusSummary,
    DocumentEntity,
    DocumentMetadata,
    DocumentPlanningArtifact,
    DocumentSearchHit,
    PlanningArtifactStatus,
    RetrievalHit,
)
from .planning_artifacts import PLANNING_ARTIFACT_VERSION, compute_planning_fingerprint

FTS_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "do",
    "for",
    "how",
    "i",
    "is",
    "me",
    "of",
    "or",
    "tell",
    "the",
    "to",
    "what",
    "when",
    "where",
    "who",
    "why",
}


class SQLiteStore:
    def __init__(self, database_path: Path) -> None:
        self.database_path = database_path

    def initialize(self) -> None:
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.executescript(
                """
                PRAGMA foreign_keys = ON;

                CREATE TABLE IF NOT EXISTS documents (
                    doc_id TEXT PRIMARY KEY,
                    source_path TEXT NOT NULL UNIQUE,
                    content_type TEXT NOT NULL,
                    checksum TEXT NOT NULL,
                    parser_version TEXT NOT NULL,
                    title TEXT NOT NULL,
                    ingested_at TEXT NOT NULL,
                    access_scope TEXT NOT NULL,
                    access_principals TEXT NOT NULL,
                    file_size_bytes INTEGER NOT NULL,
                    modified_at TEXT NOT NULL,
                    ingest_mode TEXT NOT NULL DEFAULT 'local',
                    ingest_model TEXT NOT NULL DEFAULT 'local-heuristic',
                    ingest_fingerprint TEXT NOT NULL DEFAULT 'legacy-local',
                    chunking_strategy TEXT NOT NULL DEFAULT 'heuristic'
                );

                CREATE TABLE IF NOT EXISTS chunks (
                    rowid INTEGER PRIMARY KEY AUTOINCREMENT,
                    chunk_id TEXT NOT NULL UNIQUE,
                    doc_id TEXT NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
                    source_path TEXT NOT NULL,
                    content_type TEXT NOT NULL,
                    checksum TEXT NOT NULL,
                    parser_version TEXT NOT NULL,
                    title TEXT NOT NULL,
                    ingested_at TEXT NOT NULL,
                    access_scope TEXT NOT NULL,
                    access_principals TEXT NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    section_path TEXT NOT NULL,
                    text TEXT NOT NULL,
                    location_label TEXT NOT NULL,
                    page_number INTEGER,
                    line_start INTEGER,
                    line_end INTEGER,
                    token_count INTEGER NOT NULL
                );

                CREATE TABLE IF NOT EXISTS chunk_embeddings (
                    chunk_id TEXT PRIMARY KEY REFERENCES chunks(chunk_id) ON DELETE CASCADE,
                    embedding_model TEXT NOT NULL,
                    vector_dim INTEGER NOT NULL,
                    vector_blob BLOB NOT NULL
                );

                CREATE TABLE IF NOT EXISTS document_artifacts (
                    doc_id TEXT PRIMARY KEY REFERENCES documents(doc_id) ON DELETE CASCADE,
                    artifact_version TEXT NOT NULL,
                    planning_fingerprint TEXT NOT NULL,
                    normalized_title TEXT NOT NULL,
                    short_summary TEXT NOT NULL,
                    section_outline TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS document_entities (
                    rowid INTEGER PRIMARY KEY AUTOINCREMENT,
                    doc_id TEXT NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
                    entity_type TEXT NOT NULL,
                    entity_value TEXT NOT NULL,
                    normalized_value TEXT NOT NULL,
                    UNIQUE(doc_id, entity_type, normalized_value)
                );

                CREATE INDEX IF NOT EXISTS idx_document_entities_normalized
                    ON document_entities(normalized_value, entity_type);

                CREATE VIRTUAL TABLE IF NOT EXISTS chunk_fts USING fts5(
                    text,
                    title,
                    source_path,
                    section_path,
                    content='chunks',
                    content_rowid='rowid'
                );

                CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
                    INSERT INTO chunk_fts(rowid, text, title, source_path, section_path)
                    VALUES (new.rowid, new.text, new.title, new.source_path, new.section_path);
                END;

                CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
                    INSERT INTO chunk_fts(chunk_fts, rowid, text, title, source_path, section_path)
                    VALUES ('delete', old.rowid, old.text, old.title, old.source_path, old.section_path);
                END;

                CREATE TRIGGER IF NOT EXISTS chunks_au AFTER UPDATE ON chunks BEGIN
                    INSERT INTO chunk_fts(chunk_fts, rowid, text, title, source_path, section_path)
                    VALUES ('delete', old.rowid, old.text, old.title, old.source_path, old.section_path);
                    INSERT INTO chunk_fts(rowid, text, title, source_path, section_path)
                    VALUES (new.rowid, new.text, new.title, new.source_path, new.section_path);
                END;
                """
            )
            connection.executescript(
                """
                DELETE FROM chunk_embeddings
                WHERE chunk_id NOT IN (SELECT chunk_id FROM chunks);

                DELETE FROM chunks
                WHERE doc_id NOT IN (SELECT doc_id FROM documents);

                INSERT INTO chunk_fts(chunk_fts) VALUES ('rebuild');
                """
            )
            self._ensure_document_columns(connection)

    def upsert_document(self, document: DocumentMetadata) -> None:
        document.validate()
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO documents (
                    doc_id,
                    source_path,
                    content_type,
                    checksum,
                    parser_version,
                    title,
                    ingested_at,
                    access_scope,
                    access_principals,
                    file_size_bytes,
                    modified_at,
                    ingest_mode,
                    ingest_model,
                    ingest_fingerprint,
                    chunking_strategy
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(doc_id) DO UPDATE SET
                    source_path = excluded.source_path,
                    content_type = excluded.content_type,
                    checksum = excluded.checksum,
                    parser_version = excluded.parser_version,
                    title = excluded.title,
                    ingested_at = excluded.ingested_at,
                    access_scope = excluded.access_scope,
                    access_principals = excluded.access_principals,
                    file_size_bytes = excluded.file_size_bytes,
                    modified_at = excluded.modified_at,
                    ingest_mode = excluded.ingest_mode,
                    ingest_model = excluded.ingest_model,
                    ingest_fingerprint = excluded.ingest_fingerprint,
                    chunking_strategy = excluded.chunking_strategy
                """,
                (
                    document.doc_id,
                    document.source_path,
                    document.content_type,
                    document.checksum,
                    document.parser_version,
                    document.title,
                    document.ingested_at,
                    document.access_scope,
                    json.dumps(document.access_principals),
                    document.file_size_bytes,
                    document.modified_at,
                    document.ingest_mode,
                    document.ingest_model,
                    document.ingest_fingerprint,
                    document.chunking_strategy,
                ),
            )

    def replace_chunks(
        self,
        document: DocumentMetadata,
        chunks: list[ChunkRecord],
        embeddings: dict[str, list[float]],
        *,
        embedding_model: str,
        planning_artifact: DocumentPlanningArtifact | None = None,
        entities: list[DocumentEntity] | None = None,
    ) -> None:
        document.validate()
        with self._connect() as connection:
            connection.execute("BEGIN")
            connection.execute("DELETE FROM documents WHERE doc_id = ?", (document.doc_id,))
            connection.execute(
                """
                INSERT INTO documents (
                    doc_id,
                    source_path,
                    content_type,
                    checksum,
                    parser_version,
                    title,
                    ingested_at,
                    access_scope,
                    access_principals,
                    file_size_bytes,
                    modified_at,
                    ingest_mode,
                    ingest_model,
                    ingest_fingerprint,
                    chunking_strategy
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    document.doc_id,
                    document.source_path,
                    document.content_type,
                    document.checksum,
                    document.parser_version,
                    document.title,
                    document.ingested_at,
                    document.access_scope,
                    json.dumps(document.access_principals),
                    document.file_size_bytes,
                    document.modified_at,
                    document.ingest_mode,
                    document.ingest_model,
                    document.ingest_fingerprint,
                    document.chunking_strategy,
                ),
            )
            for chunk in chunks:
                chunk.validate()
                connection.execute(
                    """
                    INSERT INTO chunks (
                        chunk_id,
                        doc_id,
                        source_path,
                        content_type,
                        checksum,
                        parser_version,
                        title,
                        ingested_at,
                        access_scope,
                        access_principals,
                        chunk_index,
                        section_path,
                        text,
                        location_label,
                        page_number,
                        line_start,
                        line_end,
                        token_count
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        chunk.chunk_id,
                        chunk.doc_id,
                        chunk.source_path,
                        chunk.content_type,
                        chunk.checksum,
                        chunk.parser_version,
                        chunk.title,
                        chunk.ingested_at,
                        chunk.access_scope,
                        json.dumps(chunk.access_principals),
                        chunk.chunk_index,
                        chunk.section_path,
                        chunk.text,
                        chunk.location_label,
                        chunk.page_number,
                        chunk.line_start,
                        chunk.line_end,
                        chunk.token_count,
                    ),
                )
                vector = np.asarray(embeddings[chunk.chunk_id], dtype=np.float32)
                connection.execute(
                    """
                    INSERT INTO chunk_embeddings (
                        chunk_id,
                        embedding_model,
                        vector_dim,
                        vector_blob
                    ) VALUES (?, ?, ?, ?)
                    """,
                    (chunk.chunk_id, embedding_model, int(vector.shape[0]), vector.tobytes()),
                )
            if planning_artifact is not None:
                connection.execute(
                    """
                    INSERT INTO document_artifacts (
                        doc_id,
                        artifact_version,
                        planning_fingerprint,
                        normalized_title,
                        short_summary,
                        section_outline
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        planning_artifact.doc_id,
                        planning_artifact.artifact_version,
                        planning_artifact.planning_fingerprint,
                        planning_artifact.normalized_title,
                        planning_artifact.short_summary,
                        json.dumps(planning_artifact.section_outline),
                    ),
                )
            for entity in entities or []:
                connection.execute(
                    """
                    INSERT INTO document_entities (
                        doc_id,
                        entity_type,
                        entity_value,
                        normalized_value
                    ) VALUES (?, ?, ?, ?)
                    """,
                    (
                        entity.doc_id,
                        entity.entity_type,
                        entity.entity_value,
                        entity.normalized_value,
                    ),
                )
            connection.commit()

    def delete_document_by_source_path(self, source_path: str) -> None:
        with self._connect() as connection:
            connection.execute("DELETE FROM documents WHERE source_path = ?", (source_path,))

    def get_document_by_source_path(self, source_path: str) -> DocumentMetadata | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM documents WHERE source_path = ?",
                (source_path,),
            ).fetchone()
        if row is None:
            return None
        return self._row_to_document(row)

    def get_document_planning_artifact(self, doc_id: str) -> DocumentPlanningArtifact | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM document_artifacts WHERE doc_id = ?",
                (doc_id,),
            ).fetchone()
        if row is None:
            return None
        return DocumentPlanningArtifact(
            doc_id=row["doc_id"],
            artifact_version=row["artifact_version"],
            planning_fingerprint=row["planning_fingerprint"],
            normalized_title=row["normalized_title"],
            short_summary=row["short_summary"],
            section_outline=json.loads(row["section_outline"]),
        )

    def get_planning_artifact_status(self, *, artifact_version: str = PLANNING_ARTIFACT_VERSION) -> PlanningArtifactStatus:
        with self._connect() as connection:
            document_rows = connection.execute(
                """
                SELECT documents.doc_id, documents.ingest_fingerprint, document_artifacts.artifact_version, document_artifacts.planning_fingerprint
                FROM documents
                LEFT JOIN document_artifacts ON document_artifacts.doc_id = documents.doc_id
                """
            ).fetchall()
        if not document_rows:
            return PlanningArtifactStatus(
                document_count=0,
                ready_document_count=0,
                missing_document_count=0,
                outdated_document_count=0,
                artifact_version=artifact_version,
                available=True,
                reindex_required_for_middleweight=False,
            )
        ready = 0
        missing = 0
        outdated = 0
        for row in document_rows:
            expected_fingerprint = compute_planning_fingerprint(
                ingest_fingerprint=row["ingest_fingerprint"],
                artifact_version=artifact_version,
            )
            actual_version = row["artifact_version"]
            actual_fingerprint = row["planning_fingerprint"]
            if actual_version is None or actual_fingerprint is None:
                missing += 1
            elif actual_version != artifact_version or actual_fingerprint != expected_fingerprint:
                outdated += 1
            else:
                ready += 1
        return PlanningArtifactStatus(
            document_count=len(document_rows),
            ready_document_count=ready,
            missing_document_count=missing,
            outdated_document_count=outdated,
            artifact_version=artifact_version,
            available=ready == len(document_rows),
            reindex_required_for_middleweight=(missing + outdated) > 0,
        )

    def search_document_titles(
        self,
        query: str,
        *,
        permissions_enabled: bool,
        active_principals: list[str],
        limit: int,
    ) -> list[DocumentSearchHit]:
        catalog = self._document_search_catalog(
            permissions_enabled=permissions_enabled,
            active_principals=active_principals,
        )
        tokens = _significant_tokens(query)
        scored: list[DocumentSearchHit] = []
        for item in catalog:
            haystack = " ".join([item["title"], item["source_path"], item["normalized_title"]]).lower()
            score = sum(2.0 for token in tokens if token in item["normalized_title"])
            score += sum(1.0 for token in tokens if token in haystack)
            if score <= 0:
                continue
            scored.append(self._catalog_item_to_hit(item, score))
        scored.sort(key=lambda item: (-item.score, item.title.lower()))
        return scored[:limit]

    def search_document_metadata(
        self,
        query: str,
        *,
        permissions_enabled: bool,
        active_principals: list[str],
        limit: int,
    ) -> list[DocumentSearchHit]:
        catalog = self._document_search_catalog(
            permissions_enabled=permissions_enabled,
            active_principals=active_principals,
        )
        tokens = _significant_tokens(query)
        scored: list[DocumentSearchHit] = []
        for item in catalog:
            entity_matches = [entity for entity in item["entities"] if any(token in entity.lower() for token in tokens)]
            outline_matches = [outline for outline in item["section_outline"] if any(token in outline.lower() for token in tokens)]
            summary = item["short_summary"].lower()
            score = sum(1.6 for token in tokens if token in summary)
            score += sum(2.2 for _entity in entity_matches)
            score += sum(1.2 for _outline in outline_matches)
            if score <= 0:
                continue
            scored.append(self._catalog_item_to_hit(item, score, entity_matches=entity_matches))
        scored.sort(key=lambda item: (-item.score, item.title.lower()))
        return scored[:limit]

    def get_section_context(
        self,
        doc_id: str,
        *,
        section_path: str,
        permissions_enabled: bool,
        active_principals: list[str],
        limit: int,
    ) -> list[ChunkRecord]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT *
                FROM chunks
                WHERE doc_id = ?
                  AND LOWER(section_path) LIKE ?
                ORDER BY chunk_index
                LIMIT ?
                """,
                (doc_id, f"%{section_path.lower()}%", limit),
            ).fetchall()
        chunks = [self._row_to_chunk(row) for row in rows]
        if not permissions_enabled:
            return chunks
        return [
            chunk
            for chunk in chunks
            if _chunk_accessible(chunk, active_principals=active_principals)
        ]

    def list_chunks_for_document(
        self,
        doc_id: str,
        *,
        permissions_enabled: bool,
        active_principals: list[str],
        limit: int | None = None,
    ) -> list[ChunkRecord]:
        sql = """
            SELECT *
            FROM chunks
            WHERE doc_id = ?
            ORDER BY chunk_index
        """
        params: list[object] = [doc_id]
        if limit is not None:
            sql += " LIMIT ?"
            params.append(limit)
        with self._connect() as connection:
            rows = connection.execute(sql, params).fetchall()
        chunks = [self._row_to_chunk(row) for row in rows]
        if not permissions_enabled:
            return chunks
        return [
            chunk
            for chunk in chunks
            if _chunk_accessible(chunk, active_principals=active_principals)
        ]

    def list_document_sources(self) -> set[str]:
        with self._connect() as connection:
            rows = connection.execute("SELECT source_path FROM documents").fetchall()
        return {row["source_path"] for row in rows}

    def get_corpus_summary(self) -> CorpusSummary:
        with self._connect() as connection:
            document_row = connection.execute(
                """
                SELECT
                    COUNT(*) AS document_count,
                    SUM(CASE WHEN access_scope = 'public' THEN 1 ELSE 0 END) AS public_document_count,
                    SUM(CASE WHEN access_scope != 'public' THEN 1 ELSE 0 END) AS restricted_document_count
                FROM documents
                """
            ).fetchone()
            chunk_row = connection.execute("SELECT COUNT(*) AS chunk_count FROM chunks").fetchone()
            principal_rows = connection.execute(
                """
                SELECT DISTINCT json_each.value AS principal
                FROM documents, json_each(documents.access_principals)
                WHERE json_each.value != '*'
                ORDER BY principal
                """
            ).fetchall()
        return CorpusSummary(
            document_count=int(document_row["document_count"] or 0),
            chunk_count=int(chunk_row["chunk_count"] or 0),
            public_document_count=int(document_row["public_document_count"] or 0),
            restricted_document_count=int(document_row["restricted_document_count"] or 0),
            principals=[row["principal"] for row in principal_rows],
        )

    def get_corpus_ingest_summary(self) -> CorpusIngestSummary:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT
                    COUNT(*) AS document_count,
                    CASE
                        WHEN COUNT(DISTINCT ingest_mode) = 0 THEN NULL
                        WHEN COUNT(DISTINCT ingest_mode) = 1 THEN MIN(ingest_mode)
                        ELSE 'mixed'
                    END AS ingest_mode,
                    CASE
                        WHEN COUNT(DISTINCT ingest_model) = 0 THEN NULL
                        WHEN COUNT(DISTINCT ingest_model) = 1 THEN MIN(ingest_model)
                        ELSE 'mixed'
                    END AS ingest_model,
                    CASE
                        WHEN COUNT(DISTINCT ingest_fingerprint) = 0 THEN NULL
                        WHEN COUNT(DISTINCT ingest_fingerprint) = 1 THEN MIN(ingest_fingerprint)
                        ELSE 'mixed'
                    END AS ingest_fingerprint,
                    CASE
                        WHEN COUNT(DISTINCT chunking_strategy) = 0 THEN NULL
                        WHEN COUNT(DISTINCT chunking_strategy) = 1 THEN MIN(chunking_strategy)
                        ELSE 'mixed'
                    END AS chunking_strategy
                FROM documents
                """
            ).fetchone()
        return CorpusIngestSummary(
            document_count=int(row["document_count"] or 0),
            mode=row["ingest_mode"],
            ingest_model=row["ingest_model"],
            ingest_fingerprint=row["ingest_fingerprint"],
            chunking_strategy=row["chunking_strategy"],
        )

    def get_chunks_by_ids(self, chunk_ids: list[str]) -> dict[str, ChunkRecord]:
        if not chunk_ids:
            return {}
        placeholders = ", ".join("?" for _ in chunk_ids)
        with self._connect() as connection:
            rows = connection.execute(
                f"SELECT * FROM chunks WHERE chunk_id IN ({placeholders})",
                chunk_ids,
            ).fetchall()
        return {row["chunk_id"]: self._row_to_chunk(row) for row in rows}

    def list_all_chunks(self) -> list[ChunkRecord]:
        with self._connect() as connection:
            rows = connection.execute("SELECT * FROM chunks ORDER BY source_path, chunk_index").fetchall()
        return [self._row_to_chunk(row) for row in rows]

    def list_embeddings(self, *, embedding_model: str) -> list[tuple[str, list[float]]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT chunk_id, vector_blob, vector_dim
                FROM chunk_embeddings
                WHERE embedding_model = ?
                ORDER BY chunk_id
                """,
                (embedding_model,),
            ).fetchall()
        output: list[tuple[str, list[float]]] = []
        for row in rows:
            vector = np.frombuffer(row["vector_blob"], dtype=np.float32, count=row["vector_dim"]).tolist()
            output.append((row["chunk_id"], vector))
        return output

    def list_embedding_models(self) -> set[str]:
        with self._connect() as connection:
            rows = connection.execute("SELECT DISTINCT embedding_model FROM chunk_embeddings").fetchall()
        return {row["embedding_model"] for row in rows}

    def list_prompt_seed_chunks(
        self,
        *,
        permissions_enabled: bool,
        active_principals: list[str],
        limit_docs: int = 6,
        chunks_per_doc: int = 2,
    ) -> list[ChunkRecord]:
        document_access_clause = ""
        access_params: list[object] = []
        if permissions_enabled:
            principal_placeholders = ", ".join("?" for _ in active_principals) or "?"
            document_access_clause = f"""
                WHERE (
                    documents.access_scope = 'public'
                    OR EXISTS (
                        SELECT 1
                        FROM json_each(documents.access_principals)
                        WHERE json_each.value IN ({principal_placeholders})
                    )
                )
            """
            access_params.extend(active_principals or ["*"])

        with self._connect() as connection:
            document_rows = connection.execute(
                f"""
                SELECT doc_id
                FROM documents
                {document_access_clause}
                ORDER BY
                    CASE WHEN access_scope = 'restricted' THEN 0 ELSE 1 END,
                    title COLLATE NOCASE
                LIMIT ?
                """,
                (*access_params, limit_docs),
            ).fetchall()
            doc_ids = [row["doc_id"] for row in document_rows]
            if not doc_ids:
                return []
            placeholders = ", ".join("?" for _ in doc_ids)
            chunk_rows = connection.execute(
                f"""
                SELECT *
                FROM chunks
                WHERE doc_id IN ({placeholders})
                """,
                doc_ids,
            ).fetchall()

        selected_rows: list[sqlite3.Row] = []
        per_doc_counts: dict[str, int] = {}
        doc_order = {doc_id: index for index, doc_id in enumerate(doc_ids)}
        ordered_chunk_rows = sorted(
            chunk_rows,
            key=lambda row: (doc_order.get(row["doc_id"], len(doc_order)), row["chunk_index"]),
        )
        for row in ordered_chunk_rows:
            count = per_doc_counts.get(row["doc_id"], 0)
            if count >= chunks_per_doc:
                continue
            selected_rows.append(row)
            per_doc_counts[row["doc_id"]] = count + 1
        return [self._row_to_chunk(row) for row in selected_rows]

    def keyword_search(
        self,
        query: str,
        *,
        limit: int,
        permissions_enabled: bool,
        active_principals: list[str],
        doc_ids: list[str] | None = None,
    ) -> list[RetrievalHit]:
        if not query.strip():
            return []
        access_clause = ""
        access_params: list[object] = []
        if permissions_enabled:
            principal_placeholders = ", ".join("?" for _ in active_principals) or "?"
            access_clause = f"""
                AND (
                    chunks.access_scope = 'public'
                    OR EXISTS (
                        SELECT 1
                        FROM json_each(chunks.access_principals)
                        WHERE json_each.value IN ({principal_placeholders})
                    )
                )
            """
            access_params.extend(active_principals or ["*"])
        doc_clause = ""
        doc_params: list[object] = []
        if doc_ids:
            placeholders = ", ".join("?" for _ in doc_ids)
            doc_clause = f" AND chunks.doc_id IN ({placeholders})"
            doc_params.extend(doc_ids)
        sql = f"""
            SELECT
                chunks.*,
                bm25(chunk_fts) AS keyword_score
            FROM chunk_fts
            JOIN chunks ON chunks.rowid = chunk_fts.rowid
            WHERE chunk_fts MATCH ?
            {access_clause}
            {doc_clause}
            ORDER BY keyword_score ASC
            LIMIT ?
        """
        rows = []
        with self._connect() as connection:
            for fts_query in _keyword_query_candidates(query):
                params: list[object] = [fts_query, *access_params, *doc_params, limit]
                rows = connection.execute(sql, params).fetchall()
                if rows:
                    break
        hits: list[RetrievalHit] = []
        for rank, row in enumerate(rows, start=1):
            score = 1.0 / (1.0 + abs(float(row["keyword_score"])))
            hits.append(
                RetrievalHit(
                    chunk=self._row_to_chunk(row),
                    score=score,
                    source="keyword",
                    rank=rank,
                )
            )
        return hits

    def _document_search_catalog(
        self,
        *,
        permissions_enabled: bool,
        active_principals: list[str],
    ) -> list[dict[str, object]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT
                    documents.doc_id,
                    documents.title,
                    documents.source_path,
                    documents.access_scope,
                    documents.access_principals,
                    document_artifacts.normalized_title,
                    document_artifacts.short_summary,
                    document_artifacts.section_outline
                FROM documents
                LEFT JOIN document_artifacts ON document_artifacts.doc_id = documents.doc_id
                ORDER BY documents.title COLLATE NOCASE
                """
            ).fetchall()
            entity_rows = connection.execute(
                """
                SELECT doc_id, entity_value
                FROM document_entities
                ORDER BY doc_id, entity_value COLLATE NOCASE
                """
            ).fetchall()

        entity_lookup: dict[str, list[str]] = {}
        for row in entity_rows:
            entity_lookup.setdefault(row["doc_id"], []).append(row["entity_value"])

        catalog: list[dict[str, object]] = []
        for row in rows:
            access_principals = json.loads(row["access_principals"])
            if permissions_enabled and not _document_accessible(
                access_scope=row["access_scope"],
                access_principals=access_principals,
                active_principals=active_principals,
            ):
                continue
            catalog.append(
                {
                    "doc_id": row["doc_id"],
                    "title": row["title"],
                    "source_path": row["source_path"],
                    "access_scope": row["access_scope"],
                    "access_principals": access_principals,
                    "normalized_title": row["normalized_title"] or str(row["title"]).lower(),
                    "short_summary": row["short_summary"] or "",
                    "section_outline": json.loads(row["section_outline"]) if row["section_outline"] else [],
                    "entities": entity_lookup.get(row["doc_id"], []),
                }
            )
        return catalog

    def _catalog_item_to_hit(
        self,
        item: dict[str, object],
        score: float,
        *,
        entity_matches: list[str] | None = None,
    ) -> DocumentSearchHit:
        return DocumentSearchHit(
            doc_id=str(item["doc_id"]),
            title=str(item["title"]),
            source_path=str(item["source_path"]),
            access_scope=str(item["access_scope"]),
            access_principals=list(item["access_principals"]),
            score=round(score, 3),
            short_summary=str(item["short_summary"]),
            section_outline=list(item["section_outline"]),
            entity_matches=entity_matches or [],
        )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def _row_to_document(self, row: sqlite3.Row) -> DocumentMetadata:
        return DocumentMetadata(
            doc_id=row["doc_id"],
            source_path=row["source_path"],
            content_type=row["content_type"],
            checksum=row["checksum"],
            parser_version=row["parser_version"],
            title=row["title"],
            ingested_at=row["ingested_at"],
            access_scope=row["access_scope"],
            access_principals=json.loads(row["access_principals"]),
            file_size_bytes=row["file_size_bytes"],
            modified_at=row["modified_at"],
            ingest_mode=row["ingest_mode"],
            ingest_model=row["ingest_model"],
            ingest_fingerprint=row["ingest_fingerprint"],
            chunking_strategy=row["chunking_strategy"],
        )

    def _row_to_chunk(self, row: sqlite3.Row) -> ChunkRecord:
        return ChunkRecord(
            chunk_id=row["chunk_id"],
            doc_id=row["doc_id"],
            source_path=row["source_path"],
            content_type=row["content_type"],
            checksum=row["checksum"],
            parser_version=row["parser_version"],
            title=row["title"],
            ingested_at=row["ingested_at"],
            access_scope=row["access_scope"],
            access_principals=json.loads(row["access_principals"]),
            chunk_index=row["chunk_index"],
            section_path=row["section_path"],
            text=row["text"],
            location_label=row["location_label"],
            page_number=row["page_number"],
            line_start=row["line_start"],
            line_end=row["line_end"],
            token_count=row["token_count"],
        )

    def _ensure_document_columns(self, connection: sqlite3.Connection) -> None:
        columns = {
            row["name"]
            for row in connection.execute("PRAGMA table_info(documents)").fetchall()
        }
        required_columns = {
            "ingest_mode": "ALTER TABLE documents ADD COLUMN ingest_mode TEXT NOT NULL DEFAULT 'local'",
            "ingest_model": "ALTER TABLE documents ADD COLUMN ingest_model TEXT NOT NULL DEFAULT 'local-heuristic'",
            "ingest_fingerprint": "ALTER TABLE documents ADD COLUMN ingest_fingerprint TEXT NOT NULL DEFAULT 'legacy-local'",
            "chunking_strategy": "ALTER TABLE documents ADD COLUMN chunking_strategy TEXT NOT NULL DEFAULT 'heuristic'",
        }
        for column_name, statement in required_columns.items():
            if column_name in columns:
                continue
            connection.execute(statement)


def _document_accessible(*, access_scope: str, access_principals: list[str], active_principals: list[str]) -> bool:
    if access_scope == "public":
        return True
    if "*" in access_principals:
        return True
    return bool(set(access_principals).intersection(active_principals))


def _chunk_accessible(chunk: ChunkRecord, *, active_principals: list[str]) -> bool:
    return _document_accessible(
        access_scope=chunk.access_scope,
        access_principals=chunk.access_principals,
        active_principals=active_principals,
    )


def _to_fts_query(query: str) -> str:
    tokens = _significant_tokens(query)
    return " AND ".join(f'"{token}"' for token in tokens)


def _keyword_query_candidates(query: str) -> list[str]:
    tokens = _significant_tokens(query)
    if not tokens:
        return []
    if len(tokens) == 1:
        return [f'"{tokens[0]}"']

    strict = " AND ".join(f'"{token}"' for token in tokens)
    relaxed = " OR ".join(f'"{token}"' for token in tokens)
    prefix = " OR ".join(f'"{token}"*' for token in tokens if len(token) > 3)
    candidates = [strict, relaxed]
    if prefix and prefix not in candidates:
        candidates.append(prefix)
    return candidates


def _significant_tokens(query: str) -> list[str]:
    tokens = re.findall(r"[a-zA-Z0-9]+", query.lower())
    filtered: list[str] = []
    seen: set[str] = set()
    for token in tokens:
        if len(token) <= 1 or token in FTS_STOPWORDS:
            continue
        if token in seen:
            continue
        seen.add(token)
        filtered.append(token)
    return filtered
