from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path

import numpy as np

from .models import ChunkRecord, CorpusSummary, DocumentMetadata, RetrievalHit

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
                    modified_at TEXT NOT NULL
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
                    modified_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                    modified_at = excluded.modified_at
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
                ),
            )

    def replace_chunks(
        self,
        document: DocumentMetadata,
        chunks: list[ChunkRecord],
        embeddings: dict[str, list[float]],
        *,
        embedding_model: str,
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
                    modified_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
        )

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

    def keyword_search(
        self,
        query: str,
        *,
        limit: int,
        permissions_enabled: bool,
        active_principals: list[str],
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
        sql = f"""
            SELECT
                chunks.*,
                bm25(chunk_fts) AS keyword_score
            FROM chunk_fts
            JOIN chunks ON chunks.rowid = chunk_fts.rowid
            WHERE chunk_fts MATCH ?
            {access_clause}
            ORDER BY keyword_score ASC
            LIMIT ?
        """
        rows = []
        with self._connect() as connection:
            for fts_query in _keyword_query_candidates(query):
                params: list[object] = [fts_query, *access_params, limit]
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

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

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
