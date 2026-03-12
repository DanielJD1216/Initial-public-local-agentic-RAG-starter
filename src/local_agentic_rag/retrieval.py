from __future__ import annotations

from dataclasses import dataclass

from .clients import EmbeddingClient
from .config import AppConfig
from .models import RetrievalAttempt, RetrievalHit
from .permissions import is_accessible
from .storage import SQLiteStore
from .vector_index import VectorHit, VectorIndex


@dataclass(slots=True)
class HybridRetriever:
    config: AppConfig
    store: SQLiteStore
    vector_index: VectorIndex
    embedding_client: EmbeddingClient

    def ensure_index(self) -> None:
        self.vector_index.load()
        embedding_models = self.store.list_embedding_models()
        if not embedding_models:
            return
        if self.config.models.embedding_model not in embedding_models:
            raise RuntimeError(
                "Stored embeddings do not match the configured embedding model. "
                "Run `local-rag reindex --force-embeddings` after updating config.yaml."
            )

    def rebuild_from_store(self) -> None:
        vectors = self.store.list_embeddings(embedding_model=self.config.models.embedding_model)
        self.vector_index.build(vectors, embedding_model=self.config.models.embedding_model)

    def search(self, query: str, *, query_type: str, active_principals: list[str]) -> RetrievalAttempt:
        keyword_hits = self.store.keyword_search(
            query,
            limit=self.config.retrieval.keyword_k,
            permissions_enabled=self.config.permissions.enabled,
            active_principals=active_principals,
        )
        query_vector = self.embedding_client.embed_texts([query])[0]
        vector_hits = self._vector_search(query_vector, active_principals=active_principals)
        fused_hits = self._fuse_hits(keyword_hits, vector_hits)
        evidence_score = _evidence_score(keyword_hits, vector_hits, fused_hits)
        return RetrievalAttempt(
            query=query,
            query_type=query_type,
            keyword_hits=keyword_hits,
            vector_hits=vector_hits,
            fused_hits=fused_hits[: self.config.retrieval.top_k],
            evidence_score=evidence_score,
        )

    def _vector_search(self, query_vector: list[float], *, active_principals: list[str]) -> list[RetrievalHit]:
        overfetch = max(self.config.retrieval.vector_k * 4, 20)
        raw_hits: list[VectorHit] = self.vector_index.search(query_vector, limit=overfetch)
        chunk_lookup = self.store.get_chunks_by_ids([hit.chunk_id for hit in raw_hits])
        filtered: list[RetrievalHit] = []
        for hit in raw_hits:
            chunk = chunk_lookup.get(hit.chunk_id)
            if chunk is None:
                continue
            if not is_accessible(
                access_scope=chunk.access_scope,
                access_principals=chunk.access_principals,
                active_principals=active_principals,
                permissions_enabled=self.config.permissions.enabled,
            ):
                continue
            filtered.append(
                RetrievalHit(
                    chunk=chunk,
                    score=hit.score,
                    source="vector",
                    rank=len(filtered) + 1,
                )
            )
            if len(filtered) >= self.config.retrieval.vector_k:
                break
        return filtered

    def _fuse_hits(self, keyword_hits: list[RetrievalHit], vector_hits: list[RetrievalHit]) -> list[RetrievalHit]:
        rrf_scores: dict[str, float] = {}
        hit_lookup: dict[str, RetrievalHit] = {}
        for source_hits in (keyword_hits, vector_hits):
            for hit in source_hits:
                rrf_scores.setdefault(hit.chunk.chunk_id, 0.0)
                rrf_scores[hit.chunk.chunk_id] += 1.0 / (self.config.retrieval.rrf_k + hit.rank)
                hit_lookup.setdefault(hit.chunk.chunk_id, hit)
        ordered_ids = sorted(rrf_scores, key=lambda chunk_id: rrf_scores[chunk_id], reverse=True)
        fused_hits: list[RetrievalHit] = []
        for rank, chunk_id in enumerate(ordered_ids, start=1):
            template = hit_lookup[chunk_id]
            fused_hits.append(
                RetrievalHit(
                    chunk=template.chunk,
                    score=rrf_scores[chunk_id],
                    source="hybrid",
                    rank=rank,
                )
            )
        return fused_hits


def _evidence_score(
    keyword_hits: list[RetrievalHit],
    vector_hits: list[RetrievalHit],
    fused_hits: list[RetrievalHit],
) -> float:
    if not fused_hits:
        return 0.0
    components: list[float] = []
    if keyword_hits:
        components.append(min(1.0, len(keyword_hits) / 4.0))
    if vector_hits:
        components.append(max(0.0, min(1.0, (vector_hits[0].score + 1.0) / 2.0)))
    distinct_documents = len({hit.chunk.doc_id for hit in fused_hits[:3]})
    components.append(min(1.0, distinct_documents / 2.0))
    return round(sum(components) / len(components), 3)
