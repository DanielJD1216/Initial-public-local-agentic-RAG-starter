from __future__ import annotations

from dataclasses import dataclass

from .agent import classify_query
from .service import AppRuntime


@dataclass(slots=True)
class MCPToolset:
    runtime: AppRuntime

    def search_documents(self, query: str, *, top_k: int = 5, principals: list[str] | None = None) -> dict[str, object]:
        principals = principals or list(self.runtime.config.permissions.active_principals)
        attempt = self.runtime.retriever.search(
            query,
            query_type=classify_query(query),
            active_principals=principals,
        )
        hits = [
            {
                "chunk_id": hit.chunk.chunk_id,
                "title": hit.chunk.title,
                "source_path": hit.chunk.source_path,
                "location": hit.chunk.location_label,
                "score": hit.score,
                "text_preview": hit.chunk.text[:300],
            }
            for hit in attempt.fused_hits[:top_k]
        ]
        return {
            "query": query,
            "query_type": attempt.query_type,
            "evidence_score": attempt.evidence_score,
            "hits": hits,
        }

    def get_chunk_context(self, chunk_id: str) -> dict[str, object]:
        chunk_lookup = self.runtime.store.get_chunks_by_ids([chunk_id])
        chunk = chunk_lookup.get(chunk_id)
        if chunk is None:
            raise ValueError(f"Chunk `{chunk_id}` was not found.")
        return chunk.to_dict()

    def ask_with_citations(
        self,
        question: str,
        *,
        principals: list[str] | None = None,
        debug: bool = False,
    ) -> dict[str, object]:
        principals = principals or list(self.runtime.config.permissions.active_principals)
        result = self.runtime.agent.answer(question, active_principals=principals)
        payload = {
            "question": result.question,
            "answer": result.answer,
            "grounded": result.grounded,
            "status": result.status,
            "citations": [citation.to_dict() for citation in result.citations],
        }
        if debug:
            payload["trace"] = result.trace.to_dict()
        return payload
