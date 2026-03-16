from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .agent import classify_query
from .ingest_bridge import discover_ingest_bridge
from .service import AppRuntime


@dataclass(slots=True)
class MCPToolset:
    runtime: AppRuntime

    def get_runtime_status(self) -> dict[str, object]:
        agent_status = self.runtime.agent.runtime_status()
        if self.runtime.config.ingest.mode == "bridge":
            bridge_status = discover_ingest_bridge(
                self.runtime.config.ingest.bridge_base_url,
                model=self.runtime.config.ingest.bridge_model,
                timeout_seconds=self.runtime.config.ingest.request_timeout_seconds,
            )
        else:
            bridge_status = {
                "base_url": self.runtime.config.ingest.bridge_base_url,
                "reachable": None,
                "model": self.runtime.config.ingest.bridge_model,
                "error": None,
            }
        return {
            "project_name": self.runtime.config.project_name,
            "documents_path": str(self.runtime.config.paths.documents),
            "local_models": {
                "profile": self.runtime.config.models.profile,
                "base_url": self.runtime.config.models.base_url,
                "chat_model": self.runtime.config.models.chat_model,
                "embedding_model": self.runtime.config.models.embedding_model,
            },
            "ingest": {
                "mode": self.runtime.config.ingest.mode,
                "bridge": bridge_status if isinstance(bridge_status, dict) else bridge_status.to_dict(),
                "corpus": self.runtime.store.get_corpus_ingest_summary().to_dict(),
            },
            "agent": agent_status.to_dict(),
            "permissions_enabled": self.runtime.config.permissions.enabled,
            "default_principals": list(self.runtime.config.permissions.active_principals),
        }

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
            "failure_reason": result.failure_reason,
            "task_mode": result.task_mode,
            "stop_reason": result.stop_reason,
            "citations": [citation.to_dict() for citation in result.citations],
        }
        if debug:
            payload["trace"] = result.trace.to_dict()
        return payload

    def ingest_path(
        self,
        path: str,
        *,
        prune_missing: bool = True,
        force_embeddings: bool = False,
    ) -> dict[str, object]:
        candidate = Path(path).expanduser().resolve()
        if not candidate.exists():
            raise ValueError(f"Documents path does not exist: {candidate}")
        if not candidate.is_dir():
            raise ValueError(f"Documents path is not a directory: {candidate}")
        self.runtime.config.paths.documents = candidate
        report = self.runtime.ingestion.ingest(prune_missing=prune_missing, force_embeddings=force_embeddings)
        return {
            "status": self.get_runtime_status(),
            "report": {
                "processed": report.processed,
                "skipped": report.skipped,
                "deleted": report.deleted,
                "errors": report.errors,
            },
        }
