from __future__ import annotations

from dataclasses import dataclass

from .config import AppConfig
from .models import ChunkRecord, DocumentSearchHit, RetrievalAttempt, ToolEvent
from .retrieval import HybridRetriever
from .storage import SQLiteStore


@dataclass(slots=True)
class ToolOutcome:
    event: ToolEvent
    attempt: RetrievalAttempt | None = None
    documents: list[DocumentSearchHit] | None = None
    chunks: list[ChunkRecord] | None = None
    blocked_principals: list[str] | None = None


@dataclass(slots=True)
class AgentToolDispatcher:
    config: AppConfig
    store: SQLiteStore
    retriever: HybridRetriever

    def semantic_search(
        self,
        query: str,
        *,
        active_principals: list[str],
        doc_ids: list[str] | None = None,
    ) -> ToolOutcome:
        attempt = self.retriever.semantic_search(
            query,
            query_type="semantic_search",
            active_principals=active_principals,
            doc_ids=doc_ids,
        )
        return ToolOutcome(
            attempt=attempt,
            event=ToolEvent(
                tool_name="semantic_search",
                status="ok",
                query=query,
                summary=f"Returned {len(attempt.vector_hits)} vector hits.",
                result_count=len(attempt.vector_hits),
                doc_ids=_doc_ids_from_attempt(attempt),
                chunk_ids=[hit.chunk.chunk_id for hit in attempt.vector_hits],
            ),
        )

    def keyword_search(
        self,
        query: str,
        *,
        active_principals: list[str],
        doc_ids: list[str] | None = None,
    ) -> ToolOutcome:
        attempt = self.retriever.keyword_search(
            query,
            query_type="keyword_search",
            active_principals=active_principals,
            doc_ids=doc_ids,
        )
        return ToolOutcome(
            attempt=attempt,
            event=ToolEvent(
                tool_name="keyword_search",
                status="ok",
                query=query,
                summary=f"Returned {len(attempt.keyword_hits)} keyword hits.",
                result_count=len(attempt.keyword_hits),
                doc_ids=_doc_ids_from_attempt(attempt),
                chunk_ids=[hit.chunk.chunk_id for hit in attempt.keyword_hits],
            ),
        )

    def title_search(self, query: str, *, active_principals: list[str], limit: int = 5) -> ToolOutcome:
        documents = self.store.search_document_titles(
            query,
            permissions_enabled=self.config.permissions.enabled,
            active_principals=active_principals,
            limit=limit,
        )
        return ToolOutcome(
            documents=documents,
            event=ToolEvent(
                tool_name="title_search",
                status="ok",
                query=query,
                summary=f"Matched {len(documents)} document titles.",
                result_count=len(documents),
                doc_ids=[item.doc_id for item in documents],
            ),
        )

    def metadata_search(self, query: str, *, active_principals: list[str], limit: int = 5) -> ToolOutcome:
        documents = self.store.search_document_metadata(
            query,
            permissions_enabled=self.config.permissions.enabled,
            active_principals=active_principals,
            limit=limit,
        )
        return ToolOutcome(
            documents=documents,
            event=ToolEvent(
                tool_name="metadata_search",
                status="ok",
                query=query,
                summary=f"Matched {len(documents)} metadata candidates.",
                result_count=len(documents),
                doc_ids=[item.doc_id for item in documents],
            ),
        )

    def get_document_outline(self, doc_id: str) -> ToolOutcome:
        artifact = self.store.get_document_planning_artifact(doc_id)
        outline = artifact.section_outline if artifact else []
        return ToolOutcome(
            event=ToolEvent(
                tool_name="get_document_outline",
                status="ok" if artifact else "missing",
                query=doc_id,
                summary=f"Outline contains {len(outline)} section labels." if artifact else "Planning artifact missing.",
                result_count=len(outline),
                doc_ids=[doc_id] if artifact else [],
            ),
        )

    def expand_section_context(
        self,
        *,
        doc_id: str,
        section_path: str,
        active_principals: list[str],
        limit: int = 4,
    ) -> ToolOutcome:
        chunks = self.store.get_section_context(
            doc_id,
            section_path=section_path,
            permissions_enabled=self.config.permissions.enabled,
            active_principals=active_principals,
            limit=limit,
        )
        return ToolOutcome(
            chunks=chunks,
            event=ToolEvent(
                tool_name="expand_section_context",
                status="ok" if chunks else "empty",
                query=section_path,
                summary=f"Expanded to {len(chunks)} section-adjacent chunks.",
                result_count=len(chunks),
                doc_ids=[doc_id] if chunks else [],
                chunk_ids=[chunk.chunk_id for chunk in chunks],
            ),
        )

    def explain_access(self, query: str, *, active_principals: list[str]) -> ToolOutcome:
        blocked_principals = self.retriever.detect_permission_block(
            query,
            query_type="permission_explanation",
            active_principals=active_principals,
        )
        return ToolOutcome(
            blocked_principals=blocked_principals,
            event=ToolEvent(
                tool_name="explain_access",
                status="blocked" if blocked_principals else "clear",
                query=query,
                summary=(
                    f"Blocked by principals: {', '.join(blocked_principals)}."
                    if blocked_principals
                    else "No permission block detected."
                ),
                result_count=len(blocked_principals),
                doc_ids=[],
            ),
        )

    def collect_evidence_set(
        self,
        query: str,
        *,
        active_principals: list[str],
        doc_ids: list[str] | None = None,
    ) -> ToolOutcome:
        attempt = self.retriever.search(
            query,
            query_type="collect_evidence_set",
            active_principals=active_principals,
            doc_ids=doc_ids,
        )
        return ToolOutcome(
            attempt=attempt,
            event=ToolEvent(
                tool_name="collect_evidence_set",
                status="ok" if attempt.fused_hits else "empty",
                query=query,
                summary=f"Collected {len(attempt.fused_hits)} fused evidence hits.",
                result_count=len(attempt.fused_hits),
                doc_ids=_doc_ids_from_attempt(attempt),
                chunk_ids=[hit.chunk.chunk_id for hit in attempt.fused_hits],
            ),
        )


def _doc_ids_from_attempt(attempt: RetrievalAttempt) -> list[str]:
    seen: list[str] = []
    for hit in attempt.fused_hits:
        if hit.chunk.doc_id in seen:
            continue
        seen.append(hit.chunk.doc_id)
    return seen
