from __future__ import annotations

import json
import re
from dataclasses import dataclass

from .citations import build_citations
from .clients import ChatClient
from .config import AppConfig
from .models import AgentTrace, AnswerResult, RetrievalAttempt
from .retrieval import HybridRetriever

GROUNDING_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "for",
    "how",
    "is",
    "me",
    "of",
    "or",
    "the",
    "to",
    "what",
    "when",
    "who",
}


@dataclass(slots=True)
class TransparentRAGAgent:
    config: AppConfig
    retriever: HybridRetriever
    chat_client: ChatClient

    def answer(self, question: str, *, active_principals: list[str]) -> AnswerResult:
        query_type = classify_query(question)
        trace = AgentTrace(initial_query=question, query_type=query_type, rewritten_query=None)

        first_attempt = self.retriever.search(question, query_type=query_type, active_principals=active_principals)
        trace.attempts.append(first_attempt)

        best_attempt = first_attempt
        if should_retry(first_attempt, threshold=self.config.retrieval.min_evidence_score):
            rewritten_query = self._rewrite_query(question, first_attempt)
            trace.rewritten_query = rewritten_query
            if rewritten_query and rewritten_query != question:
                second_attempt = self.retriever.search(
                    rewritten_query,
                    query_type=query_type,
                    active_principals=active_principals,
                )
                second_attempt.rewritten = True
                trace.attempts.append(second_attempt)
                if second_attempt.evidence_score >= first_attempt.evidence_score:
                    best_attempt = second_attempt

        if best_attempt.evidence_score < (self.config.retrieval.min_evidence_score / 2):
            trace.verification_notes.append("Evidence score below abstention threshold.")
            return AnswerResult(
                question=question,
                answer="I couldn't ground a reliable answer in the indexed documents. Try a more specific question or reindex the corpus.",
                grounded=False,
                citations=[],
                trace=trace,
                retrieved_chunks=[hit.chunk for hit in best_attempt.fused_hits],
                status="abstained",
            )

        response = self._grounded_answer(question, best_attempt)
        available_chunks = {hit.chunk.chunk_id: hit.chunk for hit in best_attempt.fused_hits}
        citation_ids = [item["chunk_id"] for item in response.get("citations", []) if item.get("chunk_id") in available_chunks]
        citation_reasons = {
            item["chunk_id"]: item.get("reason", "")
            for item in response.get("citations", [])
            if item.get("chunk_id") in available_chunks
        }
        citations = build_citations(citation_ids, available_chunks, citation_reasons)
        citation_chunks = [available_chunks[citation.chunk_id] for citation in citations]
        grounded = (
            bool(response.get("grounded"))
            and bool(citations)
            and bool(response.get("answer", "").strip())
            and has_keyword_grounding(question, citation_chunks)
        )
        if not grounded:
            trace.verification_notes.append("Model response was not sufficiently grounded or cited.")
            return AnswerResult(
                question=question,
                answer="I found related material, but I couldn't verify a fully grounded answer. Try narrowing the question or inspecting the trace.",
                grounded=False,
                citations=[],
                trace=trace,
                retrieved_chunks=[hit.chunk for hit in best_attempt.fused_hits],
                status="abstained",
            )

        trace.verification_notes.append("Answer grounded with citations from retrieved chunks.")
        return AnswerResult(
            question=question,
            answer=response["answer"].strip(),
            grounded=True,
            citations=citations,
            trace=trace,
            retrieved_chunks=[hit.chunk for hit in best_attempt.fused_hits],
        )

    def _rewrite_query(self, question: str, attempt: RetrievalAttempt) -> str:
        context_lines = [f"- {hit.chunk.title}: {hit.chunk.section_path}" for hit in attempt.fused_hits[:5]]
        payload = self.chat_client.chat_json(
            system_prompt=(
                "You rewrite weak retrieval queries for document search. "
                "Return JSON with keys rewritten_query and reason."
            ),
            user_prompt=(
                "Question:\n"
                f"{question}\n\n"
                "Retrieved context titles:\n"
                f"{chr(10).join(context_lines) if context_lines else '- none'}\n\n"
                "Rewrite the question into one sharper search query. "
                "Keep proper nouns and dates. If no rewrite helps, repeat the original."
            ),
        )
        rewritten = str(payload.get("rewritten_query", "")).strip()
        return rewritten or question

    def _grounded_answer(self, question: str, attempt: RetrievalAttempt) -> dict[str, object]:
        context_blocks = []
        for hit in attempt.fused_hits:
            context_blocks.append(
                {
                    "chunk_id": hit.chunk.chunk_id,
                    "citation": hit.chunk.citation_label(),
                    "text": hit.chunk.text,
                }
            )
        payload = self.chat_client.chat_json(
            system_prompt=(
                "You answer only from the provided evidence. "
                "Return JSON with keys answer, grounded, and citations. "
                "citations must be a list of objects with chunk_id and reason. "
                "If evidence is insufficient, grounded must be false."
            ),
            user_prompt=(
                "Question:\n"
                f"{question}\n\n"
                "Evidence:\n"
                f"{json.dumps(context_blocks, indent=2)}\n\n"
                "Answer using only the evidence. Cite every important claim."
            ),
        )
        return payload


def classify_query(question: str) -> str:
    normalized = question.strip().lower()
    if len(normalized.split()) <= 4:
        return "ambiguous"
    if any(token in normalized for token in ["compare", "versus", "difference", "across", "between"]):
        return "multi_hop"
    if any(
        phrase in normalized
        for phrase in ["tell me about", "what should i know", "give me an overview", "summarize", "anything about"]
    ):
        return "broad"
    if re.search(r"\b(and|or)\b", normalized) and len(normalized.split()) > 10:
        return "multi_hop"
    return "simple"


def should_retry(attempt: RetrievalAttempt, *, threshold: float) -> bool:
    return attempt.evidence_score < threshold or attempt.query_type in {"ambiguous", "broad"}


def has_keyword_grounding(question: str, citation_chunks) -> bool:
    keywords = [
        token
        for token in re.findall(r"[a-z0-9]+", question.lower())
        if token not in GROUNDING_STOPWORDS and len(token) > 2
    ]
    if not keywords:
        return True
    combined_text = " ".join(chunk.text.lower() for chunk in citation_chunks)
    overlap = {keyword for keyword in keywords if keyword in combined_text}
    required_overlap = 1 if len(keywords) <= 3 else 2
    return len(overlap) >= required_overlap
