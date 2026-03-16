from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from .agent_tools import AgentToolDispatcher
from .citations import build_citations
from .clients import ChatClient
from .config import AppConfig
from .models import (
    AgentRuntimeStatus,
    AgentTrace,
    AnswerResult,
    PlanStep,
    RetrievalAttempt,
    RetrievalHit,
    ToolEvent,
    VerifierSummary,
)
from .planning_artifacts import PLANNING_ARTIFACT_VERSION
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

STRUCTURAL_AMBIGUITY_PATTERN = re.compile(
    r"\b(?:it|this|that|they|them|those|these|step|steps|next|previous|\d+(?:st|nd|rd|th)\s+step|first\s+step|second\s+step|third\s+step)\b",
    re.I,
)
CAPITALIZED_SUBJECT_PATTERN = re.compile(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2}\b")
DATE_OR_NUMBER_PATTERN = re.compile(
    r"\b(?:\d{1,2}[/-]\d{1,2}(?:[/-]\d{2,4})?|"
    r"\d+(?:\.\d+)?\s*(?:business\s+)?(?:hours?|days?|weeks?|months?)|"
    r"(?:january|february|march|april|may|june|july|august|september|october|november|december)\b(?:\s+\d{1,2})?)",
    re.I,
)


@dataclass(slots=True)
class PlannedSubquestion:
    step_id: str
    title: str
    subquestion: str
    notes: list[str] = field(default_factory=list)


@dataclass(slots=True)
class TransparentRAGAgent:
    config: AppConfig
    retriever: HybridRetriever
    chat_client: ChatClient
    dispatcher: AgentToolDispatcher = field(init=False)

    def __post_init__(self) -> None:
        self.dispatcher = AgentToolDispatcher(
            config=self.config,
            store=self.retriever.store,
            retriever=self.retriever,
        )

    def runtime_status(self) -> AgentRuntimeStatus:
        configured_mode = self._normalized_mode(self.config.agent.mode)
        artifact_status = self.retriever.store.get_planning_artifact_status(
            artifact_version=PLANNING_ARTIFACT_VERSION
        )
        active_mode = configured_mode
        downgrade_reason = None
        if configured_mode == "middleweight" and not artifact_status.available:
            active_mode = "lightweight"
            downgrade_reason = "Middle-weight planning artifacts are missing or outdated. Reindex to activate them."
        return AgentRuntimeStatus(
            configured_mode=configured_mode,
            active_mode=active_mode,
            planning_artifacts_available=artifact_status.available,
            reindex_required_for_middleweight=artifact_status.reindex_required_for_middleweight,
            downgrade_reason=downgrade_reason,
            artifact_version=artifact_status.artifact_version,
        )

    def answer(self, question: str, *, active_principals: list[str]) -> AnswerResult:
        query_type = classify_query(question)
        task_mode = classify_task_mode(question)
        runtime_status = self.runtime_status()
        trace = AgentTrace(
            initial_query=question,
            query_type=query_type,
            rewritten_query=None,
            configured_mode=runtime_status.configured_mode,
            active_mode=runtime_status.active_mode,
            downgrade_reason=runtime_status.downgrade_reason,
            task_mode=task_mode,
            reindex_required_for_middleweight=runtime_status.reindex_required_for_middleweight,
        )

        if runtime_status.active_mode == "lightweight":
            return self._answer_lightweight(question, active_principals=active_principals, trace=trace)
        return self._answer_middleweight(question, active_principals=active_principals, trace=trace)

    def _answer_lightweight(
        self,
        question: str,
        *,
        active_principals: list[str],
        trace: AgentTrace,
    ) -> AnswerResult:
        step = PlanStep(
            step_id="lightweight-hybrid-search",
            title="Run lightweight hybrid retrieval",
            subquestion=question,
            status="running",
            selected_tool="collect_evidence_set",
        )
        trace.plan_steps.append(step)

        first_attempt = self.retriever.search(
            question,
            query_type=trace.query_type,
            active_principals=active_principals,
        )
        trace.attempts.append(first_attempt)
        trace.tool_events.append(
            ToolEvent(
                tool_name="collect_evidence_set",
                status="ok" if first_attempt.fused_hits else "empty",
                query=question,
                summary=f"Collected {len(first_attempt.fused_hits)} fused evidence hits.",
                result_count=len(first_attempt.fused_hits),
                doc_ids=[hit.chunk.doc_id for hit in first_attempt.fused_hits],
                chunk_ids=[hit.chunk.chunk_id for hit in first_attempt.fused_hits],
            )
        )

        best_attempt = first_attempt
        retries_used = 0
        while (
            retries_used < self.config.agent.max_rewrites
            and should_retry(best_attempt, threshold=self.config.retrieval.min_evidence_score)
        ):
            rewritten_query = self._rewrite_query(question, best_attempt)
            trace.rewritten_query = rewritten_query
            if not rewritten_query or rewritten_query == best_attempt.query:
                break
            retries_used += 1
            second_attempt = self.retriever.search(
                rewritten_query,
                query_type=trace.query_type,
                active_principals=active_principals,
            )
            second_attempt.rewritten = True
            trace.attempts.append(second_attempt)
            if second_attempt.evidence_score >= best_attempt.evidence_score:
                best_attempt = second_attempt

        if best_attempt.evidence_score < (self.config.retrieval.min_evidence_score / 2):
            blocked_principals = self.retriever.detect_permission_block(
                best_attempt.query,
                query_type=best_attempt.query_type,
                active_principals=active_principals,
            )
            if blocked_principals:
                trace.verification_notes.append("Matching evidence exists but is blocked by permission filtering.")
                trace.stop_reason = "permission_blocked"
                step.status = "blocked"
                return self._restricted_result(
                    question=question,
                    trace=trace,
                    attempt=best_attempt,
                    blocked_principals=blocked_principals,
                )
            trace.verification_notes.append("Evidence score below abstention threshold.")
            step.status = "empty"
            trace.stop_reason = "no_evidence"
            trace.verifier_summary = VerifierSummary(
                status="no_evidence",
                citation_coverage_ok=False,
                contradiction_detected=False,
                completion_ok=False,
                notes=["Evidence score stayed below the abstention threshold."],
            )
            return self._build_result(
                question=question,
                answer="I couldn't ground a reliable answer in the indexed documents. Try a more specific question or reindex the corpus.",
                grounded=False,
                trace=trace,
                citations=[],
                retrieved_chunks=[hit.chunk for hit in best_attempt.fused_hits],
                status="no_evidence",
                failure_reason="no_evidence",
            )

        try:
            response = self._grounded_answer(question, best_attempt, task_mode=trace.task_mode)
        except RuntimeError as exc:
            trace.verification_notes.append(f"Structured generation failed: {exc}")
            trace.stop_reason = "generation_failure"
            step.status = "generation_failure"
            trace.verifier_summary = VerifierSummary(
                status="generation_failure",
                citation_coverage_ok=False,
                contradiction_detected=False,
                completion_ok=False,
                notes=[str(exc)],
            )
            return self._build_result(
                question=question,
                answer=(
                    "I found related material, but the active local chat model could not produce a structured grounded answer. "
                    "Try a different chat model or inspect the runtime settings."
                ),
                grounded=False,
                trace=trace,
                citations=[],
                retrieved_chunks=[hit.chunk for hit in best_attempt.fused_hits],
                status="generation_failure",
                failure_reason="generation_failure",
            )

        step.status = "completed"
        return self._finalize_model_response(
            question=question,
            trace=trace,
            attempt=best_attempt,
            response=response,
            active_principals=active_principals,
        )

    def _answer_middleweight(
        self,
        question: str,
        *,
        active_principals: list[str],
        trace: AgentTrace,
    ) -> AnswerResult:
        clarification_prompt = build_clarification_prompt(question)
        if clarification_prompt and self.config.agent.clarification_policy != "none":
            access_check = self.dispatcher.explain_access(question, active_principals=active_principals)
            trace.tool_events.append(access_check.event)
            if access_check.blocked_principals:
                trace.stop_reason = "permission_blocked"
                return self._restricted_result(
                    question=question,
                    trace=trace,
                    attempt=RetrievalAttempt(
                        query=question,
                        query_type=trace.query_type,
                        keyword_hits=[],
                        vector_hits=[],
                        fused_hits=[],
                        evidence_score=0.0,
                    ),
                    blocked_principals=access_check.blocked_principals,
                )
            trace.stop_reason = "clarification_required"
            trace.clarification_prompt = clarification_prompt
            trace.plan_steps.append(
                PlanStep(
                    step_id="clarify-scope",
                    title="Clarify the target scope",
                    subquestion=question,
                    status="clarification_required",
                    selected_tool="explain_access",
                    notes=[clarification_prompt],
                )
            )
            trace.verifier_summary = VerifierSummary(
                status="clarification_required",
                citation_coverage_ok=False,
                contradiction_detected=False,
                completion_ok=False,
                notes=["A structural clarification is required before evidence gathering can continue."],
            )
            return self._build_result(
                question=question,
                answer=f"Clarification needed. {clarification_prompt}",
                grounded=False,
                trace=trace,
                citations=[],
                retrieved_chunks=[],
                status="clarification_required",
                failure_reason="clarification_required",
                clarification_prompt=clarification_prompt,
            )

        plan = build_subquestion_plan(
            question,
            task_mode=trace.task_mode,
            max_subquestions=self.config.agent.max_subquestions,
        )
        trace.plan_steps.extend(
            PlanStep(
                step_id=item.step_id,
                title=item.title,
                subquestion=item.subquestion,
                status="pending",
                notes=list(item.notes),
            )
            for item in plan
        )

        tool_calls = 0
        steps_run = 0
        collected_hits: dict[str, RetrievalHit] = {}
        collected_attempts: list[RetrievalAttempt] = []

        for step in trace.plan_steps:
            if steps_run >= self.config.agent.max_steps or tool_calls >= self.config.agent.max_tool_calls:
                step.status = "skipped"
                trace.stop_reason = "tool_budget_exhausted"
                break
            steps_run += 1
            subquestion = step.subquestion or question
            candidate_doc_ids: list[str] = []
            hits_before = len(collected_hits)

            if self._should_run_title_search(subquestion, trace.task_mode):
                title_outcome = self.dispatcher.title_search(subquestion, active_principals=active_principals)
                trace.tool_events.append(title_outcome.event)
                tool_calls += 1
                candidate_doc_ids = _merge_doc_ids(candidate_doc_ids, title_outcome.documents or [])

            if tool_calls < self.config.agent.max_tool_calls and self._should_run_metadata_search(trace.task_mode):
                metadata_outcome = self.dispatcher.metadata_search(subquestion, active_principals=active_principals)
                trace.tool_events.append(metadata_outcome.event)
                tool_calls += 1
                candidate_doc_ids = _merge_doc_ids(candidate_doc_ids, metadata_outcome.documents or [])

            if candidate_doc_ids and tool_calls < self.config.agent.max_tool_calls:
                for doc_id in candidate_doc_ids[:2]:
                    outline_outcome = self.dispatcher.get_document_outline(doc_id)
                    trace.tool_events.append(outline_outcome.event)
                    tool_calls += 1
                    if tool_calls >= self.config.agent.max_tool_calls:
                        break

            evidence_outcome = self.dispatcher.collect_evidence_set(
                subquestion,
                active_principals=active_principals,
                doc_ids=candidate_doc_ids or None,
            )
            trace.tool_events.append(evidence_outcome.event)
            tool_calls += 1
            if evidence_outcome.attempt is not None:
                if subquestion != question and trace.rewritten_query is None:
                    trace.rewritten_query = subquestion
                    evidence_outcome.attempt.rewritten = True
                collected_attempts.append(evidence_outcome.attempt)
                trace.attempts.append(evidence_outcome.attempt)
                _collect_hits(collected_hits, evidence_outcome.attempt.fused_hits)

                if (
                    evidence_outcome.attempt.fused_hits
                    and tool_calls < self.config.agent.max_tool_calls
                    and trace.task_mode != "simple_lookup"
                ):
                    top_hit = evidence_outcome.attempt.fused_hits[0]
                    expanded = self.dispatcher.expand_section_context(
                        doc_id=top_hit.chunk.doc_id,
                        section_path=top_hit.chunk.section_path,
                        active_principals=active_principals,
                    )
                    trace.tool_events.append(expanded.event)
                    tool_calls += 1
                    for chunk in expanded.chunks or []:
                        if chunk.chunk_id not in collected_hits:
                            collected_hits[chunk.chunk_id] = RetrievalHit(
                                chunk=chunk,
                                score=max(top_hit.score - 0.05, 0.01),
                                source="expanded",
                                rank=len(collected_hits) + 1,
                            )

            if (
                evidence_outcome.attempt is not None
                and not evidence_outcome.attempt.fused_hits
                and tool_calls < self.config.agent.max_tool_calls
            ):
                semantic_outcome = self.dispatcher.semantic_search(
                    subquestion,
                    active_principals=active_principals,
                    doc_ids=candidate_doc_ids or None,
                )
                trace.tool_events.append(semantic_outcome.event)
                tool_calls += 1
                if semantic_outcome.attempt is not None:
                    collected_attempts.append(semantic_outcome.attempt)
                    trace.attempts.append(semantic_outcome.attempt)
                    _collect_hits(collected_hits, semantic_outcome.attempt.vector_hits)

            step.selected_tool = "collect_evidence_set"
            step.status = "completed" if len(collected_hits) > hits_before else "empty"

        if not collected_hits:
            blocked_principals = self.retriever.detect_permission_block(
                question,
                query_type=trace.query_type,
                active_principals=active_principals,
            )
            if blocked_principals:
                trace.stop_reason = "permission_blocked"
                return self._restricted_result(
                    question=question,
                    trace=trace,
                    attempt=RetrievalAttempt(
                        query=question,
                        query_type=trace.query_type,
                        keyword_hits=[],
                        vector_hits=[],
                        fused_hits=[],
                        evidence_score=0.0,
                    ),
                    blocked_principals=blocked_principals,
                )
            trace.stop_reason = trace.stop_reason or "no_evidence"
            trace.verifier_summary = VerifierSummary(
                status="no_evidence",
                citation_coverage_ok=False,
                contradiction_detected=False,
                completion_ok=False,
                notes=["The planner finished without collecting enough evidence to answer safely."],
            )
            return self._build_result(
                question=question,
                answer="I couldn't find enough supporting evidence to answer that safely. Try naming the document, subject, or exact fact you need.",
                grounded=False,
                trace=trace,
                citations=[],
                retrieved_chunks=[],
                status="no_evidence",
                failure_reason="no_evidence",
            )

        attempt = self._planned_attempt(question, trace.query_type, collected_attempts, list(collected_hits.values()))
        try:
            response = self._grounded_answer(question, attempt, task_mode=trace.task_mode)
        except RuntimeError as exc:
            trace.verification_notes.append(f"Structured generation failed: {exc}")
            trace.stop_reason = "generation_failure"
            trace.verifier_summary = VerifierSummary(
                status="generation_failure",
                citation_coverage_ok=False,
                contradiction_detected=False,
                completion_ok=False,
                notes=[str(exc)],
            )
            return self._build_result(
                question=question,
                answer=(
                    "I gathered supporting material, but the active local chat model could not produce a structured grounded answer. "
                    "Try a different chat model or inspect the trace."
                ),
                grounded=False,
                trace=trace,
                citations=[],
                retrieved_chunks=[hit.chunk for hit in attempt.fused_hits],
                status="generation_failure",
                failure_reason="generation_failure",
            )

        return self._finalize_model_response(
            question=question,
            trace=trace,
            attempt=attempt,
            response=response,
            active_principals=active_principals,
        )

    def _planned_attempt(
        self,
        question: str,
        query_type: str,
        attempts: list[RetrievalAttempt],
        hits: list[RetrievalHit],
    ) -> RetrievalAttempt:
        ordered_hits = sorted(hits, key=lambda item: item.score, reverse=True)
        fused_hits = [
            RetrievalHit(
                chunk=hit.chunk,
                score=hit.score,
                source="planner",
                rank=index + 1,
            )
            for index, hit in enumerate(ordered_hits[: self.config.retrieval.top_k + 2])
        ]
        evidence_score = max((attempt.evidence_score for attempt in attempts), default=0.0)
        return RetrievalAttempt(
            query=question,
            query_type=query_type,
            keyword_hits=[],
            vector_hits=[],
            fused_hits=fused_hits[: self.config.retrieval.top_k],
            evidence_score=evidence_score,
        )

    def _finalize_model_response(
        self,
        *,
        question: str,
        trace: AgentTrace,
        attempt: RetrievalAttempt,
        response: dict[str, object],
        active_principals: list[str],
    ) -> AnswerResult:
        available_chunks = {hit.chunk.chunk_id: hit.chunk for hit in attempt.fused_hits}
        citation_ids = [item["chunk_id"] for item in response.get("citations", []) if item.get("chunk_id") in available_chunks]
        citation_reasons = {
            item["chunk_id"]: item.get("reason", "")
            for item in response.get("citations", [])
            if item.get("chunk_id") in available_chunks
        }
        if not response.get("grounded") or not citation_ids or not str(response.get("answer", "")).strip():
            fallback_response = self._extractive_fallback_response(
                question=question,
                attempt=attempt,
                task_mode=trace.task_mode,
            )
            if fallback_response is not None:
                trace.verification_notes.append(
                    "Used extractive fallback because the model response did not satisfy the grounded-answer schema."
                )
                response = fallback_response
                citation_ids = [
                    item["chunk_id"] for item in response.get("citations", []) if item.get("chunk_id") in available_chunks
                ]
                citation_reasons = {
                    item["chunk_id"]: item.get("reason", "")
                    for item in response.get("citations", [])
                    if item.get("chunk_id") in available_chunks
                }
        citations = build_citations(citation_ids, available_chunks, citation_reasons)
        citation_chunks = [available_chunks[citation.chunk_id] for citation in citations]
        blocked_principals = self.retriever.detect_permission_block(
            attempt.query,
            query_type=attempt.query_type,
            active_principals=active_principals,
        )

        if not response.get("grounded") or not citations or not str(response.get("answer", "")).strip():
            if blocked_principals:
                trace.verification_notes.append("Matching evidence exists but is blocked by permission filtering.")
                trace.stop_reason = "permission_blocked"
                return self._restricted_result(
                    question=question,
                    trace=trace,
                    attempt=attempt,
                    blocked_principals=blocked_principals,
                )
            trace.stop_reason = "partial_evidence"
            trace.verification_notes.append("The model draft did not satisfy citation or grounding requirements.")
            trace.verifier_summary = VerifierSummary(
                status="partial_evidence",
                citation_coverage_ok=False,
                contradiction_detected=False,
                completion_ok=False,
                notes=["The synthesized draft was missing citations or explicit grounded support."],
            )
            return self._build_result(
                question=question,
                answer="I found related material, but I couldn't verify a fully grounded answer. Try naming the document or narrowing the question.",
                grounded=False,
                trace=trace,
                citations=[],
                retrieved_chunks=[hit.chunk for hit in attempt.fused_hits],
                status="partial_evidence",
                failure_reason="partial_evidence",
            )

        if not has_keyword_grounding(question, citation_chunks):
            if blocked_principals:
                trace.verification_notes.append("Matching evidence exists but is blocked by permission filtering.")
                trace.stop_reason = "permission_blocked"
                return self._restricted_result(
                    question=question,
                    trace=trace,
                    attempt=attempt,
                    blocked_principals=blocked_principals,
                )
            trace.stop_reason = "partial_evidence"
            trace.verification_notes.append("The cited chunks did not overlap enough with the user query.")
            trace.verifier_summary = VerifierSummary(
                status="partial_evidence",
                citation_coverage_ok=False,
                contradiction_detected=False,
                completion_ok=False,
                notes=["The cited evidence did not overlap enough with the request terms."],
            )
            return self._build_result(
                question=question,
                answer="I found related material, but the cited evidence did not line up closely enough with your request to answer safely.",
                grounded=False,
                trace=trace,
                citations=[],
                retrieved_chunks=[hit.chunk for hit in attempt.fused_hits],
                status="partial_evidence",
                failure_reason="partial_evidence",
            )

        if _has_conflicting_evidence(question, citation_chunks, task_mode=trace.task_mode):
            trace.stop_reason = "conflicting_sources"
            trace.verification_notes.append("Cited evidence contains conflicting date or numeric signals.")
            trace.verifier_summary = VerifierSummary(
                status="conflicting_sources",
                citation_coverage_ok=True,
                contradiction_detected=True,
                completion_ok=False,
                notes=["The evidence set appears to contain multiple conflicting factual variants."],
            )
            return self._build_result(
                question=question,
                answer="I found conflicting evidence across the matching documents, so I withheld a definitive answer. Inspect the trace and citations to compare the conflicting sources.",
                grounded=False,
                trace=trace,
                citations=citations,
                retrieved_chunks=[hit.chunk for hit in attempt.fused_hits],
                status="conflicting_sources",
                failure_reason="conflicting_sources",
            )

        if trace.task_mode in {"comparison", "cross_document_analysis"}:
            cited_doc_count = len({chunk.doc_id for chunk in citation_chunks})
            if cited_doc_count < 2:
                trace.stop_reason = "partial_evidence"
                trace.verification_notes.append("Cross-document task cited fewer than two source documents.")
                trace.verifier_summary = VerifierSummary(
                    status="partial_evidence",
                    citation_coverage_ok=True,
                    contradiction_detected=False,
                    completion_ok=False,
                    notes=["Cross-document analysis did not gather evidence from enough distinct documents."],
                )
                return self._build_result(
                    question=question,
                    answer="I found related material, but not enough distinct document evidence to finish the cross-document analysis confidently.",
                    grounded=False,
                    trace=trace,
                    citations=citations,
                    retrieved_chunks=[hit.chunk for hit in attempt.fused_hits],
                    status="partial_evidence",
                    failure_reason="partial_evidence",
                )

        if trace.task_mode == "timeline" and not any(DATE_OR_NUMBER_PATTERN.search(chunk.text) for chunk in citation_chunks):
            trace.stop_reason = "partial_evidence"
            trace.verification_notes.append("Timeline task lacked explicit date or sequencing evidence.")
            trace.verifier_summary = VerifierSummary(
                status="partial_evidence",
                citation_coverage_ok=True,
                contradiction_detected=False,
                completion_ok=False,
                notes=["Timeline mode needs dated or ordered evidence to complete safely."],
            )
            return self._build_result(
                question=question,
                answer="I found related material, but not enough dated or ordered evidence to build a reliable timeline.",
                grounded=False,
                trace=trace,
                citations=citations,
                retrieved_chunks=[hit.chunk for hit in attempt.fused_hits],
                status="partial_evidence",
                failure_reason="partial_evidence",
            )

        trace.stop_reason = trace.stop_reason or "answer_grounded"
        trace.verification_notes.append("Answer grounded with citations from retrieved chunks.")
        trace.verifier_summary = VerifierSummary(
            status="grounded",
            citation_coverage_ok=True,
            contradiction_detected=False,
            completion_ok=True,
            notes=["The answer satisfied the grounding and citation checks."],
        )
        return self._build_result(
            question=question,
            answer=str(response["answer"]).strip(),
            grounded=True,
            trace=trace,
            citations=citations,
            retrieved_chunks=[hit.chunk for hit in attempt.fused_hits],
            status="grounded",
        )

    def _restricted_result(
        self,
        *,
        question: str,
        trace: AgentTrace,
        attempt: RetrievalAttempt,
        blocked_principals: list[str],
    ) -> AnswerResult:
        principal_hint = ", ".join(blocked_principals)
        next_step = (
            f"Try Access view > Custom and enable: {principal_hint}."
            if principal_hint
            else "Try an access view with the required principals."
        )
        trace.verifier_summary = VerifierSummary(
            status="restricted",
            citation_coverage_ok=False,
            contradiction_detected=False,
            completion_ok=False,
            notes=["Matching evidence is present, but it is blocked by the current access view."],
        )
        return self._build_result(
            question=question,
            answer=(
                "Restricted. Your current access view does not allow one or more matching documents. "
                f"{next_step}"
            ),
            grounded=False,
            trace=trace,
            citations=[],
            retrieved_chunks=[hit.chunk for hit in attempt.fused_hits],
            status="restricted",
            failure_reason="restricted",
            blocked_principals=blocked_principals,
        )

    def _rewrite_query(self, question: str, attempt: RetrievalAttempt) -> str:
        context_lines = [f"- {hit.chunk.title}: {hit.chunk.section_path}" for hit in attempt.fused_hits[:5]]
        try:
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
        except RuntimeError:
            return question
        rewritten = str(payload.get("rewritten_query", "")).strip()
        return rewritten or question

    def _grounded_answer(self, question: str, attempt: RetrievalAttempt, *, task_mode: str) -> dict[str, object]:
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
                "Return JSON with keys answer, grounded, and citations only. "
                "citations must be a list of objects with chunk_id and reason. "
                "If evidence is insufficient, grounded must be false. "
                "Do not echo task_mode, question, or evidence back to the user."
            ),
            user_prompt=(
                "Task mode:\n"
                f"{task_mode}\n\n"
                "Question:\n"
                f"{question}\n\n"
                "Evidence:\n"
                f"{json.dumps(context_blocks, indent=2)}\n\n"
                "Answer using only the evidence. Cite every important claim."
            ),
        )
        return payload

    def _extractive_fallback_response(
        self,
        *,
        question: str,
        attempt: RetrievalAttempt,
        task_mode: str,
    ) -> dict[str, object] | None:
        if task_mode not in {"simple_lookup", "ownership_policy"}:
            return None
        keywords = _question_keywords(question)
        if not keywords:
            return None

        best_candidate: tuple[float, RetrievalHit, str] | None = None
        required_overlap = 1 if len(keywords) <= 3 else 2
        question_lower = question.lower().strip()
        for hit in attempt.fused_hits[:3]:
            for sentence in _candidate_sentences(hit.chunk.text):
                sentence_lower = sentence.lower()
                overlap = len({keyword for keyword in keywords if keyword in sentence_lower})
                if overlap < required_overlap:
                    continue
                score = float(overlap * 10 - hit.rank)
                if question_lower.startswith("when") and DATE_OR_NUMBER_PATTERN.search(sentence):
                    score += 6
                if question_lower.startswith("who") and re.search(r"\b(?:is|owner|lead|manager|responsible)\b", sentence_lower):
                    score += 4
                if question_lower.startswith("what") and re.search(
                    r"\b(?:is|are|receive|receives|within|costs|planned)\b",
                    sentence_lower,
                ):
                    score += 2
                if hit.rank == 1:
                    score += 2
                if best_candidate is None or score > best_candidate[0]:
                    best_candidate = (score, hit, sentence.strip())

        if best_candidate is None:
            return None

        best_score, best_hit, best_sentence = best_candidate
        minimum_score = 14 if len(keywords) > 4 else 10
        if best_score < minimum_score:
            return None

        return {
            "answer": best_sentence,
            "grounded": True,
            "citations": [
                {
                    "chunk_id": best_hit.chunk.chunk_id,
                    "reason": "Extractive fallback selected the best-matching evidence sentence.",
                }
            ],
        }

    def _build_result(
        self,
        *,
        question: str,
        answer: str,
        grounded: bool,
        trace: AgentTrace,
        citations,
        retrieved_chunks,
        status: str,
        failure_reason: str | None = None,
        blocked_principals: list[str] | None = None,
        clarification_prompt: str | None = None,
    ) -> AnswerResult:
        return AnswerResult(
            question=question,
            answer=answer,
            grounded=grounded,
            citations=list(citations),
            trace=trace,
            retrieved_chunks=list(retrieved_chunks),
            status=status,
            blocked_principals=list(blocked_principals or []),
            task_mode=trace.task_mode,
            failure_reason=failure_reason,
            clarification_prompt=clarification_prompt or trace.clarification_prompt,
            plan_summary=list(trace.plan_steps),
            tool_events=list(trace.tool_events),
            stop_reason=trace.stop_reason,
            verifier_summary=trace.verifier_summary,
            configured_mode=trace.configured_mode,
            active_mode=trace.active_mode,
            downgrade_reason=trace.downgrade_reason,
            reindex_required_for_middleweight=trace.reindex_required_for_middleweight,
        )

    def _normalized_mode(self, mode: str) -> str:
        return "middleweight" if str(mode).strip().lower() == "middleweight" else "lightweight"

    def _should_run_title_search(self, question: str, task_mode: str) -> bool:
        return bool(extract_named_subjects(question)) or '"' in question or task_mode in {"comparison", "timeline"}

    def _should_run_metadata_search(self, task_mode: str) -> bool:
        return task_mode in {"comparison", "cross_document_analysis", "timeline", "ownership_policy"}


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


def classify_task_mode(question: str) -> str:
    normalized = question.strip().lower()
    if not normalized:
        return "simple_lookup"
    if any(
        phrase in normalized
        for phrase in ["tell me about", "what should i know", "give me an overview", "summarize", "anything about"]
    ):
        return "cross_document_analysis"
    if any(token in normalized for token in ["permission", "access view", "allowed", "restricted", "who can see"]):
        return "permission_explanation"
    if any(token in normalized for token in ["compare", "versus", "vs", "difference", "between"]):
        return "comparison"
    if any(token in normalized for token in ["timeline", "history", "chronology", "sequence", "ordered"]):
        return "timeline"
    if any(token in normalized for token in ["owner", "owns", "responsible", "policy", "deadline", "due", "postmortem"]):
        return "ownership_policy"
    if len(normalized.split()) <= 4 or build_clarification_prompt(question):
        return "clarification_required"
    if any(token in normalized for token in ["across", "summarize", "all documents", "cross-document"]):
        return "cross_document_analysis"
    return "simple_lookup"


def should_retry(attempt: RetrievalAttempt, *, threshold: float) -> bool:
    return attempt.evidence_score < threshold or attempt.query_type in {"ambiguous", "broad"}


def has_keyword_grounding(question: str, citation_chunks) -> bool:
    keywords = _question_keywords(question)
    if not keywords:
        return True
    combined_text = " ".join(chunk.text.lower() for chunk in citation_chunks)
    overlap = {keyword for keyword in keywords if keyword in combined_text}
    required_overlap = 1 if len(keywords) <= 3 else 2
    return len(overlap) >= required_overlap


def build_subquestion_plan(question: str, *, task_mode: str, max_subquestions: int) -> list[PlannedSubquestion]:
    if task_mode == "comparison":
        subjects = extract_named_subjects(question)
        items = [
            PlannedSubquestion(
                step_id=f"compare-{index + 1}",
                title=f"Gather evidence for {subject}",
                subquestion=f"What does the corpus say about {subject}?",
            )
            for index, subject in enumerate(subjects[: max_subquestions - 1] or ["the first side", "the second side"])
        ]
        items.append(
            PlannedSubquestion(
                step_id="compare-summary",
                title="Compare the matching evidence",
                subquestion=question,
                notes=["Use evidence from multiple documents if available."],
            )
        )
        return items[:max_subquestions]
    if task_mode == "timeline":
        focus = condensed_focus_query(question)
        return [
            PlannedSubquestion(
                step_id="timeline-events",
                title="Gather dated events",
                subquestion=f"What dated or ordered events are documented about {focus}?",
            ),
            PlannedSubquestion(
                step_id="timeline-sequence",
                title="Assemble the sequence",
                subquestion=question,
            ),
        ][:max_subquestions]
    if task_mode == "ownership_policy":
        focus = condensed_focus_query(question)
        return [
            PlannedSubquestion(
                step_id="ownership-lookup",
                title="Find ownership or policy statements",
                subquestion=question,
            ),
            PlannedSubquestion(
                step_id="ownership-deadline",
                title="Find due dates or related policy details",
                subquestion=f"What deadlines, due dates, or policy conditions are documented for {focus}?",
            ),
        ][:max_subquestions]
    if task_mode == "cross_document_analysis":
        focus = condensed_focus_query(question)
        return [
            PlannedSubquestion(
                step_id="crossdoc-broad",
                title="Gather the best matching evidence",
                subquestion=question,
            ),
            PlannedSubquestion(
                step_id="crossdoc-focus",
                title="Gather supporting detail",
                subquestion=focus,
            ),
        ][:max_subquestions]
    return [
        PlannedSubquestion(
            step_id="lookup-primary",
            title="Gather the primary evidence",
            subquestion=question,
        )
    ]


def build_clarification_prompt(question: str) -> str | None:
    normalized = question.strip()
    if not normalized:
        return "What would you like me to look up?"
    word_count = len(normalized.split())
    meaningful_tokens = [
        token
        for token in re.findall(r"[a-z0-9]+", normalized.lower())
        if token not in GROUNDING_STOPWORDS and len(token) > 2
    ]
    if word_count <= 4 and len(meaningful_tokens) <= 1:
        return "Which document, subject, or procedure are you referring to?"
    if STRUCTURAL_AMBIGUITY_PATTERN.search(normalized) and not extract_named_subjects(question):
        return "Which document or named subject should I use for that request?"
    return None


def condensed_focus_query(question: str) -> str:
    keywords = [
        token
        for token in re.findall(r"[a-z0-9]+", question.lower())
        if token not in GROUNDING_STOPWORDS and len(token) > 2
    ]
    if not keywords:
        return question
    return " ".join(keywords[:6])


def extract_named_subjects(question: str) -> list[str]:
    quoted = [item.strip() for item in re.findall(r'"([^"]+)"', question) if item.strip()]
    capitalized = [item.strip() for item in CAPITALIZED_SUBJECT_PATTERN.findall(question) if item.strip()]
    ordered: list[str] = []
    for item in [*quoted, *capitalized]:
        if item in ordered:
            continue
        ordered.append(item)
    return ordered


def _collect_hits(target: dict[str, RetrievalHit], hits: list[RetrievalHit]) -> None:
    for hit in hits:
        existing = target.get(hit.chunk.chunk_id)
        if existing is None or hit.score > existing.score:
            target[hit.chunk.chunk_id] = hit


def _question_keywords(question: str) -> list[str]:
    return [
        token
        for token in re.findall(r"[a-z0-9]+", question.lower())
        if token not in GROUNDING_STOPWORDS and len(token) > 2
    ]


def _candidate_sentences(text: str) -> list[str]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    sentences: list[str] = []
    for line in lines:
        parts = [item.strip() for item in re.split(r"(?<=[.!?])\s+", line) if item.strip()]
        sentences.extend(parts or [line])
    return sentences


def _merge_doc_ids(existing: list[str], hits) -> list[str]:
    merged = list(existing)
    for item in hits:
        if item.doc_id not in merged:
            merged.append(item.doc_id)
    return merged


def _has_conflicting_evidence(question: str, citation_chunks, *, task_mode: str) -> bool:
    if task_mode == "comparison":
        return False
    question_keywords = {
        token
        for token in re.findall(r"[a-z0-9]+", question.lower())
        if token not in GROUNDING_STOPWORDS and len(token) > 2
    }
    signals: set[str] = set()
    signal_docs: set[str] = set()
    for chunk in citation_chunks:
        for sentence in re.split(r"(?<=[.!?])\s+", chunk.text):
            lowered = sentence.lower()
            if question_keywords and not any(keyword in lowered for keyword in question_keywords):
                continue
            matches = DATE_OR_NUMBER_PATTERN.findall(sentence)
            for match in matches:
                signals.add(match.strip().lower())
                signal_docs.add(chunk.doc_id)
    return len(signals) > 1 and len(signal_docs) > 1
