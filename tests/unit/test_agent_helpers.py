from local_agentic_rag.agent import TransparentRAGAgent, classify_query, has_keyword_grounding, should_retry
from local_agentic_rag.models import ChunkRecord, RetrievalAttempt
from tests.conftest import build_test_runtime


def test_query_classification_covers_broad_and_multi_hop() -> None:
    assert classify_query("Tell me about support") == "ambiguous"
    assert classify_query("Compare support coverage and postmortem deadlines") == "multi_hop"


def test_retry_when_evidence_is_low() -> None:
    attempt = RetrievalAttempt(
        query="support",
        query_type="ambiguous",
        keyword_hits=[],
        vector_hits=[],
        fused_hits=[],
        evidence_score=0.1,
    )
    assert should_retry(attempt, threshold=0.3)


def test_keyword_grounding_rejects_irrelevant_citations() -> None:
    chunk = ChunkRecord(
        chunk_id="chunk-1",
        doc_id="doc-1",
        source_path="/tmp/handbook.md",
        content_type="text/markdown",
        checksum="checksum",
        parser_version="v1",
        title="Handbook",
        ingested_at="2026-03-12T00:00:00+00:00",
        access_scope="public",
        access_principals=["*"],
        chunk_index=0,
        section_path="Escalation",
        text="Escalation updates should be sent every 2 hours until resolution.",
        location_label="Escalation | lines 1-2",
    )
    assert not has_keyword_grounding("When is the salary adjustment review window planned?", [chunk])


class FailingChatClient:
    def chat_json(self, *, system_prompt: str, user_prompt: str):  # type: ignore[no-untyped-def]
        raise RuntimeError("model returned invalid json")


class EchoingAnswerChatClient:
    def chat_json(self, *, system_prompt: str, user_prompt: str):  # type: ignore[no-untyped-def]
        if "rewrite weak retrieval queries" in system_prompt:
            return {"rewritten_query": _extract_question(user_prompt), "reason": "no rewrite"}
        return {
            "task_mode": "simple_lookup",
            "question": _extract_question(user_prompt),
            "evidence": _extract_evidence(user_prompt),
        }


def test_agent_gracefully_abstains_when_chat_json_fails(tmp_path) -> None:
    runtime, _config_path, _docs_path = build_test_runtime(tmp_path)
    runtime.ingestion.ingest(prune_missing=False)
    agent = TransparentRAGAgent(
        config=runtime.config,
        retriever=runtime.retriever,
        chat_client=FailingChatClient(),
    )

    result = agent.answer("What is the standard support first response time?", active_principals=["*"])
    assert not result.grounded
    assert result.status == "generation_failure"
    assert result.failure_reason == "generation_failure"
    assert "could not produce a structured grounded answer" in result.answer


def test_agent_uses_extractive_fallback_when_model_returns_wrong_schema(tmp_path) -> None:
    runtime, _config_path, _docs_path = build_test_runtime(tmp_path, permissions_enabled=True)
    runtime.ingestion.ingest(prune_missing=False)
    agent = TransparentRAGAgent(
        config=runtime.config,
        retriever=runtime.retriever,
        chat_client=EchoingAnswerChatClient(),
    )

    result = agent.answer("When is the salary adjustment review window planned?", active_principals=["owners"])

    assert result.grounded
    assert result.status == "grounded"
    assert "third week of June" in result.answer
    assert result.citations
    assert result.citations[0].chunk_id == "doc-restricted-internal-roadmap-md-chunk-0000"


def test_agent_requests_clarification_for_structural_ambiguity(tmp_path) -> None:
    runtime, _config_path, _docs_path = build_test_runtime(tmp_path)
    runtime.ingestion.ingest(prune_missing=False)

    result = runtime.agent.answer("what's the 3rd step to take?", active_principals=["*"])

    assert not result.grounded
    assert result.status == "clarification_required"
    assert result.failure_reason == "clarification_required"
    assert result.clarification_prompt is not None


def _extract_question(user_prompt: str) -> str:
    marker = "Question:\n"
    if marker not in user_prompt:
        return user_prompt.strip()
    return user_prompt.split(marker, 1)[1].split("\n\n", 1)[0].strip()


def _extract_evidence(user_prompt: str):  # type: ignore[no-untyped-def]
    marker = "Evidence:\n"
    if marker not in user_prompt:
        return []
    raw_payload = user_prompt.split(marker, 1)[1].split("\n\nAnswer", 1)[0]
    import json

    return json.loads(raw_payload)
