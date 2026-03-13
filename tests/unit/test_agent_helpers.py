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
    assert result.status == "generation_error"
    assert "could not produce a structured grounded answer" in result.answer
