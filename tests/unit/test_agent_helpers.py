from local_agentic_rag.agent import classify_query, has_keyword_grounding, should_retry
from local_agentic_rag.models import ChunkRecord, RetrievalAttempt


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
