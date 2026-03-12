from local_agentic_rag.citations import format_citation
from local_agentic_rag.models import ChunkRecord


def test_citation_format_uses_title_and_location() -> None:
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
        text="The support lead is Maya Chen.",
        location_label="Escalation | lines 1-2",
    )
    assert format_citation(chunk) == "Handbook — Escalation | lines 1-2"
