from local_agentic_rag.chunking import build_chunks
from local_agentic_rag.models import DocumentMetadata, ParsedSection


def test_chunking_respects_section_boundaries() -> None:
    metadata = DocumentMetadata(
        doc_id="doc-test",
        source_path="/tmp/test.md",
        content_type="text/markdown",
        checksum="abc123",
        parser_version="v1",
        title="Test Doc",
        ingested_at="2026-03-12T00:00:00+00:00",
        access_scope="public",
        access_principals=["*"],
        file_size_bytes=100,
        modified_at="2026-03-12T00:00:00+00:00",
    )
    section = ParsedSection(
        text="Paragraph one.\n\nParagraph two with more words.\n\nParagraph three wraps this up.",
        section_path="Overview",
        line_start=1,
        line_end=9,
    )
    chunks = build_chunks(metadata, [section], max_chunk_tokens=8, overlap_tokens=2)
    assert len(chunks) >= 2
    assert all(chunk.section_path == "Overview" for chunk in chunks)
    assert all("Overview" in chunk.location_label for chunk in chunks)
