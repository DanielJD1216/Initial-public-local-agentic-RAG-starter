import pytest

from local_agentic_rag.models import ChunkRecord, DocumentMetadata


def test_document_metadata_requires_access_principals() -> None:
    metadata = DocumentMetadata(
        doc_id="doc-test",
        source_path="/tmp/test.md",
        content_type="text/markdown",
        checksum="abc123",
        parser_version="v1",
        title="Test Doc",
        ingested_at="2026-03-12T00:00:00+00:00",
        access_scope="public",
        access_principals=[],
        file_size_bytes=100,
        modified_at="2026-03-12T00:00:00+00:00",
        ingest_mode="local",
        ingest_model="local-heuristic",
        ingest_fingerprint="fingerprint",
        chunking_strategy="heuristic",
    )
    with pytest.raises(ValueError):
        metadata.validate()


def test_chunk_metadata_requires_location_label() -> None:
    chunk = ChunkRecord(
        chunk_id="chunk-1",
        doc_id="doc-1",
        source_path="/tmp/test.md",
        content_type="text/plain",
        checksum="abc123",
        parser_version="v1",
        title="Doc",
        ingested_at="2026-03-12T00:00:00+00:00",
        access_scope="public",
        access_principals=["*"],
        chunk_index=0,
        section_path="Body",
        text="hello",
        location_label="",
    )
    with pytest.raises(ValueError):
        chunk.validate()
